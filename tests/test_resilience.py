"""Mojin 客服模块健壮性测试 — 并发 / 异常 / 兜底 / 懒加载 / 边界。

全部用例不依赖外网、不依赖 model server（fake embed / 假 LLM / 临时 Milvus Lite）。
参照 tests/test_mojin_chat.py 的 fake/临时库模式；本模块独立设置，不受其 skipif 影响。
"""

import hashlib
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
from rag_toolkit.storage.milvus_manager import MilvusManager

from app.knowledge.config import kb_settings
from app.knowledge.loader import ensure_collection, load_knowledge
from app.routers import mojin_chat as mc

try:
    from pymilvus import MilvusClient
except ImportError:  # pragma: no cover
    MilvusClient = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════
#  公共 fake 组件
# ═══════════════════════════════════════════════════════════════════

def _fake_embed(texts):
    """确定性 fake embed（1024 维），不依赖 model server。"""
    out = []
    for t in texts:
        seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
        vec = [float((seed >> (i % 32)) & 1) * 0.1 + 0.01 for i in range(1024)]
        out.append(vec)
    return out


class _FakeMultiQuery:
    """假 MultiQueryGenerator：确定性改写，不调 DeepSeek。"""

    def generate(self, question, count=3):
        return [f"{question} (expanded 1)", f"{question} (expanded 2)"]


class _FakeReranker:
    """假 XReranker：确定性分数（足够高于 RERANK_MIN_SCORE=0.15），不调 model server。"""

    def rerank(self, documents, query):
        return [(i, 0.9 - i * 0.01) for i in range(len(documents))]


# ═══════════════════════════════════════════════════════════════════
#  fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_env(monkeypatch):
    """注入 fake embed / multi-query / reranker（manager 与 llm 由用例自定）。"""
    monkeypatch.setattr(mc, "_embed", _fake_embed)
    monkeypatch.setattr(mc, "_get_multi_query", lambda: _FakeMultiQuery())
    monkeypatch.setattr(mc, "_get_reranker", lambda: _FakeReranker())


@pytest.fixture
def kb_manager(tmp_path):
    """临时 Milvus：en 语言 kb 集合（FAQ + 政策，fake embed 摄入）。"""
    mgr = MilvusManager(client=MilvusClient(uri=str(tmp_path / "kb.db")))
    load_knowledge(manager=mgr, embed_fn=_fake_embed, langs=["en"], scope="kb", rebuild=True)
    return mgr


@pytest.fixture
def empty_manager(tmp_path):
    """临时 Milvus：空库（无任何集合）。"""
    return MilvusManager(client=MilvusClient(uri=str(tmp_path / "empty.db")))


def _use_manager(monkeypatch, mgr):
    monkeypatch.setattr(mc, "_get_manager", lambda: mgr)
    return mgr


# ═══════════════════════════════════════════════════════════════════
#  a. 并发稳定性
# ═══════════════════════════════════════════════════════════════════

def test_concurrent_answer_calls(fake_env, monkeypatch, kb_manager):
    """并发 5 个 _run_answer（线程池 + fake embed + 假 LLM + 临时 Milvus），
    全部返回结构合法、无异常泄漏。"""
    _use_manager(monkeypatch, kb_manager)
    monkeypatch.setattr(mc, "_llm_chat", lambda msgs: "ok")

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(mc._run_answer, "shipping policy", "en") for _ in range(5)]
        results = [f.result(timeout=60) for f in futures]  # 任一异常都会在此泄漏

    assert len(results) == 5
    for r in results:
        assert set(r) >= {"answer", "sources", "hit_count", "elapsed_ms"}
        assert r["hit_count"] >= 0
        assert isinstance(r["elapsed_ms"], int)
    assert all(r["answer"] for r in results)


# ═══════════════════════════════════════════════════════════════════
#  b. model server 不可达
# ═══════════════════════════════════════════════════════════════════

def test_embed_failure_returns_friendly_error_http(fake_env, monkeypatch, empty_manager, client):
    """model server 不可达（_embed 抛 requests 异常）→ HTTP 端点返回友好错误 + error 字段，不抛 500。"""
    import requests

    _use_manager(monkeypatch, empty_manager)

    def boom(texts):
        raise requests.ConnectionError("model server down")

    monkeypatch.setattr(mc, "_embed", boom)

    resp = client.get("/mojin_chat/answer_question", params={"query": "hi", "lang": "en"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == "retrieval service unavailable"
    assert data["hit_count"] == 0
    assert "WhatsApp" in data["answer"]  # 转人工引导


def test_embed_failure_direct_call(fake_env, monkeypatch, empty_manager):
    """同场景直接调 _run_answer：结构合法、无异常。"""
    import requests

    _use_manager(monkeypatch, empty_manager)

    def boom(texts):
        raise requests.ConnectionError("model server down")

    monkeypatch.setattr(mc, "_embed", boom)

    r = mc._run_answer("hi", "en")
    assert r["error"] == "retrieval service unavailable"
    assert set(r) >= {"answer", "sources", "hit_count", "elapsed_ms"}


# ═══════════════════════════════════════════════════════════════════
#  c. DeepSeek 不可达（假 llm 抛异常）
# ═══════════════════════════════════════════════════════════════════

def test_llm_failure_returns_handoff(fake_env, monkeypatch, kb_manager):
    """知识库有命中但 LLM 抛异常 → _friendly_error 兜底，answer 含转人工引导。"""
    _use_manager(monkeypatch, kb_manager)

    def boom(msgs):
        raise RuntimeError("deepseek down")

    monkeypatch.setattr(mc, "_llm_chat", boom)

    r = mc._run_answer("shipping policy", "en")
    assert r["error"] == "llm service unavailable"
    assert r["hit_count"] == 0
    assert "WhatsApp" in r["answer"]
    assert "internal service error" in r["answer"].lower()


# ═══════════════════════════════════════════════════════════════════
#  d. 懒加载幂等
# ═══════════════════════════════════════════════════════════════════

def test_lazy_clients_instantiated_once(monkeypatch, tmp_path):
    """_clients 中 manager / llm / reranker 连续两次调用只实例化一次，且返回同一实例。"""
    from unittest import mock

    monkeypatch.setattr(mc, "_clients", {})
    # kb_settings 是 pydantic 单例，禁止任意 setattr，改为替换模块引用
    monkeypatch.setattr(mc, "kb_settings", types.SimpleNamespace(
        MILVUS_URI=str(tmp_path / "lazy.db"),
        require_llm_key=lambda: "test-key",
        LLM_URL="http://llm.test/v1",
        RERANKER_URL="http://rerank.test",
        RERANKER_MODEL="bge-reranker-v2-m3",
        RERANK_TOP_K=5,
    ))

    with (
        mock.patch("rag_toolkit.storage.milvus_manager.MilvusManager") as mgr_cls,
        mock.patch("pymilvus.MilvusClient") as client_cls,
        mock.patch("openai.OpenAI") as llm_cls,
        mock.patch("rag_toolkit.pipelines.reranker.XReranker") as reranker_cls,
    ):
        mgr1, mgr2 = mc._get_manager(), mc._get_manager()
        llm1, llm2 = mc._get_llm(), mc._get_llm()
        rrk1, rrk2 = mc._get_reranker(), mc._get_reranker()

    assert mgr_cls.call_count == 1 and mgr1 is mgr2
    assert llm_cls.call_count == 1 and llm1 is llm2
    assert reranker_cls.call_count == 1 and rrk1 is rrk2
    # manager 构造时只建了一次 client
    assert client_cls.call_count == 1


# ═══════════════════════════════════════════════════════════════════
#  e. 边界输入
# ═══════════════════════════════════════════════════════════════════

def test_empty_query_not_crash(fake_env, monkeypatch, empty_manager):
    """空 query 不崩溃：返回转人工（无命中路径），结构合法。"""
    _use_manager(monkeypatch, empty_manager)

    r = mc._run_answer("", "en")
    assert set(r) >= {"answer", "sources", "hit_count", "elapsed_ms"}
    assert r["hit_count"] == 0
    assert "WhatsApp" in r["answer"]


def test_very_long_query_not_crash(fake_env, monkeypatch, kb_manager):
    """2000 字超长 query 不崩溃：结构合法、无异常泄漏。"""
    _use_manager(monkeypatch, kb_manager)
    monkeypatch.setattr(mc, "_llm_chat", lambda msgs: "ok")

    long_query = "手套" * 1000  # 2000 字
    r = mc._run_answer(long_query, "en")
    assert set(r) >= {"answer", "sources", "hit_count", "elapsed_ms"}
    assert r["answer"]


# ═══════════════════════════════════════════════════════════════════
#  f. order_help 空结果集（知识库只有 about 分类）
# ═══════════════════════════════════════════════════════════════════

def test_order_help_empty_category_set(fake_env, monkeypatch, tmp_path):
    """库中只有 about 分类时，order_help（限定 shipping/support）→ 转人工，不调 LLM。"""
    mgr = MilvusManager(client=MilvusClient(uri=str(tmp_path / "about.db")))
    col = kb_settings.kb_collection("en")
    ensure_collection(mgr, col, kb_settings.EMBEDDING_DIM, "kb", rebuild=True)

    vec = _fake_embed(["about"])[0]
    rows = [
        {"id": "faq-q1-en", "vector": vec, "text": "Q: Who is Mojin? A: A brand.",
         "sparse_text": "Who is Mojin", "doc_id": "faq-q1", "title": "Who", "category": "about", "lang": "en"},
        {"id": "faq-q2-en", "vector": vec, "text": "Q: Where are you? A: Riyadh.",
         "sparse_text": "Where are you", "doc_id": "faq-q2", "title": "Where", "category": "about", "lang": "en"},
    ]
    mgr.upsert(col, rows)
    _use_manager(monkeypatch, mgr)

    llm_called: list = []
    monkeypatch.setattr(mc, "_llm_chat", lambda msgs: llm_called.append(1) or "x")

    r = mc._run_answer("where is my shipping?", "en", ["shipping", "support"])
    assert r["hit_count"] == 0
    assert "WhatsApp" in r["answer"]  # 转人工引导
    assert not llm_called  # 关键：空结果集不调 LLM
