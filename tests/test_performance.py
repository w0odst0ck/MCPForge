"""性能优化测试 — 答案 TTL-LRU 缓存 + MultiQuery 开关。

全部用例不依赖外网、不依赖 model server（fake embed / 假 LLM / 临时 Milvus Lite）。
参照 tests/test_resilience.py 的 fake/临时库模式；conftest 的 autouse fixture
会在每个测试前后清空答案缓存，防止用例间串缓存。
"""

import hashlib
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
from rag_toolkit.storage.milvus_manager import MilvusManager

from app.knowledge.config import kb_settings
from app.knowledge.loader import load_knowledge
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

    def __init__(self, calls: list | None = None):
        self.calls = [] if calls is None else calls

    def generate(self, question, count=3):
        self.calls.append(question)
        return [f"{question} (expanded 1)", f"{question} (expanded 2)"]


class _FakeReranker:
    """假 XReranker：确定性分数（足够高于 RERANK_MIN_SCORE=0.15），不调 model server。"""

    def rerank(self, documents, query):
        return [(i, 0.9 - i * 0.01) for i in range(len(documents))]


# ═══════════════════════════════════════════════════════════════════
#  fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def kb_manager(tmp_path):
    """临时 Milvus：en 语言 kb 集合（FAQ + 政策，fake embed 摄入）。"""
    mgr = MilvusManager(client=MilvusClient(uri=str(tmp_path / "perf_kb.db")))
    load_knowledge(manager=mgr, embed_fn=_fake_embed, langs=["en"], scope="kb", rebuild=True)
    return mgr


@pytest.fixture
def fake_env(monkeypatch, kb_manager):
    """注入 fake embed / reranker / manager（multi-query 由用例自定）。"""
    monkeypatch.setattr(mc, "_get_manager", lambda: kb_manager)
    monkeypatch.setattr(mc, "_embed", _fake_embed)
    monkeypatch.setattr(mc, "_get_reranker", lambda: _FakeReranker())


def _ok_result(answer="cached answer", hit_count=2):
    """构造一个成功（可缓存）的结果 dict。"""
    return {
        "answer": answer,
        "sources": [{"doc_id": "faq-q1", "title": "t", "category": "shipping", "score": 0.9}],
        "hit_count": hit_count,
        "elapsed_ms": 1234,
    }


# ═══════════════════════════════════════════════════════════════════
#  a. _TTLAnswerCache 单元测试（TTL / LRU 淘汰 / 线程安全）
# ═══════════════════════════════════════════════════════════════════

def test_cache_basic_get_set():
    cache = mc._TTLAnswerCache(capacity=10, ttl=600)
    key = ("q", "en", "answer")
    assert cache.get(key) is None  # 未命中
    cache.set(key, {"answer": "a"})
    assert cache.get(key) == {"answer": "a"}


def test_cache_ttl_expiry():
    cache = mc._TTLAnswerCache(capacity=10, ttl=0.05)
    key = ("q", "en", "answer")
    cache.set(key, {"answer": "a"})
    assert cache.get(key) is not None
    time.sleep(0.08)
    assert cache.get(key) is None  # 过期后未命中


def test_cache_lru_eviction():
    """容量淘汰：超过 capacity 淘汰最久未使用；get 刷新 LRU 顺序。"""
    cache = mc._TTLAnswerCache(capacity=2, ttl=600)
    cache.set(("a", "en", "answer"), {"answer": "A"})
    cache.set(("b", "en", "answer"), {"answer": "B"})
    assert cache.get(("a", "en", "answer")) is not None  # 刷新 a → a 变为最新
    cache.set(("c", "en", "answer"), {"answer": "C"})     # 淘汰最旧 b
    assert cache.get(("b", "en", "answer")) is None
    assert cache.get(("a", "en", "answer")) is not None
    assert cache.get(("c", "en", "answer")) is not None


def test_cache_clear():
    cache = mc._TTLAnswerCache(capacity=10, ttl=600)
    cache.set(("a", "en", "answer"), {"answer": "A"})
    cache.clear()
    assert cache.get(("a", "en", "answer")) is None


# ═══════════════════════════════════════════════════════════════════
#  b. _run_answer 缓存行为（命中标记 / 只缓存成功 / mode 区分）
# ═══════════════════════════════════════════════════════════════════

def test_answer_cached_on_second_call(monkeypatch):
    """同一 query 第二次调用命中缓存：impl 只执行一次，第二次带 cached=True。"""
    impl_calls: list = []
    monkeypatch.setattr(mc, "_run_answer_impl",
                        lambda q, lang, categories=None: impl_calls.append(q) or _ok_result())

    r1 = mc._run_answer("shipping policy?", "en")
    r2 = mc._run_answer("shipping policy?", "en")

    assert len(impl_calls) == 1          # 第二次走了缓存
    assert "cached" not in r1            # 首次结果无命中标记（保持原结构）
    assert r2["cached"] is True
    assert r2["answer"] == r1["answer"] == "cached answer"
    assert isinstance(r2["elapsed_ms"], int) and r2["elapsed_ms"] >= 0


def test_cache_key_separates_mode_and_lang(monkeypatch):
    """mode（answer/order_help）与 lang 不同 → 不共享缓存。"""
    calls: list = []
    monkeypatch.setattr(mc, "_run_answer_impl",
                        lambda q, lang, categories=None: calls.append((q, lang, categories)) or _ok_result())

    mc._run_answer("q", "en")
    mc._run_answer("q", "en", ["shipping", "support"])  # order_help
    mc._run_answer("q", "cn")                           # 不同语言

    assert len(calls) == 3  # 三次全链路都执行，互不命中


def test_error_result_not_cached(monkeypatch):
    """错误兜底（含 error 键）不缓存：第二次仍走全链路。"""
    results = iter([
        {**_ok_result(), "error": "llm service unavailable"},
        _ok_result("second"),
    ])
    monkeypatch.setattr(mc, "_run_answer_impl", lambda q, lang, categories=None: next(results))

    r1 = mc._run_answer("q", "en")
    r2 = mc._run_answer("q", "en")

    assert "error" in r1
    assert r2["answer"] == "second"  # 第二次重新执行而非命中旧错误


def test_no_hit_handoff_not_cached(monkeypatch):
    """无命中转人工（hit_count=0）不缓存：知识库更新后不会返回旧转人工。"""
    results = iter([
        {"answer": "human handoff text", "sources": [], "hit_count": 0, "elapsed_ms": 50},
        _ok_result("now hit"),
    ])
    monkeypatch.setattr(mc, "_run_answer_impl", lambda q, lang, categories=None: next(results))

    r1 = mc._run_answer("q", "en")
    r2 = mc._run_answer("q", "en")

    assert r1["hit_count"] == 0
    assert r2["answer"] == "now hit"  # 未命中缓存


def test_answer_cache_ttl_expiry_reruns(monkeypatch):
    """TTL 过期后重新走全链路（短 TTL 缓存实例替换模块缓存）。"""
    monkeypatch.setattr(mc, "_answer_cache", mc._TTLAnswerCache(capacity=10, ttl=0.05))
    calls: list = []
    monkeypatch.setattr(mc, "_run_answer_impl",
                        lambda q, lang, categories=None: calls.append(1) or _ok_result())

    mc._run_answer("q", "en")
    mc._run_answer("q", "en")          # 命中
    time.sleep(0.08)
    r = mc._run_answer("q", "en")      # 过期 → 重跑

    assert len(calls) == 2
    assert "cached" not in r


def test_answer_cache_lru_capacity(monkeypatch):
    """容量 200：超过后最旧键被淘汰，重新走全链路。"""
    monkeypatch.setattr(mc, "_answer_cache", mc._TTLAnswerCache(capacity=2, ttl=600))
    calls: list = []
    monkeypatch.setattr(mc, "_run_answer_impl",
                        lambda q, lang, categories=None: calls.append(q) or _ok_result())

    mc._run_answer("q1", "en")   # miss → 缓存 {q1}
    mc._run_answer("q2", "en")   # miss → 缓存 {q1,q2}
    mc._run_answer("q1", "en")   # hit，q1 刷新为最新 → {q2,q1}
    mc._run_answer("q3", "en")   # miss → set q3 淘汰最旧 q2 → {q1,q3}
    mc._run_answer("q1", "en")   # hit（q1 仍在缓存）
    mc._run_answer("q2", "en")   # miss（q2 已被淘汰）→ 重跑

    assert calls == ["q1", "q2", "q3", "q2"]


def test_answer_cache_concurrent_safe(fake_env, monkeypatch):
    """线程池并发同键调用：全部返回合法结构、无异常泄漏、无死锁。"""
    monkeypatch.setattr(mc, "_llm_chat", lambda msgs: "ok")

    def slow_impl(q, lang, categories=None):
        time.sleep(0.1)
        return _ok_result(answer=f"answer:{q}")

    monkeypatch.setattr(mc, "_run_answer_impl", slow_impl)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(mc._run_answer, "concurrent q", "en") for _ in range(5)]
        results = [f.result(timeout=30) for f in futures]

    assert len(results) == 5
    for r in results:
        assert set(r) >= {"answer", "sources", "hit_count", "elapsed_ms"}
        assert r["answer"] == "answer:concurrent q"


# ═══════════════════════════════════════════════════════════════════
#  c. MultiQuery 开关
# ═══════════════════════════════════════════════════════════════════

def test_multi_query_disabled_by_default(fake_env, monkeypatch):
    """默认配置（MULTI_QUERY_ENABLED=False）：_retrieve_kb 不调 MultiQueryGenerator。"""
    mq = _FakeMultiQuery()
    monkeypatch.setattr(mc, "_get_multi_query", lambda: mq)

    merged = mc._retrieve_kb("shipping policy?", "en")

    assert mq.calls == []            # 改写未被调用
    assert merged                     # 只用原文检索仍有结果


def test_multi_query_enabled_calls_expander(fake_env, monkeypatch):
    """MULTI_QUERY_ENABLED=True 且块数足够：恢复调用 MultiQueryGenerator。"""
    mq = _FakeMultiQuery()
    monkeypatch.setattr(mc, "_get_multi_query", lambda: mq)
    # kb_settings 是 pydantic 单例，禁止 setattr；替换模块引用为 SimpleNamespace
    # （MIN_BLOCKS=0 保证块数判断恒通过，不依赖真实 count）
    monkeypatch.setattr(mc, "kb_settings", types.SimpleNamespace(
        MULTI_QUERY_ENABLED=True,
        MULTI_QUERY_MIN_BLOCKS=0,
        MULTI_QUERY_COUNT=3,
        kb_collection=kb_settings.kb_collection,
        RAG_HYBRID_TOP_K=8,
    ))

    merged = mc._retrieve_kb("shipping policy?", "en")

    assert mq.calls == ["shipping policy?"]  # 改写被调用
    assert merged


def test_multi_query_enabled_but_small_kb_skips(fake_env, monkeypatch):
    """开启但块数不足 MIN_BLOCKS：跳过改写（块数走 _kb_block_count 5 分钟缓存）。"""
    mq = _FakeMultiQuery()
    monkeypatch.setattr(mc, "_get_multi_query", lambda: mq)
    monkeypatch.setattr(mc, "kb_settings", types.SimpleNamespace(
        MULTI_QUERY_ENABLED=True,
        MULTI_QUERY_MIN_BLOCKS=50,
        MULTI_QUERY_COUNT=3,
        kb_collection=kb_settings.kb_collection,
        RAG_HYBRID_TOP_K=8,
    ))
    monkeypatch.setattr(mc, "_kb_block_count", lambda lang: 10)  # 小库：10 块 < 50

    merged = mc._retrieve_kb("shipping policy?", "en")

    assert mq.calls == []    # 块数不足 → 跳过改写
    assert merged


def test_use_multi_query_unit(monkeypatch):
    """_use_multi_query 开关逻辑：默认关；开启+块数足→True；开启+块数不足→False。"""
    assert mc._use_multi_query("en") is False  # 默认关（kb_settings.MULTI_QUERY_ENABLED=False）

    monkeypatch.setattr(mc, "kb_settings", types.SimpleNamespace(
        MULTI_QUERY_ENABLED=True,
        MULTI_QUERY_MIN_BLOCKS=50,
    ))
    monkeypatch.setattr(mc, "_kb_block_count", lambda lang: 100)
    assert mc._use_multi_query("en") is True

    monkeypatch.setattr(mc, "_kb_block_count", lambda lang: 10)
    assert mc._use_multi_query("en") is False


def test_kb_block_count_cached(monkeypatch, kb_manager):
    """块数查询 5 分钟缓存：两次调用只 count 一次。"""
    counts: list = []
    real_stats = kb_manager.milvus.get_collection_stats

    def counting_stats(name):
        counts.append(name)
        return real_stats(name)

    monkeypatch.setattr(mc, "_block_count_cache", {})
    monkeypatch.setattr(kb_manager.milvus, "get_collection_stats", counting_stats)
    monkeypatch.setattr(mc, "_get_manager", lambda: kb_manager)

    c1 = mc._kb_block_count("en")
    c2 = mc._kb_block_count("en")

    assert c1 == c2 > 0
    assert len(counts) == 1  # 第二次命中缓存，未再 count
