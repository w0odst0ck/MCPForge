"""Mojin 客服多轮对话 + 问答可观测性测试。

全部单元测试：不依赖外网 / model server / DeepSeek：
- SQLite 会话表 + 日志表 CRUD、PII 脱敏、运营查询（高频问题/转人工率）
- 多轮对话：历史拼接（指代消解）、截断、session_id 原样回传、不缓存
- 问答日志：三个端点落库、is_handoff 标记、sources JSON
- RAG 冒烟：fake embed + 临时 Milvus + 假 LLM 走真实检索链路（不依赖 model server）
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest
from rag_toolkit.storage.milvus_manager import MilvusManager

from app.knowledge.config import kb_settings
from app.knowledge.loader import load_knowledge
from app.knowledge.store import PII_MASKED, ChatStore, sanitize_query
from app.routers import mojin_chat as mc

try:
    from pymilvus import MilvusClient
except ImportError:  # pragma: no cover
    MilvusClient = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════
#  fixtures / helpers
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def chat_store(tmp_path, monkeypatch):
    """临时 SQLite 会话/日志库，并让 mojin_chat 使用它（避免污染 data/chat.db）。"""
    store = ChatStore(tmp_path / "chat.db")
    store.init_db()
    monkeypatch.setattr(mc, "_get_store", lambda: store)
    return store


def _entity(doc_id: str, category: str = "support", text: str = "some knowledge text") -> dict:
    """构造 Milvus entity（含 hybrid 距离），供假检索返回。"""
    return {
        "id": f"{doc_id}-en-1", "doc_id": doc_id, "title": doc_id,
        "category": category, "text": text, "_distance": 0.9,
    }


class _RaisingReranker:
    """rerank 抛异常 → _run_answer_impl 降级为 hybrid 顺序（无需 model server）。"""

    def rerank(self, texts, query):
        raise RuntimeError("no model server")


def _patch_rag(monkeypatch, retrieve_fn, llm_answer: str = "ok") -> dict:
    """monkeypatch 检索/reranker/LLM，使 _run_answer_impl 无外网可跑。返回调用记录。"""
    state: dict = {"retrieved": [], "llm_calls": []}

    def fake_retrieve(query, lang, categories=None):
        state["retrieved"].append(query)
        return retrieve_fn(query, lang)

    def fake_llm(messages):
        state["llm_calls"].append(messages)
        return llm_answer

    monkeypatch.setattr(mc, "_retrieve_kb", fake_retrieve)
    monkeypatch.setattr(mc, "_get_reranker", lambda: _RaisingReranker())
    monkeypatch.setattr(mc, "_llm_chat", fake_llm)
    return state


def _fake_embed(texts):
    """确定性 fake embed（1024 维），测试不依赖 model server。"""
    import hashlib

    out = []
    for t in texts:
        seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
        vec = [float((seed >> (i % 32)) & 1) * 0.1 + 0.01 for i in range(1024)]
        out.append(vec)
    return out


# ═══════════════════════════════════════════════════════════════════
#  PII 脱敏（沙特 PDPL）
# ═══════════════════════════════════════════════════════════════════

def test_pii_sanitize_masks_long_digits():
    """query 含连续 8+ 数字（疑似手机号）→ [PII 已脱敏]。"""
    assert sanitize_query("请回电 1389915060") == PII_MASKED
    assert sanitize_query("order #12345678 please") == PII_MASKED
    assert sanitize_query("沙特 966501234567") == PII_MASKED


def test_pii_sanitize_keeps_short_digits():
    """少于 8 位数字 / 无数字 → 原样保留。"""
    assert sanitize_query("How long is 7 days?") == "How long is 7 days?"
    assert sanitize_query("尺寸 M 码") == "尺寸 M 码"
    assert sanitize_query("") == ""


# ═══════════════════════════════════════════════════════════════════
#  多轮检索 query 构造 / 历史截断
# ═══════════════════════════════════════════════════════════════════

def test_build_conversation_query_empty_history():
    """无历史：检索 query = 当前问题原文。"""
    assert mc._build_conversation_query("那退换呢", []) == "那退换呢"


def test_build_conversation_query_appends_last_round():
    """有历史：只拼接最近一轮（user_q + assistant_a），更早轮次不进检索。"""
    history = [
        {"role": "user", "content": "配送多久"},
        {"role": "assistant", "content": "3-7 天"},
        {"role": "user", "content": "运费多少"},
        {"role": "assistant", "content": "免费"},
    ]
    q = mc._build_conversation_query("那退换呢", history)
    assert q.startswith("那退换呢")
    assert "运费多少" in q and "免费" in q
    assert "配送多久" not in q and "3-7 天" not in q  # 只取最近一轮


def test_build_conversation_query_truncates_long_messages():
    """历史段超长按字符截断，避免 embedding 超长。"""
    history = [{"role": "user", "content": "x" * 500}]
    q = mc._build_conversation_query("hi", history)
    assert q.startswith("hi")
    assert len(q) < 500 + 100


def test_append_history_turn_appends_pair():
    """追加一轮 = user + assistant 两条消息。"""
    h = mc._append_history_turn([], "q", "a", max_rounds=6)
    assert h == [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]


def test_append_history_turn_truncates_to_max_rounds():
    """超长截断：只保留最近 max_rounds 轮。"""
    h: list = []
    for i in range(5):
        h = mc._append_history_turn(h, f"q{i}", f"a{i}", max_rounds=2)
    assert len(h) == 4  # 最近 2 轮 = 4 条
    assert h[0] == {"role": "user", "content": "q3"}
    assert h[-1] == {"role": "assistant", "content": "a4"}


# ═══════════════════════════════════════════════════════════════════
#  SQLite 会话表
# ═══════════════════════════════════════════════════════════════════

def test_store_session_roundtrip(chat_store):
    """会话写入/读取闭环：首次创建，后续续接更新。"""
    chat_store.upsert_session("s1", "en", [{"role": "user", "content": "hi"}])
    s = chat_store.get_session("s1")
    assert s["session_id"] == "s1" and s["lang"] == "en"
    assert s["history"] == [{"role": "user", "content": "hi"}]
    # 续接更新
    chat_store.upsert_session("s1", "en", [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
    ])
    assert len(chat_store.get_session("s1")["history"]) == 2


def test_store_session_missing_returns_none(chat_store):
    """不存在的 session_id → None。"""
    assert chat_store.get_session("nope") is None


def test_delete_expired_sessions(chat_store):
    """只删除 updated_at 早于 TTL 的会话，新鲜会话保留。"""
    chat_store._execute(
        "INSERT INTO chat_sessions (session_id, lang, created_at, updated_at, history) "
        "VALUES (?, ?, datetime('now','localtime','-2 days'), datetime('now','localtime','-2 days'), '[]')",
        ("old", "en"),
    )
    chat_store.upsert_session("fresh", "en", [])
    assert chat_store.delete_expired_sessions(ttl_hours=24) == 1
    assert chat_store.get_session("old") is None
    assert chat_store.get_session("fresh") is not None


# ═══════════════════════════════════════════════════════════════════
#  SQLite 问答日志表
# ═══════════════════════════════════════════════════════════════════

def test_append_log_fields_and_sources_json(chat_store):
    """日志字段完整：session_id/lang/query/answer/hit_count/elapsed_ms/sources/is_handoff。"""
    chat_store.append_log(
        "s1", "en", "how long?", "3-7 days",
        hit_count=2, elapsed_ms=123, sources=["doc-a", "doc-b"], is_handoff=0,
    )
    row = chat_store._query_one("SELECT * FROM chat_logs")
    assert row["session_id"] == "s1" and row["lang"] == "en"
    assert row["query"] == "how long?" and row["answer"] == "3-7 days"
    assert row["hit_count"] == 2 and row["elapsed_ms"] == 123
    assert json.loads(row["sources"]) == ["doc-a", "doc-b"]
    assert row["is_handoff"] == 0 and row["created_at"]


def test_delete_expired_logs(chat_store):
    """只删除超过保留期（30 天）的日志。"""
    chat_store._execute(
        "INSERT INTO chat_logs (session_id, lang, query, answer, hit_count, elapsed_ms, "
        "sources, is_handoff, created_at) "
        "VALUES (?, ?, ?, ?, 1, 1, '[]', 0, datetime('now','localtime','-40 days'))",
        ("s", "en", "old q", "old a"),
    )
    chat_store.append_log("s", "en", "new q", "new a", hit_count=1)
    assert chat_store.delete_expired_logs(retention_days=30) == 1
    rows = chat_store._query_all("SELECT query FROM chat_logs")
    assert [r["query"] for r in rows] == ["new q"]


def test_top_questions_ranking(chat_store):
    """高频问题 TopN：按 query 计数降序。"""
    for q in ["how long?", "how long?", "what is return?", "how long?"]:
        chat_store.append_log(None, "en", q, "a", hit_count=1)
    top = chat_store.top_questions(limit=2)
    assert top[0]["query"] == "how long?" and top[0]["cnt"] == 3
    assert len(top) == 2


def test_handoff_rate(chat_store):
    """转人工率 = is_handoff=1 条数 / 总条数；空库返回 None。"""
    assert chat_store.handoff_rate() is None
    chat_store.append_log(None, "en", "q1", "a1", hit_count=1)
    chat_store.append_log(None, "en", "q2", "handoff", hit_count=0, is_handoff=1)
    chat_store.append_log(None, "en", "q3", "a3", hit_count=2)
    assert chat_store.handoff_rate() == pytest.approx(1 / 3)


# ═══════════════════════════════════════════════════════════════════
#  conversation 端点（HTTP / MCP）
# ═══════════════════════════════════════════════════════════════════

def test_conversation_tool_registered():
    """conversation 注册为 MCP 工具，旧工具不受影响。"""
    names = {t.name for t in asyncio.run(mc.mcp.list_tools())}
    assert "conversation" in names
    assert {"answer_question", "search_products", "order_help"} <= names


def test_conversation_bad_lang_400(client):
    """非法 lang → 400。"""
    resp = client.get("/mojin_chat/conversation", params={"session_id": "s", "query": "hi", "lang": "xx"})
    assert resp.status_code == 400


def test_conversation_missing_params_422(client):
    """缺必填参数（session_id / query）→ 422。"""
    resp = client.get("/mojin_chat/conversation", params={"query": "hi"})
    assert resp.status_code == 422
    resp2 = client.get("/mojin_chat/conversation", params={"session_id": "s"})
    assert resp2.status_code == 422


def test_conversation_multi_turn_reference_resolution(client, chat_store, monkeypatch):
    """核心验收：问「配送多久」→ 答 → 问「那退换呢」→ 命中退换政策。

    第二轮检索 query 必须拼接最近一轮历史（指代消解），答案来自 return-policy 而非配送。
    """

    def retrieve(query, lang):
        if "return" in query.lower():
            return [_entity("policy-return-policy-en-1", "support", "Return window is 7 days after delivery.")]
        return [_entity("policy-shipping-en-1", "shipping", "Delivery takes 3-7 business days.")]

    state = _patch_rag(monkeypatch, retrieve, llm_answer="RETURN_POLICY_ANSWER")

    # 首轮：配送
    r1 = client.get(
        "/mojin_chat/conversation",
        params={"session_id": "sess-1", "query": "How long is delivery?", "lang": "en"},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["session_id"] == "sess-1"          # session_id 原样回传
    assert body1["hit_count"] == 1
    assert state["retrieved"][0] == "How long is delivery?"  # 首轮无历史 = 原文

    # 次轮：退换（指代需结合上轮）
    r2 = client.get(
        "/mojin_chat/conversation",
        params={"session_id": "sess-1", "query": "What about returns?", "lang": "en"},
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["answer"] == "RETURN_POLICY_ANSWER"
    assert body2["sources"][0]["doc_id"] == "policy-return-policy-en-1"  # 命中退换政策
    # 关键：第二轮检索 query 拼接了最近一轮历史（user_q + assistant_a）
    assert "What about returns?" in state["retrieved"][1]
    assert "How long is delivery?" in state["retrieved"][1]
    assert "RETURN_POLICY_ANSWER" in state["retrieved"][1]

    # 会话历史：2 轮 4 条消息
    session = chat_store.get_session("sess-1")
    assert session["history"] == [
        {"role": "user", "content": "How long is delivery?"},
        {"role": "assistant", "content": "RETURN_POLICY_ANSWER"},
        {"role": "user", "content": "What about returns?"},
        {"role": "assistant", "content": "RETURN_POLICY_ANSWER"},
    ]


def test_conversation_not_cached(client, chat_store, monkeypatch):
    """conversation 不缓存：相同 session 相同 query 重复问，LLM 每次都被调用（简单正确优先）。"""

    def retrieve(query, lang):
        return [_entity("policy-shipping-en-1", "shipping", "Delivery takes 3-7 business days.")]

    state = _patch_rag(monkeypatch, retrieve, llm_answer="delivery answer")
    for _ in range(2):
        resp = client.get(
            "/mojin_chat/conversation",
            params={"session_id": "sess-c", "query": "How long?", "lang": "en"},
        )
        assert resp.status_code == 200
    assert len(state["llm_calls"]) == 2  # 未命中缓存


def test_conversation_no_hit_handoff_logged(client, chat_store, monkeypatch):
    """无命中 → 转人工引导文案 + 日志 is_handoff=1，且不调 LLM。"""
    state = _patch_rag(monkeypatch, retrieve_fn=lambda q, _lang: [], llm_answer="x")
    resp = client.get(
        "/mojin_chat/conversation",
        params={"session_id": "sess-h", "query": "meaning of life", "lang": "en"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["hit_count"] == 0 and "WhatsApp" in body["answer"]
    assert not state["llm_calls"]  # 关键：无命中不调 LLM
    rows = chat_store._query_all("SELECT * FROM chat_logs")
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess-h" and rows[0]["is_handoff"] == 1


def test_answer_question_writes_log(client, chat_store, monkeypatch):
    """单轮 answer_question 也写日志：session_id 为空、hit_count/sources 正确。"""
    _patch_rag(
        monkeypatch,
        retrieve_fn=lambda q, _lang: [_entity("policy-shipping-en-1", "shipping", "text")],
        llm_answer="ship answer",
    )
    resp = client.get("/mojin_chat/answer_question", params={"query": "How long is delivery?", "lang": "en"})
    assert resp.status_code == 200
    rows = chat_store._query_all("SELECT * FROM chat_logs")
    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] is None
    assert row["query"] == "How long is delivery?"
    assert row["hit_count"] == 1 and row["is_handoff"] == 0
    assert json.loads(row["sources"]) == ["policy-shipping-en-1"]


def test_order_help_writes_log(client, chat_store, monkeypatch):
    """order_help 也写日志（向后兼容不受影响）。"""
    _patch_rag(
        monkeypatch,
        retrieve_fn=lambda q, _lang: [_entity("policy-return-policy-en-1", "support", "text")],
        llm_answer="return answer",
    )
    resp = client.get("/mojin_chat/order_help", params={"query": "return policy", "lang": "en"})
    assert resp.status_code == 200
    rows = chat_store._query_all("SELECT * FROM chat_logs")
    assert len(rows) == 1 and rows[0]["session_id"] is None and rows[0]["is_handoff"] == 0


def test_conversation_pii_masked_in_log(client, chat_store, monkeypatch):
    """PII 脱敏落库：query 含连续 8+ 数字 → 日志存 [PII 已脱敏]。"""
    _patch_rag(
        monkeypatch,
        retrieve_fn=lambda q, _lang: [_entity("policy-shipping-en-1", "shipping", "text")],
        llm_answer="a",
    )
    resp = client.get(
        "/mojin_chat/conversation",
        params={"session_id": "sess-p", "query": "我的手机是1389915060请回电", "lang": "cn"},
    )
    assert resp.status_code == 200
    row = chat_store._query_one("SELECT query FROM chat_logs ORDER BY id DESC LIMIT 1")
    assert row["query"] == PII_MASKED


# ═══════════════════════════════════════════════════════════════════
#  RAG 冒烟：fake embed + 临时 Milvus + 假 LLM（不依赖 model server）
# ═══════════════════════════════════════════════════════════════════

def test_conversation_rag_smoke(chat_store, monkeypatch):
    """真实检索链路冒烟：临时 Milvus + fake embed 走 hybrid search → rerank 降级 → 假 LLM。"""
    if MilvusClient is None:
        pytest.skip("pymilvus 不可用")
    carewell = Path(kb_settings.CAREWELL_PATH)
    if not carewell.is_dir():
        pytest.skip("carewell-shop 知识源不在，跳过冒烟")

    db = os.path.join(tempfile.mkdtemp(), "conv_kb.db")
    mgr = MilvusManager(client=MilvusClient(uri=db))
    counts = load_knowledge(manager=mgr, embed_fn=_fake_embed, langs=["en"], scope="kb", rebuild=True)
    assert counts["mojin_kb_en"] >= 10

    monkeypatch.setattr(mc, "_get_manager", lambda: mgr)
    monkeypatch.setattr(mc, "_embed", _fake_embed)
    monkeypatch.setattr(mc, "_get_reranker", lambda: _RaisingReranker())
    monkeypatch.setattr(mc, "_llm_chat", lambda msgs: "SMOKE_ANSWER")

    result = mc._run_conversation("sess-smoke", "How long does delivery take?", "en")
    assert result["session_id"] == "sess-smoke"
    assert result["answer"] == "SMOKE_ANSWER"
    assert result["hit_count"] > 0 and result["sources"]
    assert chat_store.get_session("sess-smoke") is not None
