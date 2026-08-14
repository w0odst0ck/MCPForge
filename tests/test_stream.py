"""Mojin 客服 SSE 流式输出测试（Step 5 交付物）。

全部单元测试：不依赖外网 / model server / DeepSeek：
- 检索/reranker/LLM 全部 monkeypatch（复用 test_conversation.py 的假组件模式）
- 流式事件序列：start → token* → done；失败/超时 → error + 转人工收尾
- 非 stream 请求保持 JSON（向后兼容）
"""

import json
import time

from app.routers import mojin_chat as mc

# ═══════════════════════════════════════════════════════════════════
#  helpers / fixtures
# ═══════════════════════════════════════════════════════════════════

def _entity(doc_id: str, category: str = "shipping", text: str = "some knowledge text") -> dict:
    """构造 Milvus entity（含 hybrid 距离），供假检索返回。"""
    return {
        "id": f"{doc_id}-en-1", "doc_id": doc_id, "title": doc_id,
        "category": category, "text": text, "_distance": 0.9,
    }


class _RaisingReranker:
    """rerank 抛异常 → _prepare_answer 降级为 hybrid 顺序（无需 model server）。"""

    def rerank(self, texts, query):
        raise RuntimeError("no model server")


def _retrieve_hits(query, lang):
    """假检索：固定返回两条 shipping 命中。"""
    return [
        _entity("faq-shipping-1", "shipping", "Shipping takes 3-7 business days."),
        _entity("faq-shipping-2", "shipping", "Delivery is free over 200 SAR."),
    ]


def _retrieve_empty(query, lang):
    """假检索：空知识库（无命中 → 转人工）。"""
    return []


def _patch_rag(monkeypatch, retrieve_fn, stream_deltas=None, stream_fail=None, slow_first_token=False):
    """monkeypatch 检索/reranker/LLM 流，使 _answer_stream 无外网可跑。返回调用记录。

    stream_deltas: 假 LLM 逐 token 输出；stream_fail: 非 None 则 LLM 抛该异常；
    slow_first_token: 首个 token 前 sleep（配合超时用例）。
    """
    state: dict = {"retrieved": [], "llm_calls": []}

    def fake_retrieve(query, lang, categories=None):
        state["retrieved"].append(query)
        return retrieve_fn(query, lang)

    def fake_llm_stream(messages):
        state["llm_calls"].append(messages)
        if stream_fail is not None:
            raise stream_fail
        if slow_first_token:
            time.sleep(0.05)
        for d in (stream_deltas if stream_deltas is not None else ["d1", "d2"]):
            yield d

    monkeypatch.setattr(mc, "_retrieve_kb", fake_retrieve)
    monkeypatch.setattr(mc, "_get_reranker", lambda: _RaisingReranker())
    monkeypatch.setattr(mc, "_llm_chat", lambda messages: state["llm_calls"].append(messages) or "ok")
    monkeypatch.setattr(mc, "_llm_chat_stream", fake_llm_stream)
    return state


def _parse_sse(text: str) -> list:
    """解析 SSE 文本 → [(event, data_dict)]，data 行必须是单行 JSON。"""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event is not None:
            events.append((event, data))
    return events


# ═══════════════════════════════════════════════════════════════════
#  流式：事件序列与数据完整性
# ═══════════════════════════════════════════════════════════════════

def test_stream_returns_sse_event_sequence(client, monkeypatch):
    """stream=true → text/event-stream，事件序列 start → token* → done（token ≥1）。"""
    _patch_rag(monkeypatch, _retrieve_hits, stream_deltas=["We ", "ship ", "in 3-7 days."])
    resp = client.get(
        "/mojin_chat/answer_question",
        params={"query": "How long is shipping?", "lang": "en", "stream": "true"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    names = [e for e, _ in events]
    assert names[0] == "start"
    assert names[-1] == "done"
    tokens = [d["delta"] for e, d in events if e == "token"]
    assert tokens == ["We ", "ship ", "in 3-7 days."]  # 逐 token 推送，保序


def test_stream_done_has_full_answer(client, monkeypatch):
    """done 事件的 answer 等于全部 token 拼接，且带 elapsed_ms。"""
    _patch_rag(monkeypatch, _retrieve_hits, stream_deltas=["你好", "，", "客服"])
    resp = client.get(
        "/mojin_chat/answer_question",
        params={"query": "你好", "lang": "cn", "stream": "true"},
    )
    events = _parse_sse(resp.text)
    done = [d for e, d in events if e == "done"][0]
    assert done["answer"] == "你好，客服"
    assert done["elapsed_ms"] >= 0


def test_stream_start_has_hit_count_and_sources(client, monkeypatch):
    """start 事件带 hit_count 与 sources（doc_id/title/category/score）。"""
    _patch_rag(monkeypatch, _retrieve_hits)
    resp = client.get(
        "/mojin_chat/answer_question",
        params={"query": "How long is shipping?", "lang": "en", "stream": "true"},
    )
    events = _parse_sse(resp.text)
    start = [d for e, d in events if e == "start"][0]
    assert start["hit_count"] == 2
    assert len(start["sources"]) == 2
    assert set(start["sources"][0]) >= {"doc_id", "title", "category", "score"}
    assert all(s["doc_id"].startswith("faq-") for s in start["sources"])


def test_stream_no_hit_returns_handoff_without_llm(client, monkeypatch):
    """无命中 → start(hit_count=0) + done(转人工文案)，LLM 不被调用。"""
    state = _patch_rag(monkeypatch, _retrieve_empty)
    resp = client.get(
        "/mojin_chat/answer_question",
        params={"query": "meaning of life", "lang": "en", "stream": "true"},
    )
    events = _parse_sse(resp.text)
    start = [d for e, d in events if e == "start"][0]
    done = [d for e, d in events if e == "done"][0]
    assert start["hit_count"] == 0 and start["sources"] == []
    assert "WhatsApp" in done["answer"]  # 转人工引导文案
    assert not state["llm_calls"]  # 关键：无命中不调 LLM
    assert not [e for e, _ in events if e == "token"]


# ═══════════════════════════════════════════════════════════════════
#  流式：失败兜底 / 超时强制收尾
# ═══════════════════════════════════════════════════════════════════

def test_stream_llm_failure_yields_error_event(client, monkeypatch):
    """LLM 流抛异常 → error 事件（含转人工文案），done 的 answer 为转人工文案。"""
    _patch_rag(monkeypatch, _retrieve_hits, stream_fail=RuntimeError("deepseek down"))
    resp = client.get(
        "/mojin_chat/answer_question",
        params={"query": "How long is shipping?", "lang": "en", "stream": "true"},
    )
    events = _parse_sse(resp.text)
    errors = [d for e, d in events if e == "error"]
    assert len(errors) == 1
    assert errors[0]["message"] == "llm service unavailable"
    assert "WhatsApp" in errors[0]["answer"]  # 转人工文案
    done = [d for e, d in events if e == "done"][0]
    assert done["answer"] == errors[0]["answer"]
    assert not [e for e, _ in events if e == "token"]


def test_stream_timeout_forced_close(client, monkeypatch):
    """SSE 流超过 STREAM_TIMEOUT_S → error(stream timeout) 强制收尾，不再吐 token。"""
    _patch_rag(monkeypatch, _retrieve_hits, slow_first_token=True)
    monkeypatch.setattr(mc, "STREAM_TIMEOUT_S", 0.01)  # 超时压到 10ms
    resp = client.get(
        "/mojin_chat/answer_question",
        params={"query": "How long is shipping?", "lang": "en", "stream": "true"},
    )
    events = _parse_sse(resp.text)
    errors = [d for e, d in events if e == "error"]
    assert len(errors) == 1
    assert errors[0]["message"] == "stream timeout"
    assert not [e for e, _ in events if e == "token"]  # 超时后无后续 token
    done = [d for e, d in events if e == "done"][0]
    assert "WhatsApp" in done["answer"]


# ═══════════════════════════════════════════════════════════════════
#  向后兼容与参数校验
# ═══════════════════════════════════════════════════════════════════

def test_stream_false_keeps_json_response(client, monkeypatch):
    """不传 stream（默认 false）→ 返回 JSON 而非 SSE（行为与 Step 4 完全一致）。"""
    _patch_rag(monkeypatch, _retrieve_hits)
    resp = client.get(
        "/mojin_chat/answer_question",
        params={"query": "How long is shipping?", "lang": "en"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["hit_count"] > 0
    assert body["answer"]
    assert "sources" in body and "elapsed_ms" in body


def test_stream_invalid_lang_400(client):
    """stream=true + 非法 lang → 400（与 JSON 端点一致）。"""
    resp = client.get(
        "/mojin_chat/answer_question",
        params={"query": "hi", "lang": "xx", "stream": "true"},
    )
    assert resp.status_code == 400


def test_stream_missing_query_422(client):
    """stream=true + 缺必填参数 query → 422。"""
    resp = client.get(
        "/mojin_chat/answer_question",
        params={"lang": "en", "stream": "true"},
    )
    assert resp.status_code == 422
