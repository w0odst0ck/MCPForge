"""Mojin 客服模块测试。

单元测试：不依赖外网、不依赖 model server（fake embed / 本地 Milvus Lite）。
集成测试：model server 可达时运行（真实 embed + 临时 Milvus + 假 LLM），否则 pytest.skip。
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
from rag_toolkit.storage.milvus_manager import MilvusManager

from app.knowledge.config import kb_settings
from app.knowledge.loader import (
    chunk_faqs,
    chunk_policy,
    chunk_products,
    ensure_collection,
    extract_policy_parts,
    load_knowledge,
)
from app.routers import mojin_chat as mc

try:
    from pymilvus import MilvusClient
except ImportError:  # pragma: no cover
    MilvusClient = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════
#  单元测试（无外网 / 无 model server）
# ═══════════════════════════════════════════════════════════════════

def test_tools_registered():
    """三个工具名存在于 mcp 工具列表。"""
    names = {t.name for t in asyncio.run(mc.mcp.list_tools())}
    assert {"answer_question", "search_products", "order_help"} <= names


def test_lang_validation_http_400(client):
    """HTTP 端点非法 lang → 400。"""
    resp = client.get("/mojin_chat/answer_question", params={"query": "hi", "lang": "xx"})
    assert resp.status_code == 400


def test_missing_query_422(client):
    """HTTP 端点缺必填参数 query → 422。"""
    resp = client.get("/mojin_chat/answer_question")
    assert resp.status_code == 422


def test_lang_validation_unit():
    """lang 校验逻辑：非法 lang 抛 HTTPException(400)，合法 lang 放行。"""
    from fastapi import HTTPException

    for lang in ("en", "cn", "ar"):
        assert mc._check_lang(lang) == lang
    with pytest.raises(HTTPException) as exc:
        mc._check_lang("fr")
    assert exc.value.status_code == 400


def test_chunk_faqs():
    """FAQ 分块：数量、doc_id、category、lang、文本格式。"""
    data = {
        "faqs": [
            {
                "id": "q1",
                "category": "shipping",
                "en": {"question": "How long?", "answer": "3-7 days."},
                "cn": {"question": "多久？", "answer": "3-7 天。"},
            },
            {
                "id": "q2",
                "category": "about",
                "en": {"question": "Who?", "answer": "Mojin."},
                # 无 cn 条目 → 回退 en
            },
        ]
    }
    blocks = chunk_faqs(data, "en")
    assert len(blocks) == 2
    b = blocks[0]
    assert b["doc_id"] == "faq-q1" and b["id"] == "faq-q1-en"
    assert b["category"] == "shipping" and b["lang"] == "en"
    assert b["text"] == "Q: How long? A: 3-7 days."
    assert b["title"] == "How long?"
    # 缺失语言回退到 en
    assert chunk_faqs(data, "cn")[1]["text"].startswith("Q: Who?")


def test_chunk_policy():
    """政策分块：doc_id 带 lang/序号、块大小上限、category 映射。"""
    parts = ["Shipping Coverage", "We ship to all regions of Saudi Arabia."] * 40
    blocks = chunk_policy(parts, "shipping", "en")
    assert len(blocks) >= 3
    assert blocks[0]["id"].startswith("policy-shipping-en-")
    assert all(len(b["text"]) <= 600 for b in blocks)
    assert all(b["category"] == "shipping" for b in blocks)
    # return-policy 归 support（order_help 检索范围）
    rp = chunk_policy(["Return window", "7 days"], "return-policy", "en")
    assert rp and rp[0]["category"] == "support"


def test_chunk_products():
    """产品分块：每 SKU 每语言一块，结构化字段完整，desc 分隔符替换。"""
    rows = [
        {
            "sku": "glove", "name_en": "Nitrile Gloves", "name_cn": "丁腈手套", "name_ar": "قفازات نيتريل",
            "sub_en": "100 pcs", "sub_cn": "100只装", "sub_ar": "100 قطعة",
            "cat_en": "Gloves", "cat_cn": "手套", "cat_ar": "قفازات",
            "desc_en": "Product Overview|Key Features", "desc_cn": "产品概述|主要特点", "desc_ar": "نظرة عامة|المميزات",
            "price": "28", "old_price": "35", "unit": "/ 100 pcs", "sale": "-20%",
            "rating": "4.7", "reviews": "18", "sizes": "S,M", "colors": "Blue,Black",
            "image": "images/cap-round.webp",
        }
    ]
    blocks = chunk_products(rows, "cn")
    assert len(blocks) == 1
    b = blocks[0]
    assert b["id"] == "glove-cn" and b["sku"] == "glove"
    assert b["name"] == "丁腈手套" and b["price"] == "28" and b["old_price"] == "35"
    assert b["sizes"] == "S,M" and b["colors"] == "Blue,Black"
    assert "产品概述" in b["text"] and "主要特点" in b["text"]  # | 已替换


def test_extract_policy_parts_real_files():
    """真实政策页：提取 content-en 区块含关键信息；cn 回退用 en 区块。"""
    p = Path(kb_settings.CAREWELL_PATH) / "shipping.html"
    if not p.exists():
        pytest.skip("carewell-shop 知识源不在")
    parts = extract_policy_parts(p, "en")
    assert parts and any("Riyadh" in x for x in parts)
    parts_cn = extract_policy_parts(p, "cn")
    assert parts_cn == parts  # cn 无区块 → 用 en


def _fake_embed(texts):
    """确定性 fake embed（1024 维），测试不依赖 model server。"""
    import hashlib

    out = []
    for t in texts:
        seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
        vec = [float((seed >> (i % 32)) & 1) * 0.1 + 0.01 for i in range(1024)]
        out.append(vec)
    return out


def test_load_write_read_roundtrip():
    """loader 写读闭环：fake embed + 临时 Milvus → 分块数量与字段落库验证。"""
    db = os.path.join(tempfile.mkdtemp(), "roundtrip.db")
    mgr = MilvusManager(client=MilvusClient(uri=db))
    counts = load_knowledge(manager=mgr, embed_fn=_fake_embed, langs=["en"], scope="all", rebuild=True)
    assert counts["mojin_kb_en"] >= 10          # 9 FAQ + 政策 ≥10
    assert counts["mojin_products_en"] == 10    # 10 SKU

    # 检索验证（dense）
    hits = mgr.search(
        "mojin_products_en",
        vectors=[_fake_embed(["x"])[0]],
        anns_field="vector",
        limit=5,
        output_fields=["sku", "name"],
    )
    assert len(hits[0]) == 5
    assert all(h["entity"]["sku"] for h in hits[0])

    # KB 集合字段完整
    q = mgr.query("mojin_kb_en", expr="doc_id == 'faq-q3'", output_fields=["id", "doc_id", "category", "lang", "text"])
    assert q and q[0]["category"] == "shipping" and q[0]["lang"] == "en"

    # 幂等：重复 upsert 不增行
    counts2 = load_knowledge(manager=mgr, embed_fn=_fake_embed, langs=["en"], scope="kb")
    assert counts2["mojin_kb_en"] == counts["mojin_kb_en"]


# ═══════════════════════════════════════════════════════════════════
#  集成测试（model server 可达时运行）
# ═══════════════════════════════════════════════════════════════════

def _model_server_up() -> bool:
    import requests

    try:
        r = requests.post(
            f"{kb_settings.EMBEDDING_URL}/embeddings",
            json={"texts": ["probe"], "return_dense": True, "return_sparse": False},
            timeout=3,
        )
        return r.ok
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _model_server_up(), reason="本地 model server 不可达")


class _FakeMultiQuery:
    """假 MultiQueryGenerator：确定性改写，不调 DeepSeek。"""

    def generate(self, question, count=3):
        return [f"{question} (expanded 1)", f"{question} (expanded 2)"]


@pytest.fixture
def tmp_kb_manager():
    """真实 embed + 临时 Milvus 的知识库 manager。"""
    db = os.path.join(tempfile.mkdtemp(), "it_kb.db")
    mgr = MilvusManager(client=MilvusClient(uri=db))
    load_knowledge(manager=mgr, langs=["en"], scope="all", rebuild=True)
    return mgr


class TestIntegration:
    def test_rag_chain_with_fake_llm(self, tmp_kb_manager, monkeypatch):
        """真实 embed + 真实 rerank + 假 LLM：验证 RAG 链路返回结构与命中逻辑。"""
        monkeypatch.setattr(mc, "_get_manager", lambda: tmp_kb_manager)
        monkeypatch.setattr(mc, "_get_multi_query", lambda: _FakeMultiQuery())
        llm_calls: list = []

        def fake_llm(messages):
            llm_calls.append(messages)
            return "Delivery within Saudi Arabia takes 3-7 business days."

        monkeypatch.setattr(mc, "_llm_chat", fake_llm)

        result = mc._run_answer("How long is shipping to Riyadh?", "en")
        assert result["answer"] == "Delivery within Saudi Arabia takes 3-7 business days."
        assert result["hit_count"] > 0
        assert result["sources"][0]["doc_id"]
        assert set(result["sources"][0]) >= {"doc_id", "title", "category", "score"}
        assert result["elapsed_ms"] >= 0
        assert llm_calls  # LLM 确被调用，system prompt 含知识
        assert "mojin" in llm_calls[0][0]["content"].lower()

    def test_no_hit_returns_handoff_without_llm(self, monkeypatch):
        """空知识库 → 无命中 → 不调 LLM，直接转人工引导。"""
        db = os.path.join(tempfile.mkdtemp(), "empty.db")
        mgr = MilvusManager(client=MilvusClient(uri=db))
        ensure_collection(mgr, kb_settings.kb_collection("en"), kb_settings.EMBEDDING_DIM, "kb", rebuild=True)

        monkeypatch.setattr(mc, "_get_manager", lambda: mgr)
        monkeypatch.setattr(mc, "_get_multi_query", lambda: _FakeMultiQuery())
        llm_called: list = []
        monkeypatch.setattr(mc, "_llm_chat", lambda msgs: llm_called.append(1) or "x")

        result = mc._run_answer("What is the meaning of life?", "en")
        assert result["hit_count"] == 0
        assert result["sources"] == []
        assert "WhatsApp" in result["answer"]  # 转人工引导文案
        assert not llm_called  # 关键：无命中不调 LLM

    def test_order_help_filters_categories(self, tmp_kb_manager, monkeypatch):
        """order_help 限定 category in (shipping, support)：about 分类块绝不进入结果。"""
        monkeypatch.setattr(mc, "_get_manager", lambda: tmp_kb_manager)
        merged = mc._retrieve_kb("return policy", "en", categories=["shipping", "support"])
        assert merged, "shipping/support 分类下应能检索到政策块"
        assert all(e["category"] in ("shipping", "support") for e in merged)
        # 全库检索同问题会命中 about 类（q1/q2），证明过滤确实生效
        merged_all = mc._retrieve_kb("Who is Mojin?", "en")
        assert any(e["category"] == "about" for e in merged_all)
        merged_filtered = mc._retrieve_kb("Who is Mojin?", "en", categories=["shipping", "support"])
        assert all(e["category"] != "about" for e in merged_filtered)

    def test_search_products_semantic(self, tmp_kb_manager, monkeypatch):
        """语义检索：搜手套命中 glove。"""
        monkeypatch.setattr(mc, "_get_manager", lambda: tmp_kb_manager)
        result = mc._run_search_products("gloves", "en")
        assert result["hit_count"] > 0
        assert any(item["sku"] == "glove" for item in result["results"])
        item = result["results"][0]
        assert set(item) >= {"sku", "name", "price", "old_price", "unit", "sale",
                             "rating", "sizes", "colors", "cat", "image"}
