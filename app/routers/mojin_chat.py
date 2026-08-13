"""Mojin AI 客服模块 — 基于 RAG 的三语问答 / 产品检索 / 订单售后。

仿照 app/routers/example.py 范式：每个业务函数同时是 MCP 工具（@mcp.tool()）
与 FastAPI GET 端点（@router.get）。

三个工具：
- answer_question(query, lang)：RAG 全链路问答（multi-query → hybrid search → rerank → 压缩 → LLM）
- search_products(query, lang)：语义检索产品（embed → dense search → 可选 rerank）
- order_help(query, lang)：同 answer_question，但检索限定 category in (shipping, support)

设计要点：
- 懒初始化：Milvus / LLM / Reranker client 首次调用才创建（import 即连库是大忌）
- 异常兜底：model server / DeepSeek 失败返回友好错误 + 转人工引导，不抛 500 裸异常
- 日志只记 query/lang/elapsed/hit_count/来源，绝不记录 API key
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastmcp import FastMCP
from starlette.concurrency import run_in_threadpool

from app.knowledge.config import kb_settings
from app.utils.log import log

# ── MCP 实例 ──────────────────────────────────────────────────
mcp = FastMCP(name="mojin_chat")

# ── FastAPI 路由 ──────────────────────────────────────────────
router = APIRouter(prefix="/mojin_chat", tags=["AI客服"])

LANGS: tuple[str, ...] = ("en", "cn", "ar")
LANG_NAMES: dict[str, str] = {"en": "English", "cn": "Chinese", "ar": "Arabic"}

# 转人工引导文案（无命中 / 异常兜底用，三语）
HUMAN_HANDOFF: dict[str, str] = {
    "en": (
        "I couldn't find the answer in our knowledge base. "
        "Please contact our human support on WhatsApp or email info@mojin.shop — we'll be happy to help you."
    ),
    "cn": (
        "抱歉，知识库中没有找到对应信息。请联系我们的 WhatsApp 人工客服，"
        "或发送邮件至 info@mojin.shop，我们会尽快为您处理。"
    ),
    "ar": (
        "عذرًا، لم أجد الإجابة في قاعدة المعرفة لدينا. "
        "يرجى التواصل مع فريق الدعم البشري عبر واتساب أو البريد الإلكتروني info@mojin.shop وسنسعد بمساعدتك."
    ),
}

# ── 懒加载客户端 ──────────────────────────────────────────────
# _clients 可能被线程池（run_in_threadpool）中的多个请求并发首次访问，
# 用锁保证每个 client 只构造一次（否则并发下可能重复建 MilvusClient 等昂贵对象）。
_clients: dict[str, object] = {}
_clients_lock = threading.Lock()


def _get_manager():
    """MilvusManager（懒加载，首次调用才连库）。"""
    with _clients_lock:
        if "manager" not in _clients:
            from pymilvus import MilvusClient
            from rag_toolkit.storage.milvus_manager import MilvusManager

            Path(kb_settings.MILVUS_URI).parent.mkdir(parents=True, exist_ok=True)
            _clients["manager"] = MilvusManager(client=MilvusClient(uri=kb_settings.MILVUS_URI))
    return _clients["manager"]  # type: ignore[return-value]


def _get_reranker():
    """XReranker（懒加载）。"""
    with _clients_lock:
        if "reranker" not in _clients:
            from rag_toolkit.pipelines.reranker import XReranker

            _clients["reranker"] = XReranker(
                base_url=kb_settings.RERANKER_URL,
                model_name=kb_settings.RERANKER_MODEL,
                top_k=kb_settings.RERANK_TOP_K,
            )
    return _clients["reranker"]  # type: ignore[return-value]


def _get_multi_query():
    """MultiQueryGenerator（懒加载，DeepSeek 驱动）。"""
    with _clients_lock:
        if "multi_query" not in _clients:
            from rag_toolkit.pipelines.query_expander import MultiQueryGenerator

            # 短文本场景控制 token：max_tokens=128
            _clients["multi_query"] = MultiQueryGenerator(
                api_key=kb_settings.require_llm_key(),
                base_url=kb_settings.LLM_URL,
                model=kb_settings.LLM_MODEL,
                max_tokens=128,
                temperature=0.0,
            )
    return _clients["multi_query"]  # type: ignore[return-value]


def _get_llm():
    """OpenAI-compatible DeepSeek client（懒加载）。key 缺失时抛清晰错误。"""
    with _clients_lock:
        if "llm" not in _clients:
            from openai import OpenAI

            _clients["llm"] = OpenAI(
                api_key=kb_settings.require_llm_key(),
                base_url=kb_settings.LLM_URL,
            )
    return _clients["llm"]  # type: ignore[return-value]


# ── 高频问题 TTL-LRU 缓存 ──────────────────────────────────────────
# 相同 query+lang+mode 的答案短期缓存（TTL 10 分钟，容量 200），省一次
# DeepSeek 调用（~1-3s + token 成本）。仅缓存成功结果（有 answer 且无 error、
# 有命中），错误兜底与转人工不缓存，避免把故障状态缓存 10 分钟。
class _TTLAnswerCache:
    """线程安全 TTL-LRU 缓存：OrderedDict 保序 + popitem(last=False) 淘汰最旧。"""

    def __init__(self, capacity: int = 200, ttl: float = 600.0):
        self._capacity = capacity
        self._ttl = ttl
        self._data: OrderedDict[tuple, tuple[float, Dict]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[Dict]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, result = item
            if time.monotonic() - ts > self._ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)  # LRU 刷新
            return result

    def set(self, key: tuple, result: Dict) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), result)
            self._data.move_to_end(key)
            while len(self._data) > self._capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_answer_cache = _TTLAnswerCache()


def _embed(texts: List[str]) -> List[List[float]]:
    """调用自建 model server /embeddings（bge-m3 dense 向量）。失败向上抛，由调用方兜底。"""
    import requests

    resp = requests.post(
        f"{kb_settings.EMBEDDING_URL}/embeddings",
        json={"texts": texts, "return_dense": True, "return_sparse": False},
        timeout=120,
    )
    resp.raise_for_status()
    return [item["dense_vector"]["vector"] for item in resp.json()]


def _llm_chat(messages: List[dict]) -> str:
    """DeepSeek 对话生成（失败向上抛，由调用方兜底）。"""
    resp = _get_llm().chat.completions.create(
        model=kb_settings.LLM_MODEL,
        messages=messages,
        max_tokens=kb_settings.LLM_MAX_TOKENS,
        temperature=kb_settings.LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content or ""


# ── 校验 ──────────────────────────────────────────────────────

def _check_lang(lang: str) -> str:
    """lang 校验：en/cn/ar，非法抛 400。"""
    if lang not in LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"lang 必须是 en|cn|ar 之一，got {lang!r}",
        )
    return lang


def _system_prompt(lang: str, context: str) -> str:
    """三语 system prompt：只依据知识回答；未覆盖时明确转人工；用用户语言回答。"""
    return (
        "You are the AI customer service assistant of Mojin (mojin.shop), a protective equipment brand "
        "serving Saudi Arabia. Answer ONLY based on the knowledge base below. Never invent facts, prices, "
        "or policies. If the knowledge does not cover the question, clearly say you cannot answer and guide "
        f"the customer to human support via WhatsApp. Keep the answer short and friendly, "
        f"always in {LANG_NAMES[lang]}. "
        "Knowledge base:\n{context}"
    ).format(context=context)


def _friendly_error(stage: str, t0: float, lang: str = "en") -> Dict:
    """异常兜底：按请求语言返回友好错误 + 转人工引导，不抛 500 裸异常。"""
    log.error("mojin_chat {} failed | 转人工兜底", stage)
    suffix = {
        "en": "(An internal service error occurred, please try again later.)",
        "cn": "（服务暂时不可用，请稍后重试。）",
        "ar": "(حدث خطأ داخلي في الخدمة، يرجى المحاولة لاحقًا.)",
    }
    return {
        "answer": HUMAN_HANDOFF.get(lang, HUMAN_HANDOFF["en"]) + " " + suffix.get(lang, suffix["en"]),
        "sources": [],
        "hit_count": 0,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "error": f"{stage} service unavailable",
    }


# ── RAG 检索核心 ──────────────────────────────────────────────

# 知识库块数缓存（5 分钟）：避免每次请求都 count Milvus
_block_count_cache: dict[str, tuple[float, int]] = {}


def _kb_block_count(lang: str) -> int:
    """当前语言知识库块数（5 分钟缓存）。查询失败返回 0（视为小库）。"""
    now = time.monotonic()
    cached = _block_count_cache.get(lang)
    if cached is not None and now - cached[0] < 300:
        return cached[1]
    try:
        stats = _get_manager().milvus.get_collection_stats(kb_settings.kb_collection(lang))
        count = int(stats.get("row_count", 0) or 0)
    except Exception as e:
        log.warning("kb block count unavailable | lang={} err={}", lang, type(e).__name__)
        count = 0
    _block_count_cache[lang] = (now, count)
    return count


def _use_multi_query(lang: str) -> bool:
    """MultiQuery 改写开关：总开关关闭，或知识库块数不足 MIN_BLOCKS 时跳过。"""
    if not kb_settings.MULTI_QUERY_ENABLED:
        return False
    return _kb_block_count(lang) >= kb_settings.MULTI_QUERY_MIN_BLOCKS


def _hybrid_search(manager, collection: str, query: str, vec: List[float], expr: str = "") -> List[dict]:
    """单条查询的 hybrid search（dense + BM25），返回 entity 列表（附 _distance）。

    expr: Milvus 表达式过滤器（如 order_help 的 category in ['shipping', 'support']）。
    """
    hits = manager.hybrid_search(
        collection_name=collection,
        vec_data=[vec],
        text_data=[query],
        vec_limit=kb_settings.RAG_HYBRID_TOP_K * 2,
        full_limit=kb_settings.RAG_HYBRID_TOP_K * 2,
        res_limit=kb_settings.RAG_HYBRID_TOP_K * 2,
        output_fields=["id", "doc_id", "title", "category", "text"],
        vec_expr=expr,
        full_expr=expr,
    )
    entities = []
    for item in hits[0]:
        e = dict(item["entity"])
        e["_distance"] = float(item.get("distance", 0.0))
        entities.append(e)
    return entities


def _retrieve_kb(query: str, lang: str, categories: Optional[List[str]] = None) -> List[dict]:
    """RAG 检索：multi-query 改写 → 逐条 hybrid search → 合并去重。

    categories 非空时用 Milvus 表达式过滤（order_help 用 category in [shipping, support]）。
    """
    manager = _get_manager()
    collection = kb_settings.kb_collection(lang)

    # 1. MultiQueryGenerator 改写（默认关 / 小库跳过；开启时失败降级为只用原文）
    queries: List[str] = [query]
    try:
        if _use_multi_query(lang):
            mq = _get_multi_query()
            alt = mq.generate(query, count=kb_settings.MULTI_QUERY_COUNT) or []
            for q in alt:
                q = q.strip()
                if q and q not in queries:
                    queries.append(q)
    except Exception as e:
        log.warning("multi-query generation skipped | err={}", type(e).__name__)
    queries = queries[: kb_settings.MULTI_QUERY_COUNT + 1]

    # 2. 逐条查询 embed（一次 batch 请求）+ hybrid search
    vecs = _embed(queries)
    expr = f"category in {categories}" if categories else ""
    all_entities: List[dict] = []
    for q, vec in zip(queries, vecs, strict=False):
        try:
            all_entities.extend(_hybrid_search(manager, collection, q, vec, expr=expr))
        except Exception as e:
            log.warning("hybrid_search failed for sub-query | err={}", type(e).__name__)

    # 3. 合并去重（按 chunk 主键 id）
    seen: set[str] = set()
    merged: List[dict] = []
    for e in all_entities:
        if e.get("id") in seen:
            continue
        seen.add(e["id"])
        merged.append(e)
    return merged


def _rerank(merged: List[dict], query: str) -> List[tuple]:
    """XReranker 重排，返回 [(entity, score)] 按分数降序；失败降级为原顺序（score 用 distance）。"""
    if not merged:
        return []
    texts = [m["text"] for m in merged]
    try:
        results = _get_reranker().rerank(texts, query)  # [(idx, score)]
    except Exception as e:
        log.warning("rerank failed, fallback to hybrid order | err={}", type(e).__name__)
        results = []
    if results:
        scored = [(merged[idx], float(score)) for idx, score in results]
        scored = [(m, s) for m, s in scored if s >= kb_settings.RERANK_MIN_SCORE]
    else:
        # 降级：按 hybrid 距离排序（去重后顺序即距离降序）
        scored = [(m, m.get("_distance", 0.0)) for m in merged]
    return scored


def _run_answer(query: str, lang: str, categories: Optional[List[str]] = None) -> Dict:
    """RAG 全链路问答（带 10 分钟 TTL-LRU 缓存）。

    缓存键 (query, lang, mode)，mode 区分 answer / order_help（检索范围不同）。
    只缓存成功结果（有 answer、无 error、有命中）；错误兜底与转人工不缓存。
    命中时 elapsed_ms 为本次实际耗时，并带 cached=True 标记。
    """
    t0 = time.perf_counter()
    lang = _check_lang(lang)
    mode = "order_help" if categories else "answer"
    key = (query, lang, mode)

    cached = _answer_cache.get(key)
    if cached is not None:
        elapsed = int((time.perf_counter() - t0) * 1000)
        result = dict(cached)
        result["elapsed_ms"] = elapsed
        result["cached"] = True
        log.info("mojin_chat answer cache hit | query={} lang={} mode={} elapsed_ms={}", query, lang, mode, elapsed)
        return result

    result = _run_answer_impl(query, lang, categories=categories)
    if result.get("answer") and "error" not in result and result.get("hit_count", 0) > 0:
        _answer_cache.set(key, result)
    return result


def _run_answer_impl(query: str, lang: str, categories: Optional[List[str]] = None) -> Dict:
    """RAG 全链路：检索 → rerank → BM25 压缩 → DeepSeek 生成（同步执行，HTTP/MCP 侧跑在线程池）。"""
    t0 = time.perf_counter()
    lang = _check_lang(lang)
    log.info("mojin_chat answer start | query={} lang={} categories={}", query, lang, categories)

    # 1. 检索
    try:
        merged = _retrieve_kb(query, lang, categories=categories)
    except Exception as e:
        log.error("retrieval failed | err={}", type(e).__name__)
        return _friendly_error("retrieval", t0, lang)

    # 2. rerank + 无命中判定（无命中不调 LLM，直接转人工）
    scored = _rerank(merged, query)
    if not scored:
        elapsed = int((time.perf_counter() - t0) * 1000)
        log.info("mojin_chat no hit | query={} lang={} elapsed_ms={}", query, lang, elapsed)
        return {"answer": HUMAN_HANDOFF[lang], "sources": [], "hit_count": 0, "elapsed_ms": elapsed}

    top = scored[: kb_settings.RERANK_TOP_K]

    # 3. BM25 压缩上下文
    from rag_toolkit.pipelines.context_expander import BM25Compressor

    context = "\n".join(f"[{m['doc_id']}] {m['text']}" for m, _ in top)
    try:
        compressor = BM25Compressor(rate=kb_settings.COMPRESS_RATE)
        context = compressor.compress(context, query=query) or context
    except Exception as e:
        log.warning("context compression skipped | err={}", type(e).__name__)

    # 4. DeepSeek 生成（失败兜底转人工）
    try:
        answer = _llm_chat([
            {"role": "system", "content": _system_prompt(lang, context)},
            {"role": "user", "content": query},
        ])
    except Exception as e:
        log.error("llm generation failed | err={}", type(e).__name__)
        return _friendly_error("llm", t0, lang)

    elapsed = int((time.perf_counter() - t0) * 1000)
    sources = [
        {"doc_id": m["doc_id"], "title": m["title"], "category": m["category"], "score": round(s, 4)}
        for m, s in top
    ]
    log.info("mojin_chat answer done | query={} lang={} hit_count={} elapsed_ms={}",
             query, lang, len(top), elapsed)
    return {"answer": answer, "sources": sources, "hit_count": len(top), "elapsed_ms": elapsed}


def _run_search_products(query: str, lang: str) -> Dict:
    """产品语义检索：embed → dense search → 可选 rerank。"""
    t0 = time.perf_counter()
    lang = _check_lang(lang)
    log.info("mojin_chat product search | query={} lang={}", query, lang)

    manager = _get_manager()
    collection = kb_settings.product_collection(lang)

    try:
        vec = _embed([query])[0]
        hits = manager.search(
            collection_name=collection,
            vectors=[vec],
            anns_field="vector",
            limit=kb_settings.PRODUCT_SEARCH_TOP_K * 2,
            output_fields=[
                "sku", "name", "price", "old_price", "unit", "sale", "rating",
                "reviews", "sizes", "colors", "cat", "image", "text",
            ],
        )
    except Exception as e:
        log.error("product search failed | err={}", type(e).__name__)
        return {"results": [], "hit_count": 0,
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                "error": "product search unavailable"}

    entities = [dict(h["entity"]) for h in hits[0]]

    # 可选 rerank：失败或降级为原顺序
    try:
        results = _get_reranker().rerank([e["text"] for e in entities], query)
        if results:
            entities = [entities[idx] for idx, _ in results]
    except Exception as e:
        log.warning("product rerank skipped | err={}", type(e).__name__)

    entities = entities[: kb_settings.PRODUCT_SEARCH_TOP_K]
    items = [
        {
            "sku": e.get("sku", ""), "name": e.get("name", ""), "price": e.get("price", ""),
            "old_price": e.get("old_price", ""), "unit": e.get("unit", ""), "sale": e.get("sale", ""),
            "rating": e.get("rating", ""), "sizes": e.get("sizes", ""), "colors": e.get("colors", ""),
            "cat": e.get("cat", ""), "image": e.get("image", ""),
        }
        for e in entities
    ]
    elapsed = int((time.perf_counter() - t0) * 1000)
    log.info("mojin_chat product search done | query={} lang={} hit_count={} elapsed_ms={}",
             query, lang, len(items), elapsed)
    return {"results": items, "hit_count": len(items), "elapsed_ms": elapsed}


# ═══════════════════════════════════════════════════════════════
#  MCP 工具 — 同时作为 HTTP GET 端点
# ═══════════════════════════════════════════════════════════════

@mcp.tool()
@router.get("/answer_question")
async def answer_question(query: str, lang: str = "en") -> Dict:
    """基于知识库回答客户问题（RAG 全链路，三语客服）

    Args:
        query: 客户问题（支持英语/中文/阿拉伯语）
        lang: 回答语言，en | cn | ar，默认 en
    """
    _check_lang(lang)
    log.info("answer_question called | query={} lang={}", query, lang)
    return await run_in_threadpool(_run_answer, query, lang)


@mcp.tool()
@router.get("/search_products")
async def search_products(query: str, lang: str = "en") -> Dict:
    """语义检索产品目录

    Args:
        query: 产品搜索词，如帽子/gloves/قفازات/便宜/蓝色
        lang: 检索语言，en | cn | ar，默认 en
    """
    _check_lang(lang)
    log.info("search_products called | query={} lang={}", query, lang)
    return await run_in_threadpool(_run_search_products, query, lang)


@mcp.tool()
@router.get("/order_help")
async def order_help(query: str, lang: str = "en") -> Dict:
    """订单/配送/售后问题解答（检索限定 shipping + support 分类）

    Args:
        query: 订单、配送、退换货相关问题
        lang: 回答语言，en | cn | ar，默认 en
    """
    _check_lang(lang)
    log.info("order_help called | query={} lang={}", query, lang)
    return await run_in_threadpool(_run_answer, query, lang, ["shipping", "support"])
