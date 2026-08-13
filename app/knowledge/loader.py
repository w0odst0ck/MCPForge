"""知识摄入管道 — 读 3 个知识源 → 按语言分块 → embed → 写入对应语言集合。

知识源（只读，carewell-shop 仓）：
- faq-knowledge.json：9 条 FAQ，每条含 en/cn/ar 三语 question+answer，附 categories
- shipping.html / return-policy.html / privacy.html：政策页，含 id=content-en 与 id=content-ar 区块
  （无 content-cn，中文按英文文本处理，lang 字段标 cn）
- products.csv：10 SKU，列含 sku,name_*/sub_*/cat_*/desc_*、price 等

分块规则：
- FAQ：每条每语言一块，文本 = "Q: {question} A: {answer}"，doc_id 如 faq-q1
- 政策页：提取指定语言区块纯文本，按段落贪心合并成 300-600 字块，doc_id 如 policy-shipping-en-1
- 产品：每 SKU 每语言一块，文本 = 名称 + 副标题 + 描述（该语言列），结构化字段随块存储

写入：按语言分集合（mojin_kb_* / mojin_products_*），按主键 id upsert（幂等）；
--rebuild 时先 drop 再重建。集合创建走 rag_toolkit.MilvusManager.create_collection。
"""

from __future__ import annotations

import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Optional

from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType
from pymilvus.milvus_client.index import IndexParams
from rag_toolkit.storage.milvus_manager import MilvusManager

from app.knowledge.config import kb_settings

LANGS: tuple[str, ...] = ("en", "cn", "ar")

# 政策页 → (title, category) 映射（order_help 检索 shipping+support 分类）
POLICY_META: dict[str, tuple[str, str]] = {
    "shipping": ("Shipping Policy", "shipping"),
    "return-policy": ("Return Policy", "support"),
    "privacy": ("Privacy Policy", "support"),
}

EmbedFn = Callable[[list[str]], list[list[float]]]


# ═══════════════════════════════════════════════════════════════════
#  Schema
# ═══════════════════════════════════════════════════════════════════

def _kb_schema(dim: int) -> CollectionSchema:
    """知识库集合：FAQ + 政策。字段：id(str 主键)、vector、sparse_vector、text、
    sparse_text、doc_id、title、category、lang。BM25 function 由 sparse_text 生成 sparse_vector。"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="sparse_text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=8),
    ]
    return _build_schema(fields, "Mojin knowledge base (FAQ + policies)")


def _product_schema(dim: int) -> CollectionSchema:
    """产品库集合。字段：id(str 主键)、vector、sparse_vector、text、sparse_text、
    sku、name、price、old_price、unit、sale、rating、reviews、sizes、colors、cat、image、lang。"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="sparse_text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
        FieldSchema(name="sku", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="price", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="old_price", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="unit", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="sale", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="rating", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="reviews", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="sizes", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="colors", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="cat", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="image", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=8),
    ]
    return _build_schema(fields, "Mojin product catalog")


def _build_schema(fields: list[FieldSchema], description: str) -> CollectionSchema:
    bm25 = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["sparse_text"],
        output_field_names=["sparse_vector"],
    )
    return CollectionSchema(fields=fields, functions=[bm25], description=description)


def ensure_collection(manager: MilvusManager, name: str, dim: int, schema_kind: str,
                      rebuild: bool = False) -> None:
    """创建集合（含 dense + sparse 索引）。rebuild=True 时先 drop。

    MilvusManager.create_collection 传自定义 schema 时只建集合不建索引，
    此处手动补齐两个索引（与默认 schema 分支行为一致）。
    """
    schema = _kb_schema(dim) if schema_kind == "kb" else _product_schema(dim)
    if rebuild and manager.has_collection(name):
        manager.drop_collection(name)
    if not manager.has_collection(name):
        manager.create_collection(name, schema=schema)
        manager.milvus.create_index(
            name, IndexParams(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        )
        manager.milvus.create_index(
            name,
            IndexParams(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"),
        )


# ═══════════════════════════════════════════════════════════════════
#  分块
# ═══════════════════════════════════════════════════════════════════

def chunk_faqs(faq_data: dict, lang: str) -> list[dict[str, str]]:
    """FAQ 分块：每条每语言一块，文本 = Q: {question} A: {answer}。"""
    blocks: list[dict[str, str]] = []
    for faq in faq_data.get("faqs", []):
        entry = faq.get(lang) or faq.get("en")
        if not entry or not entry.get("question"):
            continue
        q = entry["question"].strip()
        a = (entry.get("answer") or "").strip()
        doc_id = f"faq-{faq['id']}"
        blocks.append({
            "id": f"{doc_id}-{lang}",
            "doc_id": doc_id,
            "title": q[:200],
            "category": faq.get("category", ""),
            "lang": lang,
            "text": f"Q: {q} A: {a}",
        })
    return blocks


class _ContentExtractor(HTMLParser):
    """提取 id=content-{lang} 的 div 区块，块级元素转为独立段落。"""

    BLOCK_TAGS = {"h2", "h3", "p", "li", "tr", "br"}

    def __init__(self, target_id: str):
        super().__init__(convert_charrefs=True)
        self.target_id = target_id
        self._capturing = False
        self._depth = 0
        self._buf: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]):
        attr_dict = dict(attrs)
        if not self._capturing and attr_dict.get("id") == self.target_id:
            self._capturing = True
            self._depth = 1
            return
        if self._capturing:
            if tag == "div":
                self._depth += 1
            if tag in self.BLOCK_TAGS:
                self._flush()

    def handle_endtag(self, tag: str):
        if not self._capturing:
            return
        if tag == "div":
            self._depth -= 1
            if self._depth <= 0:
                self._capturing = False
                self._flush()
        elif tag in self.BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str):
        if self._capturing:
            self._buf.append(data)

    def _flush(self):
        text = " ".join("".join(self._buf).split())
        self._buf = []
        if text:
            self.parts.append(text)


def extract_policy_parts(html_path: Path, lang: str) -> list[str]:
    """提取政策页指定语言区块的纯文本段落列表。cn 无区块 → 用 en 文本。"""
    target_id = "content-en" if lang == "cn" else f"content-{lang}"
    parser = _ContentExtractor(target_id)
    parser.feed(Path(html_path).read_text(encoding="utf-8"))
    return [p for p in parser.parts if p]


def _split_long(part: str, max_len: int) -> list[str]:
    """超长段落按句子/逗号切分；单段仍超长时硬切兜底。"""
    if len(part) <= max_len:
        return [part]
    chunks, cur = [], ""
    for piece in re.split(r"(?<=[。！？!?；;])\s*", part):
        if not piece:
            continue
        if len(cur) + len(piece) > max_len and cur:
            chunks.append(cur)
            cur = piece
        else:
            cur = (cur + " " + piece).strip()
    if cur:
        chunks.append(cur)
    # 兜底：标点切分后仍有单块超长（如无标点的连续文本）→ 硬切，避免超嵌入长度上限
    out: list[str] = []
    for c in chunks:
        while len(c) > max_len:
            out.append(c[:max_len])
            c = c[max_len:]
        if c:
            out.append(c)
    return out


def chunk_policy(parts: list[str], name: str, lang: str,
                 min_len: int = 300, max_len: int = 600) -> list[dict[str, str]]:
    """政策段落贪心合并为 300-600 字块，doc_id 如 policy-shipping-en-1。"""
    title, category = POLICY_META.get(name, (name, "support"))
    blocks: list[dict[str, str]] = []
    seq, cur, cur_len = 0, [], 0

    def flush():
        nonlocal seq, cur, cur_len
        if not cur:
            return
        text = " ".join(cur)
        for piece in _split_long(text, max_len):
            seq += 1
            doc_id = f"policy-{name}-{lang}-{seq}"
            blocks.append({
                "id": doc_id,
                "doc_id": doc_id,
                "title": f"{title} · {name}",
                "category": category,
                "lang": lang,
                "text": piece,
            })
        cur, cur_len = [], 0

    for part in parts:
        if cur and cur_len + len(part) > max_len and cur_len >= min_len:
            flush()
        cur.append(part)
        cur_len += len(part) + 1
    flush()
    return blocks


def chunk_products(rows: list[dict], lang: str) -> list[dict[str, str]]:
    """产品分块：每 SKU 每语言一块，文本 = 名称 + 副标题 + 描述（该语言列）。"""
    blocks: list[dict[str, str]] = []
    for r in rows:
        sku = (r.get("sku") or "").strip()
        if not sku:
            continue
        name = (r.get(f"name_{lang}") or "").strip()
        sub = (r.get(f"sub_{lang}") or "").strip()
        desc = (r.get(f"desc_{lang}") or "").replace("|", "，").strip()
        text = " ".join(p for p in (name, sub, desc) if p)
        blocks.append({
            "id": f"{sku}-{lang}",
            "sku": sku,
            "name": name,
            "price": (r.get("price") or "").strip(),
            "old_price": (r.get("old_price") or "").strip(),
            "unit": (r.get("unit") or "").strip(),
            "sale": (r.get("sale") or "").strip(),
            "rating": (r.get("rating") or "").strip(),
            "reviews": (r.get("reviews") or "").strip(),
            "sizes": (r.get("sizes") or "").strip(),
            "colors": (r.get("colors") or "").strip(),
            "cat": (r.get(f"cat_{lang}") or "").strip(),
            "image": (r.get("image") or "").strip(),
            "lang": lang,
            "text": text,
        })
    return blocks


# ═══════════════════════════════════════════════════════════════════
#  加载与写入
# ═══════════════════════════════════════════════════════════════════

def default_embed_fn(texts: list[str]) -> list[list[float]]:
    """调用自建 model server 的 /embeddings 端点（bge-m3，dense 1024 维）。"""
    import requests

    resp = requests.post(
        f"{kb_settings.EMBEDDING_URL}/embeddings",
        json={"texts": texts, "return_dense": True, "return_sparse": False},
        timeout=120,
    )
    resp.raise_for_status()
    return [item["dense_vector"]["vector"] for item in resp.json()]


def default_manager() -> MilvusManager:
    from pymilvus import MilvusClient

    uri = kb_settings.MILVUS_URI
    # Milvus Lite 要求本地 .db 文件父目录存在；远端 URI（http/tcp/file 等）跳过建目录
    if not uri.startswith(("http://", "https://", "tcp://", "file://")):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    return MilvusManager(client=MilvusClient(uri=uri))


def _collect_blocks(carewell: Path, lang: str, scope: str) -> dict[str, list[dict]]:
    """收集指定语言的 KB / 产品分块。返回 {schema_kind: blocks}。"""
    out: dict[str, list[dict]] = {}
    if scope in ("kb", "all"):
        faq = json.loads((carewell / kb_settings.FAQ_FILE).read_text(encoding="utf-8"))
        kb_blocks: list[dict] = chunk_faqs(faq, lang)
        for pfile in kb_settings.POLICY_FILES:
            name = pfile.removesuffix(".html")
            parts = extract_policy_parts(carewell / pfile, lang)
            kb_blocks.extend(chunk_policy(parts, name, lang))
        out["kb"] = kb_blocks
    if scope in ("products", "all"):
        with (carewell / kb_settings.PRODUCTS_FILE).open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out["products"] = chunk_products(rows, lang)
    return out


def _write(manager: MilvusManager, collection: str, blocks: list[dict],
           embed_fn: EmbedFn, batch: int = 32) -> int:
    """embed + upsert 一批块，返回写入条数。"""
    if not blocks:
        return 0
    texts = [b["text"] for b in blocks]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch):
        vectors.extend(embed_fn(texts[i:i + batch]))
    rows = []
    for b, vec in zip(blocks, vectors, strict=False):
        row: dict[str, Any] = {"id": b["id"], "vector": vec, "text": b["text"], "sparse_text": b["text"]}
        for k, v in b.items():
            if k not in ("id", "text"):
                row[k] = v
        rows.append(row)
    manager.upsert(collection, rows)
    return len(rows)

def load_knowledge(
    manager: Optional[MilvusManager] = None,
    embed_fn: Optional[EmbedFn] = None,
    langs: Optional[list[str]] = None,
    scope: str = "all",
    rebuild: bool = False,
) -> dict[str, int]:
    """执行知识摄入，返回 {集合名: 写入块数}。

    - manager: 可注入（测试用临时 Milvus）；默认连 kb_settings.MILVUS_URI
    - embed_fn: 可注入（测试用 fake embed）；默认调 model server
    - langs: 如 ["en","cn","ar"]；scope: kb | products | all
    """
    manager = manager or default_manager()
    embed_fn = embed_fn or default_embed_fn
    langs = langs or list(LANGS)
    if scope not in ("kb", "products", "all"):
        raise ValueError(f"scope 必须是 kb|products|all，got {scope!r}")
    carewell = Path(kb_settings.CAREWELL_PATH)
    if not carewell.is_dir():
        raise FileNotFoundError(f"知识源目录不存在（可设 MOJIN_CAREWELL_PATH 覆盖）: {carewell}")

    counts: dict[str, int] = {}
    for lang in langs:
        if lang not in LANGS:
            raise ValueError(f"lang 必须是 en|cn|ar，got {lang!r}")
        blocks_by_kind = _collect_blocks(carewell, lang, scope)
        for kind, blocks in blocks_by_kind.items():
            if kind == "kb":
                collection = kb_settings.kb_collection(lang)
                schema_kind = "kb"
            else:
                collection = kb_settings.product_collection(lang)
                schema_kind = "products"
            ensure_collection(manager, collection, kb_settings.EMBEDDING_DIM, schema_kind, rebuild=rebuild)
            counts[collection] = _write(manager, collection, blocks, embed_fn)
    return counts
