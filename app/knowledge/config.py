"""Mojin 客服配置 — Milvus / Embedding / Reranker / LLM / 知识源路径集中管理。

用法:
    from app.knowledge.config import kb_settings
    kb_settings.MILVUS_URI      # => .../data/mojin_kb.db
    kb_settings.LLM_API_KEY     # => 环境变量 DEEPSEEK_API_KEY（缺失为空串）

环境变量覆盖: 所有字段均可用 `MOJIN_` 前缀的环境变量覆盖，
如 `MOJIN_CAREWELL_PATH=/path/to/carewell-shop`；
`LLM_API_KEY` 直接读 `DEEPSEEK_API_KEY`（任务约束：不硬编码、不进文件）。
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# MCPForge 项目根目录（本文件位于 app/knowledge/ 下）
MCPFORGE_ROOT = Path(__file__).resolve().parent.parent.parent
# mcpforge-mojin 目录（MCPForge 的上一级）
MOJIN_PROJECT_ROOT = MCPFORGE_ROOT.parent
# 知识源默认路径：mcpforge-mojin/../carewell-shop
DEFAULT_CAREWELL_PATH = str(MOJIN_PROJECT_ROOT.parent / "carewell-shop")


class KnowledgeSettings(BaseSettings):
    """Mojin 客服 RAG 全部连接配置。"""

    model_config = SettingsConfigDict(
        env_prefix="MOJIN_",
        env_file=MCPFORGE_ROOT / ".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # ── Milvus Lite ──────────────────────────────────────────
    # 本地嵌入式库文件，data/ 目录已加入 .gitignore
    MILVUS_URI: str = str(MCPFORGE_ROOT / "data" / "mojin_kb.db")

    # ── Model Server（自建 bge-m3 / reranker） ────────────────
    EMBEDDING_URL: str = "http://127.0.0.1:9997"
    RERANKER_URL: str = "http://127.0.0.1:9997"
    EMBEDDING_MODEL: str = "bge-m3"
    RERANKER_MODEL: str = "bge-reranker-v2-m3"
    EMBEDDING_DIM: int = 1024

    # ── LLM（DeepSeek，OpenAI 兼容） ──────────────────────────
    LLM_URL: str = "https://api.deepseek.com/v1"
    # 只从环境变量 DEEPSEEK_API_KEY 读取（validation_alias 决定 env 名）
    LLM_API_KEY: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")
    LLM_MODEL: str = "deepseek-chat"
    LLM_MAX_TOKENS: int = 400
    LLM_TEMPERATURE: float = 0.3

    # ── 集合命名（按语言分集合，避免过滤复杂度） ──────────────
    KB_COLLECTIONS: dict[str, str] = {
        "en": "mojin_kb_en",
        "cn": "mojin_kb_cn",
        "ar": "mojin_kb_ar",
    }
    PRODUCT_COLLECTIONS: dict[str, str] = {
        "en": "mojin_products_en",
        "cn": "mojin_products_cn",
        "ar": "mojin_products_ar",
    }

    # ── 知识源路径（只读，位于 carewell-shop 仓） ─────────────
    CAREWELL_PATH: str = DEFAULT_CAREWELL_PATH
    FAQ_FILE: str = "faq-knowledge.json"
    POLICY_FILES: list[str] = ["shipping.html", "return-policy.html", "privacy.html"]
    PRODUCTS_FILE: str = "products.csv"

    # ── RAG 链路参数 ─────────────────────────────────────────
    RAG_HYBRID_TOP_K: int = 8          # 每条改写查询 hybrid search 返回条数
    RERANK_TOP_K: int = 5              # rerank 后保留条数
    RERANK_MIN_SCORE: float = 0.15     # rerank 分数下限，低于视为无命中
    PRODUCT_SEARCH_TOP_K: int = 5      # 产品检索条数
    MULTI_QUERY_COUNT: int = 3         # MultiQueryGenerator 改写条数
    MULTI_QUERY_ENABLED: bool = False  # MultiQuery 改写总开关（知识库小，默认关，省一次 DeepSeek 调用）
    MULTI_QUERY_MIN_BLOCKS: int = 50   # 知识库块数低于此值即使开启也跳过改写
    COMPRESS_RATE: float = 0.7         # BM25Compressor 保留比例

    def kb_collection(self, lang: str) -> str:
        return self.KB_COLLECTIONS[lang]

    def product_collection(self, lang: str) -> str:
        return self.PRODUCT_COLLECTIONS[lang]

    def require_llm_key(self) -> str:
        """返回 LLM API key，缺失时抛出带指引的清晰错误（不写日志）。"""
        key = (self.LLM_API_KEY or "").strip()
        if not key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY 未设置：请通过环境变量注入（export DEEPSEEK_API_KEY=...），"
                "或在 MCPForge/.env 中配置，服务不会硬编码 API key。"
            )
        return key


kb_settings = KnowledgeSettings()
