"""应用配置 — 基于 pydantic-settings 的环境变量管理。

用法:
    from app.config import settings
    settings.PROJECT_NAME    # => "mcp-server-template"
    settings.DATABASE_URL    # => 从 .env 或环境变量读取
"""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AnyUrl, BeforeValidator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors(v: Any) -> list[str] | str:
    """解析 CORS 来源：逗号分隔字符串 或 JSON 数组。"""
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    """所有配置集中管理，优先从环境变量和 .env 文件读取。"""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # ── 项目基础 ──────────────────────────────────────────
    PROJECT_NAME: str = "mcp-server-template"
    VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["dev", "test", "uat", "prod"] = "dev"

    # ── 服务器 ─────────────────────────────────────────────
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    RELOAD: bool = True  # 仅 dev 环境有效

    # ── CORS ───────────────────────────────────────────────
    BACKEND_CORS_ORIGINS: Annotated[list[AnyUrl | str] | str, BeforeValidator(parse_cors)] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    @computed_field
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS]

    # ── 数据库（可选） ──────────────────────────────────────
    DATABASE_URL: str | None = None

    # ── 日志 ───────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"


settings = Settings()
