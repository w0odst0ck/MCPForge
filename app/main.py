"""MCP Server 模板 — 应用入口。

架构说明:
    FastAPI 承载 HTTP 路由，FastMCP 挂载为子应用提供 MCP 协议端点。
    每个业务模块对应一个独立的 FastMCP 实例，通过 app.mount 挂载到指定路径。

快速开始:
    # 安装依赖
    uv sync

    # 开发模式
    uv run uvicorn app.main:app --reload

    # 生产模式
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

    # FastAPI 原生方式
    uv run fastapi dev app/main.py
"""

from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import example
from app.utils.log import Loggers, log


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。

    - 启动时：初始化资源（数据库连接、MCP session manager 等）
    - 关闭时：清理资源
    """
    async with AsyncExitStack() as stack:
        # ── 启动初始化 ────────────────────────────────────
        # 示例：启动 MCP session manager
        await stack.enter_async_context(example.mcp.session_manager.run())
        # TODO: 在此添加更多资源初始化（DB 连接池、缓存客户端等）

        log.info("Application started | env={} | host={}:{}",
                 settings.ENVIRONMENT, settings.HOST, settings.PORT)

        try:
            yield
        finally:
            log.info("Application shutting down...")


# ── FastAPI 应用 ──────────────────────────────────────────────
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ── 路由注册 ──────────────────────────────────────────────────
app.include_router(example.router)

# ── MCP 子应用挂载 ────────────────────────────────────────────
# 每个 FastMCP 实例通过 streamable_http_app() 生成 ASGI 子应用，
# 以 /{module_name} 路径挂载到 FastAPI 上。
# MCP 客户端通过 http://host:port/{module_name} 连接。
app.mount("/example", app=example.mcp.streamable_http_app())

# ── 日志 ──────────────────────────────────────────────────────
Loggers.init_config()


# ── 直接运行入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app="app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )
