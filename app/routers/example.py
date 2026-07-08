"""示例模块 — 展示如何创建一个带 MCP 工具和 FastAPI 路由的业务模块。

这个文件是模板，新建业务模块时复制它并修改：
  1. 改名（如 product.py、knowledge.py）
  2. 修改 mcp 的 name 参数
  3. 修改 APIRouter 的 prefix 和 tags
  4. 在 main.py 中注册 router 和 mount

MCP 工具注解:
    - @mcp.tool() 装饰的方法会成为 MCP 协议可调用的工具
    - 函数签名中的类型注解和文档字符串会暴露给 LLM
    - 可以同时用 @router.get/post 暴露为 HTTP 端点
"""

from typing import Dict, Optional

from fastapi import APIRouter
from mcp.server import FastMCP

from app.config import settings
from app.utils.log import log

# ── MCP 实例 ──────────────────────────────────────────────────
# name: MCP 工具命名空间，LLM 通过这个名字识别工具来源
# stateless_http: 启用无状态 HTTP 传输模式
mcp = FastMCP(name="example", stateless_http=True)

# ── FastAPI 路由 ──────────────────────────────────────────────
router = APIRouter(prefix="/example", tags=["示例"])


# ═══════════════════════════════════════════════════════════════
#  MCP 工具 — 可同时作为 HTTP 端点暴露
# ═══════════════════════════════════════════════════════════════


@mcp.tool()
@router.get("/hello")
async def hello(name: Optional[str] = None) -> str:
    """打招呼（MCP 工具 + HTTP 端点）

    Args:
        name: 称呼名字，不传则返回通用问候
    """
    log.info("hello called | name={}", name)
    target = name or "World"
    return f"Hello, {target}! Running {settings.PROJECT_NAME} v{settings.VERSION}"


@mcp.tool()
@router.get("/system_info")
async def system_info() -> Dict:
    """获取系统信息（MCP 工具 + HTTP 端点）

    返回当前运行环境的基础信息，方便调试排查。
    """
    import platform
    import os

    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
    }


@mcp.tool()
@router.get("/echo")
async def echo(message: str) -> Dict:
    """回显消息（MCP 工具 + HTTP 端点）

    用于测试连通性。

    Args:
        message: 要回显的消息
    """
    return {
        "code": 0,
        "message": message,
        "length": len(message),
    }
