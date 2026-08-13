# MCP Server Template

## 架构

```
请求 → FastAPI (HTTP) ─┬→ /example → FastMCP → MCP 工具
                       │             （/streamable-http）
                       └→ /api/... → FastAPI 路由
```

**FastAPI** 作为主容器，承载 HTTP 路由和中间件。
**FastMCP** 挂载为子应用，通过 `http_app()` 暴露 MCP 协议端点。

## 添加新模块

复制 `app/routers/example.py`，按以下步骤：

```python
# 1. 创建新文件 app/routers/your_module.py
mcp = FastMCP(name="your_module")
router = APIRouter(prefix="/your_module", tags=["你的模块"])

@mcp.tool()
@router.get("/your_tool")
async def your_tool(param: str) -> str:
    """工具描述（会给 LLM 看）"""
    return f"result: {param}"

# 2. 在 app/main.py 中注册
from fastmcp.utilities.lifespan import combine_lifespans
from app.routers import your_module

your_module_app = your_module.mcp.http_app(path="/", stateless_http=True)
app.include_router(your_module.router)
app.mount("/your_module", app=your_module_app)
# 并把 your_module_app.lifespan 合并进 FastAPI lifespan：
# app = FastAPI(lifespan=combine_lifespans(lifespan, your_module_app.lifespan))
```

## 模块可同时做两件事

- **MCP 工具** → 被 AI 智能体调用
- **HTTP 端点** → 被普通 web 客户端调用

不需要分开写两个接口，加两个装饰器就行。
