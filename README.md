# MCP Server Template

通用 MCP Server 脚手架 — **FastAPI + FastMCP 集成模板**。

## 这是什么

一个干净的起点，用来快速搭建 **MCP (Model Context Protocol) 服务**。

MCP 是 AI 模型调用外部工具的协议标准。这个模板让你用最少的配置，把任意功能暴露为 AI 可调用的工具。

```
你的业务代码 → MCP Server → AI 智能体（Cursor / Claude / 自定义应用）
```

## 架构

```
FastAPI（主容器）
├── CORS 中间件
├── 生命周期管理
├── FastAPI 路由（传统 HTTP）
└── FastMCP 子应用（MCP 协议）
    └── @mcp.tool() 工具
```

**核心特点：**
- 一个装饰器 = 同时支持 MCP 协议 + HTTP 接口
- 多模块独立挂载，各模块互不干扰
- 开箱即用的日志、配置、Docker 部署

## 快速开始

### 前置条件

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### 安装

```shell
# 克隆模板
git clone <your-repo-url> my-mcp-server
cd my-mcp-server

# 安装依赖
uv sync

# 配置环境变量
cp .env-sample .env
```

### 启动

```shell
# 方式一：uvicorn（推荐）
uv run uvicorn app.main:app --reload

# 方式二：python
uv run python -m app.main

# 方式三：fastapi CLI
uv run fastapi dev app/main.py
```

### 验证

```shell
# HTTP 端点
curl http://127.0.0.1:8000/example/hello
# => "Hello, World! Running mcp-server-template v0.1.0"

# MCP 协议端点（查看可用工具）
curl http://127.0.0.1:8000/example/mcp/tools/list
```

## 项目结构

```
mcp-server-template/
│
├── app/                    # 应用主目录
│   ├── routers/            # MCP 工具 + FastAPI 路由
│   │   ├── example.py      #   示例模块（供复制参考）
│   │   └── __init__.py
│   ├── utils/              # 工具函数
│   │   └── log.py          #   日志（loguru）
│   ├── __init__.py
│   ├── config.py           # 配置管理（pydantic-settings）
│   ├── dependencies.py     # 依赖注入
│   └── main.py             # 应用入口
│
├── docs/                   # 文档
├── migrations/             # Alembic 数据库迁移（可选）
├── scripts/                # 开发脚本
├── tests/                  # 测试
│
├── .env-sample             # 环境变量模板
├── .gitignore
├── .python-version
├── alembic.ini
├── docker-compose.yaml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## 添加新业务模块

模板内置了 `example.py` 作为参考。新建模块：

1. **复制 `app/routers/example.py`** 为 `app/routers/your_module.py`
2. **修改模块内容** — 改名、改路由前缀、写你的工具函数
3. **在 `app/main.py` 注册**

```python
# app/main.py
from app.routers import your_module
app.include_router(your_module.router)
app.mount("/your_module", app=your_module.mcp.streamable_http_app())
```

工具函数可以同时被 AI（通过 MCP）和浏览器（通过 HTTP）调用：

```python
@mcp.tool()
@router.get("/search")
async def search(query: str) -> list:
    """搜索知识库（此描述会暴露给 AI）"""
    # 你的业务逻辑
    return results
```

## 测试

```shell
uv run pytest tests/ -v
```

## Docker 部署

```shell
# 构建
docker build -t mcp-server:latest .

# 启动
docker compose up -d
```

## 配置

所有配置通过环境变量或 `.env` 文件管理，详见 `.env-sample`。

关键配置项：

| 变量 | 说明 | 默认 |
|---|---|---|
| `ENVIRONMENT` | 运行环境 | `dev` |
| `HOST` | 监听地址 | `127.0.0.1` |
| `PORT` | 监听端口 | `8000` |
| `DATABASE_URL` | 数据库连接（可选） | `None` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

## 从旧项目迁移

如果你有之前基于 FastAPI + FastMCP 的项目（如 wx-mcp），迁移步骤：

1. 复制 `app/routers/` 下的业务模块到新模板
2. 在 `app/main.py` 中注册
3. 补充缺失的依赖到 `pyproject.toml`
4. 复制 `.env` 配置

## 后续扩展方向

- **知识库模块** — 对接本地文档，提供语义搜索工具
- **数据查询模块** — 对接数据库，提供 NL→SQL 工具
- **电商分析模块** — 商品搜索、价格分析、比价工具
- **自动化模块** — 文件处理、数据导出、定时任务

## 参考

- [MCP 官方文档](https://modelcontextprotocol.io/introduction)
- [FastMCP 文档](https://github.com/jlowin/fastmcp)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
