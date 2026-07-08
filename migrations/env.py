"""Alembic 数据库迁移配置。

使用:
    # 初始化迁移仓库（仅首次）
    alembic init migrations

    # 生成迁移脚本
    alembic revision --autogenerate -m "描述"

    # 执行迁移
    alembic upgrade head

注意: 该项目默认不强制要求数据库。如需数据库支持：
    1. 在 .env 中设置 DATABASE_URL
    2. 取消 pyproject.toml 中 sqlmodel/alembic/pymysql 的注释
    3. 运行 uv sync 安装依赖
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import settings

# Alembic 配置对象
config = context.config

# 日志配置
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 元数据（从业务模型导入）
# from app.models import Base
# target_metadata = Base.metadata
target_metadata = None


def run_migrations_offline() -> None:
    """离线模式运行迁移（只生成 SQL 脚本，不连接数据库）。"""
    url = settings.DATABASE_URL or config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式运行迁移（连接数据库执行）。"""
    if not settings.DATABASE_URL:
        print("⚠ DATABASE_URL 未设置，跳过迁移。")
        return

    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = settings.DATABASE_URL
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
