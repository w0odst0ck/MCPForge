"""Alembic 迁移脚本模板。"""
revision: str = None
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """升级迁移。"""
    pass


def downgrade() -> None:
    """回滚迁移。"""
    pass
