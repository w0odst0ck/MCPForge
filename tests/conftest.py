"""测试配置 — pytest fixtures 和全局设置。"""
import pytest


@pytest.fixture
def client():
    """返回 FastAPI TestClient 实例。"""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
