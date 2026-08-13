"""测试配置 — pytest fixtures 和全局设置。"""
import pytest


@pytest.fixture
def client():
    """返回 FastAPI TestClient 实例。"""
    from fastapi.testclient import TestClient

    from app.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_mojin_answer_cache():
    """每个测试前后清空 mojin_chat 答案缓存，防止用例间串缓存（缓存是进程内单例）。"""
    from app.routers import mojin_chat as mc

    mc._answer_cache.clear()
    yield
    mc._answer_cache.clear()
