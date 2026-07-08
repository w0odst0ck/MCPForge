"""测试示例路由。"""
from fastapi.testclient import TestClient


def test_hello(client: TestClient):
    response = client.get("/example/hello")
    assert response.status_code == 200
    assert "Hello, World!" in response.text


def test_hello_with_name(client: TestClient):
    response = client.get("/example/hello?name=Test")
    assert response.status_code == 200
    assert "Hello, Test!" in response.text


def test_echo(client: TestClient):
    response = client.get("/example/echo?message=hi")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "hi"
    assert data["length"] == 2


def test_system_info(client: TestClient):
    response = client.get("/example/system_info")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "mcp-server-template"
    assert "python_version" in data
