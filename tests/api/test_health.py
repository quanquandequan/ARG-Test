"""API tests for /health and /health/ready."""


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_ready_all_components_up(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    for key in ("embedder", "vectordb", "reranker", "llm"):
        assert body["checks"][key] is True
