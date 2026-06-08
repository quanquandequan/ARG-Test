"""/query 与 /query/stream 的 API 测试：基于 Agent 的端点。"""

import src.api.dependencies as deps
from src.agent.types import AgentResult
from src.bootstrap import UnknownAgentProfileError


def test_query_returns_answer_with_citation(wired_singletons, client):
    wired_singletons["llm"].response_text = "答案见 [1]。"

    r = client.post("/query", json={"query": "RAG 是什么？", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert "答案" in body["answer"]
    assert body["iterations"] >= 1
    assert len(body["steps"]) >= 1
    # Agent 会将 [N] 标记提取为引用索引
    assert len(body["citations"]) == 1
    assert body["citations"][0]["index"] == 1


def test_query_validation_error_missing_query(client):
    r = client.post("/query", json={})
    assert r.status_code == 422


def test_query_empty_index_returns_no_answer(wired_singletons, client):
    wired_singletons["llm"].response_text = "根据目前的信息无法回答。"
    r = client.post("/query", json={"query": "无内容"})
    assert r.status_code == 200
    body = r.json()
    assert "无法回答" in body["answer"]
    assert body["citations"] == []


def test_query_stream_sse(wired_singletons, client):
    wired_singletons["llm"].response_text = "abcDONE"

    with client.stream("POST", "/query/stream", json={"query": "流"}) as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())

    assert "event: token" in body
    assert "event: answer" in body
    assert "abcDONE" in body


def test_query_includes_steps(wired_singletons, client):
    """Agent 响应包含步骤 trace，便于观测。"""
    wired_singletons["llm"].response_text = "测试回答 [1][2]"

    r = client.post("/query", json={"query": "测试"})
    assert r.status_code == 200
    body = r.json()
    assert body["iterations"] == 1
    assert len(body["steps"]) == 1
    assert body["steps"][0]["thinking"] == "测试回答 [1][2]"
    assert len(body["citations"]) == 2


def test_query_with_history(wired_singletons, client):
    """Agent 接受对话历史。"""
    wired_singletons["llm"].response_text = "基于历史继续回答。"

    r = client.post(
        "/query",
        json={
            "query": "继续",
            "history": [{"role": "user", "content": "什么是向量？"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "继续" in body["answer"]
    # 验证历史记录已传递给 LLM
    messages = wired_singletons["llm"].last_messages
    assert messages is not None
    roles = [m.role for m in messages]
    assert "user" in roles


def test_query_accepts_agent_profile(client, monkeypatch):
    seen = {}

    class FakeAgent:
        async def run(self, **kwargs):
            return AgentResult(answer="mobile ok", iterations=1)

    def fake_get_agent(profile_name=None):
        seen["profile_name"] = profile_name
        return FakeAgent()

    monkeypatch.setattr(deps, "get_agent", fake_get_agent)

    r = client.post("/query", json={"query": "执行UI自动化", "profile": "mobile_debug"})

    assert r.status_code == 200
    assert r.json()["answer"] == "mobile ok"
    assert seen["profile_name"] == "mobile_debug"


def test_query_unknown_profile_returns_400(client, monkeypatch):
    def fake_get_agent(profile_name=None):
        raise UnknownAgentProfileError(
            profile_name or "",
            ["mobile_debug", "qa_agent"],
        )

    monkeypatch.setattr(deps, "get_agent", fake_get_agent)

    r = client.post("/query", json={"query": "hi", "profile": "requirement_agent"})

    assert r.status_code == 400
    assert "requirement_agent" in r.json()["detail"]
    assert "qa_agent" in r.json()["detail"]


def test_query_stream_unknown_profile_returns_400(client, monkeypatch):
    def fake_get_agent(profile_name=None):
        raise UnknownAgentProfileError(
            profile_name or "",
            ["mobile_debug", "qa_agent"],
        )

    monkeypatch.setattr(deps, "get_agent", fake_get_agent)

    r = client.post("/query/stream", json={"query": "hi", "profile": "ghost"})

    assert r.status_code == 400
    assert "ghost" in r.json()["detail"]
