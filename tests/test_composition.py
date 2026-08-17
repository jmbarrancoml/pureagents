"""Tests for chain(), Router and Graph."""

from __future__ import annotations

import pytest

from pure_agents import END, Agent, Graph, Router, chain
from tests.conftest import FakeLLM, openai_response


def agent_with(fake: FakeLLM, **kwargs) -> Agent:
    agent = Agent(api_key="k", provider="openai", **kwargs)
    agent.client.transport = fake.transport()
    return agent


class TestChain:
    async def test_output_feeds_the_next_agent(self):
        first_api, second_api = FakeLLM(), FakeLLM()
        first = agent_with(first_api, system="research")
        second = agent_with(second_api, system="write")
        first_api.queue(openai_response("raw research"))
        second_api.queue(openai_response("polished prose"))

        result = await chain([first, second], "quantum computing")

        assert result == "polished prose"
        assert second_api.last_request()["messages"][-1]["content"] == "raw research"


class TestRouter:
    async def test_function_routing_picks_the_agent(self):
        coder_api, writer_api = FakeLLM(), FakeLLM()
        router = Router(
            agents={
                "coder": agent_with(coder_api),
                "writer": agent_with(writer_api),
            },
            route=lambda prompt: "coder" if "code" in prompt else "writer",
        )
        coder_api.queue(openai_response("here is code"))

        assert await router.run("write code") == "here is code"
        assert writer_api.call_count == 0

    async def test_llm_routing_does_not_accumulate_history(self):
        router_api, target_api = FakeLLM(), FakeLLM()
        router = Router(agents={"coder": agent_with(target_api)}, api_key="k")
        router._llm_router.client.transport = router_api.transport()

        for _ in range(3):
            router_api.queue(openai_response("coder"))
            target_api.queue(openai_response("done"))
            await router.run("a question")

        # Each routing request carries only the system prompt and one question.
        for request in router_api.requests:
            assert len(request["messages"]) == 2


class TestGraph:
    async def test_the_first_node_is_the_entry_point(self):
        research_api, write_api = FakeLLM(), FakeLLM()
        graph = Graph()
        graph.add_node("research", agent_with(research_api))
        graph.add_node("write", agent_with(write_api))
        graph.add_edge("research", "write")
        research_api.queue(openai_response("findings"))
        write_api.queue(openai_response("article"))

        state = await graph.run("AI agents")

        assert state["output"] == "article"
        assert state["visited"] == ["research", "write"]

    async def test_set_entry_still_wins(self):
        first_api, second_api = FakeLLM(), FakeLLM()
        graph = Graph()
        graph.add_node("first", agent_with(first_api))
        graph.add_node("second", agent_with(second_api))
        graph.set_entry("second")
        second_api.queue(openai_response("started at second"))

        state = await graph.run("go")

        assert state["visited"] == ["second"]
        assert first_api.call_count == 0

    async def test_runs_are_independent(self):
        api = FakeLLM()
        graph = Graph()
        graph.add_node("only", agent_with(api))
        api.queue(openai_response("first run"))
        api.queue(openai_response("second run"))

        await graph.run("a")
        await graph.run("b")

        # Without per-run copies the second request carried the first exchange.
        assert len(api.requests[1]["messages"]) == len(api.requests[0]["messages"])

    async def test_conditional_edges(self):
        api = FakeLLM()
        graph = Graph()
        graph.add_node("write", agent_with(api))
        graph.add_node("review", lambda state: {**state, "reviewed": True})
        graph.add_conditional_edge(
            "write", lambda state: "review" if state.get("output") else END
        )
        graph.add_edge("review", END)
        api.queue(openai_response("draft"))

        state = await graph.run("write something")

        assert state["reviewed"] is True

    async def test_an_empty_graph_says_so(self):
        with pytest.raises(ValueError, match="no nodes"):
            await Graph().run("go")

    async def test_an_unknown_target_names_the_known_nodes(self):
        graph = Graph()
        graph.add_node("only", lambda state: state)
        graph.add_edge("only", "ghost")

        with pytest.raises(ValueError, match="Known nodes"):
            await graph.run("go")

    async def test_a_function_node_must_return_the_state(self):
        graph = Graph()
        graph.add_node("bad", lambda state: None)

        with pytest.raises(TypeError, match="must return the state dict"):
            await graph.run("go")

    async def test_runaway_loops_are_stopped(self):
        graph = Graph()
        graph.add_node("loop", lambda state: state)
        graph.add_edge("loop", "loop")

        with pytest.raises(RuntimeError, match="infinite loop"):
            await graph.run("go")
