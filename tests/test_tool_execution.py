"""Tests for how the agent executes tool calls the model asks for."""

from __future__ import annotations

from pure_agents import tool
from tests.conftest import openai_response, openai_tool_call


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


class TestMalformedToolCalls:
    async def test_invalid_json_arguments_go_back_to_the_model(self, make_agent):
        agent, fake = make_agent(tools=[add])
        fake.queue(
            openai_response(None, [openai_tool_call("add", raw_arguments="{not json")])
        )
        fake.queue(openai_response("recovered"))

        result = await agent.run("add things")

        assert result == "recovered"
        tool_message = fake.last_request()["messages"][-1]
        assert "could not parse arguments" in tool_message["content"]

    async def test_non_object_arguments_go_back_to_the_model(self, make_agent):
        agent, fake = make_agent(tools=[add])
        fake.queue(
            openai_response(None, [openai_tool_call("add", raw_arguments="[1, 2]")])
        )
        fake.queue(openai_response("recovered"))

        assert await agent.run("add things") == "recovered"
        assert "JSON object" in fake.last_request()["messages"][-1]["content"]

    async def test_wrong_argument_names_go_back_to_the_model(self, make_agent):
        agent, fake = make_agent(tools=[add])
        fake.queue(openai_response(None, [openai_tool_call("add", {"x": 1})]))
        fake.queue(openai_response("recovered"))

        assert await agent.run("add things") == "recovered"
        assert "bad arguments" in fake.last_request()["messages"][-1]["content"]

    async def test_missing_call_id_gets_a_synthetic_one(self, make_agent):
        agent, fake = make_agent(tools=[add])
        call = openai_tool_call("add", {"a": 1, "b": 2})
        del call["id"]
        fake.queue(openai_response(None, [call]))
        fake.queue(openai_response("done"))

        assert await agent.run("add") == "done"
        assert fake.last_request()["messages"][-1]["tool_call_id"] == "call_0"

    async def test_one_bad_call_does_not_cancel_its_siblings(self, make_agent):
        agent, fake = make_agent(tools=[add, greet])
        fake.queue(
            openai_response(
                None,
                [
                    openai_tool_call("add", raw_arguments="{oops", call_id="c1"),
                    openai_tool_call("greet", {"name": "Jose"}, call_id="c2"),
                ],
            )
        )
        fake.queue(openai_response("done"))

        assert await agent.run("do both") == "done"

        tool_messages = [
            m for m in fake.last_request()["messages"] if m["role"] == "tool"
        ]
        assert len(tool_messages) == 2
        assert "could not parse" in tool_messages[0]["content"]
        assert tool_messages[1]["content"] == "Hello, Jose!"


class TestHooks:
    async def test_hooks_receive_calls_and_results(self, make_agent):
        seen_calls: list[tuple[str, dict]] = []
        seen_results: list[tuple[str, str]] = []

        agent, fake = make_agent(
            tools=[add],
            on_tool_call=lambda name, args: seen_calls.append((name, args)),
            on_tool_result=lambda name, result: seen_results.append((name, result)),
        )
        fake.queue(openai_response(None, [openai_tool_call("add", {"a": 1, "b": 2})]))
        fake.queue(openai_response("done"))

        await agent.run("add")

        assert seen_calls == [("add", {"a": 1, "b": 2})]
        assert seen_results == [("add", "3")]

    async def test_a_raising_hook_does_not_break_the_run(self, make_agent):
        def bad_hook(name, args):
            raise RuntimeError("hook exploded")

        agent, fake = make_agent(tools=[add], on_tool_call=bad_hook)
        fake.queue(openai_response(None, [openai_tool_call("add", {"a": 1, "b": 2})]))
        fake.queue(openai_response("done"))

        assert await agent.run("add") == "done"
        assert fake.last_request()["messages"][-1]["content"] == "3"
