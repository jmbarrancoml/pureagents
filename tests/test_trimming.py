"""Tests for history trimming under max_messages."""

from __future__ import annotations

from pure_agents import Agent, Message, tool
from tests.conftest import openai_response, openai_tool_call


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def make_history() -> list[Message]:
    return [
        Message(role="system", content="sys"),
        Message(role="user", content="q1"),
        Message(role="assistant", content="a1"),
        Message(
            role="user",
            content="q2",
        ),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", content="3", tool_call_id="1", name="add"),
        Message(role="assistant", content="a2"),
        Message(role="user", content="q3"),
    ]


def roles(agent: Agent) -> list[str]:
    return [m.role for m in agent.messages]


class TestTrimming:
    def test_never_leaves_an_orphan_tool_message(self):
        agent = Agent(api_key="k", max_messages=3)
        agent.messages = make_history()

        agent._trim_messages()

        assert roles(agent)[1] != "tool"
        for index, message in enumerate(agent.messages):
            if message.role == "tool":
                assert any(m.tool_calls for m in agent.messages[:index])

    def test_window_opens_on_a_user_turn(self):
        agent = Agent(api_key="k", max_messages=4)
        agent.messages = make_history()

        agent._trim_messages()

        assert agent.messages[0].role == "system"
        assert agent.messages[1].role == "user"

    def test_keeps_the_system_message(self):
        agent = Agent(api_key="k", max_messages=2)
        agent.messages = make_history()

        agent._trim_messages()

        assert agent.messages[0].role == "system"
        assert agent.messages[0].content == "sys"

    def test_keeps_the_most_recent_turn(self):
        agent = Agent(api_key="k", max_messages=3)
        agent.messages = make_history()

        agent._trim_messages()

        assert agent.messages[-1].content == "q3"

    def test_short_history_is_untouched(self):
        agent = Agent(api_key="k", max_messages=20)
        original = make_history()
        agent.messages = list(original)

        agent._trim_messages()

        assert agent.messages == original

    def test_no_limit_means_no_trimming(self):
        agent = Agent(api_key="k")
        original = make_history()
        agent.messages = list(original)

        agent._trim_messages()

        assert agent.messages == original

    def test_history_without_a_system_message(self):
        agent = Agent(api_key="k", max_messages=2)
        agent.messages = make_history()[1:]

        agent._trim_messages()

        assert agent.messages[0].role == "user"


class TestTrimmingDuringARun:
    async def test_history_stops_growing_across_calls(self, make_agent):
        agent, fake = make_agent(max_messages=4)
        sizes = []
        for i in range(8):
            fake.queue(openai_response(f"answer {i}"))
            await agent.run(f"question {i}")
            sizes.append(len(agent.messages))

        # Snapping to a turn boundary can keep a message or two more than the
        # budget, but the history plateaus instead of growing with the session.
        assert max(sizes[3:]) == min(sizes[3:])
        assert max(sizes) <= 6
        assert agent.messages[0].role == "system"
        assert agent.messages[-1].content == "answer 7"

    async def test_a_tool_loop_is_never_split_mid_turn(self, make_agent):
        # max_messages cannot cut inside the turn in flight without producing
        # a request the API would reject, so the current turn survives whole.
        agent, fake = make_agent(tools=[add], max_messages=4, max_steps=6)
        for _ in range(5):
            fake.queue(
                openai_response(None, [openai_tool_call("add", {"a": 1, "b": 1})])
            )
        fake.queue(openai_response("done"))

        assert await agent.run("keep going") == "done"

        user_messages = [m for m in agent.messages if m.role == "user"]
        assert len(user_messages) == 1
        assert agent.messages[1] is user_messages[0]

    async def test_every_request_is_well_formed(self, make_agent):
        agent, fake = make_agent(tools=[add], max_messages=3, max_steps=5)
        for _ in range(4):
            fake.queue(
                openai_response(None, [openai_tool_call("add", {"a": 1, "b": 1})])
            )
        fake.queue(openai_response("done"))

        await agent.run("keep going")

        for request in fake.requests:
            messages = request["messages"]
            for index, message in enumerate(messages):
                if message["role"] == "tool":
                    assert any(m.get("tool_calls") for m in messages[:index])
