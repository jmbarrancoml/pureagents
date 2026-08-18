"""Tests for provider-enforced structured output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import pytest

from pure_agents import (
    Agent,
    StructuredOutputError,
    TruncatedResponseError,
    register_provider,
    tool,
)
from pure_agents.schema import dataclass_schema, strict_schema
from tests.conftest import (
    FakeLLM,
    anthropic_response,
    openai_response,
    openai_tool_call,
)


@dataclass
class Review:
    sentiment: str
    score: int


@dataclass
class Config:
    name: str
    retries: int = 3
    tags: list[str] = field(default_factory=list)


@dataclass
class Nested:
    inner: Review


@dataclass
class Choice:
    verdict: Literal["yes", "no"]


@tool
def lookup(term: str) -> str:
    """Look something up."""
    return f"found {term}"


class TestStrictSchema:
    def test_objects_forbid_extra_properties(self):
        schema = strict_schema(dataclass_schema(Review))

        assert schema["additionalProperties"] is False

    def test_every_property_is_required(self):
        schema = strict_schema(dataclass_schema(Config))

        assert set(schema["required"]) == {"name", "retries", "tags"}

    def test_optional_fields_become_nullable(self):
        schema = strict_schema(dataclass_schema(Config))

        assert schema["properties"]["retries"] == {
            "anyOf": [{"type": "integer"}, {"type": "null"}]
        }
        assert schema["properties"]["name"] == {"type": "string"}

    def test_nested_objects_are_tightened_too(self):
        inner = strict_schema(dataclass_schema(Nested))["properties"]["inner"]

        assert inner["additionalProperties"] is False
        assert set(inner["required"]) == {"sentiment", "score"}

    def test_arrays_of_objects_are_tightened(self):
        schema = strict_schema(
            {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": dataclass_schema(Review)}
                },
                "required": ["items"],
            }
        )

        assert schema["properties"]["items"]["items"]["additionalProperties"] is False

    def test_enums_survive(self):
        schema = strict_schema(dataclass_schema(Choice))

        assert schema["properties"]["verdict"]["enum"] == ["yes", "no"]

    def test_open_dicts_keep_their_value_schema(self):
        schema = strict_schema(
            {"type": "object", "additionalProperties": {"type": "integer"}}
        )

        assert schema["additionalProperties"] == {"type": "integer"}

    def test_it_does_not_mutate_the_input(self):
        original = dataclass_schema(Config)
        snapshot = json.dumps(original, sort_keys=True)

        strict_schema(original)

        assert json.dumps(original, sort_keys=True) == snapshot

    def test_nullable_is_not_applied_twice(self):
        once = strict_schema(dataclass_schema(Config))
        twice = strict_schema(once)

        assert twice["properties"]["retries"] == once["properties"]["retries"]


class TestOpenAIDialect:
    async def test_the_schema_travels_in_response_format(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response('{"sentiment": "good", "score": 5}'))

        assert await agent.run("review", output=Review) == Review("good", 5)

        response_format = fake.last_request()["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["name"] == "Review"
        assert response_format["json_schema"]["schema"]["additionalProperties"] is False

    async def test_the_prompt_no_longer_carries_the_schema(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response('{"sentiment": "good", "score": 5}'))

        await agent.run("Analyse this review", output=Review)

        assert fake.last_request()["messages"][-1]["content"] == "Analyse this review"

    async def test_nothing_is_sent_without_output(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response("plain answer"))

        await agent.run("hello")

        assert "response_format" not in fake.last_request()


class TestAnthropicDialect:
    async def test_the_schema_travels_in_output_config(self, make_agent):
        agent, fake = make_agent(provider="anthropic")
        fake.queue(anthropic_response('{"sentiment": "good", "score": 5}'))

        assert await agent.run("review", output=Review) == Review("good", 5)

        output_config = fake.last_request()["output_config"]
        assert output_config["format"]["type"] == "json_schema"
        assert output_config["format"]["schema"]["additionalProperties"] is False

    async def test_nothing_is_sent_without_output(self, make_agent):
        agent, fake = make_agent(provider="anthropic")
        fake.queue(anthropic_response("plain answer"))

        await agent.run("hello")

        assert "output_config" not in fake.last_request()


class TestPromptFallback:
    def endpoint(self) -> tuple[Agent, FakeLLM]:
        register_provider(
            "legacy", base_url="https://legacy.test/v1", structured_outputs=False
        )
        agent = Agent(provider="legacy", model="old-model")
        fake = FakeLLM()
        agent.client.transport = fake.transport()
        return agent, fake

    async def test_a_provider_can_declare_it_cannot_enforce_schemas(self):
        agent, fake = self.endpoint()
        fake.queue(openai_response('{"sentiment": "good", "score": 5}'))

        assert await agent.run("review", output=Review) == Review("good", 5)

        request = fake.last_request()
        assert "response_format" not in request
        assert '"type": "integer"' in request["messages"][-1]["content"]

    def test_built_in_providers_can_enforce_schemas(self):
        assert Agent(api_key="k").native_output is True
        assert Agent(api_key="k", provider="anthropic").native_output is True

    def test_an_ad_hoc_endpoint_is_assumed_capable(self):
        agent = Agent(base_url="http://localhost:11434/v1", model="m")

        assert agent.native_output is True


class TestWithTools:
    async def test_a_tool_loop_still_reaches_a_structured_answer(self, make_agent):
        agent, fake = make_agent(tools=[lookup])
        fake.queue(openai_response(None, [openai_tool_call("lookup", {"term": "x"})]))
        fake.queue(openai_response('{"sentiment": "good", "score": 4}'))

        assert await agent.run("check it", output=Review) == Review("good", 4)
        assert fake.call_count == 2

    async def test_the_schema_rides_along_on_every_request(self, make_agent):
        agent, fake = make_agent(tools=[lookup])
        fake.queue(openai_response(None, [openai_tool_call("lookup", {"term": "x"})]))
        fake.queue(openai_response('{"sentiment": "good", "score": 4}'))

        await agent.run("check it", output=Review)

        assert all("response_format" in request for request in fake.requests)


class TestCacheInteraction:
    async def test_two_shapes_do_not_share_a_cache_entry(self, make_agent):
        @dataclass
        class Other:
            summary: str

        agent, fake = make_agent(cache=True)
        fake.queue(openai_response('{"sentiment": "good", "score": 5}'))
        fake.queue(openai_response('{"summary": "all good"}'))

        first = await agent.run("analyse", output=Review)
        agent.clear()
        second = await agent.run("analyse", output=Other)

        assert first == Review("good", 5)
        assert second == Other("all good")
        assert fake.call_count == 2


class TestErrors:
    async def test_a_reply_that_still_does_not_match_reports_clearly(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response('{"wrong": "shape"}'))

        with pytest.raises(StructuredOutputError, match="does not match Review"):
            await agent.run("review", output=Review)


class TestStrictModeIsOptional:
    """OpenAI's strict mode is an extension, not something every server wants.

    Grammar-constrained backends accept the schema and then generate badly from
    the unions strict mode forces, so it only travels where it is asked for.
    """

    def test_built_in_openai_providers_ask_for_strict(self):
        assert Agent(api_key="k", provider="openai").strict_output is True
        assert Agent(api_key="k", provider="mistral").strict_output is True

    def test_anthropic_does_not(self):
        assert Agent(api_key="k", provider="anthropic").strict_output is False

    def test_an_ad_hoc_endpoint_does_not(self):
        assert (
            Agent(base_url="http://localhost:11434/v1", model="m").strict_output
            is False
        )

    async def test_a_non_strict_endpoint_gets_no_strict_flag(self):
        agent = Agent(base_url="http://localhost:11434/v1", model="m")
        fake = FakeLLM()
        agent.client.transport = fake.transport()
        fake.queue(openai_response('{"name": "x"}'))

        await agent.run("go", output=Config)

        assert "strict" not in fake.last_request()["response_format"]["json_schema"]

    async def test_a_non_strict_endpoint_keeps_optional_fields_optional(self):
        agent = Agent(base_url="http://localhost:11434/v1", model="m")
        fake = FakeLLM()
        agent.client.transport = fake.transport()
        fake.queue(openai_response('{"name": "x"}'))

        await agent.run("go", output=Config)

        schema = fake.last_request()["response_format"]["json_schema"]["schema"]
        assert schema["required"] == ["name"]
        assert schema["properties"]["retries"] == {"type": "integer"}
        assert schema["additionalProperties"] is False

    async def test_a_registered_provider_can_opt_in(self):
        register_provider(
            "groq", base_url="https://api.groq.test/openai/v1", strict_schemas=True
        )
        agent = Agent(provider="groq", model="m")
        fake = FakeLLM()
        agent.client.transport = fake.transport()
        fake.queue(openai_response('{"name": "x", "retries": null, "tags": []}'))

        await agent.run("go", output=Config)

        json_schema = fake.last_request()["response_format"]["json_schema"]
        assert json_schema["strict"] is True
        assert set(json_schema["schema"]["required"]) == {"name", "retries", "tags"}


class TestTruncationIsReported:
    """A reply cut off at the token limit is not a reply."""

    async def test_a_truncated_structured_reply_names_the_cause(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response('{"sentiment": "goo', finish_reason="length"))

        with pytest.raises(TruncatedResponseError, match="token limit"):
            await agent.run("review", output=Review)

    async def test_a_truncated_tool_call_names_the_cause(self, make_agent):
        agent, fake = make_agent(tools=[lookup])
        fake.queue(
            openai_response(
                None,
                [openai_tool_call("lookup", {"term": "x"})],
                finish_reason="length",
            )
        )

        with pytest.raises(TruncatedResponseError, match="token limit"):
            await agent.run("go")

    async def test_truncated_plain_text_is_flagged_not_raised(self, make_agent):
        agent, fake = make_agent(max_tokens=50)
        fake.queue(openai_response("half an ans", finish_reason="length"))

        result = await agent.run("go")

        assert result.startswith("half an ans")
        assert "truncated" in result

    async def test_a_complete_reply_is_untouched(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response("all done"))

        assert await agent.run("go") == "all done"
