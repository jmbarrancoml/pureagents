"""Tests for type-hint to JSON Schema conversion and structured output."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import pytest

from pure_agents import StructuredOutputError, tool
from pure_agents.agent import _parse_structured
from pure_agents.schema import dataclass_schema, json_schema
from tests.conftest import openai_response


class Colour(str, Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class Inner:
    value: int


@dataclass
class Outer:
    inner: Inner


@dataclass
class Container:
    items: list[Inner]


class TestJsonSchema:
    def test_primitives(self):
        assert json_schema(str) == {"type": "string"}
        assert json_schema(int) == {"type": "integer"}
        assert json_schema(float) == {"type": "number"}
        assert json_schema(bool) == {"type": "boolean"}

    def test_bool_is_not_confused_with_int(self):
        assert json_schema(bool)["type"] == "boolean"

    def test_typed_list(self):
        assert json_schema(list[str]) == {
            "type": "array",
            "items": {"type": "string"},
        }
        assert json_schema(list[int])["items"] == {"type": "integer"}

    def test_bare_list(self):
        assert json_schema(list) == {"type": "array", "items": {"type": "string"}}

    def test_nested_list(self):
        schema = json_schema(list[list[int]])
        assert schema["items"]["type"] == "array"
        assert schema["items"]["items"] == {"type": "integer"}

    def test_typed_dict(self):
        assert json_schema(dict[str, int]) == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }

    def test_optional_describes_the_inner_type(self):
        assert json_schema(Optional[str]) == {"type": "string"}
        assert json_schema(str | None) == {"type": "string"}
        assert json_schema(list[str] | None)["type"] == "array"

    def test_union_of_real_types(self):
        schema = json_schema(int | str)
        assert schema["anyOf"] == [{"type": "integer"}, {"type": "string"}]

    def test_literal_becomes_an_enum(self):
        assert json_schema(Literal["a", "b"]) == {
            "type": "string",
            "enum": ["a", "b"],
        }

    def test_enum_class_becomes_an_enum(self):
        assert json_schema(Colour) == {"type": "string", "enum": ["red", "blue"]}

    def test_unknown_types_fall_back_to_string(self):
        class Whatever:
            pass

        assert json_schema(Whatever) == {"type": "string"}


class TestToolSchema:
    def test_only_parameters_without_defaults_are_required(self):
        @tool
        def search(query: str, limit: int = 10) -> str:
            """Search."""
            return ""

        schema = search.to_dict()["function"]["parameters"]
        assert schema["required"] == ["query"]
        assert set(schema["properties"]) == {"query", "limit"}

    def test_list_parameter_is_an_array(self):
        @tool
        def tag(items: list[str]) -> str:
            """Tag things."""
            return ""

        assert tag.parameters["items"] == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_optional_parameter_is_not_required(self):
        @tool
        def maybe(name: str | None = None) -> str:
            """Maybe."""
            return ""

        assert maybe.required == []
        assert maybe.parameters["name"] == {"type": "string"}

    def test_var_args_are_skipped(self):
        @tool
        def flexible(a: str, *args, **kwargs) -> str:
            """Flexible."""
            return ""

        assert list(flexible.parameters) == ["a"]

    def test_anthropic_conversion_uses_the_same_schema(self):
        from pure_agents.clients import AnthropicClient

        @tool
        def search(query: str, limit: int = 10) -> str:
            """Search."""
            return ""

        converted = AnthropicClient(api_key="k")._convert_tools([search])
        assert converted[0]["input_schema"]["required"] == ["query"]


class TestDataclassSchema:
    def test_list_field_is_an_array(self):
        @dataclass
        class Analysis:
            sentiment: str
            confidence: float
            keywords: list[str]

        schema = dataclass_schema(Analysis)
        assert schema["properties"]["keywords"] == {
            "type": "array",
            "items": {"type": "string"},
        }
        assert schema["required"] == ["sentiment", "confidence", "keywords"]

    def test_fields_with_defaults_are_optional(self):
        @dataclass
        class Config:
            name: str
            retries: int = 3
            tags: list[str] = field(default_factory=list)

        assert dataclass_schema(Config)["required"] == ["name"]

    def test_nested_dataclass(self):
        schema = dataclass_schema(Outer)
        assert schema["properties"]["inner"]["type"] == "object"
        assert schema["properties"]["inner"]["properties"]["value"] == {
            "type": "integer"
        }

    def test_list_of_dataclasses(self):
        schema = dataclass_schema(Container)
        items = schema["properties"]["items"]["items"]
        assert items["type"] == "object"
        assert items["properties"]["value"] == {"type": "integer"}


@dataclass
class Review:
    sentiment: str
    score: int


@dataclass
class Analysis:
    sentiment: str
    keywords: list[str]


class TestParseStructured:
    def test_plain_json(self):
        assert _parse_structured('{"sentiment": "good", "score": 5}', Review) == Review(
            "good", 5
        )

    def test_fenced_json(self):
        text = '```json\n{"sentiment": "good", "score": 5}\n```'
        assert _parse_structured(text, Review) == Review("good", 5)

    def test_json_wrapped_in_prose(self):
        text = 'Here you go:\n{"sentiment": "good", "score": 5}\nHope that helps!'
        assert _parse_structured(text, Review) == Review("good", 5)

    def test_invalid_json_raises_a_clear_error(self):
        with pytest.raises(StructuredOutputError, match="did not return valid JSON"):
            _parse_structured("not json at all", Review)

    def test_json_array_raises_a_clear_error(self):
        with pytest.raises(StructuredOutputError, match="not a JSON object"):
            _parse_structured("[1, 2, 3]", Review)

    def test_mismatched_fields_raise_a_clear_error(self):
        with pytest.raises(StructuredOutputError, match="does not match Review"):
            _parse_structured('{"nope": 1}', Review)

    def test_error_carries_the_raw_reply(self):
        with pytest.raises(StructuredOutputError) as excinfo:
            _parse_structured("garbage", Review)
        assert excinfo.value.raw == "garbage"


class TestStructuredOutputEndToEnd:
    async def test_the_schema_is_enforced_by_the_provider(self, make_agent):
        agent, fake = make_agent()
        fake.queue(openai_response('{"sentiment": "good", "keywords": ["a", "b"]}'))

        result = await agent.run("analyse", output=Analysis)

        assert result == Analysis(sentiment="good", keywords=["a", "b"])

        request = fake.last_request()
        schema = request["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["keywords"]["type"] == "array"
        # The prompt stays about the task; the schema travels in its own field.
        assert request["messages"][-1]["content"] == "analyse"
