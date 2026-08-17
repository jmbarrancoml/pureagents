"""Turn Python type hints into JSON Schema.

Shared by the @tool decorator and by structured outputs, so a tool parameter
and a dataclass field of the same type are described to the model identically.
"""

from __future__ import annotations

import types
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

STRING_SCHEMA: dict[str, Any] = {"type": "string"}

# Matched by identity, so bool never falls through to int's branch.
PRIMITIVES: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}

SEQUENCE_TYPES = (list, set, tuple, frozenset)


def json_schema(py_type: Any) -> dict[str, Any]:
    """Describe `py_type` as a JSON Schema fragment.

    Anything genuinely unrepresentable falls back to a string, which is what a
    model can always produce.
    """
    if py_type is None or py_type is type(None) or py_type is Any:
        return dict(STRING_SCHEMA)

    if py_type in PRIMITIVES:
        return dict(PRIMITIVES[py_type])

    origin = get_origin(py_type)

    if origin is Literal:
        return _literal_schema(get_args(py_type))

    if origin is Union or origin is getattr(types, "UnionType", ()):
        return _union_schema(get_args(py_type))

    if origin in SEQUENCE_TYPES or py_type in SEQUENCE_TYPES:
        args = get_args(py_type)
        item = json_schema(args[0]) if args else dict(STRING_SCHEMA)
        return {"type": "array", "items": item}

    if origin is dict or py_type is dict:
        args = get_args(py_type)
        if len(args) == 2:
            return {"type": "object", "additionalProperties": json_schema(args[1])}
        return {"type": "object"}

    if isinstance(py_type, type):
        if issubclass(py_type, Enum):
            return _literal_schema([member.value for member in py_type])
        if is_dataclass(py_type):
            return dataclass_schema(py_type)

    return dict(STRING_SCHEMA)


def dataclass_schema(cls: type) -> dict[str, Any]:
    """Describe a dataclass as an object schema.

    Only fields without a default are required, so `limit: int = 10` stops
    being something the model must invent a value for.
    """
    hints = _resolved_hints(cls)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in fields(cls):
        properties[field.name] = json_schema(hints.get(field.name, str))
        if field.default is MISSING and field.default_factory is MISSING:
            required.append(field.name)

    return {"type": "object", "properties": properties, "required": required}


def _literal_schema(values: list[Any]) -> dict[str, Any]:
    schema: dict[str, Any] = {"enum": list(values)}
    value_types = {type(value) for value in values}
    if len(value_types) == 1:
        primitive = PRIMITIVES.get(value_types.pop())
        if primitive:
            schema = {**primitive, **schema}
    return schema


def _union_schema(args: tuple[Any, ...]) -> dict[str, Any]:
    # Optional[X] is the common case; describe it as plain X so the model is
    # not told to produce a union it cannot express.
    non_null = [arg for arg in args if arg is not type(None)]
    if not non_null:
        return dict(STRING_SCHEMA)
    if len(non_null) == 1:
        return json_schema(non_null[0])
    return {"anyOf": [json_schema(arg) for arg in non_null]}


def _resolved_hints(cls: type) -> dict[str, Any]:
    from typing import get_type_hints

    try:
        return get_type_hints(cls)
    except Exception:
        # Unresolvable forward references should not take the whole agent down.
        return {field.name: field.type for field in fields(cls)}
