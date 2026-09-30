"""Structured decision layer compatible with laya_mlx and upstream laya.structured specification."""
from __future__ import annotations

import math
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

MAX_PROPERTIES = 32
MAX_OPTIONS = 32
MAX_SCORE_LEVELS = 10


class SchemaError(ValueError):
    """A schema cannot be expressed as Laya questions; the message names the path."""


def check_min_confidence(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError("min_confidence must be a float in [0.0, 1.0], got %r" % (v,))
    return float(v)


def flag_low_confidence(results: List[Dict[str, Any]], min_confidence: float) -> None:
    if min_confidence == 0.0:
        return
    for res in results:
        answers = res.get("answers") if isinstance(res, dict) else None
        if not isinstance(answers, dict):
            continue
        for a in answers.values():
            if not isinstance(a, dict):
                continue
            conf = a.get("answer_confidence")
            if conf is None:
                conf = a.get("confidence")
            if isinstance(conf, (int, float)) and not isinstance(conf, bool) and math.isfinite(conf) and conf < min_confidence:
                a["low_confidence"] = True


@dataclass
class DecisionResult:
    values: Dict[str, Any]
    confidence: Dict[str, float]
    probabilities: Dict[str, Dict[str, Any]]
    answers: Dict[str, Any]
    usage: Optional[Dict[str, int]] = None
    routing: Optional[Dict[str, Any]] = None


@dataclass
class _Field:
    name: str
    kind: str  # "choice" | "score" | "noul"
    question: Dict[str, Any]
    options: List[Tuple[str, Any]] = field(default_factory=list)
    minimum: Optional[int] = None


def _schema_of(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    if hasattr(model, "schema"):
        return model.schema()
    if isinstance(model, dict):
        return model
    raise SchemaError("expected a JSON schema dict or a pydantic model, got %s" % type(model).__name__)


def _enum_field(path: str, name: str, values: Sequence[Any], description: Optional[str]) -> _Field:
    if len(values) > MAX_OPTIONS:
        raise SchemaError("%s: %d options exceeds MAX_OPTIONS=%d" % (path, len(values), MAX_OPTIONS))
    if not values:
        raise SchemaError("%s: 'enum' must not be empty" % path)
    if all(isinstance(v, bool) for v in values):
        return _noul_field(path, name, description)
    options = [(("null" if v is None else str(v)), v) for v in values]
    if len({label for label, _ in options}) != len(options):
        raise SchemaError("%s: enum values produce duplicate choice labels" % path)
    criteria = {label: None for label, _ in options}
    question = {
        "type": "choice",
        "instructions": description or ("What is `%s`?" % name),
        "criteria": criteria,
    }
    return _Field(name=name, kind="choice", question=question, options=options)


def _noul_field(path: str, name: str, description: Optional[str]) -> _Field:
    question = {
        "type": "noul",
        "instructions": description or ("Is `%s` true?" % name),
    }
    return _Field(name=name, kind="noul", question=question)


def _score_field(path: str, name: str, prop: Dict[str, Any], description: Optional[str]) -> _Field:
    lo, hi = prop.get("minimum"), prop.get("maximum")
    if not isinstance(lo, int) or not isinstance(hi, int):
        raise SchemaError("%s: a numeric field needs integer 'minimum' and 'maximum' to become a score" % path)
    if hi < lo:
        raise SchemaError("%s: 'maximum' %d is below 'minimum' %d" % (path, hi, lo))
    span = hi - lo + 1
    if span > MAX_SCORE_LEVELS:
        raise SchemaError(
            "%s: %d levels exceeds MAX_SCORE_LEVELS=%d; narrow the range or use an enum"
            % (path, span, MAX_SCORE_LEVELS)
        )
    question = {
        "type": "score",
        "instructions": description or ("Score `%s` from %d to %d" % (name, lo, hi)),
        "criteria": [str(v) for v in range(lo, hi + 1)],
    }
    return _Field(name=name, kind="score", question=question, minimum=lo)


def _field(path: str, name: str, prop: Dict[str, Any]) -> _Field:
    if not isinstance(prop, dict):
        raise SchemaError("%s: property must be an object, got %s" % (path, type(prop).__name__))
    description = prop.get("description")
    if not ({"const", "enum", "type"} & set(prop)):
        union = prop.get("anyOf") or prop.get("oneOf")
        if union is not None:
            branches = [b for b in union if isinstance(b, dict) and b.get("type") != "null"]
            if len(branches) != 1:
                raise SchemaError(
                    "%s: only 'Optional[...]' unions (one non-null branch) are supported, got %d"
                    % (path, len(branches))
                )
            branch = dict(branches[0])
            branch.setdefault("description", description)
            return _field(path, name, branch)
    if "const" in prop:
        return _enum_field(path, name, [prop["const"]], description)
    if "enum" in prop:
        return _enum_field(path, name, prop["enum"], description)

    jtype = prop.get("type")
    if isinstance(jtype, list):
        non_null_types = [t for t in jtype if t != "null"]
        if len(non_null_types) > 1:
            raise SchemaError("%s: 'type' has multiple non-null types; unions are not supported" % path)
        jtype = non_null_types[0] if non_null_types else None
    if jtype == "boolean":
        return _noul_field(path, name, description)
    if jtype == "string":
        raise SchemaError("%s: a free string cannot be a fixed option set; use 'enum' or a boolean" % path)
    if jtype in ("integer", "number"):
        return _score_field(path, name, prop, description)
    if jtype == "array":
        raise SchemaError("%s: arrays are not supported; ask one field per element" % path)
    if jtype == "object":
        raise SchemaError("%s: nested objects are not supported; flatten the schema" % path)
    if "$ref" in prop:
        raise SchemaError("%s: $ref/recursion is not supported; flatten the schema" % path)
    raise SchemaError("%s: unsupported schema %r" % (path, prop))


def plan_from_json_schema(schema: Dict[str, Any]) -> List[_Field]:
    if not isinstance(schema, dict):
        raise SchemaError("expected a JSON schema object, got %s" % type(schema).__name__)
    if schema.get("type") not in (None, "object") or "properties" not in schema:
        raise SchemaError("the top level must be an object with 'properties'")
    properties = schema["properties"]
    if not isinstance(properties, dict) or not properties:
        raise SchemaError("'properties' must be a non-empty object")
    if len(properties) > MAX_PROPERTIES:
        raise SchemaError("%d properties exceeds MAX_PROPERTIES=%d" % (len(properties), MAX_PROPERTIES))
    return [_field("properties.%s" % name, name, prop) for name, prop in properties.items()]


def questions_from_json_schema(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {f.name: f.question for f in plan_from_json_schema(schema)}


def _project(answers: Dict[str, Any], fields: Sequence[_Field]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for f in fields:
        answer = answers.get(f.name)
        if answer is None:
            continue
        if answer.get("low_confidence"):
            values[f.name] = None
            continue
        if f.kind == "noul":
            values[f.name] = bool(float(answer.get("noul", 0.0)) >= 0.5)
        elif f.kind == "score":
            probs = answer.get("probabilities") or {}
            if probs:
                idx = max(range(len(probs)), key=lambda i: float(probs.get(str(i), probs.get(i, 0.0))))
            else:
                idx = int(round(float(answer.get("score", 0.0)))) - int(f.minimum or 0)
            values[f.name] = int(f.minimum or 0) + idx
        else:
            label = str(answer.get("choice"))
            values[f.name] = next((value for lbl, value in f.options if lbl == label), label)
    return values


def answers_to_json(answers: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    return _project(answers, plan_from_json_schema(schema))


def _details(values: Dict[str, Any], answers: Dict[str, Any], result: Dict[str, Any]) -> DecisionResult:
    confidence: Dict[str, float] = {}
    probabilities: Dict[str, Dict[str, Any]] = {}
    for name, answer in answers.items():
        confidence[name] = float(answer.get("confidence", 0.0))
        if answer.get("type") == "noul":
            p = float(answer.get("noul", 0.0))
            probabilities[name] = {"false": round(1.0 - p, 4), "true": round(p, 4)}
        else:
            probabilities[name] = dict(answer.get("probabilities") or {})
    return DecisionResult(
        values=values,
        confidence=confidence,
        probabilities=probabilities,
        answers=dict(answers),
        usage=result.get("usage"),
        routing=result.get("routing"),
    )


def decide(
    runner,
    state: Any,
    schema: Any = None,
    *,
    questions: Optional[Dict[str, Any]] = None,
    return_details: bool = False,
    min_confidence: Optional[float] = None,
    **predict_kwargs,
) -> Any:
    if (schema is None) == (questions is None):
        raise ValueError("pass exactly one of schema= or questions=")

    mc = check_min_confidence(min_confidence) if min_confidence is not None else None
    if mc is not None:
        predict_kwargs["min_confidence"] = mc

    fields: Optional[List[_Field]] = None
    if schema is not None:
        fields = plan_from_json_schema(_schema_of(schema))
        questions = {f.name: f.question for f in fields}

    try:
        result = runner.predict(state, questions, **predict_kwargs)
    except TypeError as e:
        if mc is not None and "unexpected keyword argument 'min_confidence'" in str(e):
            predict_kwargs.pop("min_confidence", None)
            result = runner.predict(state, questions, **predict_kwargs)
        else:
            raise

    if mc is not None and isinstance(result, dict):
        flag_low_confidence([result], mc)

    answers = result.get("answers", {}) or {}
    values = _project(answers, fields) if fields is not None else dict(answers)
    if return_details:
        return _details(values, answers, result)
    return values
