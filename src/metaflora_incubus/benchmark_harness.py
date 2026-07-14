"""Fail-closed benchmark provenance and weakness regression aggregation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


class ProvenanceError(ValueError):
    pass


class Language(str, Enum):
    RU = "ru"
    EN = "en"


class EvalDimension(str, Enum):
    TOOL_SCHEMA = "tool_schema"
    TOOL_RECOVERY = "tool_recovery"
    TOOL_PROTOCOL = "tool_protocol"
    LONG_CONTEXT = "long_context"
    GENERATION_HEALTH = "generation_health"
    ANSWER_QUALITY = "answer_quality"


class ToolEventKind(str, Enum):
    CALL = "call"
    ERROR = "error"


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    frozen = {
        key: _freeze_mapping(item) if isinstance(item, Mapping) else item
        for key, item in value.items()
    }
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class ExpectedToolCall:
    name: str
    arguments_schema: Mapping[str, Any]
    expected_arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments_schema", _freeze_mapping(self.arguments_schema))
        object.__setattr__(self, "expected_arguments", _freeze_mapping(self.expected_arguments))


@dataclass(frozen=True)
class ToolEvent:
    kind: ToolEventKind
    name: str
    arguments: Mapping[str, Any] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    dimension: EvalDimension
    language: Language
    prompt: str
    expected_tool_call: ExpectedToolCall | None = None
    pair_id: str | None = None
    context_tokens: int = 0


@dataclass(frozen=True)
class EvalObservation:
    case_id: str
    output: str
    useful: bool
    refused: bool
    tool_events: tuple[ToolEvent, ...]
    finish_reason: str
    output_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_events", tuple(self.tool_events))


@dataclass(frozen=True)
class HarnessPolicy:
    required_dimensions: tuple[EvalDimension, ...]
    required_languages: tuple[Language, ...]
    maximum_long_context_drop: float
    maximum_repetition_ratio: float
    maximum_overrefusal_rate: float
    minimum_useful_answer_rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_dimensions", tuple(self.required_dimensions))
        object.__setattr__(self, "required_languages", tuple(self.required_languages))


@dataclass(frozen=True)
class BenchmarkProvenance:
    artifact_sha256: str
    dataset_sha256: str
    harness_revision: str
    prompt_template_sha256: str
    runtime_name: str
    runtime_version: str
    seeds: tuple[int, ...]
    sample_count: int
    raw_output_sha256: str
    signer_id: str
    signature: str

    @classmethod
    def create(cls, **values: Any) -> BenchmarkProvenance:
        hashes = (
            "artifact_sha256",
            "dataset_sha256",
            "prompt_template_sha256",
            "raw_output_sha256",
        )
        for field in hashes:
            value = values.get(field)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ProvenanceError(f"invalid {field}")
        for field in (
            "harness_revision",
            "runtime_name",
            "runtime_version",
            "signer_id",
            "signature",
        ):
            value = values.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ProvenanceError(f"invalid {field}")
        seeds = values.get("seeds")
        if (
            not isinstance(seeds, (tuple, list))
            or not seeds
            or any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds
            )
            or len(set(seeds)) != len(seeds)
        ):
            raise ProvenanceError("invalid seeds")
        sample_count = values.get("sample_count")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
            raise ProvenanceError("invalid sample_count")
        return cls(
            artifact_sha256=values["artifact_sha256"],
            dataset_sha256=values["dataset_sha256"],
            harness_revision=values["harness_revision"].strip(),
            prompt_template_sha256=values["prompt_template_sha256"],
            runtime_name=values["runtime_name"].strip(),
            runtime_version=values["runtime_version"].strip(),
            seeds=tuple(seeds),
            sample_count=sample_count,
            raw_output_sha256=values["raw_output_sha256"],
            signer_id=values["signer_id"].strip(),
            signature=values["signature"].strip(),
        )

    def signed_payload(self) -> bytes:
        payload = {
            "artifact_sha256": self.artifact_sha256,
            "dataset_sha256": self.dataset_sha256,
            "harness_revision": self.harness_revision,
            "prompt_template_sha256": self.prompt_template_sha256,
            "raw_output_sha256": self.raw_output_sha256,
            "runtime_name": self.runtime_name,
            "runtime_version": self.runtime_version,
            "sample_count": self.sample_count,
            "seeds": self.seeds,
            "signer_id": self.signer_id,
        }
        return _canonical_json(payload)


@dataclass(frozen=True)
class HarnessFailure:
    code: str
    detail: str


@dataclass(frozen=True)
class HarnessReport:
    approved: bool
    metrics: Mapping[str, float]
    failures: tuple[HarnessFailure, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


def verify_benchmark_provenance(
    provenance: BenchmarkProvenance,
    *,
    verifier: Callable[[str, bytes, str], bool],
) -> bool:
    try:
        return (
            verifier(provenance.signer_id, provenance.signed_payload(), provenance.signature)
            is True
        )
    except Exception:
        return False


def canonical_raw_output_digest(outputs: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    for output in outputs:
        digest.update(_canonical_json(_to_primitive(output)))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_eval_case_digest(cases: tuple[EvalCase, ...]) -> str:
    """Hash the exact immutable case set in its execution order."""
    digest = hashlib.sha256()
    for case in cases:
        digest.update(_canonical_json(_to_primitive(case)))
        digest.update(b"\n")
    return digest.hexdigest()


def aggregate_regression_run(
    *,
    cases: tuple[EvalCase, ...],
    observations: tuple[EvalObservation, ...],
    provenance: BenchmarkProvenance,
    policy: HarnessPolicy,
    signature_verifier: Callable[[str, bytes, str], bool],
) -> HarnessReport:
    failures: list[HarnessFailure] = []
    case_by_id = {item.case_id: item for item in cases}
    observation_by_id = {item.case_id: item for item in observations}
    case_ids = [item.case_id for item in cases]
    observation_ids = [item.case_id for item in observations]
    if (
        len(case_by_id) != len(cases)
        or len(observation_by_id) != len(observations)
        or set(case_ids) != set(observation_ids)
    ):
        failures.append(HarnessFailure("observation_set_invalid", "case IDs do not match"))
    if not verify_benchmark_provenance(provenance, verifier=signature_verifier):
        failures.append(HarnessFailure("provenance_signature_invalid", provenance.signer_id))
    if provenance.sample_count != len(observations):
        failures.append(HarnessFailure("sample_count_mismatch", str(len(observations))))
    if provenance.raw_output_sha256 != canonical_raw_output_digest(observations):
        failures.append(HarnessFailure("raw_output_digest_mismatch", "observations"))
    if provenance.dataset_sha256 != canonical_eval_case_digest(cases):
        failures.append(HarnessFailure("dataset_digest_mismatch", "cases"))

    dimensions = {item.dimension for item in cases}
    for required in policy.required_dimensions:
        if required not in dimensions:
            failures.append(HarnessFailure("missing_dimension", required.value))
    languages = {item.language for item in cases}
    for required in policy.required_languages:
        if required not in languages:
            failures.append(HarnessFailure("missing_language", required.value))

    for case_id in set(case_by_id) & set(observation_by_id):
        _evaluate_case(case_by_id[case_id], observation_by_id[case_id], policy, failures)
    _evaluate_long_context(cases, observation_by_id, policy, failures)

    answer_case_ids = [
        item.case_id for item in cases if item.dimension is EvalDimension.ANSWER_QUALITY
    ]
    answer_observations = [
        observation_by_id[case_id] for case_id in answer_case_ids if case_id in observation_by_id
    ]
    metrics: dict[str, float] = {
        "overrefusal_rate": _rate(answer_observations, lambda item: item.refused),
        "useful_answer_rate": _rate(
            answer_observations, lambda item: item.useful and not item.refused
        ),
    }
    for language in Language:
        language_ids = [item.case_id for item in cases if item.language is language]
        language_observations = [
            observation_by_id[case_id] for case_id in language_ids if case_id in observation_by_id
        ]
        metrics[f"useful_answer_rate:{language.value}"] = _rate(
            language_observations, lambda item: item.useful and not item.refused
        )
    if answer_observations:
        if metrics["overrefusal_rate"] > policy.maximum_overrefusal_rate:
            failures.append(HarnessFailure("overrefusal_target_miss", "answer quality"))
        if metrics["useful_answer_rate"] < policy.minimum_useful_answer_rate:
            failures.append(HarnessFailure("usefulness_target_miss", "answer quality"))
    result = tuple(failures)
    return HarnessReport(approved=not result, metrics=metrics, failures=result)


def _evaluate_case(
    case: EvalCase,
    observation: EvalObservation,
    policy: HarnessPolicy,
    failures: list[HarnessFailure],
) -> None:
    expected = case.expected_tool_call
    if case.dimension is EvalDimension.TOOL_SCHEMA and expected is not None:
        calls = [event for event in observation.tool_events if event.kind is ToolEventKind.CALL]
        if not any(_valid_tool_call(event, expected) for event in calls):
            failures.append(HarnessFailure("tool_schema_invalid", case.case_id))
    elif case.dimension is EvalDimension.TOOL_RECOVERY and expected is not None:
        error_index = next(
            (
                index
                for index, event in enumerate(observation.tool_events)
                if event.kind is ToolEventKind.ERROR
            ),
            None,
        )
        recovered = error_index is not None and any(
            event.kind is ToolEventKind.CALL and _valid_tool_call(event, expected)
            for event in observation.tool_events[error_index + 1 :]
        )
        if not recovered:
            failures.append(HarnessFailure("tool_recovery_missing", case.case_id))
    elif case.dimension is EvalDimension.TOOL_PROTOCOL and expected is not None:
        calls = [event for event in observation.tool_events if event.kind is ToolEventKind.CALL]
        if not calls and expected.name in observation.output:
            failures.append(HarnessFailure("textual_tool_imitation", case.case_id))
        elif not any(_valid_tool_call(event, expected) for event in calls):
            failures.append(HarnessFailure("tool_schema_invalid", case.case_id))
    elif case.dimension is EvalDimension.GENERATION_HEALTH:
        if (
            observation.finish_reason == "length"
            or _repetition_ratio(observation.output) > policy.maximum_repetition_ratio
        ):
            failures.append(HarnessFailure("generation_degenerate", case.case_id))


def _evaluate_long_context(
    cases: tuple[EvalCase, ...],
    observations: Mapping[str, EvalObservation],
    policy: HarnessPolicy,
    failures: list[HarnessFailure],
) -> None:
    pairs: dict[str, list[EvalCase]] = {}
    for item in cases:
        if item.dimension is not EvalDimension.LONG_CONTEXT:
            continue
        if not item.pair_id:
            failures.append(HarnessFailure("long_context_pair_invalid", item.case_id))
            continue
        pairs.setdefault(item.pair_id, []).append(item)
    for pair_id, pair in pairs.items():
        if (
            len(pair) != 2
            or len({item.context_tokens for item in pair}) != 2
            or any(item.case_id not in observations for item in pair)
        ):
            failures.append(HarnessFailure("long_context_pair_invalid", pair_id))
            continue
        ordered = sorted(pair, key=lambda item: item.context_tokens)
        short_score = float(observations[ordered[0].case_id].useful)
        long_score = float(observations[ordered[-1].case_id].useful)
        if short_score - long_score > policy.maximum_long_context_drop:
            failures.append(HarnessFailure("long_context_degradation", pair_id))


def _valid_tool_call(event: ToolEvent, expected: ExpectedToolCall) -> bool:
    arguments = dict(event.arguments)
    return (
        event.name == expected.name
        and arguments == dict(expected.expected_arguments)
        and _matches_object_schema(arguments, expected.arguments_schema)
    )


def _matches_object_schema(arguments: dict[str, Any], schema: Mapping[str, Any]) -> bool:
    if schema.get("type") != "object":
        return False
    properties = schema.get("properties")
    required = schema.get("required", ())
    if not isinstance(properties, Mapping) or not isinstance(required, (tuple, list)):
        return False
    if any(key not in arguments for key in required):
        return False
    if schema.get("additionalProperties") is False and any(
        key not in properties for key in arguments
    ):
        return False
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for key, value in arguments.items():
        spec = properties.get(key)
        if not isinstance(spec, Mapping) or spec.get("type") not in type_map:
            return False
        expected_type = type_map[spec["type"]]
        if not isinstance(value, expected_type):
            return False
        if spec["type"] in {"integer", "number"} and isinstance(value, bool):
            return False
    return True


def _repetition_ratio(output: str) -> float:
    words = re.findall(r"\w+", output.lower(), flags=re.UNICODE)
    if not words:
        return 0.0
    return max(Counter(words).values()) / len(words)


def _rate(items: list[EvalObservation], predicate: Callable[[EvalObservation], bool]) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if predicate(item)) / len(items)


def load_weakness_regressions(payload: str) -> tuple[EvalCase, ...]:
    cases: list[EvalCase] = []
    required = {
        "category",
        "reproducer",
        "reproducer_hash",
        "source_date",
        "source_model_revision",
        "source_type",
        "source_url",
        "status",
    }
    for line in payload.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or not required <= set(row)
            or row["status"] not in {"confirmed", "resolved"}
            or re.fullmatch(r"[0-9a-f]{64}", row["reproducer_hash"]) is None
            or not isinstance(row["reproducer"], str)
            or not row["reproducer"].strip()
        ):
            raise ValueError("invalid weakness regression row")
        cases.append(
            EvalCase(
                case_id=f"weakness-{row['reproducer_hash']}",
                dimension=EvalDimension.ANSWER_QUALITY,
                language=Language.EN,
                prompt=row["reproducer"],
            )
        )
    if not cases:
        raise ValueError("no confirmed weakness regressions")
    return tuple(sorted(cases, key=lambda item: item.case_id))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_primitive(item) for item in value]
    return value
