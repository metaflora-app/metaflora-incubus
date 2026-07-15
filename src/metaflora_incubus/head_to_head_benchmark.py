"""Paired GGUF head-to-head execution and fail-closed promotion statistics."""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from metaflora_incubus.gguf_benchmark_runner import (
    BenchmarkRunnerConfig,
    run_gguf_benchmark,
)

MODEL_ROLES = ("candidate", "incumbent", "same_size", "conditional")
CAPABILITY_GROUPS: Mapping[str, tuple[str, ...]] = {
    "coding": ("coding",),
    "agentic_tool_use": ("tool_calling", "agentic_search"),
    "ru_text": ("russian",),
    "en_text": ("english",),
    "text_quality": ("text_quality",),
}
_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}")
_LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}")


class HeadToHeadError(RuntimeError):
    """Raised when a comparison cannot produce trustworthy paired evidence."""


@dataclass(frozen=True)
class ComparisonModel:
    """Exact local GGUF and its role in a private comparison."""

    model_id: str
    role: str
    model_path: Path
    model_sha256: str

    @classmethod
    def create(cls, **values: object) -> ComparisonModel:
        model_id = values.get("model_id")
        role = values.get("role")
        path = values.get("model_path")
        digest = values.get("model_sha256")
        if not isinstance(model_id, str) or _MODEL_ID.fullmatch(model_id) is None:
            raise HeadToHeadError("model_id is invalid")
        if role not in MODEL_ROLES:
            raise HeadToHeadError("model role is invalid")
        if not isinstance(path, Path):
            raise HeadToHeadError("model_path must be a Path")
        if not isinstance(digest, str) or _LOWER_HEX_64.fullmatch(digest) is None:
            raise HeadToHeadError("model_sha256 is invalid")
        return cls(model_id=model_id, role=str(role), model_path=path, model_sha256=digest)


@dataclass(frozen=True)
class HeadToHeadPolicy:
    """Predeclared statistical promotion policy."""

    bootstrap_iterations: int
    confidence_level: float
    maximum_capability_regression: float
    maximum_overrefusal_increase: float
    bootstrap_seed: int

    @classmethod
    def release(cls) -> HeadToHeadPolicy:
        return cls(
            bootstrap_iterations=5_000,
            confidence_level=0.95,
            maximum_capability_regression=0.05,
            maximum_overrefusal_increase=0.02,
            bootstrap_seed=1701,
        )

    def __post_init__(self) -> None:
        if self.bootstrap_iterations < 1_000:
            raise HeadToHeadError("bootstrap_iterations must be at least 1000")
        if not 0.8 <= self.confidence_level < 1:
            raise HeadToHeadError("confidence_level is invalid")
        if not 0 <= self.maximum_capability_regression <= 1:
            raise HeadToHeadError("maximum_capability_regression is invalid")
        if not 0 <= self.maximum_overrefusal_increase <= 1:
            raise HeadToHeadError("maximum_overrefusal_increase is invalid")
        if self.bootstrap_seed < 0:
            raise HeadToHeadError("bootstrap_seed is invalid")


@dataclass(frozen=True)
class HeadToHeadConfig:
    """One immutable execution plan shared by every compared artifact."""

    server_binary: Path
    server_sha256: str
    cases_path: Path
    output_dir: Path
    runner_code_revision: str
    models: tuple[ComparisonModel, ...]
    seed: int
    port: int
    gpu_layers: int
    health_timeout_seconds: float
    request_timeout_seconds: float

    @classmethod
    def create(cls, **values: object) -> HeadToHeadConfig:
        server = _path(values.get("server_binary"), "server_binary")
        cases = _path(values.get("cases_path"), "cases_path")
        output = _path(values.get("output_dir"), "output_dir")
        server_sha = _sha(values.get("server_sha256"), "server_sha256")
        revision = values.get("runner_code_revision")
        if not isinstance(revision, str) or _LOWER_HEX_40.fullmatch(revision) is None:
            raise HeadToHeadError("runner_code_revision is invalid")
        model_values = values.get("models")
        if not isinstance(model_values, (tuple, list)) or not model_values:
            raise HeadToHeadError("models must not be empty")
        models = tuple(model_values)
        if any(not isinstance(model, ComparisonModel) for model in models):
            raise HeadToHeadError("models contain an invalid entry")
        _validate_model_set(models)
        seed = _integer(values.get("seed"), "seed", minimum=0)
        port = _integer(values.get("port"), "port", minimum=1)
        if port + len(models) - 1 > 65535:
            raise HeadToHeadError("port range is invalid")
        gpu_layers = _integer(values.get("gpu_layers"), "gpu_layers", minimum=0)
        health_timeout = _positive_number(
            values.get("health_timeout_seconds"), "health_timeout_seconds"
        )
        request_timeout = _positive_number(
            values.get("request_timeout_seconds"), "request_timeout_seconds"
        )
        return cls(
            server_binary=server,
            server_sha256=server_sha,
            cases_path=cases,
            output_dir=output,
            runner_code_revision=revision,
            models=models,
            seed=seed,
            port=port,
            gpu_layers=gpu_layers,
            health_timeout_seconds=health_timeout,
            request_timeout_seconds=request_timeout,
        )


BenchmarkRunner = Callable[[BenchmarkRunnerConfig], dict[str, object]]


def load_head_to_head_config(path: Path) -> HeadToHeadConfig:
    """Load a private JSON execution manifest with paths relative to the manifest."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeadToHeadError("head-to-head manifest cannot be read") from exc
    if not isinstance(document, dict) or not isinstance(document.get("models"), list):
        raise HeadToHeadError("head-to-head manifest is invalid")
    base = path.resolve().parent
    models: list[ComparisonModel] = []
    for item in document["models"]:
        if not isinstance(item, dict):
            raise HeadToHeadError("head-to-head manifest model is invalid")
        models.append(
            ComparisonModel.create(
                model_id=item.get("id"),
                role=item.get("role"),
                model_path=_relative_path(base, item.get("path"), "model path"),
                model_sha256=item.get("sha256"),
            )
        )
    return HeadToHeadConfig.create(
        server_binary=_relative_path(base, document.get("server_binary"), "server_binary"),
        server_sha256=document.get("server_sha256"),
        cases_path=_relative_path(base, document.get("cases_path"), "cases_path"),
        output_dir=_relative_path(base, document.get("output_dir"), "output_dir"),
        runner_code_revision=document.get("runner_code_revision"),
        models=tuple(models),
        seed=document.get("seed"),
        port=document.get("port"),
        gpu_layers=document.get("gpu_layers"),
        health_timeout_seconds=document.get("health_timeout_seconds"),
        request_timeout_seconds=document.get("request_timeout_seconds"),
    )


def run_head_to_head(
    config: HeadToHeadConfig,
    *,
    policy: HeadToHeadPolicy | None = None,
    benchmark_runner: BenchmarkRunner = run_gguf_benchmark,
) -> dict[str, object]:
    """Run all models sequentially with identical settings and compare paired outputs."""
    runs: dict[str, tuple[dict[str, object], ...]] = {}
    roles = {model.model_id: model.role for model in config.models}
    conditional: dict[str, dict[str, str]] = {}
    for offset, model in enumerate(config.models):
        output_dir = config.output_dir / "runs" / model.model_id
        runner_config = BenchmarkRunnerConfig.create(
            server_binary=config.server_binary,
            server_sha256=config.server_sha256,
            model_path=model.model_path,
            model_sha256=model.model_sha256,
            cases_path=config.cases_path,
            output_dir=output_dir,
            seed=config.seed,
            port=config.port + offset,
            health_timeout_seconds=config.health_timeout_seconds,
            request_timeout_seconds=config.request_timeout_seconds,
            runner_code_revision=config.runner_code_revision,
            gpu_layers=config.gpu_layers,
        )
        try:
            benchmark_runner(runner_config)
            runs[model.model_id] = load_raw_run(output_dir / "benchmark-raw.jsonl")
            if model.role == "conditional":
                conditional[model.model_id] = {"status": "completed"}
        except Exception as exc:
            if model.role != "conditional":
                raise HeadToHeadError(f"mandatory benchmark failed: {model.model_id}") from exc
            conditional[model.model_id] = {
                "status": "unavailable",
                "reason": "benchmark_failed",
            }

    report = compare_runs(runs, roles, policy or HeadToHeadPolicy.release())
    report = {
        **report,
        "conditional_models": conditional,
        "execution": {
            "case_bank": str(config.cases_path),
            "gpu_layers": config.gpu_layers,
            "max_tokens": 512,
            "parallel": 1,
            "runner_code_revision": config.runner_code_revision,
            "runtime_sha256": config.server_sha256,
            "seed": config.seed,
            "temperature": 0,
        },
        "model_artifacts": {
            model.model_id: {
                "role": model.role,
                "sha256": model.model_sha256,
            }
            for model in config.models
        },
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(config.output_dir / "head-to-head-report.json", report)
    return report


def load_raw_run(path: Path) -> tuple[dict[str, object], ...]:
    """Load a completed raw run without accepting duplicate or malformed cases."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HeadToHeadError("raw benchmark output is missing") from exc
    if not lines:
        raise HeadToHeadError("raw benchmark output is empty")
    rows: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HeadToHeadError("raw benchmark output is invalid") from exc
        _validate_row(row)
        case_id = str(row["case_id"])
        if case_id in identifiers:
            raise HeadToHeadError("raw benchmark output has duplicate cases")
        identifiers.add(case_id)
        rows.append(row)
    return tuple(rows)


def compare_runs(
    runs: Mapping[str, Sequence[Mapping[str, object]]],
    roles: Mapping[str, str],
    policy: HeadToHeadPolicy,
) -> dict[str, object]:
    """Apply paired bootstrap superiority and product guardrails."""
    candidate_ids = [model_id for model_id, role in roles.items() if role == "candidate"]
    if len(candidate_ids) != 1:
        raise HeadToHeadError("comparison requires exactly one candidate")
    candidate_id = candidate_ids[0]
    required = [model_id for model_id, role in roles.items() if role in {"incumbent", "same_size"}]
    if not any(roles[item] == "incumbent" for item in required) or not any(
        roles[item] == "same_size" for item in required
    ):
        raise HeadToHeadError("comparison requires incumbent and same-size baselines")
    if candidate_id not in runs or any(model_id not in runs for model_id in required):
        raise HeadToHeadError("mandatory comparison run is missing")

    normalized = {model_id: _rows_by_case(rows) for model_id, rows in runs.items()}
    reference_ids = set(normalized[candidate_id])
    if not reference_ids or any(set(rows) != reference_ids for rows in normalized.values()):
        raise HeadToHeadError("comparison case sets are not identical")
    ordered_ids = sorted(reference_ids)
    _verify_paired_metadata(normalized, ordered_ids)

    comparisons: dict[str, object] = {}
    failures: list[str] = []
    for comparator_id, comparator_rows in normalized.items():
        if comparator_id == candidate_id:
            continue
        comparison = _compare_pair(
            normalized[candidate_id],
            comparator_rows,
            ordered_ids,
            policy,
            seed_offset=sum(comparator_id.encode("utf-8")),
        )
        comparisons[comparator_id] = comparison
        if comparator_id not in required:
            continue
        if float(comparison["overall_delta_ci95"][0]) <= 0:
            failures.append(f"overall_lead_not_significant:{comparator_id}")
        for capability, delta in comparison["capability_deltas"].items():
            if float(delta) < -policy.maximum_capability_regression:
                failures.append(f"capability_regression:{comparator_id}:{capability}")
        if float(comparison["overrefusal_delta_ci95"][1]) > policy.maximum_overrefusal_increase:
            failures.append(f"overrefusal_regression:{comparator_id}")

    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "required_comparators": required,
        "advisory_comparators": [model_id for model_id in comparisons if model_id not in required],
        "criteria": {
            "paired_bootstrap_iterations": policy.bootstrap_iterations,
            "confidence_level": policy.confidence_level,
            "overall_superiority": "lower_ci_bound_gt_zero",
            "maximum_capability_regression": policy.maximum_capability_regression,
            "maximum_benign_overrefusal_increase": policy.maximum_overrefusal_increase,
        },
        "comparisons": comparisons,
        "failures": failures,
        "verdict": "promotion_passed" if not failures else "not_demonstrated",
        "public_winner_claim": False,
    }


def _compare_pair(
    candidate: Mapping[str, Mapping[str, object]],
    comparator: Mapping[str, Mapping[str, object]],
    case_ids: list[str],
    policy: HeadToHeadPolicy,
    *,
    seed_offset: int,
) -> dict[str, object]:
    score_deltas = [
        float(candidate[case_id]["score"]) - float(comparator[case_id]["score"])
        for case_id in case_ids
    ]
    refusal_deltas = [
        float(bool(candidate[case_id]["refused"])) - float(bool(comparator[case_id]["refused"]))
        for case_id in case_ids
    ]
    capability_deltas = {
        group: round(
            _mean(
                [
                    float(candidate[case_id]["score"]) - float(comparator[case_id]["score"])
                    for case_id in case_ids
                    if str(candidate[case_id]["dimension"]) in dimensions
                ]
            ),
            6,
        )
        for group, dimensions in CAPABILITY_GROUPS.items()
    }
    return {
        "case_count": len(case_ids),
        "candidate_mean": round(
            _mean([float(candidate[case_id]["score"]) for case_id in case_ids]), 6
        ),
        "comparator_mean": round(
            _mean([float(comparator[case_id]["score"]) for case_id in case_ids]), 6
        ),
        "overall_delta": round(_mean(score_deltas), 6),
        "overall_delta_ci95": _paired_bootstrap_interval(
            score_deltas, policy, seed_offset=seed_offset
        ),
        "capability_deltas": capability_deltas,
        "overrefusal_delta": round(_mean(refusal_deltas), 6),
        "overrefusal_delta_ci95": _paired_bootstrap_interval(
            refusal_deltas, policy, seed_offset=seed_offset + 1
        ),
    }


def _paired_bootstrap_interval(
    deltas: list[float], policy: HeadToHeadPolicy, *, seed_offset: int
) -> list[float]:
    if not deltas:
        raise HeadToHeadError("paired bootstrap requires observations")
    generator = random.Random(policy.bootstrap_seed + seed_offset)
    count = len(deltas)
    estimates = sorted(
        _mean([deltas[generator.randrange(count)] for _ in range(count)])
        for _ in range(policy.bootstrap_iterations)
    )
    tail = (1 - policy.confidence_level) / 2
    lower = estimates[max(0, math.floor(tail * policy.bootstrap_iterations))]
    upper_index = min(
        policy.bootstrap_iterations - 1,
        math.ceil((1 - tail) * policy.bootstrap_iterations) - 1,
    )
    upper = estimates[upper_index]
    return [round(lower, 6), round(upper, 6)]


def _rows_by_case(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        _validate_row(row)
        case_id = str(row["case_id"])
        if case_id in result:
            raise HeadToHeadError("comparison run has duplicate cases")
        result[case_id] = row
    return result


def _validate_row(row: object) -> None:
    if not isinstance(row, dict):
        raise HeadToHeadError("benchmark row must be an object")
    for name in ("case_id", "dimension", "language"):
        if not isinstance(row.get(name), str) or not row[name]:
            raise HeadToHeadError(f"benchmark row has invalid {name}")
    score = row.get("score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or not 0 <= float(score) <= 1
    ):
        raise HeadToHeadError("benchmark row has invalid score")
    if not isinstance(row.get("refused"), bool):
        raise HeadToHeadError("benchmark row has invalid refused flag")
    if not isinstance(row.get("seed"), int) or isinstance(row["seed"], bool):
        raise HeadToHeadError("benchmark row has invalid seed")


def _verify_paired_metadata(
    runs: Mapping[str, Mapping[str, Mapping[str, object]]], case_ids: list[str]
) -> None:
    reference = next(iter(runs.values()))
    for case_id in case_ids:
        expected = (
            reference[case_id]["dimension"],
            reference[case_id]["language"],
            reference[case_id]["seed"],
        )
        if any(
            (rows[case_id]["dimension"], rows[case_id]["language"], rows[case_id]["seed"])
            != expected
            for rows in runs.values()
        ):
            raise HeadToHeadError("paired benchmark metadata does not match")


def _validate_model_set(models: tuple[ComparisonModel, ...]) -> None:
    identifiers = [model.model_id for model in models]
    if len(set(identifiers)) != len(identifiers):
        raise HeadToHeadError("model IDs must be unique")
    counts = {role: sum(model.role == role for model in models) for role in MODEL_ROLES}
    if counts["candidate"] != 1 or counts["incumbent"] != 1 or counts["same_size"] < 1:
        raise HeadToHeadError("model roles require candidate, incumbent, and same-size baseline")


def _path(value: object, label: str) -> Path:
    if not isinstance(value, Path):
        raise HeadToHeadError(f"{label} must be a Path")
    return value


def _relative_path(base: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HeadToHeadError(f"{label} is invalid")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise HeadToHeadError(f"{label} is invalid")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise HeadToHeadError(f"{label} is invalid")
    return value


def _positive_number(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise HeadToHeadError(f"{label} is invalid")
    return float(value)


def _mean(values: list[float]) -> float:
    if not values:
        raise HeadToHeadError("metric group is empty")
    return sum(values) / len(values)


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
