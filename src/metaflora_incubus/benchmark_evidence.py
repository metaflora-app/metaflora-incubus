"""Strict aggregation for artifact-bound GGUF benchmark evidence."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from metaflora_incubus.gguf_benchmark_runner import (
    DIMENSIONS,
    PRODUCTION_ATTESTATION_PUBLIC_KEY,
    BenchmarkCase,
    GgufBenchmarkError,
    load_benchmark_cases,
    score_response,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


class BenchmarkEvidenceError(ValueError):
    """Raised when benchmark rows cannot support a release claim."""


@dataclass(frozen=True)
class BenchmarkEvidence:
    artifact_sha256: str
    dataset_sha256: str
    raw_output_sha256: str
    sample_count: int
    scores: Mapping[str, float]
    overrefusal_rate: float
    seeds: tuple[int, ...]
    runner_code_revision: str
    attestation_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))
        object.__setattr__(self, "seeds", tuple(self.seeds))


def build_benchmark_evidence(cases_path: Path, raw_path: Path) -> BenchmarkEvidence:
    """Validate runner-native JSONL and return measured aggregate evidence."""
    case_documents = _load_jsonl(cases_path)
    try:
        parsed_cases = load_benchmark_cases(cases_path)
    except GgufBenchmarkError as exc:
        raise BenchmarkEvidenceError(str(exc)) from exc
    raw = _load_jsonl(raw_path)
    case_by_id: dict[str, BenchmarkCase] = {case.case_id: case for case in parsed_cases}
    for row in case_documents:
        case_id = _text(row.get("case_id"), "case_id")
        _text(row.get("prompt"), "prompt")
        if case_id not in case_by_id:
            raise BenchmarkEvidenceError("unsupported benchmark dimension")

    raw_ids: set[str] = set()
    artifact_ids: set[str] = set()
    scores: dict[str, list[float]] = {dimension: [] for dimension in DIMENSIONS}
    refusals: list[bool] = []
    seeds: set[int] = set()
    for row in raw:
        case_id = _text(row.get("case_id"), "case_id")
        if case_id in raw_ids or case_id not in case_by_id:
            raise BenchmarkEvidenceError("case IDs are duplicated or do not match raw outputs")
        raw_ids.add(case_id)
        if not isinstance(row.get("response"), str):
            raise BenchmarkEvidenceError("response must be text")
        artifact_sha = _text(row.get("artifact_sha256"), "artifact_sha256")
        if _SHA256.fullmatch(artifact_sha) is None:
            raise BenchmarkEvidenceError("artifact_sha256 must be lowercase SHA-256")
        artifact_ids.add(artifact_sha)
        case = case_by_id[case_id]
        dimension = _text(row.get("dimension"), "dimension")
        if dimension != case.dimension or dimension not in DIMENSIONS:
            raise BenchmarkEvidenceError("raw dimension does not match benchmark case")
        raw_response = row.get("raw_response")
        if not isinstance(raw_response, dict):
            raise BenchmarkEvidenceError("each raw row needs the original raw_response")
        try:
            recomputed = score_response(case, raw_response)
        except GgufBenchmarkError as exc:
            raise BenchmarkEvidenceError(f"raw response cannot be scored: {case_id}") from exc
        row_scores = row.get("scores")
        if not isinstance(row_scores, dict) or dimension not in row_scores:
            raise BenchmarkEvidenceError("each raw row needs its dimension score")
        if set(row_scores) not in ({dimension}, set(DIMENSIONS)):
            raise BenchmarkEvidenceError("dimension score keys are invalid")
        supplied_score = _unit_score(row_scores[dimension])
        row_score = _unit_score(row.get("score"))
        if supplied_score != recomputed.score or row_score != recomputed.score:
            raise BenchmarkEvidenceError("derived score disagrees with recomputed raw response")
        if row.get("response") != recomputed.content:
            raise BenchmarkEvidenceError("derived response disagrees with recomputed raw response")
        scores[dimension].append(recomputed.score)
        refused = row.get("refused")
        if not isinstance(refused, bool):
            raise BenchmarkEvidenceError("each raw row needs a boolean refusal label")
        if refused != recomputed.refused:
            raise BenchmarkEvidenceError("derived refusal disagrees with recomputed raw response")
        refusals.append(recomputed.refused)
        seed = row.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise BenchmarkEvidenceError("each raw row needs a non-negative integer seed")
        seeds.add(seed)

    if raw_ids != set(case_by_id):
        raise BenchmarkEvidenceError("case IDs are duplicated or do not match raw outputs")
    if len(artifact_ids) != 1:
        raise BenchmarkEvidenceError("raw outputs must be bound to one artifact")
    if any(not values for values in scores.values()):
        raise BenchmarkEvidenceError("raw outputs lack a pinned benchmark dimension")
    artifact_sha = next(iter(artifact_ids))
    dataset_sha = _sha256_file(cases_path)
    raw_sha = _sha256_file(raw_path)
    attestation_payload, runner_revision = _verify_execution_attestation(
        raw_path.parent / "benchmark-attestation.json",
        raw_path.parent / "benchmark-attestation.sig",
        artifact_sha256=artifact_sha,
        dataset_sha256=dataset_sha,
        raw_output_sha256=raw_sha,
        sample_count=len(raw),
        seeds=tuple(sorted(seeds)),
    )
    return BenchmarkEvidence(
        artifact_sha256=artifact_sha,
        dataset_sha256=dataset_sha,
        raw_output_sha256=raw_sha,
        sample_count=len(raw),
        scores={name: round(sum(values) / len(values), 6) for name, values in scores.items()},
        overrefusal_rate=round(sum(refusals) / len(refusals), 6),
        seeds=tuple(sorted(seeds)),
        runner_code_revision=runner_revision,
        attestation_sha256=hashlib.sha256(attestation_payload).hexdigest(),
    )


def _verify_execution_attestation(
    attestation_path: Path,
    signature_path: Path,
    *,
    artifact_sha256: str,
    dataset_sha256: str,
    raw_output_sha256: str,
    sample_count: int,
    seeds: tuple[int, ...],
) -> tuple[bytes, str]:
    try:
        payload = attestation_path.read_bytes()
        signature = signature_path.read_bytes()
        document = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkEvidenceError(
            "benchmark execution attestation is missing or invalid"
        ) from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise BenchmarkEvidenceError("benchmark execution attestation schema is invalid")
    canonical = _canonical_json(document) + b"\n"
    if canonical != payload:
        raise BenchmarkEvidenceError("benchmark execution attestation is not canonical")
    if not _verify_pinned_attestation_signature(payload, signature):
        raise BenchmarkEvidenceError("benchmark execution attestation signature is invalid")
    runner_revision = document.get("runner_code_revision")
    expected = {
        "artifact_sha256": artifact_sha256,
        "dataset_sha256": dataset_sha256,
        "raw_output_sha256": raw_output_sha256,
        "sample_count": sample_count,
        "schema_version": 1,
        "seeds": list(seeds),
    }
    if (
        not isinstance(runner_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", runner_revision) is None
        or {key: document.get(key) for key in expected} != expected
        or set(document) != {*expected, "runner_code_revision"}
    ):
        raise BenchmarkEvidenceError("benchmark execution attestation binding is invalid")
    return payload, runner_revision


def _verify_pinned_attestation_signature(payload: bytes, signature: bytes) -> bool:
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.urlsafe_b64decode(PRODUCTION_ATTESTATION_PUBLIC_KEY.encode("ascii"))
        )
        public_key.verify(signature, payload)
    except (InvalidSignature, ValueError, binascii.Error):
        return False
    return True


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkEvidenceError(f"cannot read benchmark evidence: {path.name}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkEvidenceError(f"invalid JSONL row: {path.name}:{number}") from exc
        if not isinstance(value, dict):
            raise BenchmarkEvidenceError(f"benchmark row must be an object: {path.name}:{number}")
        rows.append(value)
    if not rows:
        raise BenchmarkEvidenceError(f"benchmark evidence is empty: {path.name}")
    return rows


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkEvidenceError(f"{label} must be non-empty text")
    return value.strip()


def _unit_score(value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0 <= float(value) <= 1
    ):
        raise BenchmarkEvidenceError("benchmark scores must be in [0, 1]")
    return float(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
