from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from metaflora_incubus.benchmark_evidence import (
    BenchmarkEvidenceError,
    _verify_pinned_attestation_signature,
    build_benchmark_evidence,
)
from metaflora_incubus.gguf_benchmark_runner import PRODUCTION_ATTESTATION_PUBLIC_KEY

DIMENSIONS = (
    "coding",
    "tool_calling",
    "agentic_search",
    "text_quality",
    "russian",
    "english",
)


@pytest.fixture(autouse=True)
def accept_ephemeral_attestations_only_in_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "metaflora_incubus.benchmark_evidence._verify_pinned_attestation_signature",
        lambda payload, signature: signature == b"s" * 64,
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def runner_native_evidence(root: Path) -> tuple[Path, Path, str]:
    artifact_sha = hashlib.sha256(b"GGUF-candidate").hexdigest()
    cases = []
    raw = []
    for dimension in DIMENSIONS:
        for index in range(8):
            case_id = f"{dimension}-{index}"
            case = {
                "case_id": case_id,
                "dimension": dimension,
                "language": "ru" if index % 2 == 0 else "en",
                "prompt": "prompt",
                "required_terms": ["measured"],
                "forbidden_terms": [],
            }
            raw_response: dict[str, object] = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "measured response"},
                    }
                ]
            }
            if dimension in {"tool_calling", "agentic_search"}:
                case["expected_tool_name"] = "lookup"
                case["expected_tool_arguments"] = {"query": "value"}
                raw_response = {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"query":"value"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            cases.append(case)
            raw.append(
                {
                    "artifact_sha256": artifact_sha,
                    "case_id": case_id,
                    "dimension": dimension,
                    "raw_response": raw_response,
                    "response": (
                        "measured response"
                        if dimension not in {"tool_calling", "agentic_search"}
                        else ""
                    ),
                    "score": 1.0,
                    "scores": {dimension: 1.0},
                    "refused": False,
                    "seed": 4242,
                }
            )
    cases_path = root / "benchmark-cases.jsonl"
    raw_path = root / "benchmark-raw.jsonl"
    write_jsonl(cases_path, cases)
    write_jsonl(raw_path, raw)
    attestation = {
        "artifact_sha256": artifact_sha,
        "dataset_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "raw_output_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "runner_code_revision": "1" * 40,
        "sample_count": len(raw),
        "schema_version": 1,
        "seeds": [4242],
    }
    (root / "benchmark-attestation.json").write_text(
        json.dumps(attestation, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (root / "benchmark-attestation.sig").write_bytes(b"s" * 64)
    return cases_path, raw_path, artifact_sha


def test_accepts_native_runner_rows_and_aggregates_each_dimension(tmp_path: Path) -> None:
    cases, raw, artifact_sha = runner_native_evidence(tmp_path)

    evidence = build_benchmark_evidence(cases, raw)

    assert evidence.artifact_sha256 == artifact_sha
    assert evidence.sample_count == 48
    assert evidence.scores == {
        "coding": 1.0,
        "tool_calling": 1.0,
        "agentic_search": 1.0,
        "text_quality": 1.0,
        "russian": 1.0,
        "english": 1.0,
    }
    assert evidence.overrefusal_rate == 0.0
    assert evidence.seeds == (4242,)


def test_rejects_raw_rows_not_bound_to_one_artifact(tmp_path: Path) -> None:
    cases, raw, _artifact_sha = runner_native_evidence(tmp_path)
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    rows[-1]["artifact_sha256"] = "f" * 64
    write_jsonl(raw, rows)

    with pytest.raises(BenchmarkEvidenceError, match="one artifact"):
        build_benchmark_evidence(cases, raw)


def test_rejects_dimension_score_mismatch_in_runner_row(tmp_path: Path) -> None:
    cases, raw, _artifact_sha = runner_native_evidence(tmp_path)
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    rows[0]["scores"] = {"english": 0.5}
    write_jsonl(raw, rows)

    with pytest.raises(BenchmarkEvidenceError, match="dimension score"):
        build_benchmark_evidence(cases, raw)


@pytest.mark.parametrize(
    ("field", "tampered"),
    (("score", 0.0), ("scores", {"coding": 0.0}), ("refused", True)),
)
def test_recomputes_and_rejects_tampered_derived_fields(
    tmp_path: Path, field: str, tampered: object
) -> None:
    cases, raw, _artifact_sha = runner_native_evidence(tmp_path)
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = tampered
    write_jsonl(raw, rows)

    with pytest.raises(BenchmarkEvidenceError, match="recomputed"):
        build_benchmark_evidence(cases, raw)


def test_production_verifier_is_pinned_and_rejects_ephemeral_test_keys() -> None:
    payload = b'{"schema_version":1}\n'
    signature = Ed25519PrivateKey.generate().sign(payload)

    assert PRODUCTION_ATTESTATION_PUBLIC_KEY == "eqUEQBjrmtGSwGRtxYBiui3L7s0MzV_mx28PFLjTUA8="
    assert _verify_pinned_attestation_signature(payload, signature) is False
