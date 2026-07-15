from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from metaflora_incubus.huggingface_publication import (
    MODEL_NAME,
    PublicationPolicy,
    evaluate_publication_bundle,
)
from metaflora_incubus.release_bundle import (
    BoundBenchmarkInput,
    ReleaseBundleError,
    ReleaseBundleInputs,
    build_release_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "benchmarks" / "gguf-v1-cases.jsonl"

DIMENSIONS = (
    "coding",
    "tool_calling",
    "agentic_search",
    "text_quality",
    "russian",
    "english",
)


@pytest.fixture(autouse=True)
def accept_ephemeral_attestations_only_in_bundle_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "metaflora_incubus.benchmark_evidence._verify_pinned_attestation_signature",
        lambda payload, signature: signature == b"s" * 64 or signature.startswith(b"valid:"),
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def bound_evidence(
    root: Path,
    artifact: Path,
    artifact_id: str,
    *,
    passing: bool = True,
    refusal_every_dimension: bool = False,
) -> BoundBenchmarkInput:
    root.mkdir()
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines()]
    raw = []
    dimension_counts: dict[str, int] = {}
    for case in cases:
        dimension = case["dimension"]
        index = dimension_counts.get(dimension, 0)
        dimension_counts[dimension] = index + 1
        case_id = case["case_id"]
        content = ""
        score = 0.0
        refused = refusal_every_dimension and index == 0
        if refused:
            content = "I cannot help"
        raw_response: dict[str, object] = {
            "choices": [{"finish_reason": "stop", "message": {"content": content}}]
        }
        if passing and dimension in {"tool_calling", "agentic_search"}:
            score = 1.0
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
                                        "name": case["expected_tool_name"],
                                        "arguments": json.dumps(
                                            case["expected_tool_arguments"], sort_keys=True
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        elif passing:
            score = 1.0
            content = " ".join(case["required_terms"]) or "measured response"
            raw_response = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        raw.append(
            {
                "artifact_sha256": artifact_sha,
                "case_id": case_id,
                "dimension": dimension,
                "raw_response": raw_response,
                "response": content,
                "score": score,
                "scores": {dimension: score},
                "refused": refused,
                "seed": 4242,
            }
        )
    (root / "benchmark-cases.jsonl").write_bytes(CASES_PATH.read_bytes())
    write_jsonl(root / "benchmark-raw.jsonl", raw)
    attestation = {
        "artifact_sha256": artifact_sha,
        "dataset_sha256": hashlib.sha256((root / "benchmark-cases.jsonl").read_bytes()).hexdigest(),
        "raw_output_sha256": hashlib.sha256(
            (root / "benchmark-raw.jsonl").read_bytes()
        ).hexdigest(),
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
    return BoundBenchmarkInput(
        artifact_path=artifact,
        evidence_dir=root,
        artifact_id=artifact_id,
        suite_id="incubus-release-v1",
    )


def release_inputs(tmp_path: Path, *, deployable_passing: bool = True) -> ReleaseBundleInputs:
    model = tmp_path / "candidate.gguf"
    model.write_bytes(b"GGUF" + b"x" * 12)
    full = tmp_path / "candidate-full.bin"
    full.write_bytes(b"full-precision-artifact")
    reference = tmp_path / "reference.bin"
    reference.write_bytes(b"reference-artifact")
    competitor_a = tmp_path / "competitor-a.bin"
    competitor_a.write_bytes(b"competitor-a-artifact")
    competitor_b = tmp_path / "competitor-b.bin"
    competitor_b.write_bytes(b"competitor-b-artifact")
    deployable = bound_evidence(
        tmp_path / "deployable-evidence", model, "candidate-q5", passing=deployable_passing
    )
    license_path = tmp_path / "approved-license.txt"
    notices_path = tmp_path / "approved-notices.txt"
    license_path.write_text("Approved distribution license\n", encoding="utf-8")
    notices_path.write_text("Approved third-party notices\n", encoding="utf-8")
    return ReleaseBundleInputs(
        model_path=model,
        evidence_dir=deployable.evidence_dir,
        output_dir=tmp_path / "bundle",
        full_precision_report=bound_evidence(tmp_path / "full-evidence", full, "candidate-full"),
        baselines={
            "reference": bound_evidence(
                tmp_path / "reference-evidence",
                reference,
                "reference",
                passing=False,
                refusal_every_dimension=True,
            ),
            "competitor_a": bound_evidence(
                tmp_path / "competitor-a-evidence",
                competitor_a,
                "competitor-a",
                passing=False,
            ),
            "competitor_b": bound_evidence(
                tmp_path / "competitor-b-evidence",
                competitor_b,
                "competitor-b",
                passing=False,
            ),
        },
        runtime_id="llama.cpp-b7001",
        harness_revision="1" * 40,
        license_path=license_path,
        notices_path=notices_path,
    )


def signer(purpose: str, payload: bytes) -> bytes:
    return f"valid:{purpose}:".encode() + hashlib.sha256(payload).digest()


def verifier(purpose: str, payload: bytes, signature: bytes) -> bool:
    return signature == signer(purpose, payload)


def test_builds_a_complete_release_gated_hugging_face_bundle(tmp_path: Path) -> None:
    inputs = release_inputs(tmp_path)
    policy = PublicationPolicy(
        repo_id="metaflora/incubus",
        min_model_bytes=8,
        max_model_bytes=32,
        prohibited_identifiers=("private-source-name", "private-teacher-name"),
    )

    result = build_release_bundle(
        inputs,
        signer=signer,
        signature_verifier=verifier,
        publication_policy=policy,
    )

    assert result.output_dir == inputs.output_dir
    assert (inputs.output_dir / MODEL_NAME).read_bytes().startswith(b"GGUF")
    assert result.release_decision.approved is True
    publication = evaluate_publication_bundle(
        inputs.output_dir,
        policy=policy,
        signature_verifier=verifier,
    )
    assert publication.approved is True
    card = (inputs.output_dir / "README.md").read_text(encoding="utf-8")
    assert "1.000000" in card
    assert "private-source-name" not in card
    manifest = json.loads((inputs.output_dir / "release-manifest.json").read_text())
    artifacts = {artifact["path"]: artifact for artifact in manifest["artifacts"]}
    assert set(artifacts) == {MODEL_NAME, "LICENSE", "THIRD_PARTY_NOTICES"}
    for name in artifacts:
        payload = (inputs.output_dir / name).read_bytes()
        assert artifacts[name]["sha256"] == hashlib.sha256(payload).hexdigest()
        assert artifacts[name]["size_bytes"] == len(payload)
    checksums = (inputs.output_dir / "SHA256SUMS").read_text().splitlines()
    assert {line.split("  ", 1)[1] for line in checksums} == set(artifacts)


def test_rejects_runner_revision_not_approved_by_release_inputs(tmp_path: Path) -> None:
    inputs = replace(release_inputs(tmp_path), harness_revision="2" * 40)
    policy = PublicationPolicy(
        repo_id="metaflora/incubus",
        min_model_bytes=8,
        max_model_bytes=32,
        prohibited_identifiers=("private-source-name",),
    )

    with pytest.raises(ReleaseBundleError, match="runner revision"):
        build_release_bundle(
            inputs,
            signer=signer,
            signature_verifier=verifier,
            publication_policy=policy,
        )


def test_fails_closed_without_creating_bundle_when_benchmarks_miss_gate(tmp_path: Path) -> None:
    inputs = release_inputs(tmp_path, deployable_passing=False)
    policy = PublicationPolicy(
        repo_id="metaflora/incubus",
        min_model_bytes=8,
        max_model_bytes=32,
        prohibited_identifiers=("private-source-name",),
    )

    with pytest.raises(ReleaseBundleError, match="release gate"):
        build_release_bundle(
            inputs,
            signer=signer,
            signature_verifier=verifier,
            publication_policy=policy,
        )

    assert not inputs.output_dir.exists()


def test_fails_closed_when_full_precision_artifact_changes_after_benchmark(tmp_path: Path) -> None:
    inputs = release_inputs(tmp_path)
    inputs.full_precision_report.artifact_path.write_bytes(b"tampered-full-precision-artifact")
    policy = PublicationPolicy(
        repo_id="metaflora/incubus",
        min_model_bytes=8,
        max_model_bytes=32,
        prohibited_identifiers=("private-source-name",),
    )

    with pytest.raises(ReleaseBundleError, match="not bound to its artifact"):
        build_release_bundle(
            inputs,
            signer=signer,
            signature_verifier=verifier,
            publication_policy=policy,
        )

    assert not inputs.output_dir.exists()
