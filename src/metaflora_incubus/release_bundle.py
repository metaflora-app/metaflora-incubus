"""Build a complete, measured, fail-closed Hugging Face release bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from metaflora_incubus.benchmark_evidence import BenchmarkEvidence, build_benchmark_evidence
from metaflora_incubus.huggingface_publication import (
    MODEL_NAME,
    PublicationPolicy,
    SignatureVerifier,
    evaluate_publication_bundle,
    pinned_v1_release_policy,
)
from metaflora_incubus.release_gates import (
    BenchmarkReport,
    ReleaseDecision,
    evaluate_release,
)

BundleSigner = Callable[[str, bytes], bytes]
PINNED_V1_DATASET_SHA256 = "9f18aba6bed35a1165cb5015ab10302f6a219c10ea1e564838a77cc3bcd75d49"


class ReleaseBundleError(RuntimeError):
    """Raised when measured inputs are incomplete or fail promotion."""


@dataclass(frozen=True)
class BoundBenchmarkInput:
    artifact_path: Path
    evidence_dir: Path
    artifact_id: str
    suite_id: str


@dataclass(frozen=True)
class ReleaseBundleInputs:
    model_path: Path
    evidence_dir: Path
    output_dir: Path
    full_precision_report: BoundBenchmarkInput
    baselines: Mapping[str, BoundBenchmarkInput]
    runtime_id: str
    harness_revision: str
    license_path: Path
    notices_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "baselines", MappingProxyType(dict(self.baselines)))


@dataclass(frozen=True)
class ReleaseBundleResult:
    output_dir: Path
    artifact_sha256: str
    release_decision: ReleaseDecision


def build_release_bundle(
    inputs: ReleaseBundleInputs,
    *,
    signer: BundleSigner,
    signature_verifier: SignatureVerifier,
    publication_policy: PublicationPolicy,
) -> ReleaseBundleResult:
    """Render a release directory only after artifact-bound benchmarks pass."""
    _validate_inputs(inputs)
    evidence = build_benchmark_evidence(
        inputs.evidence_dir / "benchmark-cases.jsonl",
        inputs.evidence_dir / "benchmark-raw.jsonl",
    )
    artifact_sha = _sha256_file(inputs.model_path)
    if evidence.artifact_sha256 != artifact_sha:
        raise ReleaseBundleError("benchmark evidence is not bound to the release model")
    if evidence.dataset_sha256 != PINNED_V1_DATASET_SHA256:
        raise ReleaseBundleError("deployable benchmark does not use the pinned v1 case bank")
    deployable = BenchmarkReport(
        artifact_id=f"incubus-v1-q5-{artifact_sha[:12]}",
        suite_id="incubus-release-v1",
        scores=evidence.scores,
        overrefusal_rate=evidence.overrefusal_rate,
    )
    full_precision, full_binding = _materialize_bound_report(inputs.full_precision_report)
    baselines_with_bindings = {
        name: _materialize_bound_report(value) for name, value in inputs.baselines.items()
    }
    baselines = {name: value[0] for name, value in baselines_with_bindings.items()}
    policy = pinned_v1_release_policy()
    decision = evaluate_release(
        full_precision,
        deployable,
        baselines,
        policy,
    )
    if not decision.approved:
        codes = ",".join(failure.code for failure in decision.failures)
        raise ReleaseBundleError(f"release gate failed: {codes}")
    if inputs.output_dir.exists():
        raise ReleaseBundleError("release output directory already exists")

    inputs.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".incubus-release-", dir=inputs.output_dir.parent))
    try:
        _render_bundle(
            stage,
            inputs=inputs,
            full_precision=(full_precision, full_binding),
            baselines=baselines_with_bindings,
            deployable=deployable,
            evidence=evidence,
            artifact_sha=artifact_sha,
            signer=signer,
            publication_policy=publication_policy,
        )
        publication_decision = evaluate_publication_bundle(
            stage,
            policy=publication_policy,
            signature_verifier=signature_verifier,
        )
        if not publication_decision.approved:
            codes = ",".join(blocker.code for blocker in publication_decision.blockers)
            raise ReleaseBundleError(f"rendered publication bundle is invalid: {codes}")
        stage.replace(inputs.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return ReleaseBundleResult(inputs.output_dir, artifact_sha, decision)


def _render_bundle(
    stage: Path,
    *,
    inputs: ReleaseBundleInputs,
    full_precision: tuple[BenchmarkReport, BenchmarkEvidence],
    baselines: Mapping[str, tuple[BenchmarkReport, BenchmarkEvidence]],
    deployable: BenchmarkReport,
    evidence: BenchmarkEvidence,
    artifact_sha: str,
    signer: BundleSigner,
    publication_policy: PublicationPolicy,
) -> None:
    model_target = stage / MODEL_NAME
    try:
        os.link(inputs.model_path, model_target)
    except OSError:
        shutil.copy2(inputs.model_path, model_target)
    shutil.copy2(inputs.license_path, stage / "LICENSE")
    shutil.copy2(inputs.notices_path, stage / "THIRD_PARTY_NOTICES")
    shutil.copy2(inputs.evidence_dir / "benchmark-cases.jsonl", stage / "benchmark-cases.jsonl")
    shutil.copy2(inputs.evidence_dir / "benchmark-raw.jsonl", stage / "benchmark-raw.jsonl")
    shutil.copy2(
        inputs.evidence_dir / "benchmark-attestation.json",
        stage / "benchmark-attestation.json",
    )
    shutil.copy2(
        inputs.evidence_dir / "benchmark-attestation.sig",
        stage / "benchmark-attestation.sig",
    )

    release_manifest = {
        "schema_version": 1,
        "release_id": "incubus-v1",
        "artifacts": [
            {
                "path": MODEL_NAME,
                "gguf_quantization": "Q5_K_M",
                "sha256": artifact_sha,
                "size_bytes": model_target.stat().st_size,
            }
        ],
    }
    report = {
        "schema_version": 1,
        "artifact_sha256": artifact_sha,
        "suite_id": "incubus-release-v1",
        "status": "passed",
        "gate_input": {
            "candidate": _report_document(*full_precision),
            "deployable_candidate": _report_document(deployable, evidence),
            "baselines": {
                name: _report_document(*value) for name, value in sorted(baselines.items())
            },
            "policy": _policy_document(),
        },
    }
    _write_signed_json(stage, "release-manifest", release_manifest, "release_manifest", signer)
    report_payload = _write_json(stage / "benchmark-report.json", report)
    provenance = {
        "schema_version": 1,
        "artifact_sha256": artifact_sha,
        "report_sha256": hashlib.sha256(report_payload).hexdigest(),
        "harness_revision": inputs.harness_revision,
        "dataset_sha256": evidence.dataset_sha256,
        "raw_output_sha256": evidence.raw_output_sha256,
        "sample_count": evidence.sample_count,
        "runtime": inputs.runtime_id,
        "seeds": list(evidence.seeds),
        "attestation_sha256": evidence.attestation_sha256,
    }
    provenance_payload = _write_signed_json(
        stage, "benchmark-provenance", provenance, "benchmark_provenance", signer
    )
    benchmark_decision = {
        "schema_version": 1,
        "approved": True,
        "artifact_sha256": artifact_sha,
        "report_sha256": hashlib.sha256(report_payload).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_payload).hexdigest(),
    }
    _write_signed_json(
        stage, "benchmark-decision", benchmark_decision, "benchmark_decision", signer
    )
    smoke_case_id, smoke_request, smoke_response = _attested_smoke(inputs.evidence_dir)
    smoke = {
        "schema_version": 1,
        "artifact_sha256": artifact_sha,
        "runtime": inputs.runtime_id,
        "status": "passed",
        "case_id": smoke_case_id,
        "request": smoke_request,
        "response": smoke_response,
    }
    _write_signed_json(stage, "smoke-test", smoke, "smoke_test", signer)
    (stage / "SHA256SUMS").write_text(f"{artifact_sha}  {MODEL_NAME}\n", encoding="utf-8")
    (stage / "Modelfile").write_text(
        f"FROM ./{MODEL_NAME}\nPARAMETER num_ctx 8192\n", encoding="utf-8"
    )
    (stage / "README.md").write_text(
        _model_card(deployable, artifact_sha, publication_policy.repo_id), encoding="utf-8"
    )


def _validate_inputs(inputs: ReleaseBundleInputs) -> None:
    required_files = (
        inputs.model_path,
        inputs.evidence_dir / "benchmark-cases.jsonl",
        inputs.evidence_dir / "benchmark-raw.jsonl",
        inputs.license_path,
        inputs.notices_path,
    )
    if any(not path.is_file() or path.is_symlink() for path in required_files):
        raise ReleaseBundleError("release input file is missing or unsafe")
    with inputs.model_path.open("rb") as model_stream:
        model_magic = model_stream.read(4)
    if model_magic != b"GGUF":
        raise ReleaseBundleError("release model is not GGUF")
    if not inputs.runtime_id.strip() or not inputs.harness_revision.strip():
        raise ReleaseBundleError("runtime and harness revision are required")


def _materialize_bound_report(
    source: BoundBenchmarkInput,
) -> tuple[BenchmarkReport, BenchmarkEvidence]:
    if not source.artifact_path.is_file() or source.artifact_path.is_symlink():
        raise ReleaseBundleError("benchmark artifact is missing or unsafe")
    evidence = build_benchmark_evidence(
        source.evidence_dir / "benchmark-cases.jsonl",
        source.evidence_dir / "benchmark-raw.jsonl",
    )
    if evidence.artifact_sha256 != _sha256_file(source.artifact_path):
        raise ReleaseBundleError("benchmark report is not bound to its artifact")
    if evidence.dataset_sha256 != PINNED_V1_DATASET_SHA256:
        raise ReleaseBundleError("benchmark report does not use the pinned v1 case bank")
    return (
        BenchmarkReport(
            artifact_id=source.artifact_id,
            suite_id=source.suite_id,
            scores=evidence.scores,
            overrefusal_rate=evidence.overrefusal_rate,
        ),
        evidence,
    )


def _report_document(
    report: BenchmarkReport, evidence: BenchmarkEvidence | None = None
) -> dict[str, object]:
    document: dict[str, object] = {
        "artifact_id": report.artifact_id,
        "scores": dict(report.scores),
        "overrefusal_rate": report.overrefusal_rate,
    }
    if report.asr_wer is not None:
        document["asr_wer"] = report.asr_wer
    if evidence is not None:
        document["evidence_binding"] = {
            "artifact_sha256": evidence.artifact_sha256,
            "dataset_sha256": evidence.dataset_sha256,
            "raw_output_sha256": evidence.raw_output_sha256,
            "sample_count": evidence.sample_count,
            "seeds": list(evidence.seeds),
            "runner_code_revision": evidence.runner_code_revision,
            "attestation_sha256": evidence.attestation_sha256,
        }
    return document


def _attested_smoke(evidence_dir: Path) -> tuple[str, str, str]:
    cases = {
        row["case_id"]: row
        for row in _read_jsonl(evidence_dir / "benchmark-cases.jsonl")
    }
    for row in _read_jsonl(evidence_dir / "benchmark-raw.jsonl"):
        response = row.get("response")
        case_id = row.get("case_id")
        if isinstance(case_id, str) and isinstance(response, str) and response.strip():
            prompt = cases.get(case_id, {}).get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                return case_id, prompt, response
    raise ReleaseBundleError("attested benchmark has no usable smoke row")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ReleaseBundleError("benchmark JSONL row is invalid")
    return rows


def _policy_document() -> dict[str, object]:
    policy = pinned_v1_release_policy()
    return {
        "required_baselines": list(policy.required_baselines),
        "required_score_targets": dict(policy.required_score_targets),
        "minimum_lead_over_each_baseline": policy.minimum_lead_over_each_baseline,
        "minimum_overrefusal_reduction": policy.minimum_overrefusal_reduction,
        "maximum_quantization_drop": policy.maximum_quantization_drop,
        "require_asr": policy.require_asr,
        "maximum_asr_wer": policy.maximum_asr_wer,
        "minimum_asr_lead_over_each_baseline": policy.minimum_asr_lead_over_each_baseline,
    }


def _write_signed_json(
    root: Path,
    stem: str,
    document: object,
    purpose: str,
    signer: BundleSigner,
) -> bytes:
    payload = _write_json(root / f"{stem}.json", document)
    signature = signer(purpose, payload)
    if not isinstance(signature, bytes) or not signature:
        raise ReleaseBundleError(f"signer returned no signature for {purpose}")
    (root / f"{stem}.sig").write_bytes(signature)
    return payload


def _write_json(path: Path, document: object) -> bytes:
    payload = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return payload


def _model_card(report: BenchmarkReport, artifact_sha: str, repo_id: str) -> str:
    rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {score:.6f} |"
        for name, score in report.scores.items()
    )
    return f"""---
language:
  - ru
  - en
pipeline_tag: text-generation
tags:
  - gguf
  - local
  - tool-use
---

# Metaflora Incubus v1

Compact local text model for code, structured tool calls, search workflows,
and Russian and English text. The values below are measured on the published
Q5_K_M artifact by the bundled deterministic release suite.

| Area | Score |
| --- | ---: |
{rows}
| Benign refusal rate | {report.overrefusal_rate:.6f} |

Artifact SHA-256: `{artifact_sha}`  
Repository: `{repo_id}`

## Ollama

```sh
ollama create metaflora-incubus:v1 -f Modelfile
ollama run metaflora-incubus:v1
```

## OpenAI-compatible runtime

Run the bundled GGUF with a recent `llama-server`, bound to `127.0.0.1`.
Use model ID `metaflora-incubus-v1`. Benchmark cases, raw outputs, provenance,
checksums, and the signed promotion decision are included in this repository.

Model outputs can be incorrect. Review generated code and tool actions before
execution.
"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
