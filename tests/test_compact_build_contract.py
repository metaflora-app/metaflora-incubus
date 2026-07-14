from dataclasses import FrozenInstanceError, replace

import pytest

from metaflora_incubus.benchmark_harness import BenchmarkProvenance, HarnessReport
from metaflora_incubus.training_contract import (
    ArtifactSizeError,
    BuildArtifact,
    BuildReleaseEvidence,
    ComputeProfile,
    ParityJob,
    QuantizationFormat,
    QuantizationMatrix,
    ReleaseEvidenceError,
    ResourceEstimateInput,
    estimate_resources,
)

GIB = 1024**3
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def provenance() -> BenchmarkProvenance:
    return BenchmarkProvenance.create(
        artifact_sha256=SHA_A,
        dataset_sha256=SHA_B,
        harness_revision="0123456789abcdef0123456789abcdef01234567",
        prompt_template_sha256=SHA_C,
        runtime_name="local-runtime",
        runtime_version="v1.2.3",
        seeds=(17, 29, 43),
        sample_count=100,
        raw_output_sha256=SHA_D,
        signer_id="release-key-v1",
        signature="signed-benchmark-report",
    )


def test_quantization_matrix_requires_bf16_q8_q6_q5_q4_and_q5_is_deployable() -> None:
    matrix = QuantizationMatrix.default()

    assert matrix.formats == (
        QuantizationFormat.BF16,
        QuantizationFormat.Q8,
        QuantizationFormat.Q6,
        QuantizationFormat.Q5,
        QuantizationFormat.Q4,
    )
    assert matrix.deployable_format is QuantizationFormat.Q5
    assert matrix.parity_jobs == tuple(
        ParityJob(
            baseline_format=QuantizationFormat.BF16,
            candidate_format=candidate,
            required_metrics=(
                "coding",
                "agentic_search",
                "text_quality",
                "russian",
                "english",
                "overrefusal_rate",
            ),
        )
        for candidate in (
            QuantizationFormat.Q8,
            QuantizationFormat.Q6,
            QuantizationFormat.Q5,
            QuantizationFormat.Q4,
        )
    )


def test_quantization_contract_is_immutable_and_cannot_skip_q5_parity() -> None:
    matrix = QuantizationMatrix.default()

    with pytest.raises(FrozenInstanceError):
        matrix.deployable_format = QuantizationFormat.Q4  # type: ignore[misc]
    with pytest.raises(ValueError, match="Q5"):
        QuantizationMatrix.create(
            formats=matrix.formats,
            deployable_format=QuantizationFormat.Q5,
            parity_jobs=tuple(
                job
                for job in matrix.parity_jobs
                if job.candidate_format is not QuantizationFormat.Q5
            ),
        )
    with pytest.raises(ValueError, match="deployable"):
        QuantizationMatrix.create(
            formats=matrix.formats,
            deployable_format=QuantizationFormat.Q4,
            parity_jobs=matrix.parity_jobs,
        )


@pytest.mark.parametrize("size_bytes", (3 * GIB, 5 * GIB))
def test_deployable_q5_artifact_accepts_compact_three_to_five_gib_boundaries(
    size_bytes: int,
) -> None:
    artifact = BuildArtifact.create(
        artifact_id="incubus-v1-q5",
        format=QuantizationFormat.Q5,
        path="dist/incubus-v1-q5.gguf",
        size_bytes=size_bytes,
        sha256=SHA_A,
    )

    assert artifact.size_bytes == size_bytes


@pytest.mark.parametrize("size_bytes", (3 * GIB - 1, 5 * GIB + 1))
def test_deployable_q5_artifact_rejects_size_outside_three_to_five_gib(
    size_bytes: int,
) -> None:
    with pytest.raises(ArtifactSizeError):
        BuildArtifact.create(
            artifact_id="incubus-v1-q5",
            format=QuantizationFormat.Q5,
            path="dist/incubus-v1-q5.gguf",
            size_bytes=size_bytes,
            sha256=SHA_A,
        )


def test_resource_estimator_returns_reproducible_train_build_and_runtime_profiles() -> None:
    request = ResourceEstimateInput(
        parameter_count=9_000_000_000,
        trainable_parameter_count=180_000_000,
        sequence_length=32768,
        effective_batch_size=128,
        dataset_tokens=2_000_000_000,
        checkpoint_count=8,
        target_format=QuantizationFormat.Q5,
    )

    first = estimate_resources(request)
    second = estimate_resources(request)

    assert first == second
    assert first.training.gpu_vram_bytes > 0
    assert first.training.disk_bytes > first.build.disk_bytes
    assert first.build.disk_bytes >= 6 * GIB
    assert first.runtime.disk_bytes == 5 * GIB
    assert first.runtime.ram_bytes >= first.runtime.disk_bytes
    assert first.recommended_profile in (ComputeProfile.SINGLE_GPU, ComputeProfile.MULTI_GPU)


@pytest.mark.parametrize(
    "field",
    (
        "parameter_count",
        "trainable_parameter_count",
        "sequence_length",
        "effective_batch_size",
        "dataset_tokens",
        "checkpoint_count",
    ),
)
def test_resource_estimator_rejects_zero_or_negative_inputs(field: str) -> None:
    request = ResourceEstimateInput(
        parameter_count=9_000_000_000,
        trainable_parameter_count=180_000_000,
        sequence_length=32768,
        effective_batch_size=128,
        dataset_tokens=2_000_000_000,
        checkpoint_count=8,
        target_format=QuantizationFormat.Q5,
    )

    with pytest.raises(ValueError, match=field):
        estimate_resources(replace(request, **{field: 0}))


def test_release_evidence_requires_a_real_approved_harness_report_and_provenance() -> None:
    report = HarnessReport(
        approved=True,
        metrics={
            "coding": 0.84,
            "agentic_search": 0.81,
            "text_quality": 0.86,
            "russian": 0.88,
            "english": 0.87,
            "overrefusal_rate": 0.04,
        },
        failures=(),
    )

    evidence = BuildReleaseEvidence.create(
        artifact_sha256=SHA_A,
        harness_report=report,
        provenance=provenance(),
        raw_outputs_path="benchmarks/raw-output.jsonl",
    )

    assert evidence.harness_report is report
    assert evidence.provenance.artifact_sha256 == evidence.artifact_sha256


def test_release_evidence_rejects_dicts_failed_reports_and_artifact_mismatch() -> None:
    approved = HarnessReport(approved=True, metrics={"coding": 0.9}, failures=())
    failed = HarnessReport(approved=False, metrics={"coding": 0.2}, failures=())

    with pytest.raises(ReleaseEvidenceError, match="HarnessReport"):
        BuildReleaseEvidence.create(
            artifact_sha256=SHA_A,
            harness_report={"approved": True, "metrics": {"coding": 1.0}},
            provenance=provenance(),
            raw_outputs_path="benchmarks/raw-output.jsonl",
        )
    with pytest.raises(ReleaseEvidenceError, match="approved"):
        BuildReleaseEvidence.create(
            artifact_sha256=SHA_A,
            harness_report=failed,
            provenance=provenance(),
            raw_outputs_path="benchmarks/raw-output.jsonl",
        )
    with pytest.raises(ReleaseEvidenceError, match="artifact_sha256"):
        BuildReleaseEvidence.create(
            artifact_sha256="f" * 64,
            harness_report=approved,
            provenance=provenance(),
            raw_outputs_path="benchmarks/raw-output.jsonl",
        )
