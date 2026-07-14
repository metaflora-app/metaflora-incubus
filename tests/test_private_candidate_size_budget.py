from __future__ import annotations

from pathlib import Path

import pytest

import metaflora_incubus.private_candidate as candidate_module
from metaflora_incubus.private_candidate import (
    CandidateConstraintError,
    CandidateSelectionPolicy,
    IncumbentArtifactPin,
    PrivateCandidatePin,
    bind_candidate_reports,
    select_private_candidate,
    verify_private_candidate,
)
from metaflora_incubus.release_gates import BenchmarkReport, ReleaseGatePolicy

GIB = 1024**3
CANDIDATE_SHA = "a" * 64
INCUMBENT_SHA = "b" * 64
SOURCE_SHA = "c" * 64
REFERENCE_SHA = "d" * 64
LICENSE_SHA = "e" * 64


def report(artifact_id: str, score: float, overrefusal_rate: float) -> BenchmarkReport:
    return BenchmarkReport(
        artifact_id=artifact_id,
        suite_id="release-suite-v1",
        scores={
            "coding": score,
            "tool_calling": score,
            "agentic_search": score,
            "text_quality": score,
            "russian": score,
            "english": score,
        },
        overrefusal_rate=overrefusal_rate,
    )


def policy() -> CandidateSelectionPolicy:
    return CandidateSelectionPolicy(
        release_policy=ReleaseGatePolicy(
            required_baselines=("reference",),
            required_score_targets={
                "coding": 0.75,
                "tool_calling": 0.75,
                "agentic_search": 0.75,
                "text_quality": 0.75,
                "russian": 0.75,
                "english": 0.75,
            },
            minimum_lead_over_each_baseline=0.01,
            minimum_overrefusal_reduction=0.05,
            maximum_quantization_drop=0.02,
            require_asr=False,
            maximum_asr_wer=0.12,
            minimum_asr_lead_over_each_baseline=0.01,
        ),
        minimum_candidate_size_bytes=3 * GIB,
        maximum_candidate_size_bytes=5 * GIB,
        minimum_overrefusal_improvement=0.05,
        allowed_license_ids=("Apache-2.0",),
    )


def verified_candidate(monkeypatch: pytest.MonkeyPatch, size_bytes: int):
    pin = PrivateCandidatePin.create(
        private_repo_id="private-owner/candidate",
        revision="1" * 40,
        artifact_filename="candidate.gguf",
        artifact_sha256=CANDIDATE_SHA,
        artifact_size_bytes=size_bytes,
        license_id="Apache-2.0",
        license_url="https://licenses.example/license",
        license_filename="LICENSE",
        license_sha256=LICENSE_SHA,
    )

    def evidence(path: Path, *, magic_size: int, label: str):
        del path, magic_size
        if label == "artifact":
            return size_bytes, CANDIDATE_SHA, b"GGUF"
        return 1, LICENSE_SHA, b""

    monkeypatch.setattr(candidate_module, "_read_file_evidence", evidence)
    return verify_private_candidate(
        artifact_path=Path("candidate.gguf"),
        license_path=Path("LICENSE"),
        pin=pin,
        allowed_license_ids=("Apache-2.0",),
    )


def test_quality_first_candidate_can_be_larger_than_incumbent_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = verified_candidate(monkeypatch, 4 * GIB)
    reports = bind_candidate_reports(
        candidate=candidate,
        full_precision_report=report(SOURCE_SHA, 0.85, 0.05),
        deployable_report=report(CANDIDATE_SHA, 0.84, 0.05),
    )

    decision = select_private_candidate(
        candidate=candidate,
        candidate_reports=reports,
        incumbent=IncumbentArtifactPin(INCUMBENT_SHA, 3 * GIB + GIB // 2),
        incumbent_report=report(INCUMBENT_SHA, 0.80, 0.15),
        baselines={"reference": report(REFERENCE_SHA, 0.70, 0.25)},
        policy=policy(),
    )

    assert decision.selected is True
    assert decision.failures == ()


@pytest.mark.parametrize("size_bytes", (3 * GIB - 1, 5 * GIB + 1))
def test_candidate_outside_deployable_size_budget_is_rejected(
    monkeypatch: pytest.MonkeyPatch, size_bytes: int
) -> None:
    candidate = verified_candidate(monkeypatch, size_bytes)
    reports = bind_candidate_reports(
        candidate=candidate,
        full_precision_report=report(SOURCE_SHA, 0.85, 0.05),
        deployable_report=report(CANDIDATE_SHA, 0.84, 0.05),
    )

    decision = select_private_candidate(
        candidate=candidate,
        candidate_reports=reports,
        incumbent=IncumbentArtifactPin(INCUMBENT_SHA, 4 * GIB),
        incumbent_report=report(INCUMBENT_SHA, 0.80, 0.15),
        baselines={"reference": report(REFERENCE_SHA, 0.70, 0.25)},
        policy=policy(),
    )

    assert decision.selected is False
    assert "candidate_size_out_of_bounds" in {failure.code for failure in decision.failures}


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    ((0, 5 * GIB), (3 * GIB, 3 * GIB - 1), (True, 5 * GIB)),
)
def test_candidate_size_budget_must_be_valid(minimum: int, maximum: int) -> None:
    with pytest.raises(CandidateConstraintError, match="candidate size bounds"):
        CandidateSelectionPolicy(
            release_policy=policy().release_policy,
            minimum_candidate_size_bytes=minimum,
            maximum_candidate_size_bytes=maximum,
            minimum_overrefusal_improvement=0.05,
            allowed_license_ids=("Apache-2.0",),
        )


def test_policy_requires_strict_overrefusal_advantage() -> None:
    with pytest.raises(CandidateConstraintError, match="overrefusal"):
        CandidateSelectionPolicy(
            release_policy=policy().release_policy,
            minimum_candidate_size_bytes=3 * GIB,
            maximum_candidate_size_bytes=5 * GIB,
            minimum_overrefusal_improvement=0.0,
            allowed_license_ids=("Apache-2.0",),
        )


def test_release_policy_pins_three_to_five_gib_without_caller_size_inputs() -> None:
    release = CandidateSelectionPolicy.release(
        release_policy=policy().release_policy,
        minimum_overrefusal_improvement=0.05,
        allowed_license_ids=("Apache-2.0",),
    )

    assert release.minimum_candidate_size_bytes == 3 * GIB
    assert release.maximum_candidate_size_bytes == 5 * GIB
