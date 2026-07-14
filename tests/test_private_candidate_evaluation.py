from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from metaflora_incubus.private_candidate import (
    CandidateConstraintError,
    CandidateSelectionPolicy,
    IncumbentArtifactPin,
    PrivateCandidatePin,
    VerifiedCandidateReports,
    VerifiedPrivateCandidate,
    bind_candidate_reports,
    select_private_candidate,
    verify_private_candidate,
)
from metaflora_incubus.release_gates import (
    BenchmarkReport,
    ReleaseGatePolicy,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(artifact: Path, license_path: Path, **overrides: object) -> PrivateCandidatePin:
    values: dict[str, object] = {
        "private_repo_id": "private-owner/private-candidate",
        "revision": "1" * 40,
        "artifact_filename": "candidate.gguf",
        "artifact_sha256": sha256(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "license_id": "Apache-2.0",
        "license_url": "https://licenses.example/apache-2.0",
        "license_filename": "LICENSE.txt",
        "license_sha256": sha256(license_path),
    }
    values.update(overrides)
    return PrivateCandidatePin.create(**values)


def report(artifact_id: str, score: float, overrefusal: float) -> BenchmarkReport:
    return BenchmarkReport(
        artifact_id=artifact_id,
        suite_id="incubus-release-v1",
        scores={
            "coding": score,
            "tool_calling": score,
            "agentic_search": score,
            "text_quality": score,
            "russian": score,
            "english": score,
        },
        overrefusal_rate=overrefusal,
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
            minimum_overrefusal_reduction=0.10,
            maximum_quantization_drop=0.02,
            require_asr=False,
            maximum_asr_wer=0.12,
            minimum_asr_lead_over_each_baseline=0.01,
        ),
        minimum_size_reduction_bytes=1,
        minimum_overrefusal_improvement=0.0,
        allowed_license_ids=("Apache-2.0", "MIT"),
    )


def bound_reports(
    candidate: VerifiedPrivateCandidate,
    deployable_score: float,
    deployable_overrefusal: float,
    *,
    source_score: float | None = None,
) -> VerifiedCandidateReports:
    return bind_candidate_reports(
        candidate=candidate,
        full_precision_report=report(
            SHA_C,
            deployable_score if source_score is None else source_score,
            deployable_overrefusal,
        ),
        deployable_report=report(
            candidate.artifact_sha256,
            deployable_score,
            deployable_overrefusal,
        ),
    )


def test_private_pin_is_exact_immutable_and_redacts_sensitive_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "candidate.gguf"
    artifact.write_bytes(b"GGUFcandidate")
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("license text")

    candidate_pin = pin(artifact, license_path)

    assert candidate_pin.revision == "1" * 40
    assert (
        candidate_pin.repo_id_sha256
        == hashlib.sha256(b"private-owner/private-candidate").hexdigest()
    )
    assert "private-owner" not in repr(candidate_pin)
    assert candidate_pin.revision not in repr(candidate_pin)
    assert candidate_pin.license_url not in repr(candidate_pin)
    with pytest.raises(FrozenInstanceError):
        candidate_pin.revision = "2" * 40  # type: ignore[misc]


@pytest.mark.parametrize(
    "override",
    (
        {"revision": "main"},
        {"artifact_sha256": "bad"},
        {"artifact_size_bytes": 0},
        {"artifact_filename": "../candidate.gguf"},
        {"license_url": "http://licenses.example/license"},
        {"license_filename": "../LICENSE"},
        {"license_url": "https://licenses.example/license?token=secret"},
        {"private_repo_id": 3},
        {"license_id": 3},
    ),
)
def test_private_pin_rejects_unpinned_or_unsafe_metadata(
    tmp_path: Path, override: dict[str, object]
) -> None:
    artifact = tmp_path / "candidate.gguf"
    artifact.write_bytes(b"GGUFcandidate")
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("license text")

    with pytest.raises(CandidateConstraintError):
        pin(artifact, license_path, **override)


def test_candidate_verification_checks_gguf_sha_size_and_private_license(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "candidate.gguf"
    artifact.write_bytes(b"GGUFcandidate")
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("license text")
    candidate_pin = pin(artifact, license_path)

    verified = verify_private_candidate(
        artifact_path=artifact,
        license_path=license_path,
        pin=candidate_pin,
        allowed_license_ids=("Apache-2.0",),
    )

    assert verified.artifact_sha256 == sha256(artifact)
    assert verified.size_bytes == artifact.stat().st_size
    assert verified.license_verified is True
    assert not hasattr(verified, "private_repo_id")

    artifact.write_bytes(b"GGUFtampered")
    with pytest.raises(CandidateConstraintError, match="artifact"):
        verify_private_candidate(
            artifact_path=artifact,
            license_path=license_path,
            pin=candidate_pin,
            allowed_license_ids=("Apache-2.0",),
        )


def test_candidate_is_selected_only_when_same_gates_and_size_beat_incumbent(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "candidate.gguf"
    artifact.write_bytes(b"GGUFcandidate")
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("license text")
    verified = verify_private_candidate(
        artifact_path=artifact,
        license_path=license_path,
        pin=pin(artifact, license_path),
        allowed_license_ids=policy().allowed_license_ids,
    )
    candidate_reports = bound_reports(verified, 0.83, 0.10, source_score=0.84)
    incumbent_report = report(SHA_A, 0.80, 0.12)
    baselines = {"reference": report(SHA_B, 0.70, 0.25)}

    decision = select_private_candidate(
        candidate=verified,
        candidate_reports=candidate_reports,
        incumbent=IncumbentArtifactPin(SHA_A, artifact.stat().st_size + 1),
        incumbent_report=incumbent_report,
        baselines=baselines,
        policy=policy(),
    )

    assert decision.selected is True
    assert decision.selected_artifact_sha256 == verified.artifact_sha256
    assert decision.failures == ()


def test_candidate_verification_uses_one_open_and_fstat_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "candidate.gguf"
    artifact.write_bytes(b"GGUFcandidate")
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("license text")
    candidate_pin = pin(artifact, license_path)
    original_open = Path.open
    opened: list[Path] = []

    def tracked_open(path: Path, *args: object, **kwargs: object):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    def forbidden_stat(path: Path, *args: object, **kwargs: object):
        raise AssertionError(f"path stat used for {path}")

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(Path, "stat", forbidden_stat)

    verify_private_candidate(
        artifact_path=artifact,
        license_path=license_path,
        pin=candidate_pin,
        allowed_license_ids=("Apache-2.0",),
    )

    assert opened == [artifact, license_path]


def test_candidate_selection_fails_closed_on_equal_size_quality_or_report_binding(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "candidate.gguf"
    artifact.write_bytes(b"GGUFcandidate")
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("license text")
    active_policy = policy()
    verified = verify_private_candidate(
        artifact_path=artifact,
        license_path=license_path,
        pin=pin(artifact, license_path),
        allowed_license_ids=active_policy.allowed_license_ids,
    )
    incumbent = report(SHA_A, 0.80, 0.12)
    baselines = {"reference": report(SHA_B, 0.70, 0.25)}

    equal_size = select_private_candidate(
        candidate=verified,
        candidate_reports=bound_reports(verified, 0.83, 0.10),
        incumbent=IncumbentArtifactPin(SHA_A, verified.size_bytes),
        incumbent_report=incumbent,
        baselines=baselines,
        policy=active_policy,
    )
    assert equal_size.selected is False
    assert "size_not_better" in {failure.code for failure in equal_size.failures}

    weaker = select_private_candidate(
        candidate=verified,
        candidate_reports=bound_reports(verified, 0.80, 0.12),
        incumbent=IncumbentArtifactPin(SHA_A, verified.size_bytes + 1),
        incumbent_report=incumbent,
        baselines=baselines,
        policy=active_policy,
    )
    assert weaker.selected is False
    assert "baseline_not_beaten" in {failure.code for failure in weaker.failures}

    with pytest.raises(CandidateConstraintError, match="deployable report"):
        bind_candidate_reports(
            candidate=verified,
            full_precision_report=report(SHA_C, 0.90, 0.01),
            deployable_report=report("wrong-artifact", 0.90, 0.01),
        )


def test_forged_candidate_and_mismatched_incumbent_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "candidate.gguf"
    artifact.write_bytes(b"GGUFcandidate")
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("license text")
    verified = verify_private_candidate(
        artifact_path=artifact,
        license_path=license_path,
        pin=pin(artifact, license_path),
        allowed_license_ids=policy().allowed_license_ids,
    )
    candidate_reports = bound_reports(verified, 0.90, 0.01)
    forged = VerifiedPrivateCandidate(
        artifact_sha256=sha256(artifact),
        size_bytes=artifact.stat().st_size,
        license_id="Apache-2.0",
        license_verified=True,
        _verification_proof=b"forged",
    )
    incumbent_report = report(SHA_A, 0.80, 0.12)

    forged_decision = select_private_candidate(
        candidate=forged,
        candidate_reports=candidate_reports,
        incumbent=IncumbentArtifactPin(SHA_A, forged.size_bytes + 1),
        incumbent_report=incumbent_report,
        baselines={"reference": report(SHA_B, 0.70, 0.25)},
        policy=policy(),
    )
    assert "candidate_not_verified" in {failure.code for failure in forged_decision.failures}

    mismatch_decision = select_private_candidate(
        candidate=verified,
        candidate_reports=bound_reports(verified, 0.90, 0.01),
        incumbent=IncumbentArtifactPin(SHA_B, verified.size_bytes + 1),
        incumbent_report=incumbent_report,
        baselines={"reference": report(SHA_B, 0.70, 0.25)},
        policy=policy(),
    )
    assert "incumbent_report_mismatch" in {failure.code for failure in mismatch_decision.failures}


def test_quantization_drop_and_forged_report_evidence_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "candidate.gguf"
    artifact.write_bytes(b"GGUFcandidate")
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("license text")
    verified = verify_private_candidate(
        artifact_path=artifact,
        license_path=license_path,
        pin=pin(artifact, license_path),
        allowed_license_ids=policy().allowed_license_ids,
    )
    incumbent_report = report(SHA_A, 0.80, 0.12)
    baselines = {"reference": report(SHA_B, 0.70, 0.25)}

    excessive_drop = select_private_candidate(
        candidate=verified,
        candidate_reports=bound_reports(verified, 0.83, 0.10, source_score=0.90),
        incumbent=IncumbentArtifactPin(SHA_A, verified.size_bytes + 1),
        incumbent_report=incumbent_report,
        baselines=baselines,
        policy=policy(),
    )
    assert "quantization_regression" in {failure.code for failure in excessive_drop.failures}

    forged = VerifiedCandidateReports(
        full_precision_report=report(SHA_C, 0.84, 0.10),
        deployable_report=report(verified.artifact_sha256, 0.83, 0.10),
        _verification_proof=b"forged",
    )
    forged_decision = select_private_candidate(
        candidate=verified,
        candidate_reports=forged,
        incumbent=IncumbentArtifactPin(SHA_A, verified.size_bytes + 1),
        incumbent_report=incumbent_report,
        baselines=baselines,
        policy=policy(),
    )
    assert "candidate_reports_not_verified" in {
        failure.code for failure in forged_decision.failures
    }
