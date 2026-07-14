"""Private verification and release selection for an external GGUF artifact."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from metaflora_incubus.release_gates import (
    BenchmarkReport,
    GateFailure,
    ReleaseGatePolicy,
    evaluate_release,
)

_GGUF_MAGIC = b"GGUF"
_HASH_CHUNK_BYTES = 1024 * 1024
_INCUMBENT_BASELINE_ID = "incumbent_deployable"
_VERIFICATION_KEY = secrets.token_bytes(32)


class CandidateConstraintError(ValueError):
    """Raised when pinned candidate evidence is incomplete or invalid."""


@dataclass(frozen=True)
class PrivateCandidatePin:
    """Exact private metadata required to admit an artifact for evaluation."""

    private_repo_id: str = field(repr=False)
    revision: str = field(repr=False)
    artifact_filename: str = field(repr=False)
    artifact_sha256: str
    artifact_size_bytes: int
    license_id: str = field(repr=False)
    license_url: str = field(repr=False)
    license_filename: str = field(repr=False)
    license_sha256: str = field(repr=False)
    repo_id_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_repo_id(self.private_repo_id)
        _require_lower_hex(self.revision, 40, "revision")
        _require_safe_relative_path(self.artifact_filename, "artifact_filename")
        _require_lower_hex(self.artifact_sha256, 64, "artifact_sha256")
        _require_positive_int(self.artifact_size_bytes, "artifact_size_bytes")
        if not isinstance(self.license_id, str) or not self.license_id.strip():
            raise CandidateConstraintError("license_id must not be empty")
        _require_https_url(self.license_url, "license_url")
        _require_safe_relative_path(self.license_filename, "license_filename")
        _require_lower_hex(self.license_sha256, 64, "license_sha256")
        object.__setattr__(
            self,
            "repo_id_sha256",
            hashlib.sha256(self.private_repo_id.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def create(
        cls,
        *,
        private_repo_id: str,
        revision: str,
        artifact_filename: str,
        artifact_sha256: str,
        artifact_size_bytes: int,
        license_id: str,
        license_url: str,
        license_filename: str,
        license_sha256: str,
    ) -> PrivateCandidatePin:
        """Create an immutable pin after validating every private field."""
        return cls(
            private_repo_id=private_repo_id,
            revision=revision,
            artifact_filename=artifact_filename,
            artifact_sha256=artifact_sha256,
            artifact_size_bytes=artifact_size_bytes,
            license_id=license_id,
            license_url=license_url,
            license_filename=license_filename,
            license_sha256=license_sha256,
        )


@dataclass(frozen=True)
class VerifiedPrivateCandidate:
    """Non-identifying evidence produced after local artifact verification."""

    artifact_sha256: str
    size_bytes: int
    license_id: str
    license_verified: bool
    _verification_proof: bytes = field(repr=False)


@dataclass(frozen=True)
class IncumbentArtifactPin:
    """Exact identity and size of the incumbent deployable artifact."""

    artifact_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _require_lower_hex(self.artifact_sha256, 64, "artifact_sha256")
        _require_positive_int(self.size_bytes, "size_bytes")


@dataclass(frozen=True)
class VerifiedCandidateReports:
    """HMAC-bound full-precision and deployable benchmark reports."""

    full_precision_report: BenchmarkReport
    deployable_report: BenchmarkReport
    _verification_proof: bytes = field(repr=False)


@dataclass(frozen=True)
class CandidateSelectionPolicy:
    """Additional constraints for replacing the incumbent artifact."""

    release_policy: ReleaseGatePolicy
    minimum_candidate_size_bytes: int
    maximum_candidate_size_bytes: int
    minimum_overrefusal_improvement: float
    allowed_license_ids: tuple[str, ...]

    @classmethod
    def release(
        cls,
        *,
        release_policy: ReleaseGatePolicy,
        minimum_overrefusal_improvement: float,
        allowed_license_ids: tuple[str, ...],
    ) -> CandidateSelectionPolicy:
        """Build the fixed v1 product policy without caller-controlled size bounds."""
        return cls(
            release_policy=release_policy,
            minimum_candidate_size_bytes=3 * 1024**3,
            maximum_candidate_size_bytes=5 * 1024**3,
            minimum_overrefusal_improvement=minimum_overrefusal_improvement,
            allowed_license_ids=allowed_license_ids,
        )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.minimum_candidate_size_bytes, int)
            or isinstance(self.minimum_candidate_size_bytes, bool)
            or not isinstance(self.maximum_candidate_size_bytes, int)
            or isinstance(self.maximum_candidate_size_bytes, bool)
            or self.minimum_candidate_size_bytes < 1
            or self.maximum_candidate_size_bytes < self.minimum_candidate_size_bytes
        ):
            raise CandidateConstraintError("candidate size bounds are invalid")
        if (
            not isinstance(self.minimum_overrefusal_improvement, (int, float))
            or isinstance(self.minimum_overrefusal_improvement, bool)
            or not isfinite(self.minimum_overrefusal_improvement)
            or not 0 < self.minimum_overrefusal_improvement <= 1
        ):
            raise CandidateConstraintError("minimum_overrefusal_improvement must be within (0, 1]")
        allowed = tuple(self.allowed_license_ids)
        if not allowed or any(not isinstance(item, str) or not item.strip() for item in allowed):
            raise CandidateConstraintError("allowed_license_ids must not be empty")
        if len(set(allowed)) != len(allowed):
            raise CandidateConstraintError("allowed_license_ids must be unique")
        object.__setattr__(self, "allowed_license_ids", allowed)


@dataclass(frozen=True)
class CandidateSelectionDecision:
    """Fail-closed selection result without private source metadata."""

    selected: bool
    selected_artifact_sha256: str | None
    failures: tuple[GateFailure, ...]


def verify_private_candidate(
    *,
    artifact_path: Path,
    license_path: Path,
    pin: PrivateCandidatePin,
    allowed_license_ids: Sequence[str],
) -> VerifiedPrivateCandidate:
    """Verify exact artifact and license evidence without exposing its source."""
    allowed = tuple(allowed_license_ids)
    if pin.license_id not in allowed:
        raise CandidateConstraintError("license is not allowed")
    artifact_size, artifact_sha256, artifact_magic = _read_file_evidence(
        artifact_path, magic_size=len(_GGUF_MAGIC), label="artifact"
    )
    if artifact_size != pin.artifact_size_bytes:
        raise CandidateConstraintError("artifact size does not match pin")
    if artifact_magic != _GGUF_MAGIC:
        raise CandidateConstraintError("artifact is not a GGUF file")
    if artifact_sha256 != pin.artifact_sha256:
        raise CandidateConstraintError("artifact SHA-256 does not match pin")
    _, license_sha256, _ = _read_file_evidence(license_path, magic_size=0, label="license")
    if license_sha256 != pin.license_sha256:
        raise CandidateConstraintError("license SHA-256 does not match pin")
    proof = _verification_proof(
        pin.artifact_sha256,
        pin.artifact_size_bytes,
        pin.license_id,
    )
    return VerifiedPrivateCandidate(
        artifact_sha256=pin.artifact_sha256,
        size_bytes=pin.artifact_size_bytes,
        license_id=pin.license_id,
        license_verified=True,
        _verification_proof=proof,
    )


def bind_candidate_reports(
    *,
    candidate: VerifiedPrivateCandidate,
    full_precision_report: BenchmarkReport,
    deployable_report: BenchmarkReport,
) -> VerifiedCandidateReports:
    """Bind independently measured full-precision and GGUF reports."""
    if not _has_valid_verification_proof(candidate):
        raise CandidateConstraintError("candidate must be verified before reports are bound")
    if deployable_report.artifact_id != candidate.artifact_sha256:
        raise CandidateConstraintError("deployable report does not match candidate artifact")
    if full_precision_report.suite_id != deployable_report.suite_id:
        raise CandidateConstraintError("candidate report suites do not match")
    _require_lower_hex(full_precision_report.artifact_id, 64, "full_precision_artifact_id")
    proof = _reports_proof(candidate.artifact_sha256, full_precision_report, deployable_report)
    return VerifiedCandidateReports(
        full_precision_report=full_precision_report,
        deployable_report=deployable_report,
        _verification_proof=proof,
    )


def select_private_candidate(
    *,
    candidate: VerifiedPrivateCandidate,
    candidate_reports: VerifiedCandidateReports,
    incumbent: IncumbentArtifactPin,
    incumbent_report: BenchmarkReport,
    baselines: Mapping[str, BenchmarkReport],
    policy: CandidateSelectionPolicy,
) -> CandidateSelectionDecision:
    """Select a verified artifact only if it beats the incumbent on every gate."""
    failures: list[GateFailure] = []
    if not _has_valid_verification_proof(candidate):
        failures.append(GateFailure("candidate_not_verified", "candidate"))
    if not candidate.license_verified or candidate.license_id not in policy.allowed_license_ids:
        failures.append(GateFailure("license_not_verified", "candidate"))
    reports_verified = _has_valid_reports_proof(candidate, candidate_reports)
    if not reports_verified:
        failures.append(GateFailure("candidate_reports_not_verified", "candidate"))
    deployable_report = candidate_reports.deployable_report
    if deployable_report.artifact_id != candidate.artifact_sha256:
        failures.append(GateFailure("report_artifact_mismatch", "candidate"))
    if incumbent_report.artifact_id != incumbent.artifact_sha256:
        failures.append(GateFailure("incumbent_report_mismatch", "incumbent"))
    if not (
        policy.minimum_candidate_size_bytes
        <= candidate.size_bytes
        <= policy.maximum_candidate_size_bytes
    ):
        failures.append(
            GateFailure(
                "candidate_size_out_of_bounds",
                "candidate",
            )
        )

    improvement = incumbent_report.overrefusal_rate - deployable_report.overrefusal_rate
    if not isfinite(improvement) or improvement < policy.minimum_overrefusal_improvement:
        failures.append(GateFailure("overrefusal_not_better", "candidate"))

    if _INCUMBENT_BASELINE_ID in baselines:
        failures.append(GateFailure("invalid_baselines", _INCUMBENT_BASELINE_ID))
    elif reports_verified and deployable_report.artifact_id == candidate.artifact_sha256:
        gate_policy = _with_incumbent_baseline(policy.release_policy)
        gate_baselines = {**baselines, _INCUMBENT_BASELINE_ID: incumbent_report}
        failures.extend(
            evaluate_release(
                candidate_reports.full_precision_report,
                deployable_report,
                gate_baselines,
                gate_policy,
            ).failures
        )

    result = tuple(failures)
    return CandidateSelectionDecision(
        selected=not result,
        selected_artifact_sha256=candidate.artifact_sha256 if not result else None,
        failures=result,
    )


def _with_incumbent_baseline(policy: ReleaseGatePolicy) -> ReleaseGatePolicy:
    return ReleaseGatePolicy(
        required_baselines=(*policy.required_baselines, _INCUMBENT_BASELINE_ID),
        required_score_targets=policy.required_score_targets,
        minimum_lead_over_each_baseline=policy.minimum_lead_over_each_baseline,
        minimum_overrefusal_reduction=policy.minimum_overrefusal_reduction,
        maximum_quantization_drop=policy.maximum_quantization_drop,
        require_asr=policy.require_asr,
        maximum_asr_wer=policy.maximum_asr_wer,
        minimum_asr_lead_over_each_baseline=policy.minimum_asr_lead_over_each_baseline,
    )


def _read_file_evidence(path: Path, *, magic_size: int, label: str) -> tuple[int, str, bytes]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise CandidateConstraintError(f"{label} is not a regular file")
            magic = stream.read(magic_size)
            digest.update(magic)
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CandidateConstraintError(f"{label} file cannot be opened") from exc
    return metadata.st_size, digest.hexdigest(), magic


def _verification_proof(artifact_sha256: str, size_bytes: int, license_id: str) -> bytes:
    payload = f"{artifact_sha256}:{size_bytes}:{license_id}".encode()
    return hmac.digest(_VERIFICATION_KEY, payload, "sha256")


def _has_valid_verification_proof(candidate: VerifiedPrivateCandidate) -> bool:
    if (
        not isinstance(candidate.size_bytes, int)
        or isinstance(candidate.size_bytes, bool)
        or candidate.size_bytes < 1
        or not isinstance(candidate.artifact_sha256, str)
        or not isinstance(candidate.license_id, str)
        or not isinstance(candidate._verification_proof, bytes)
    ):
        return False
    expected = _verification_proof(
        candidate.artifact_sha256,
        candidate.size_bytes,
        candidate.license_id,
    )
    return hmac.compare_digest(candidate._verification_proof, expected)


def _reports_proof(
    candidate_sha256: str,
    full_precision_report: BenchmarkReport,
    deployable_report: BenchmarkReport,
) -> bytes:
    payload = {
        "candidate_sha256": candidate_sha256,
        "deployable": _report_payload(deployable_report),
        "full_precision": _report_payload(full_precision_report),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hmac.digest(_VERIFICATION_KEY, encoded, "sha256")


def _report_payload(report: BenchmarkReport) -> dict[str, object]:
    return {
        "artifact_id": report.artifact_id,
        "suite_id": report.suite_id,
        "scores": dict(report.scores),
        "overrefusal_rate": report.overrefusal_rate,
        "asr_wer": report.asr_wer,
    }


def _has_valid_reports_proof(
    candidate: VerifiedPrivateCandidate, reports: VerifiedCandidateReports
) -> bool:
    if (
        not isinstance(reports, VerifiedCandidateReports)
        or not isinstance(reports.full_precision_report, BenchmarkReport)
        or not isinstance(reports.deployable_report, BenchmarkReport)
        or not isinstance(reports._verification_proof, bytes)
    ):
        return False
    try:
        expected = _reports_proof(
            candidate.artifact_sha256,
            reports.full_precision_report,
            reports.deployable_report,
        )
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(reports._verification_proof, expected)


def _require_lower_hex(value: str, length: int, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CandidateConstraintError(f"{field_name} must be {length} lowercase hex characters")


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CandidateConstraintError(f"{field_name} must be a positive integer")


def _require_safe_relative_path(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CandidateConstraintError(f"{field_name} must be a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CandidateConstraintError(f"{field_name} must be a safe relative path")


def _require_https_url(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise CandidateConstraintError(f"{field_name} must be an HTTPS URL without credentials")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise CandidateConstraintError(f"{field_name} must be an HTTPS URL without credentials")


def _require_repo_id(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character.isspace() for character in value)
    ):
        raise CandidateConstraintError("private_repo_id is invalid")
