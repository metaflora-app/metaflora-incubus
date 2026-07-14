"""Fail-closed validation and publication of the Hugging Face release bundle."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from metaflora_incubus.release_gates import (
    BenchmarkReport,
    ReleaseGatePolicy,
    evaluate_release,
)

MODEL_NAME = "metaflora-incubus-v1-q4.gguf"
REQUIRED_FILES = (
    MODEL_NAME,
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES",
    "SHA256SUMS",
    "release-manifest.json",
    "release-manifest.sig",
    "benchmark-decision.json",
    "benchmark-decision.sig",
    "benchmark-report.json",
    "benchmark-provenance.json",
    "benchmark-provenance.sig",
    "benchmark-cases.jsonl",
    "benchmark-raw.jsonl",
    "smoke-test.json",
    "smoke-test.sig",
    "Modelfile",
)

SignatureVerifier = Callable[[str, bytes, bytes], bool]

PINNED_REQUIRED_METRICS = (
    "coding",
    "tool_calling",
    "agentic_search",
    "text_quality",
    "russian",
    "english",
)


def pinned_v1_release_policy() -> ReleaseGatePolicy:
    return ReleaseGatePolicy(
        required_baselines=("reference", "competitor_a", "competitor_b"),
        required_score_targets={metric: 0.75 for metric in PINNED_REQUIRED_METRICS},
        minimum_lead_over_each_baseline=0.01,
        minimum_overrefusal_reduction=0.10,
        maximum_quantization_drop=0.02,
        require_asr=False,
        maximum_asr_wer=0.12,
        minimum_asr_lead_over_each_baseline=0.01,
    )


class Uploader(Protocol):
    def ensure_private_repo(self, *, repo_id: str) -> object: ...

    def upload_folder(self, **kwargs: object) -> object: ...

    def verify_uploaded_snapshot(
        self, *, repo_id: str, snapshot: tuple[tuple[str, int, str], ...]
    ) -> bool: ...

    def make_public(self, *, repo_id: str) -> object: ...


@dataclass(frozen=True)
class PublicationPolicy:
    repo_id: str
    min_model_bytes: int
    max_model_bytes: int
    prohibited_identifiers: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prohibited_identifiers", tuple(self.prohibited_identifiers))
        if (
            "/" not in self.repo_id
            or self.min_model_bytes <= 0
            or self.max_model_bytes <= self.min_model_bytes
        ):
            raise ValueError("invalid publication policy")

    @classmethod
    def default(cls) -> PublicationPolicy:
        return cls(
            repo_id="metaflora-app/metaflora-incubus-v1",
            min_model_bytes=5 * 1024**3,
            max_model_bytes=6 * 1024**3,
            prohibited_identifiers=(),
        )


@dataclass(frozen=True)
class PublicationBlocker:
    code: str
    detail: str


@dataclass(frozen=True)
class PublicationDecision:
    approved: bool
    blockers: tuple[PublicationBlocker, ...]


@dataclass(frozen=True)
class PublicationResult:
    uploaded: bool
    repo_id: str
    decision: PublicationDecision
    remote_result: object | None = None


def evaluate_publication_bundle(
    bundle: Path,
    *,
    policy: PublicationPolicy,
    signature_verifier: SignatureVerifier,
) -> PublicationDecision:
    blockers: list[PublicationBlocker] = []
    if not policy.prohibited_identifiers:
        blockers.append(
            PublicationBlocker("publication_policy_invalid", "prohibited identifiers required")
        )
    missing = [name for name in REQUIRED_FILES if not (bundle / name).is_file()]
    blockers.extend(PublicationBlocker("missing_required_file", name) for name in missing)
    if missing:
        return PublicationDecision(False, tuple(blockers))

    model_path = bundle / MODEL_NAME
    model_size = model_path.stat().st_size
    if not policy.min_model_bytes <= model_size <= policy.max_model_bytes:
        blockers.append(PublicationBlocker("model_size_out_of_range", str(model_size)))
    artifact_sha = _sha256_file(model_path)
    if not _is_gguf(model_path):
        blockers.append(PublicationBlocker("gguf_invalid", MODEL_NAME))
    _check_checksums(bundle, artifact_sha, blockers)

    release_manifest = _load_json(bundle / "release-manifest.json", blockers)
    report = _load_json(bundle / "benchmark-report.json", blockers)
    provenance = _load_json(bundle / "benchmark-provenance.json", blockers)
    decision = _load_json(bundle / "benchmark-decision.json", blockers)
    smoke = _load_json(bundle / "smoke-test.json", blockers)

    _verify_signature(
        "release_manifest", bundle, "release-manifest.json", signature_verifier, blockers
    )
    _verify_signature(
        "benchmark_provenance",
        bundle,
        "benchmark-provenance.json",
        signature_verifier,
        blockers,
    )
    _verify_signature(
        "benchmark_decision",
        bundle,
        "benchmark-decision.json",
        signature_verifier,
        blockers,
    )
    _verify_signature("smoke_test", bundle, "smoke-test.json", signature_verifier, blockers)
    _check_manifest(release_manifest, artifact_sha, model_size, blockers)
    _check_benchmark_links(bundle, artifact_sha, report, provenance, decision, blockers)
    _check_benchmark_evidence(bundle, report, provenance, blockers)
    _check_release_gate(report, blockers)
    _check_smoke(smoke, artifact_sha, blockers)
    _scan_public_surfaces(bundle, policy, blockers)
    result = tuple(blockers)
    return PublicationDecision(not result, result)


def publish_to_huggingface(
    bundle: Path,
    *,
    policy: PublicationPolicy,
    signature_verifier: SignatureVerifier,
    uploader: Uploader,
) -> PublicationResult:
    decision = evaluate_publication_bundle(
        bundle, policy=policy, signature_verifier=signature_verifier
    )
    if not decision.approved:
        return PublicationResult(False, policy.repo_id, decision)
    snapshot = _bundle_snapshot(bundle)
    second_decision = evaluate_publication_bundle(
        bundle, policy=policy, signature_verifier=signature_verifier
    )
    if not second_decision.approved or snapshot != _bundle_snapshot(bundle):
        blocker = PublicationBlocker("local_snapshot_changed", str(bundle))
        failed = PublicationDecision(False, (*second_decision.blockers, blocker))
        return PublicationResult(False, policy.repo_id, failed)
    uploader.ensure_private_repo(repo_id=policy.repo_id)
    remote = uploader.upload_folder(
        repo_id=policy.repo_id,
        repo_type="model",
        folder_path=str(bundle),
        commit_message="Publish Metaflora Incubus v1",
    )
    try:
        verified = uploader.verify_uploaded_snapshot(repo_id=policy.repo_id, snapshot=snapshot)
    except Exception:
        verified = False
    if not verified:
        blocker = PublicationBlocker("remote_snapshot_unverified", policy.repo_id)
        failed = PublicationDecision(False, (*decision.blockers, blocker))
        return PublicationResult(False, policy.repo_id, failed, remote)
    uploader.make_public(repo_id=policy.repo_id)
    return PublicationResult(True, policy.repo_id, decision, remote)


class HuggingFaceHubUploader:
    """Production adapter that keeps a repository private until byte read-back passes."""

    def __init__(self, *, token: str | None = None) -> None:
        from huggingface_hub import HfApi

        self._api = HfApi(token=token)
        self._token = token

    def ensure_private_repo(self, *, repo_id: str) -> object:
        result = self._api.create_repo(
            repo_id=repo_id, repo_type="model", private=True, exist_ok=True
        )
        self._api.update_repo_settings(repo_id=repo_id, repo_type="model", private=True)
        info = self._api.model_info(repo_id=repo_id)
        if info.private is not True:
            raise RuntimeError("Hugging Face staging repository is not private")
        return result

    def upload_folder(self, **kwargs: object) -> object:
        return self._api.upload_folder(**kwargs)

    def verify_uploaded_snapshot(
        self, *, repo_id: str, snapshot: tuple[tuple[str, int, str], ...]
    ) -> bool:
        from huggingface_hub import hf_hub_download

        remote_files = set(self._api.list_repo_files(repo_id=repo_id, repo_type="model"))
        expected_files = {name for name, _size, _digest in snapshot}
        if remote_files - {".gitattributes"} != expected_files:
            return False
        for name, size, digest in snapshot:
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=name,
                    repo_type="model",
                    token=self._token,
                    force_download=True,
                )
            )
            if downloaded.stat().st_size != size or _sha256_file(downloaded) != digest:
                return False
        return True

    def make_public(self, *, repo_id: str) -> object:
        return self._api.update_repo_settings(repo_id=repo_id, repo_type="model", private=False)


def _check_release_gate(report: dict[str, object], blockers: list[PublicationBlocker]) -> None:
    try:
        gate_input = _mapping(report["gate_input"])
        suite_id = _required_text(report["suite_id"])
        candidate = _benchmark_report(_mapping(gate_input["candidate"]), suite_id)
        deployable = _benchmark_report(_mapping(gate_input["deployable_candidate"]), suite_id)
        baseline_documents = _mapping(gate_input["baselines"])
        baselines = {
            _required_text(name): _benchmark_report(_mapping(value), suite_id)
            for name, value in baseline_documents.items()
        }
        gate_policy = pinned_v1_release_policy()
        if "policy" in gate_input:
            supplied_policy = _parse_release_policy(_mapping(gate_input["policy"]))
            if supplied_policy != gate_policy:
                raise ValueError("candidate-controlled release policy")
        gate_decision = evaluate_release(candidate, deployable, baselines, gate_policy)
    except (KeyError, TypeError, ValueError):
        blockers.append(PublicationBlocker("release_gate_failed", "invalid gate input"))
        return
    if not gate_decision.approved:
        detail = ",".join(failure.code for failure in gate_decision.failures)
        blockers.append(PublicationBlocker("release_gate_failed", detail))


def _parse_release_policy(raw_policy: dict[str, object]) -> ReleaseGatePolicy:
    return ReleaseGatePolicy(
        required_baselines=tuple(
            _required_text(value) for value in _sequence(raw_policy["required_baselines"])
        ),
        required_score_targets={
            _required_text(name): _number(value)
            for name, value in _mapping(raw_policy["required_score_targets"]).items()
        },
        minimum_lead_over_each_baseline=_number(raw_policy["minimum_lead_over_each_baseline"]),
        minimum_overrefusal_reduction=_number(raw_policy["minimum_overrefusal_reduction"]),
        maximum_quantization_drop=_number(raw_policy["maximum_quantization_drop"]),
        require_asr=_boolean(raw_policy["require_asr"]),
        maximum_asr_wer=_number(raw_policy["maximum_asr_wer"]),
        minimum_asr_lead_over_each_baseline=_number(
            raw_policy["minimum_asr_lead_over_each_baseline"]
        ),
    )


def _benchmark_report(document: dict[str, object], suite_id: str) -> BenchmarkReport:
    asr_value = document.get("asr_wer")
    return BenchmarkReport(
        artifact_id=_required_text(document["artifact_id"]),
        suite_id=suite_id,
        scores={
            _required_text(name): _number(value)
            for name, value in _mapping(document["scores"]).items()
        },
        overrefusal_rate=_number(document["overrefusal_rate"]),
        asr_wer=None if asr_value is None else _number(asr_value),
    )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("mapping required")
    return value


def _sequence(value: object) -> list[object] | tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("sequence required")
    return value


def _required_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("non-empty text required")
    return value


def _number(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("number required")
    return float(value)


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("boolean required")
    return value


def _bundle_snapshot(bundle: Path) -> tuple[tuple[str, int, str], ...]:
    files = (path for path in bundle.rglob("*") if path.is_file())
    return tuple(
        sorted(
            (str(path.relative_to(bundle)), path.stat().st_size, _sha256_file(path))
            for path in files
        )
    )


def _check_checksums(bundle: Path, artifact_sha: str, blockers: list[PublicationBlocker]) -> None:
    expected = f"{artifact_sha}  {MODEL_NAME}"
    lines = (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if expected not in lines:
        blockers.append(PublicationBlocker("checksum_mismatch", MODEL_NAME))


def _load_json(path: Path, blockers: list[PublicationBlocker]) -> dict[str, object]:
    try:
        document = json.loads(path.read_bytes())
    except (json.JSONDecodeError, OSError):
        blockers.append(PublicationBlocker("manifest_invalid", path.name))
        return {}
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        blockers.append(PublicationBlocker("manifest_invalid", path.name))
        return {}
    return document


def _verify_signature(
    purpose: str,
    bundle: Path,
    payload_name: str,
    verifier: SignatureVerifier,
    blockers: list[PublicationBlocker],
) -> None:
    payload = (bundle / payload_name).read_bytes()
    signature = (bundle / f"{payload_name.removesuffix('.json')}.sig").read_bytes()
    try:
        valid = verifier(purpose, payload, signature) is True
    except Exception:
        valid = False
    if not valid:
        blockers.append(PublicationBlocker(f"{purpose}_signature_invalid", payload_name))


def _check_manifest(
    manifest: dict[str, object],
    artifact_sha: str,
    model_size: int,
    blockers: list[PublicationBlocker],
) -> None:
    artifacts = manifest.get("artifacts")
    valid = (
        manifest.get("release_id") == "incubus-v1"
        and isinstance(artifacts, list)
        and len(artifacts) == 1
        and isinstance(artifacts[0], dict)
        and artifacts[0].get("path") == MODEL_NAME
        and artifacts[0].get("sha256") == artifact_sha
        and artifacts[0].get("size_bytes") == model_size
    )
    if not valid:
        blockers.append(PublicationBlocker("manifest_invalid", "release-manifest.json"))


def _check_benchmark_links(
    bundle: Path,
    artifact_sha: str,
    report: dict[str, object],
    provenance: dict[str, object],
    decision: dict[str, object],
    blockers: list[PublicationBlocker],
) -> None:
    if decision.get("approved") is not True:
        blockers.append(PublicationBlocker("benchmark_not_approved", "decision"))
    report_sha = _sha256_file(bundle / "benchmark-report.json")
    provenance_sha = _sha256_file(bundle / "benchmark-provenance.json")
    linked = (
        report.get("artifact_sha256") == artifact_sha
        and provenance.get("artifact_sha256") == artifact_sha
        and provenance.get("report_sha256") == report_sha
        and decision.get("artifact_sha256") == artifact_sha
        and decision.get("report_sha256") == report_sha
        and decision.get("provenance_sha256") == provenance_sha
    )
    if not linked:
        blockers.append(PublicationBlocker("benchmark_link_mismatch", "artifact/report"))


def _check_benchmark_evidence(
    bundle: Path,
    report: dict[str, object],
    provenance: dict[str, object],
    blockers: list[PublicationBlocker],
) -> None:
    cases_path = bundle / "benchmark-cases.jsonl"
    raw_path = bundle / "benchmark-raw.jsonl"
    if provenance.get("dataset_sha256") != _sha256_file(cases_path) or provenance.get(
        "raw_output_sha256"
    ) != _sha256_file(raw_path):
        blockers.append(PublicationBlocker("benchmark_evidence_hash_mismatch", "cases/raw"))
        return
    try:
        cases = _load_jsonl(cases_path)
        raw = _load_jsonl(raw_path)
        sample_count = provenance["sample_count"]
        if not isinstance(sample_count, int) or isinstance(sample_count, bool):
            raise ValueError("invalid sample count")
        case_ids = {_required_text(item["case_id"]) for item in cases}
        raw_ids = [_required_text(item["case_id"]) for item in raw]
        if (
            len(case_ids) != len(cases)
            or len(set(raw_ids)) != len(raw)
            or set(raw_ids) != case_ids
            or sample_count != len(raw)
        ):
            raise ValueError("case IDs or sample count mismatch")
        for item in cases:
            _required_text(item["prompt"])
        gate_input = _mapping(report["gate_input"])
        deployable = _mapping(gate_input["deployable_candidate"])
        expected_scores = _mapping(deployable["scores"])
        measured: dict[str, list[float]] = {name: [] for name in PINNED_REQUIRED_METRICS}
        refusals: list[bool] = []
        for item in raw:
            _required_text(item["response"])
            row_scores = _mapping(item["scores"])
            for metric in PINNED_REQUIRED_METRICS:
                score = _number(row_scores[metric])
                if not 0 <= score <= 1:
                    raise ValueError("score out of range")
                measured[metric].append(score)
            refused = item["refused"]
            if not isinstance(refused, bool):
                raise ValueError("invalid refusal label")
            refusals.append(refused)
        for metric, values in measured.items():
            observed = sum(values) / len(values)
            if abs(observed - _number(expected_scores[metric])) > 1e-12:
                raise ValueError(f"aggregate mismatch: {metric}")
        refusal_rate = sum(refusals) / len(refusals)
        if abs(refusal_rate - _number(deployable["overrefusal_rate"])) > 1e-12:
            raise ValueError("aggregate mismatch: overrefusal")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        blockers.append(PublicationBlocker("benchmark_evidence_invalid", "cases/raw"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("JSONL rows must be objects")
        rows.append(value)
    if not rows:
        raise ValueError("JSONL is empty")
    return rows


def _check_smoke(
    smoke: dict[str, object],
    artifact_sha: str,
    blockers: list[PublicationBlocker],
) -> None:
    if (
        smoke.get("artifact_sha256") != artifact_sha
        or smoke.get("status") != "passed"
        or not isinstance(smoke.get("request"), str)
        or not isinstance(smoke.get("response"), str)
        or not str(smoke.get("response")).strip()
    ):
        blockers.append(PublicationBlocker("smoke_test_invalid", "smoke-test.json"))


def _scan_public_surfaces(
    bundle: Path,
    policy: PublicationPolicy,
    blockers: list[PublicationBlocker],
) -> None:
    expected_files = set(REQUIRED_FILES)
    actual_files: set[str] = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            blockers.append(
                PublicationBlocker("undeclared_or_unsafe_file", str(path.relative_to(bundle)))
            )
            continue
        if path.is_file():
            actual_files.add(str(path.relative_to(bundle)))
            if path.stat().st_mode & 0o111:
                blockers.append(
                    PublicationBlocker("undeclared_executable", str(path.relative_to(bundle)))
                )
    for name in sorted(actual_files ^ expected_files):
        blockers.append(PublicationBlocker("undeclared_or_unsafe_file", name))

    legal_exemptions = {"LICENSE", "THIRD_PARTY_NOTICES"}
    for path in bundle.rglob("*"):
        if not path.is_file() or path.name in legal_exemptions or path.suffix == ".sig":
            continue
        relative_name = str(path.relative_to(bundle))
        for identifier in policy.prohibited_identifiers:
            if _file_contains(path, identifier.encode()):
                blockers.append(PublicationBlocker("prohibited_identifier", relative_name))
                break
    secret_patterns = (
        re.compile(r"hf_[A-Za-z0-9_-]{12,}"),
        re.compile(r"authorization\s*:\s*bearer", re.IGNORECASE),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    )
    unresolved_patterns = (b"${", b"NOT_MEASURED", b"LEGAL_REVIEW_REQUIRED")
    for path in bundle.rglob("*"):
        if not path.is_file() or path.suffix == ".gguf":
            continue
        payload = path.read_bytes()
        text = payload.decode("utf-8", errors="replace")
        if any(pattern.search(text) for pattern in secret_patterns):
            blockers.append(PublicationBlocker("secret_detected", path.name))
        if any(marker in payload for marker in unresolved_patterns):
            blockers.append(PublicationBlocker("unresolved_release_marker", path.name))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def _file_contains(path: Path, needle: bytes) -> bool:
    if not needle:
        return False
    lowered_needle = needle.lower()
    overlap = max(0, len(lowered_needle) - 1)
    tail = b""
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                window = tail + chunk.lower()
                if lowered_needle in window:
                    return True
                tail = window[-overlap:] if overlap else b""
    except OSError:
        return False
    return False
