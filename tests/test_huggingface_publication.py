from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

MODEL_NAME = "metaflora-incubus-v1-q4.gguf"
DEFAULT_REPO_ID = "metaflora/incubus"
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


def publication_module():
    return importlib.import_module("metaflora_incubus.huggingface_publication")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, document: object) -> bytes:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(payload)
    return payload


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> bytes:
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
    )
    path.write_bytes(payload)
    return payload


def release_scores(value: float) -> dict[str, float]:
    return {
        "coding": value,
        "tool_calling": value,
        "agentic_search": value,
        "text_quality": value,
        "russian": value,
        "english": value,
    }


def build_valid_bundle(root: Path, *, model_bytes: bytes = b"GGUFcompact-model") -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / MODEL_NAME
    model_path.write_bytes(model_bytes)
    artifact_sha = sha256_bytes(model_bytes)

    (root / "README.md").write_text(
        "# Metaflora Incubus v1\n\nA compact local model for code, tools and text.\n",
        encoding="utf-8",
    )
    (root / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    # Legal attribution is mandatory and is not a product/model-card naming surface.
    (root / "THIRD_PARTY_NOTICES").write_text(
        "Required attribution for forbidden-source components.\n",
        encoding="utf-8",
    )
    (root / "SHA256SUMS").write_text(f"{artifact_sha}  {MODEL_NAME}\n", encoding="utf-8")
    (root / "Modelfile").write_text(f"FROM ./{MODEL_NAME}\n", encoding="utf-8")

    write_json(
        root / "release-manifest.json",
        {
            "schema_version": 1,
            "release_id": "incubus-v1",
            "artifacts": [
                {
                    "path": MODEL_NAME,
                    "sha256": artifact_sha,
                    "size_bytes": len(model_bytes),
                }
            ],
        },
    )
    (root / "release-manifest.sig").write_bytes(b"valid:release_manifest")

    report_payload = write_json(
        root / "benchmark-report.json",
        {
            "schema_version": 1,
            "artifact_sha256": artifact_sha,
            "suite_id": "incubus-release-v1",
            "gate_input": {
                "candidate": {
                    "artifact_id": "candidate-full",
                    "scores": release_scores(0.94),
                    "overrefusal_rate": 0.03,
                },
                "deployable_candidate": {
                    "artifact_id": "candidate-q4",
                    "scores": release_scores(0.93),
                    "overrefusal_rate": 0.04,
                },
                "baselines": {
                    "reference": {
                        "artifact_id": "reference",
                        "scores": release_scores(0.88),
                        "overrefusal_rate": 0.20,
                    },
                    "competitor_a": {
                        "artifact_id": "competitor-a",
                        "scores": release_scores(0.70),
                        "overrefusal_rate": 0.18,
                    },
                    "competitor_b": {
                        "artifact_id": "competitor-b",
                        "scores": release_scores(0.71),
                        "overrefusal_rate": 0.17,
                    },
                },
                "policy": {
                    "required_baselines": ["reference", "competitor_a", "competitor_b"],
                    "required_score_targets": release_scores(0.75),
                    "minimum_lead_over_each_baseline": 0.01,
                    "minimum_overrefusal_reduction": 0.10,
                    "maximum_quantization_drop": 0.02,
                    "require_asr": False,
                    "maximum_asr_wer": 0.12,
                    "minimum_asr_lead_over_each_baseline": 0.01,
                },
            },
        },
    )
    case_rows = [
        {"case_id": f"release-{index:03d}", "prompt": f"Release case {index}"}
        for index in range(100)
    ]
    raw_rows = [
        {
            "case_id": f"release-{index:03d}",
            "response": f"Measured response {index}",
            "scores": release_scores(1.0 if index < 93 else 0.0),
            "refused": index < 4,
        }
        for index in range(100)
    ]
    cases_payload = write_jsonl(root / "benchmark-cases.jsonl", case_rows)
    raw_payload = write_jsonl(root / "benchmark-raw.jsonl", raw_rows)
    provenance_payload = write_json(
        root / "benchmark-provenance.json",
        {
            "schema_version": 1,
            "artifact_sha256": artifact_sha,
            "report_sha256": sha256_bytes(report_payload),
            "harness_revision": "0123456789abcdef0123456789abcdef01234567",
            "dataset_sha256": sha256_bytes(cases_payload),
            "raw_output_sha256": sha256_bytes(raw_payload),
            "sample_count": len(raw_rows),
            "runtime": "llama.cpp-b7001",
            "seeds": [17, 29, 43],
        },
    )
    (root / "benchmark-provenance.sig").write_bytes(b"valid:benchmark_provenance")
    write_json(
        root / "benchmark-decision.json",
        {
            "schema_version": 1,
            "approved": True,
            "artifact_sha256": artifact_sha,
            "report_sha256": sha256_bytes(report_payload),
            "provenance_sha256": sha256_bytes(provenance_payload),
        },
    )
    (root / "smoke-test.sig").write_bytes(b"valid:smoke_test")
    (root / "benchmark-decision.sig").write_bytes(b"valid:benchmark_decision")
    write_json(
        root / "smoke-test.json",
        {
            "schema_version": 1,
            "artifact_sha256": artifact_sha,
            "runtime": "llama.cpp-b7001",
            "status": "passed",
            "request": "Reply with the word ready.",
            "response": "ready",
        },
    )
    return {
        "artifact_sha256": artifact_sha,
        "report_sha256": sha256_bytes(report_payload),
        "provenance_sha256": sha256_bytes(provenance_payload),
    }


def policy(*, repo_id: str = DEFAULT_REPO_ID):
    publications = publication_module()
    return publications.PublicationPolicy(
        repo_id=repo_id,
        min_model_bytes=8,
        max_model_bytes=32,
        prohibited_identifiers=("forbidden-source", "forbidden-teacher"),
    )


class RecordingVerifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes, bytes]] = []

    def __call__(self, purpose: str, payload: bytes, signature: bytes) -> bool:
        self.calls.append((purpose, payload, signature))
        return signature == f"valid:{purpose}".encode()


class RecordingUploader:
    def __init__(self, *, remote_verified: bool = True) -> None:
        self.calls: list[dict[str, object]] = []
        self.verification_calls: list[dict[str, object]] = []
        self.private_calls: list[dict[str, object]] = []
        self.public_calls: list[dict[str, object]] = []
        self.remote_verified = remote_verified

    def ensure_private_repo(self, **kwargs: object) -> dict[str, bool]:
        self.private_calls.append(kwargs)
        return {"private": True}

    def upload_folder(self, **kwargs: object) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"commit_url": "https://huggingface.co/example/commit/abc"}

    def verify_uploaded_snapshot(self, **kwargs: object) -> bool:
        self.verification_calls.append(kwargs)
        return self.remote_verified

    def make_public(self, **kwargs: object) -> dict[str, bool]:
        self.public_calls.append(kwargs)
        return {"private": False}


def blocker_codes(decision: object) -> set[str]:
    return {blocker.code for blocker in decision.blockers}  # type: ignore[attr-defined]


def test_default_policy_is_three_to_five_gib_and_uses_the_public_repository() -> None:
    publications = publication_module()

    release_policy = publications.PublicationPolicy.default()

    assert release_policy.repo_id == DEFAULT_REPO_ID
    assert release_policy.min_model_bytes == 3 * 1024**3
    assert release_policy.max_model_bytes == 5 * 1024**3
    assert release_policy.min_model_bytes < release_policy.max_model_bytes


def test_real_size_bounds_can_be_exercised_with_a_sparse_gguf(tmp_path: Path) -> None:
    publications = publication_module()
    release_policy = publications.PublicationPolicy.default()
    sparse_model = tmp_path / MODEL_NAME
    sparse_model.touch()
    with sparse_model.open("r+b") as handle:
        handle.truncate(release_policy.min_model_bytes)

    assert sparse_model.stat().st_size == 3 * 1024**3


def test_complete_verified_bundle_is_approved(tmp_path: Path) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    verifier = RecordingVerifier()

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=verifier,
    )

    assert decision.approved is True
    assert decision.blockers == ()
    assert {purpose for purpose, _payload, _signature in verifier.calls} == {
        "release_manifest",
        "benchmark_decision",
        "benchmark_provenance",
        "smoke_test",
    }


@pytest.mark.parametrize("missing_name", REQUIRED_FILES)
def test_every_required_release_file_is_fail_closed(tmp_path: Path, missing_name: str) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    (tmp_path / missing_name).unlink()

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert "missing_required_file" in blocker_codes(decision)


@pytest.mark.parametrize("model_bytes", (b"tiny", b"x" * 33))
def test_gguf_outside_configured_size_window_is_blocked(tmp_path: Path, model_bytes: bytes) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path, model_bytes=model_bytes)

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert "model_size_out_of_range" in blocker_codes(decision)


def test_model_with_correct_size_but_invalid_gguf_header_is_blocked(tmp_path: Path) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path, model_bytes=b"not-a-real-gguf")

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert "gguf_invalid" in blocker_codes(decision)


def test_publication_policy_requires_explicit_build_input_denylist(tmp_path: Path) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    unsafe_policy = publications.PublicationPolicy(
        repo_id=DEFAULT_REPO_ID,
        min_model_bytes=8,
        max_model_bytes=32,
        prohibited_identifiers=(),
    )

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=unsafe_policy,
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert "publication_policy_invalid" in blocker_codes(decision)


@pytest.mark.parametrize("tampered_surface", ("SHA256SUMS", "release-manifest.json"))
def test_artifact_hash_must_match_checksums_and_manifest(
    tmp_path: Path, tampered_surface: str
) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    path = tmp_path / tampered_surface
    path.write_bytes(path.read_bytes().replace(b"a", b"c", 1))

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert {"checksum_mismatch", "manifest_invalid"} & blocker_codes(decision)


@pytest.mark.parametrize(
    ("signature_name", "expected_code"),
    (
        ("release-manifest.sig", "release_manifest_signature_invalid"),
        ("benchmark-decision.sig", "benchmark_decision_signature_invalid"),
        ("benchmark-provenance.sig", "benchmark_provenance_signature_invalid"),
        ("smoke-test.sig", "smoke_test_signature_invalid"),
    ),
)
def test_invalid_signatures_block_publication(
    tmp_path: Path, signature_name: str, expected_code: str
) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    (tmp_path / signature_name).write_bytes(b"forged")

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert expected_code in blocker_codes(decision)


@pytest.mark.parametrize(
    ("file_name", "field", "value", "expected_code"),
    (
        ("benchmark-decision.json", "approved", False, "benchmark_not_approved"),
        ("benchmark-decision.json", "report_sha256", "0" * 64, "benchmark_link_mismatch"),
        (
            "benchmark-decision.json",
            "provenance_sha256",
            "0" * 64,
            "benchmark_link_mismatch",
        ),
        ("benchmark-report.json", "artifact_sha256", "0" * 64, "benchmark_link_mismatch"),
        (
            "benchmark-provenance.json",
            "artifact_sha256",
            "0" * 64,
            "benchmark_link_mismatch",
        ),
    ),
)
def test_approved_decision_report_and_provenance_must_bind_to_the_exact_artifact(
    tmp_path: Path,
    file_name: str,
    field: str,
    value: object,
    expected_code: str,
) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    path = tmp_path / file_name
    document = json.loads(path.read_text(encoding="utf-8"))
    document[field] = value
    write_json(path, document)

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert expected_code in blocker_codes(decision)


def test_signed_approved_flag_cannot_bypass_real_release_gates(tmp_path: Path) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    report_path = tmp_path / "benchmark-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["gate_input"]["deployable_candidate"]["scores"]["coding"] = 0.10
    report_payload = write_json(report_path, report)
    decision_path = tmp_path / "benchmark-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["report_sha256"] = sha256_bytes(report_payload)
    write_json(decision_path, decision)

    result = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert result.approved is False
    assert "release_gate_failed" in blocker_codes(result)


def test_candidate_cannot_weaken_pinned_release_policy(tmp_path: Path) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    report_path = tmp_path / "benchmark-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["gate_input"]["policy"]["required_baselines"] = ["reference"]
    report["gate_input"]["policy"]["required_score_targets"] = {"coding": 0.0}
    report_payload = write_json(report_path, report)
    decision_path = tmp_path / "benchmark-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["report_sha256"] = sha256_bytes(report_payload)
    write_json(decision_path, decision)

    result = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert result.approved is False
    assert "release_gate_failed" in blocker_codes(result)


def test_signed_raw_outputs_must_reproduce_reported_metrics(tmp_path: Path) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    raw_path = tmp_path / "benchmark-raw.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    raw_rows[0]["scores"] = release_scores(0.0)
    raw_payload = write_jsonl(raw_path, raw_rows)
    provenance_path = tmp_path / "benchmark-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["raw_output_sha256"] = sha256_bytes(raw_payload)
    provenance_payload = write_json(provenance_path, provenance)
    decision_path = tmp_path / "benchmark-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["provenance_sha256"] = sha256_bytes(provenance_payload)
    write_json(decision_path, decision)

    result = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert result.approved is False
    assert "benchmark_evidence_invalid" in blocker_codes(result)


@pytest.mark.parametrize(
    ("field", "value"),
    (("status", "failed"), ("artifact_sha256", "0" * 64), ("response", "")),
)
def test_smoke_transcript_must_show_a_real_pass_for_the_exact_gguf(
    tmp_path: Path, field: str, value: object
) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    path = tmp_path / "smoke-test.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document[field] = value
    write_json(path, document)

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert "smoke_test_invalid" in blocker_codes(decision)


@pytest.mark.parametrize("surface", ("README.md", "benchmark-report.json", "smoke-test.json"))
@pytest.mark.parametrize("identifier", ("forbidden-source", "FORBIDDEN-TEACHER"))
def test_product_facing_files_are_brand_only(tmp_path: Path, surface: str, identifier: str) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    path = tmp_path / surface
    path.write_text(path.read_text(encoding="utf-8") + identifier, encoding="utf-8")

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert "prohibited_identifier" in blocker_codes(decision)


def test_nested_extra_file_cannot_bypass_identifier_scan(tmp_path: Path) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    nested = tmp_path / "extras" / "metadata.txt"
    nested.parent.mkdir()
    nested.write_text("forbidden-source", encoding="utf-8")

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert "prohibited_identifier" in blocker_codes(decision)


def test_undeclared_file_blocks_publication_even_when_content_is_benign(tmp_path: Path) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    (tmp_path / "undeclared.bin").write_bytes(b"benign")

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert "undeclared_or_unsafe_file" in blocker_codes(decision)


@pytest.mark.parametrize("marker", ("${UNRESOLVED}", "NOT_MEASURED", "LEGAL_REVIEW_REQUIRED"))
def test_unresolved_template_and_legal_markers_block_publication(
    tmp_path: Path, marker: str
) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    (tmp_path / "THIRD_PARTY_NOTICES").write_text(marker, encoding="utf-8")

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert "unresolved_release_marker" in blocker_codes(decision)


@pytest.mark.parametrize(
    "secret",
    (
        "hf_" + "a" * 32,
        "Authorization" + ": Bearer should-never-be-published",
        "-----BEGIN " + "PRIVATE KEY-----",
    ),
)
def test_secret_material_anywhere_in_public_text_blocks_upload(tmp_path: Path, secret: str) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    (tmp_path / "README.md").write_text(f"# Metaflora Incubus v1\n{secret}\n", encoding="utf-8")

    decision = publications.evaluate_publication_bundle(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
    )

    assert decision.approved is False
    assert "secret_detected" in blocker_codes(decision)


def test_uploader_is_injected_and_receives_configured_repository_once(tmp_path: Path) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    uploader = RecordingUploader()
    target = "metaflora-app/incubus-release-candidate"

    result = publications.publish_to_huggingface(
        tmp_path,
        policy=policy(repo_id=target),
        signature_verifier=RecordingVerifier(),
        uploader=uploader,
    )

    assert result.uploaded is True
    assert result.repo_id == target
    assert len(uploader.calls) == 1
    assert uploader.calls[0]["repo_id"] == target
    assert Path(uploader.calls[0]["folder_path"]) == tmp_path
    assert "token" not in uploader.calls[0]
    assert len(uploader.verification_calls) == 1
    assert uploader.verification_calls[0]["repo_id"] == target
    assert uploader.private_calls == [{"repo_id": target}]
    assert uploader.public_calls == [{"repo_id": target}]


def test_upload_is_not_reported_as_success_until_remote_bytes_are_verified(
    tmp_path: Path,
) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    uploader = RecordingUploader(remote_verified=False)

    result = publications.publish_to_huggingface(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
        uploader=uploader,
    )

    assert result.uploaded is False
    assert result.decision.approved is False
    assert "remote_snapshot_unverified" in blocker_codes(result.decision)
    assert uploader.private_calls == [{"repo_id": DEFAULT_REPO_ID}]
    assert uploader.public_calls == []


@pytest.mark.parametrize(
    "break_bundle",
    (
        lambda root: (root / "LICENSE").unlink(),
        lambda root: (root / "release-manifest.sig").write_bytes(b"forged"),
        lambda root: (root / "README.md").write_text("hf_" + "secret-value" * 2, encoding="utf-8"),
    ),
)
def test_uploader_is_never_called_when_any_blocker_exists(tmp_path: Path, break_bundle) -> None:
    publications = publication_module()
    build_valid_bundle(tmp_path)
    break_bundle(tmp_path)
    uploader = RecordingUploader()

    result = publications.publish_to_huggingface(
        tmp_path,
        policy=policy(),
        signature_verifier=RecordingVerifier(),
        uploader=uploader,
    )

    assert result.uploaded is False
    assert result.decision.approved is False
    assert uploader.calls == []


def test_cli_exposes_explicit_hugging_face_bundle_repo_and_dry_run_options() -> None:
    from metaflora_incubus.cli import build_parser

    args = build_parser().parse_args(
        [
            "publish-hf",
            "--bundle",
            "/release/incubus-v1",
            "--repo-id",
            "metaflora-app/custom-incubus",
            "--dry-run",
        ]
    )

    assert args.command == "publish-hf"
    assert args.bundle == Path("/release/incubus-v1")
    assert args.repo_id == "metaflora-app/custom-incubus"
    assert args.dry_run is True
