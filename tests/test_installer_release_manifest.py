from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest


def _manifest_module():
    return importlib.import_module("metaflora_incubus.installer.release_manifest")


def _manifest_payload() -> bytes:
    return (
        b'{"schema_version":1,"release_id":"v1.0.0","artifacts":['
        b'{"artifact_id":"incubus-v1-compact","platform":"darwin",'
        b'"architecture":"arm64","download_url":"https://releases.metaflora.ai/'
        b'incubus/v1/compact.tar.zst","sha256":"'
        + b"a"
        * 64
        + b'","download_bytes":1,"required_ram_bytes":1,"required_disk_bytes":1,'
        b'"priority":10}]}'
    )


def test_accepts_manifest_only_after_hash_and_signature_are_verified() -> None:
    manifests = _manifest_module()
    payload = _manifest_payload()
    calls: list[tuple[bytes, bytes, bytes]] = []

    def verifier(public_key: bytes, signature: bytes, signed_payload: bytes) -> bool:
        calls.append((public_key, signature, signed_payload))
        return True

    result = manifests.verify_release_manifest(
        payload,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        signature=b"detached-signature",
        public_key=b"embedded-release-public-key",
        signature_verifier=verifier,
    )

    assert result.release_id == "v1.0.0"
    assert result.artifacts[0].artifact_id == "incubus-v1-compact"
    assert calls == [(b"embedded-release-public-key", b"detached-signature", payload)]


def test_rejects_hash_mismatch_before_signature_verification() -> None:
    manifests = _manifest_module()
    verifier_called = False

    def verifier(_public_key: bytes, _signature: bytes, _payload: bytes) -> bool:
        nonlocal verifier_called
        verifier_called = True
        return True

    with pytest.raises(manifests.ManifestIntegrityError):
        manifests.verify_release_manifest(
            _manifest_payload(),
            expected_sha256="0" * 64,
            signature=b"detached-signature",
            public_key=b"embedded-release-public-key",
            signature_verifier=verifier,
        )

    assert verifier_called is False


def test_rejects_manifest_with_invalid_detached_signature() -> None:
    manifests = _manifest_module()
    payload = _manifest_payload()

    with pytest.raises(manifests.ManifestSignatureError):
        manifests.verify_release_manifest(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            signature=b"forged-signature",
            public_key=b"embedded-release-public-key",
            signature_verifier=lambda *_args: False,
        )


def test_rejects_downloaded_artifact_when_its_hash_differs(tmp_path: Path) -> None:
    manifests = _manifest_module()
    artifact = tmp_path / "incubus.tar.zst"
    artifact.write_bytes(b"tampered artifact")

    with pytest.raises(manifests.ArtifactIntegrityError):
        manifests.verify_artifact_sha256(artifact, expected_sha256="f" * 64)


def test_accepts_downloaded_artifact_with_expected_hash(tmp_path: Path) -> None:
    manifests = _manifest_module()
    artifact = tmp_path / "incubus.tar.zst"
    artifact.write_bytes(b"trusted artifact")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert manifests.verify_artifact_sha256(artifact, expected_sha256=expected) is None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact_id", "../../escape"),
        ("platform", "plan9"),
        ("architecture", "mips"),
        ("priority", True),
    ),
)
def test_rejects_unsafe_or_ambiguous_artifact_metadata(field: str, value: object) -> None:
    manifests = _manifest_module()
    document = json.loads(_manifest_payload())
    document["artifacts"][0][field] = value
    payload = json.dumps(document).encode()

    with pytest.raises(manifests.ManifestIntegrityError):
        manifests.verify_release_manifest(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            signature=b"signature",
            public_key=b"key",
            signature_verifier=lambda *_args: True,
        )


def test_rejects_duplicate_artifact_ids() -> None:
    manifests = _manifest_module()
    document = json.loads(_manifest_payload())
    document["artifacts"].append(dict(document["artifacts"][0]))
    payload = json.dumps(document).encode()

    with pytest.raises(manifests.ManifestIntegrityError):
        manifests.verify_release_manifest(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            signature=b"signature",
            public_key=b"key",
            signature_verifier=lambda *_args: True,
        )
