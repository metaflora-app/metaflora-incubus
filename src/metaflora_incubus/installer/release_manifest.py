"""Integrity checks and parsing for signed release manifests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from metaflora_incubus.installer.artifacts import ReleaseArtifact

SignatureVerifier = Callable[[bytes, bytes, bytes], bool]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUPPORTED_PLATFORMS = {"darwin", "linux", "windows"}
_SUPPORTED_ARCHITECTURES = {"arm64", "amd64"}
_TRUSTED_ARTIFACT_HOSTS = {"releases.metaflora.ai", "huggingface.co"}
_MAX_ARTIFACT_BYTES = 20 * 1024**3


class ManifestIntegrityError(ValueError):
    pass


class ManifestSignatureError(ValueError):
    pass


class ArtifactIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class ReleaseManifest:
    schema_version: int
    release_id: str
    artifacts: tuple[ReleaseArtifact, ...]


def verify_release_manifest(
    payload: bytes,
    *,
    expected_sha256: str,
    signature: bytes,
    public_key: bytes,
    signature_verifier: SignatureVerifier,
) -> ReleaseManifest:
    actual_hash = hashlib.sha256(payload).hexdigest()
    if not _valid_sha256(expected_sha256) or actual_hash != expected_sha256.lower():
        raise ManifestIntegrityError("release manifest SHA-256 mismatch")
    if not signature_verifier(public_key, signature, payload):
        raise ManifestSignatureError("release manifest signature is invalid")
    try:
        document = json.loads(payload)
        manifest = _parse_manifest(document)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestIntegrityError("release manifest has an invalid schema") from exc
    return manifest


def verify_artifact_sha256(path: Path, *, expected_sha256: str) -> None:
    if not _valid_sha256(expected_sha256):
        raise ArtifactIntegrityError("artifact SHA-256 is invalid")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256.lower():
        raise ArtifactIntegrityError("downloaded artifact SHA-256 mismatch")


def _parse_manifest(document: object) -> ReleaseManifest:
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    release_id = document["release_id"]
    raw_artifacts = document["artifacts"]
    if not isinstance(release_id, str) or not _SAFE_ID.fullmatch(release_id):
        raise ValueError("missing release id")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("manifest has no artifacts")
    artifacts = tuple(_parse_artifact(item) for item in raw_artifacts)
    artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("duplicate artifact id")
    return ReleaseManifest(schema_version=1, release_id=release_id, artifacts=artifacts)


def _parse_artifact(document: object) -> ReleaseArtifact:
    if not isinstance(document, dict):
        raise TypeError("artifact must be an object")
    artifact = ReleaseArtifact(
        artifact_id=_safe_id(document["artifact_id"]),
        platform=_allowed_value(document["platform"], _SUPPORTED_PLATFORMS),
        architecture=_allowed_value(document["architecture"], _SUPPORTED_ARCHITECTURES),
        download_url=_https_url(document["download_url"]),
        sha256=_sha256(document["sha256"]),
        download_bytes=_positive_int(document["download_bytes"]),
        required_ram_bytes=_positive_int(document["required_ram_bytes"]),
        required_disk_bytes=_positive_int(document["required_disk_bytes"]),
        priority=_bounded_int(document["priority"], minimum=-100, maximum=100),
    )
    return artifact


def _nonempty_string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected a non-empty string")
    return value


def _https_url(value: object) -> str:
    url = _nonempty_string(value)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("artifact URL must be an HTTPS URL without credentials")
    if parsed.hostname.lower() not in _TRUSTED_ARTIFACT_HOSTS:
        raise ValueError("artifact URL host is not trusted")
    return url


def _sha256(value: object) -> str:
    digest = _nonempty_string(value).lower()
    if not _valid_sha256(digest):
        raise ValueError("invalid SHA-256")
    return digest


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _positive_int(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or value > _MAX_ARTIFACT_BYTES
    ):
        raise ValueError("expected a positive integer")
    return value


def _safe_id(value: object) -> str:
    identifier = _nonempty_string(value)
    if not _SAFE_ID.fullmatch(identifier):
        raise ValueError("unsafe identifier")
    return identifier


def _allowed_value(value: object, allowed: set[str]) -> str:
    normalized = _nonempty_string(value)
    if normalized not in allowed:
        raise ValueError("unsupported artifact target")
    return normalized


def _bounded_int(value: object, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError("integer is outside allowed bounds")
    return value
