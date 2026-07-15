"""Install one opaque cloud bootstrap without exposing private values."""

from __future__ import annotations

import base64
import binascii
import json
import os
import tempfile
from collections.abc import MutableMapping
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from metaflora_incubus.cloud_training import CloudConstraintError

BOOTSTRAP_ENV_NAMES = (
    "INCUBUS_CHECKPOINT_HMAC_KEY",
    "INCUBUS_SOURCE_REPO",
    "INCUBUS_SOURCE_REVISION",
    "INCUBUS_DATASET_REPO",
    "INCUBUS_DATASET_REVISION",
    "INCUBUS_DATASET_SHA256",
    "INCUBUS_PARAMETER_COUNT",
    "INCUBUS_BENCHMARK_SIGNING_KEY",
)
_ENCRYPTED_BOOTSTRAP_MAGIC = b"INCUBUS1"
_ENCRYPTED_BOOTSTRAP_AAD = b"metaflora-incubus-cloud-bootstrap-v1"


def decrypt_cloud_bootstrap(ciphertext: bytes, encoded_key: str) -> bytes:
    """Decrypt the public opaque bootstrap with a short private Colab key."""
    if not isinstance(ciphertext, bytes) or not ciphertext.startswith(_ENCRYPTED_BOOTSTRAP_MAGIC):
        raise CloudConstraintError("encrypted cloud bootstrap is invalid")
    if not isinstance(encoded_key, str) or len(encoded_key.strip()) > 128:
        raise CloudConstraintError("cloud bootstrap key is invalid")
    try:
        key = base64.b64decode(encoded_key.strip(), altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CloudConstraintError("cloud bootstrap key is not valid base64") from exc
    if len(key) != 32:
        raise CloudConstraintError("cloud bootstrap key must contain 32 bytes")
    body = ciphertext[len(_ENCRYPTED_BOOTSTRAP_MAGIC) :]
    if len(body) < 13:
        raise CloudConstraintError("encrypted cloud bootstrap is truncated")
    nonce, encrypted = body[:12], body[12:]
    try:
        return AESGCM(key).decrypt(nonce, encrypted, _ENCRYPTED_BOOTSTRAP_AAD)
    except Exception as exc:
        raise CloudConstraintError("cloud bootstrap key did not decrypt the payload") from exc


def _private_atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except Exception:
            os.close(descriptor)
            raise
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def install_cloud_bootstrap(
    payload: bytes, *, home: Path, environment: MutableMapping[str, str]
) -> int:
    if not payload or len(payload) > 1024 * 1024:
        raise CloudConstraintError("cloud bootstrap has an invalid size")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloudConstraintError("cloud bootstrap is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "hf_token",
        "hf_stored_tokens",
        "environment",
    }:
        raise CloudConstraintError("cloud bootstrap schema is invalid")
    values = document["environment"]
    if not isinstance(values, dict) or set(values) != set(BOOTSTRAP_ENV_NAMES):
        raise CloudConstraintError("cloud bootstrap environment is invalid")
    for name in BOOTSTRAP_ENV_NAMES:
        value = values[name]
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise CloudConstraintError("cloud bootstrap environment contains an invalid value")
    token = document["hf_token"]
    stored_tokens = document["hf_stored_tokens"]
    if not isinstance(token, str) or not token.strip() or "\x00" in token:
        raise CloudConstraintError("cached Hugging Face token is invalid")
    if not isinstance(stored_tokens, str) or not stored_tokens.strip() or "\x00" in stored_tokens:
        raise CloudConstraintError("cached Hugging Face stored tokens are invalid")
    try:
        parameter_count = int(values["INCUBUS_PARAMETER_COUNT"])
    except ValueError as exc:
        raise CloudConstraintError("parameter count must be an integer") from exc
    if parameter_count <= 0 or parameter_count > 7_500_000_000:
        raise CloudConstraintError("parameter count is outside the compact cloud profile")
    if not _benchmark_signing_key_matches_production(values["INCUBUS_BENCHMARK_SIGNING_KEY"]):
        raise CloudConstraintError("benchmark signing key does not match the production key")

    cache = home / ".cache" / "huggingface"
    _private_atomic_write(cache / "token", token)
    _private_atomic_write(cache / "stored_tokens", stored_tokens)
    for name in BOOTSTRAP_ENV_NAMES:
        environment[name] = values[name]
    environment.pop("HF_TOKEN", None)
    return parameter_count


def _benchmark_signing_key_matches_production(encoded_key: str) -> bool:
    from metaflora_incubus.gguf_benchmark_runner import PRODUCTION_ATTESTATION_PUBLIC_KEY

    try:
        private_key = Ed25519PrivateKey.from_private_bytes(
            base64.urlsafe_b64decode(encoded_key.encode("ascii"))
        )
        actual = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        expected = base64.urlsafe_b64decode(PRODUCTION_ATTESTATION_PUBLIC_KEY.encode("ascii"))
    except (ValueError, binascii.Error, UnicodeEncodeError):
        return False
    return actual == expected
