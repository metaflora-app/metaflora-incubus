"""Install one opaque cloud bootstrap without exposing private values."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import MutableMapping
from pathlib import Path

from metaflora_incubus.cloud_training import CloudConstraintError

BOOTSTRAP_ENV_NAMES = (
    "INCUBUS_CHECKPOINT_HMAC_KEY",
    "INCUBUS_SOURCE_REPO",
    "INCUBUS_SOURCE_REVISION",
    "INCUBUS_DATASET_REPO",
    "INCUBUS_DATASET_REVISION",
    "INCUBUS_DATASET_SHA256",
    "INCUBUS_PARAMETER_COUNT",
)


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

    cache = home / ".cache" / "huggingface"
    _private_atomic_write(cache / "token", token)
    _private_atomic_write(cache / "stored_tokens", stored_tokens)
    for name in BOOTSTRAP_ENV_NAMES:
        environment[name] = values[name]
    environment.pop("HF_TOKEN", None)
    return parameter_count
