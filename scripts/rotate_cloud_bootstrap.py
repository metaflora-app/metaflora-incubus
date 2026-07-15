#!/usr/bin/env python3
"""Re-encrypt the opaque Colab bootstrap after rotating cached Hub auth."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"INCUBUS1"
AAD = b"metaflora-incubus-cloud-bootstrap-v1"


def _read_nonempty(path: Path) -> str:
    value = path.read_text(encoding="utf-8")
    if not value.strip() or "\x00" in value:
        raise ValueError(f"private input is empty or invalid: {path.name}")
    return value


def _decode_key(path: Path) -> bytes:
    encoded = _read_nonempty(path).strip()
    try:
        key = base64.b64decode(encoded, altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("bootstrap key is not valid base64") from exc
    if len(key) != 32:
        raise ValueError("bootstrap key must contain 32 bytes")
    return key


def rotate_bootstrap(
    *,
    document_path: Path,
    token_path: Path,
    stored_tokens_path: Path,
    key_path: Path,
    output_path: Path,
) -> None:
    document = json.loads(document_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "hf_token",
        "hf_stored_tokens",
        "environment",
    }:
        raise ValueError("bootstrap document schema is invalid")
    environment = document["environment"]
    if not isinstance(environment, dict) or not environment:
        raise ValueError("bootstrap environment is invalid")
    rotated = {
        "hf_token": _read_nonempty(token_path),
        "hf_stored_tokens": _read_nonempty(stored_tokens_path),
        "environment": environment,
    }
    payload = json.dumps(rotated, sort_keys=True, separators=(",", ":")).encode()
    nonce = os.urandom(12)
    ciphertext = MAGIC + nonce + AESGCM(_decode_key(key_path)).encrypt(nonce, payload, AAD)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(ciphertext)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--token", type=Path, required=True)
    parser.add_argument("--stored-tokens", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    rotate_bootstrap(
        document_path=arguments.document,
        token_path=arguments.token,
        stored_tokens_path=arguments.stored_tokens,
        key_path=arguments.key,
        output_path=arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
