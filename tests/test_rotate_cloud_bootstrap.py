from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path

import pytest

from metaflora_incubus.cloud_bootstrap import decrypt_cloud_bootstrap

SCRIPT = Path(__file__).parents[1] / "scripts" / "rotate_cloud_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("rotate_cloud_bootstrap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rotate_bootstrap_preserves_environment_and_replaces_auth(tmp_path: Path) -> None:
    document = tmp_path / "document.json"
    token = tmp_path / "token"
    stored = tmp_path / "stored_tokens"
    key = tmp_path / "key"
    output = tmp_path / "bootstrap.enc"
    environment = {"INCUBUS_PARAMETER_COUNT": "7000000000"}
    document.write_text(
        json.dumps(
            {"hf_token": "old", "hf_stored_tokens": "old-cache", "environment": environment}
        ),
        encoding="utf-8",
    )
    token.write_text("fresh-token", encoding="utf-8")
    stored.write_text("fresh-cache", encoding="utf-8")
    encoded_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
    key.write_text(encoded_key, encoding="utf-8")

    MODULE.rotate_bootstrap(
        document_path=document,
        token_path=token,
        stored_tokens_path=stored,
        key_path=key,
        output_path=output,
    )

    rotated = json.loads(decrypt_cloud_bootstrap(output.read_bytes(), encoded_key))
    assert rotated == {
        "hf_token": "fresh-token",
        "hf_stored_tokens": "fresh-cache",
        "environment": environment,
    }
    assert b"fresh-token" not in output.read_bytes()


def test_rotate_bootstrap_rejects_wrong_key_size(tmp_path: Path) -> None:
    document = tmp_path / "document.json"
    document.write_text(
        json.dumps({"hf_token": "old", "hf_stored_tokens": "old", "environment": {"x": "y"}}),
        encoding="utf-8",
    )
    token = tmp_path / "token"
    stored = tmp_path / "stored"
    key = tmp_path / "key"
    token.write_text("fresh", encoding="utf-8")
    stored.write_text("fresh", encoding="utf-8")
    key.write_text(base64.urlsafe_b64encode(b"short").decode(), encoding="utf-8")

    with pytest.raises(ValueError, match="32 bytes"):
        MODULE.rotate_bootstrap(
            document_path=document,
            token_path=token,
            stored_tokens_path=stored,
            key_path=key,
            output_path=tmp_path / "out.enc",
        )
