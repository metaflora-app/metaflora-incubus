from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _opencode_module():
    return importlib.import_module("metaflora_incubus.installer.opencode_config")


def _incubus_provider() -> dict:
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Metaflora Incubus v1",
        "options": {"baseURL": "http://127.0.0.1:8080/v1"},
        "models": {
            "metaflora-incubus-v1": {
                "name": "Metaflora Incubus v1",
            }
        },
    }


def test_merges_provider_into_jsonc_and_creates_byte_exact_backup(tmp_path: Path) -> None:
    opencode = _opencode_module()
    config_path = tmp_path / "opencode.jsonc"
    original = b"""{
  // Keep the user's existing provider.
  "provider": {
    "foreign": {"name": "Foreign",},
  },
  "theme": "system",
}
"""
    config_path.write_bytes(original)

    backup_path = opencode.merge_provider_config(
        config_path,
        provider_id="metaflora-incubus",
        provider_config=_incubus_provider(),
    )

    merged = json.loads(config_path.read_text(encoding="utf-8"))
    assert merged["provider"]["foreign"] == {"name": "Foreign"}
    assert merged["provider"]["metaflora-incubus"] == _incubus_provider()
    assert merged["theme"] == "system"
    assert backup_path.read_bytes() == original


def test_repeated_merge_is_idempotent_and_does_not_replace_original_backup(
    tmp_path: Path,
) -> None:
    opencode = _opencode_module()
    config_path = tmp_path / "opencode.json"
    config_path.write_text('{"provider":{"foreign":{"name":"Foreign"}}}\n', encoding="utf-8")

    backup_path = opencode.merge_provider_config(
        config_path,
        provider_id="metaflora-incubus",
        provider_config=_incubus_provider(),
    )
    first_config = config_path.read_bytes()
    first_backup = backup_path.read_bytes()

    second_backup = opencode.merge_provider_config(
        config_path,
        provider_id="metaflora-incubus",
        provider_config=_incubus_provider(),
    )

    assert config_path.read_bytes() == first_config
    assert second_backup == backup_path
    assert backup_path.read_bytes() == first_backup


def test_invalid_jsonc_leaves_original_config_untouched(tmp_path: Path) -> None:
    opencode = _opencode_module()
    config_path = tmp_path / "opencode.jsonc"
    original = b'{"provider": { this is not jsonc } }\n'
    config_path.write_bytes(original)

    with pytest.raises(opencode.OpenCodeConfigError):
        opencode.merge_provider_config(
            config_path,
            provider_id="metaflora-incubus",
            provider_config=_incubus_provider(),
        )

    assert config_path.read_bytes() == original
    assert not list(tmp_path.glob("opencode.jsonc.bak*"))


def test_backup_can_restore_config_after_a_later_install_failure(tmp_path: Path) -> None:
    opencode = _opencode_module()
    config_path = tmp_path / "opencode.json"
    original = b'{"provider":{"foreign":{"name":"Foreign"}}}\n'
    config_path.write_bytes(original)
    backup_path = opencode.merge_provider_config(
        config_path,
        provider_id="metaflora-incubus",
        provider_config=_incubus_provider(),
    )

    opencode.restore_provider_backup(config_path, backup_path)

    assert config_path.read_bytes() == original
    assert not backup_path.exists()


def test_comment_removal_cannot_turn_invalid_jsonc_into_a_different_value(
    tmp_path: Path,
) -> None:
    opencode = _opencode_module()
    config_path = tmp_path / "opencode.jsonc"
    original = b'{"value": 1/**/2}'
    config_path.write_bytes(original)

    with pytest.raises(opencode.OpenCodeConfigError):
        opencode.merge_provider_config(
            config_path,
            provider_id="metaflora-incubus",
            provider_config=_incubus_provider(),
        )

    assert config_path.read_bytes() == original
