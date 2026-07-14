"""Atomic, reversible OpenCode provider configuration."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path


class OpenCodeConfigError(ValueError):
    pass


class ProviderOwnershipError(OpenCodeConfigError):
    pass


def merge_provider_config(
    config_path: Path,
    *,
    provider_id: str,
    provider_config: dict,
) -> Path:
    original = config_path.read_bytes() if config_path.exists() else b"{}\n"
    try:
        document = _load_jsonc(original)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise OpenCodeConfigError("OpenCode configuration is not valid JSON/JSONC") from exc
    if not isinstance(document, dict):
        raise OpenCodeConfigError("OpenCode configuration root must be an object")
    provider = document.get("provider", {})
    if not isinstance(provider, dict):
        raise OpenCodeConfigError("OpenCode provider field must be an object")

    backup_path = config_path.with_name(f"{config_path.name}.bak.incubus")
    if provider.get(provider_id) == provider_config:
        return backup_path

    merged_provider = {**provider, provider_id: provider_config}
    merged_document = {**document, "provider": merged_provider}
    rendered = (json.dumps(merged_document, indent=2, ensure_ascii=False) + "\n").encode()

    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        _atomic_write(backup_path, original)
    _atomic_write(config_path, rendered)
    return backup_path


def remove_provider_config(config_path: Path, *, provider_id: str) -> None:
    if not config_path.exists():
        return
    original = config_path.read_bytes()
    try:
        document = _load_jsonc(original)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise OpenCodeConfigError("OpenCode configuration is not valid JSON/JSONC") from exc
    if not isinstance(document, dict):
        raise OpenCodeConfigError("OpenCode configuration root must be an object")
    provider = document.get("provider")
    if not isinstance(provider, dict) or provider_id not in provider:
        return
    remaining = {key: value for key, value in provider.items() if key != provider_id}
    updated = {**document, "provider": remaining}
    rendered = (json.dumps(updated, indent=2, ensure_ascii=False) + "\n").encode()
    _atomic_write(config_path, rendered)


def revert_owned_provider_config(
    config_path: Path,
    *,
    provider_id: str,
    installed_provider_config: object,
    previous_provider_config: object | None,
) -> None:
    """Revert only the exact provider value written by this installation."""
    if not config_path.exists():
        return
    original = config_path.read_bytes()
    try:
        document = _load_jsonc(original)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise OpenCodeConfigError("OpenCode configuration is not valid JSON/JSONC") from exc
    if not isinstance(document, dict):
        raise OpenCodeConfigError("OpenCode configuration root must be an object")
    providers = document.get("provider")
    if not isinstance(providers, dict):
        raise ProviderOwnershipError("owned OpenCode provider is missing")
    current = providers.get(provider_id)
    if current != installed_provider_config:
        raise ProviderOwnershipError(
            "OpenCode provider changed after installation; refusing to overwrite it"
        )
    if previous_provider_config is None:
        updated_providers = {key: value for key, value in providers.items() if key != provider_id}
    else:
        restored = (
            dict(previous_provider_config)
            if isinstance(previous_provider_config, Mapping)
            else previous_provider_config
        )
        updated_providers = {**providers, provider_id: restored}
    updated = {**document, "provider": updated_providers}
    rendered = (json.dumps(updated, indent=2, ensure_ascii=False) + "\n").encode()
    _atomic_write(config_path, rendered)


def restore_provider_backup(config_path: Path, backup_path: Path) -> None:
    if not backup_path.exists():
        raise OpenCodeConfigError("OpenCode backup does not exist")
    _atomic_write(config_path, backup_path.read_bytes())
    backup_path.unlink()


def _load_jsonc(payload: bytes) -> object:
    text = payload.decode("utf-8")
    without_comments = _strip_jsonc_comments(text)
    without_trailing_commas = _strip_trailing_commas(without_comments)
    return json.loads(without_trailing_commas)


def _strip_jsonc_comments(text: str) -> str:
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        character = text[index]
        next_character = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            output.append(character)
            index += 1
            continue
        if character == "/" and next_character == "/":
            output.extend((" ", " "))
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                output.append(" ")
                index += 1
            continue
        if character == "/" and next_character == "*":
            end = text.find("*/", index + 2)
            if end < 0:
                raise json.JSONDecodeError("unterminated comment", text, index)
            output.extend("\n" if value == "\n" else " " for value in text[index : end + 2])
            index = end + 2
            continue
        output.append(character)
        index += 1
    return "".join(output)


def _strip_trailing_commas(text: str) -> str:
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        character = text[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            output.append(character)
            index += 1
            continue
        if character == ",":
            cursor = index + 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] in "}]":
                index += 1
                continue
        output.append(character)
        index += 1
    return "".join(output)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
