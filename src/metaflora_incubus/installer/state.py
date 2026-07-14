"""Immutable installer state and scope-limited uninstall operations."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from metaflora_incubus.installer.opencode_config import (
    ProviderOwnershipError,
    _atomic_write,
    revert_owned_provider_config,
)

__all__ = ["ProviderOwnershipError"]

PRODUCT_PROVIDER_ID = "metaflora-incubus"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class UnsafeManagedPathError(ValueError):
    pass


@dataclass(frozen=True)
class InstallState:
    schema_version: int
    release_id: str
    manifest_sha256: str
    artifact_id: str
    install_root: Path
    managed_paths: tuple[Path, ...]
    provider_id: str
    previous_provider_config: Mapping[str, object] | None
    installed_provider_config: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "install_root", Path(self.install_root))
        object.__setattr__(self, "managed_paths", tuple(Path(path) for path in self.managed_paths))
        if self.previous_provider_config is not None:
            object.__setattr__(
                self,
                "previous_provider_config",
                MappingProxyType(dict(self.previous_provider_config)),
            )
        object.__setattr__(
            self,
            "installed_provider_config",
            MappingProxyType(dict(self.installed_provider_config)),
        )


def save_install_state(path: Path, state: InstallState) -> None:
    _validate_state(state)
    payload = {
        "schema_version": state.schema_version,
        "release_id": state.release_id,
        "manifest_sha256": state.manifest_sha256,
        "artifact_id": state.artifact_id,
        "install_root": str(state.install_root),
        "managed_paths": [str(managed_path) for managed_path in state.managed_paths],
        "provider_id": state.provider_id,
        "previous_provider_config": (
            None if state.previous_provider_config is None else dict(state.previous_provider_config)
        ),
        "installed_provider_config": dict(state.installed_provider_config),
    }
    rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() and path.read_bytes() == rendered:
        return
    _atomic_write(path, rendered)


def load_install_state(path: Path) -> InstallState:
    document = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "release_id",
        "manifest_sha256",
        "artifact_id",
        "install_root",
        "managed_paths",
        "provider_id",
        "previous_provider_config",
        "installed_provider_config",
    }
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise ValueError("invalid install-state schema")
    if document.get("schema_version") != 1:
        raise ValueError("unsupported install-state schema")
    if not isinstance(document["managed_paths"], list) or not all(
        isinstance(item, str) for item in document["managed_paths"]
    ):
        raise ValueError("managed_paths must be a list of strings")
    previous = document["previous_provider_config"]
    installed = document["installed_provider_config"]
    if previous is not None and not isinstance(previous, dict):
        raise ValueError("previous provider config must be an object or null")
    if not isinstance(installed, dict):
        raise ValueError("installed provider config must be an object")
    state = InstallState(
        schema_version=1,
        release_id=str(document["release_id"]),
        manifest_sha256=str(document["manifest_sha256"]),
        artifact_id=str(document["artifact_id"]),
        install_root=Path(document["install_root"]),
        managed_paths=tuple(Path(item) for item in document["managed_paths"]),
        provider_id=str(document["provider_id"]),
        previous_provider_config=previous,
        installed_provider_config=installed,
    )
    _validate_state(state)
    return state


def uninstall_from_state(*, state_path: Path, opencode_config_path: Path) -> None:
    state = load_install_state(state_path)
    install_root = state.install_root.resolve(strict=False)
    paths = state.managed_paths
    if state_path.resolve(strict=False).parent != install_root:
        raise UnsafeManagedPathError("install state is outside its install root")
    if any(not _safe_managed_leaf(path, install_root) for path in paths):
        raise UnsafeManagedPathError("managed path escapes install root")

    revert_owned_provider_config(
        opencode_config_path,
        provider_id=state.provider_id,
        installed_provider_config=state.installed_provider_config,
        previous_provider_config=state.previous_provider_config,
    )
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            path.rmdir()
        else:
            path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_managed_leaf(path: Path, root: Path) -> bool:
    if not path.is_absolute():
        return False
    # Resolve the parent to detect traversal and symlinked parents, but never resolve
    # the leaf: if it is a symlink, uninstall must unlink the link itself.
    resolved_parent = path.parent.resolve(strict=False)
    lexical_leaf = resolved_parent / path.name
    return _is_within(lexical_leaf, root) and lexical_leaf != root


def _validate_state(state: InstallState) -> None:
    if state.schema_version != 1:
        raise ValueError("unsupported install-state schema")
    if not _IDENTIFIER.fullmatch(state.release_id):
        raise ValueError("invalid release id")
    if not _IDENTIFIER.fullmatch(state.artifact_id):
        raise ValueError("invalid artifact id")
    if not _SHA256.fullmatch(state.manifest_sha256):
        raise ValueError("invalid manifest SHA-256")
    if state.provider_id != PRODUCT_PROVIDER_ID:
        raise ValueError("invalid provider id")
    if not state.install_root.is_absolute():
        raise ValueError("install root must be absolute")
    if not state.installed_provider_config:
        raise ValueError("installed provider config must not be empty")
