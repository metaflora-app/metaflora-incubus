from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _state_module():
    return importlib.import_module("metaflora_incubus.installer.state")


def _install_state(states, install_root: Path):
    return states.InstallState(
        schema_version=1,
        release_id="v1.0.0",
        manifest_sha256="a" * 64,
        artifact_id="incubus-v1-compact",
        install_root=install_root,
        managed_paths=(
            install_root / "bin" / "incubus-runtime",
            install_root / "models" / "incubus-v1.gguf",
        ),
        provider_id="metaflora-incubus",
        previous_provider_config=None,
        installed_provider_config={"name": "Metaflora Incubus v1"},
    )


def test_install_state_round_trip_is_idempotent(tmp_path: Path) -> None:
    states = _state_module()
    state_path = tmp_path / "state.json"
    state = _install_state(states, tmp_path / "incubus")

    states.save_install_state(state_path, state)
    first_write = state_path.read_bytes()
    states.save_install_state(state_path, state)

    assert state_path.read_bytes() == first_write
    assert states.load_install_state(state_path) == state


def test_uninstall_removes_only_managed_files_and_owned_provider(tmp_path: Path) -> None:
    states = _state_module()
    install_root = tmp_path / "incubus"
    managed_runtime = install_root / "bin" / "incubus-runtime"
    managed_model = install_root / "models" / "incubus-v1.gguf"
    foreign_note = install_root / "notes.txt"
    for path in (managed_runtime, managed_model, foreign_note):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    state_path = install_root / "state.json"
    states.save_install_state(state_path, _install_state(states, install_root))

    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        json.dumps(
            {
                "provider": {
                    "metaflora-incubus": {"name": "Metaflora Incubus v1"},
                    "foreign": {"name": "Foreign"},
                },
                "theme": "system",
            }
        ),
        encoding="utf-8",
    )

    states.uninstall_from_state(state_path=state_path, opencode_config_path=config_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config == {"provider": {"foreign": {"name": "Foreign"}}, "theme": "system"}
    assert not managed_runtime.exists()
    assert not managed_model.exists()
    assert not state_path.exists()
    assert foreign_note.read_text(encoding="utf-8") == "notes.txt"


def test_uninstall_rejects_managed_path_outside_install_root(tmp_path: Path) -> None:
    states = _state_module()
    install_root = tmp_path / "incubus"
    install_root.mkdir()
    foreign_file = tmp_path / "foreign.txt"
    foreign_file.write_text("keep", encoding="utf-8")
    state_path = install_root / "state.json"
    unsafe_state = states.InstallState(
        schema_version=1,
        release_id="v1.0.0",
        manifest_sha256="a" * 64,
        artifact_id="incubus-v1-compact",
        install_root=install_root,
        managed_paths=(foreign_file,),
        provider_id="metaflora-incubus",
        previous_provider_config=None,
        installed_provider_config={"name": "Metaflora Incubus v1"},
    )
    states.save_install_state(state_path, unsafe_state)
    config_path = tmp_path / "opencode.json"
    config_path.write_text('{"provider":{"foreign":{"name":"Foreign"}}}', encoding="utf-8")

    with pytest.raises(states.UnsafeManagedPathError):
        states.uninstall_from_state(state_path=state_path, opencode_config_path=config_path)

    assert foreign_file.read_text(encoding="utf-8") == "keep"
    assert state_path.exists()


def test_uninstall_removes_managed_symlink_without_deleting_its_target(
    tmp_path: Path,
) -> None:
    states = _state_module()
    install_root = tmp_path / "incubus"
    install_root.mkdir()
    foreign_file = install_root / "user-owned.txt"
    foreign_file.write_text("keep", encoding="utf-8")
    managed_link = install_root / "managed-link"
    managed_link.symlink_to(foreign_file)
    state_path = install_root / "state.json"
    state = states.InstallState(
        schema_version=1,
        release_id="v1.0.0",
        manifest_sha256="a" * 64,
        artifact_id="incubus-v1-compact",
        install_root=install_root,
        managed_paths=(managed_link,),
        provider_id="metaflora-incubus",
        previous_provider_config=None,
        installed_provider_config={"name": "Metaflora Incubus v1"},
    )
    states.save_install_state(state_path, state)
    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        '{"provider":{"metaflora-incubus":{"name":"Metaflora Incubus v1"}}}',
        encoding="utf-8",
    )

    states.uninstall_from_state(state_path=state_path, opencode_config_path=config_path)

    assert not managed_link.exists()
    assert not managed_link.is_symlink()
    assert foreign_file.read_text(encoding="utf-8") == "keep"


def test_uninstall_restores_preexisting_provider_value(tmp_path: Path) -> None:
    states = _state_module()
    install_root = tmp_path / "incubus"
    install_root.mkdir()
    previous = {"name": "User-owned provider", "options": {"baseURL": "https://example.org"}}
    installed = {"name": "Metaflora Incubus v1"}
    state = states.InstallState(
        schema_version=1,
        release_id="v1.0.0",
        manifest_sha256="a" * 64,
        artifact_id="incubus-v1-compact",
        install_root=install_root,
        managed_paths=(),
        provider_id="metaflora-incubus",
        previous_provider_config=previous,
        installed_provider_config=installed,
    )
    state_path = install_root / "state.json"
    states.save_install_state(state_path, state)
    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        json.dumps({"provider": {"metaflora-incubus": installed}}), encoding="utf-8"
    )

    states.uninstall_from_state(state_path=state_path, opencode_config_path=config_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["provider"]["metaflora-incubus"] == previous


def test_uninstall_refuses_to_overwrite_provider_modified_after_install(
    tmp_path: Path,
) -> None:
    states = _state_module()
    install_root = tmp_path / "incubus"
    install_root.mkdir()
    installed = {"name": "Metaflora Incubus v1"}
    state = states.InstallState(
        schema_version=1,
        release_id="v1.0.0",
        manifest_sha256="a" * 64,
        artifact_id="incubus-v1-compact",
        install_root=install_root,
        managed_paths=(),
        provider_id="metaflora-incubus",
        previous_provider_config=None,
        installed_provider_config=installed,
    )
    state_path = install_root / "state.json"
    states.save_install_state(state_path, state)
    config_path = tmp_path / "opencode.json"
    modified = {"name": "Changed by user"}
    config_path.write_text(
        json.dumps({"provider": {"metaflora-incubus": modified}}), encoding="utf-8"
    )

    with pytest.raises(states.ProviderOwnershipError):
        states.uninstall_from_state(state_path=state_path, opencode_config_path=config_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["provider"]["metaflora-incubus"] == modified
    assert state_path.exists()
