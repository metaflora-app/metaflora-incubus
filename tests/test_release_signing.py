from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from metaflora_incubus.release_signing import ReleaseKeyError, load_release_private_key


def _write_key(path: Path, *, mode: int = 0o600) -> None:
    path.write_text(base64.b64encode(b"k" * 32).decode("ascii"), encoding="ascii")
    path.chmod(mode)


def test_loads_private_key_only_from_private_file_outside_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    key = tmp_path / "release.key"
    _write_key(key)

    loaded = load_release_private_key(key, repository_root=repository)

    assert len(loaded.private_bytes_raw()) == 32


@pytest.mark.parametrize("mode", (0o604, 0o640, 0o644))
def test_rejects_private_key_with_group_or_other_permissions(tmp_path: Path, mode: int) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    key = tmp_path / "release.key"
    _write_key(key, mode=mode)

    with pytest.raises(ReleaseKeyError, match="0600"):
        load_release_private_key(key, repository_root=repository)


def test_rejects_symlinked_or_repository_local_private_key(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside.key"
    _write_key(outside)
    symlink = tmp_path / "linked.key"
    os.symlink(outside, symlink)

    with pytest.raises(ReleaseKeyError, match="regular non-symlink"):
        load_release_private_key(symlink, repository_root=repository)

    inside = repository / "release.key"
    _write_key(inside)
    with pytest.raises(ReleaseKeyError, match="outside the repository"):
        load_release_private_key(inside, repository_root=repository)
