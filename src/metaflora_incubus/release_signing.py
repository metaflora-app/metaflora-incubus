"""Safe loading of the offline Ed25519 release-signing key."""

from __future__ import annotations

import base64
import binascii
import stat
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class ReleaseKeyError(ValueError):
    """Raised when a release key does not meet the offline-key policy."""


def load_release_private_key(path: Path, *, repository_root: Path) -> Ed25519PrivateKey:
    """Load a raw base64 Ed25519 key only from a private, external file."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReleaseKeyError("release key is missing or unreadable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ReleaseKeyError("release key must be a regular non-symlink file")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise ReleaseKeyError("release key permissions must be 0600")

    resolved_path = path.resolve()
    resolved_repository = repository_root.resolve()
    if resolved_path == resolved_repository or resolved_repository in resolved_path.parents:
        raise ReleaseKeyError("release key must be stored outside the repository")

    try:
        encoded = resolved_path.read_text(encoding="ascii").strip()
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != 32:
            raise ValueError("invalid key length")
        return Ed25519PrivateKey.from_private_bytes(raw)
    except (OSError, UnicodeError, binascii.Error, ValueError) as exc:
        raise ReleaseKeyError("release key is not a valid base64 Ed25519 private key") from exc
