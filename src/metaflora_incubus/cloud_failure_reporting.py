"""Sanitized private failure evidence for unattended cloud runs."""

from __future__ import annotations

import io
import json
import re
import traceback
from collections.abc import Callable, Mapping
from typing import TypeVar

from metaflora_incubus.cloud_training import CheckpointBackend, RemoteCheckpointTarget

_T = TypeVar("_T")
_REDACTED = "[REDACTED]"
_TOKEN_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[^\s,;]+"),
    re.compile(r"\bhf_[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)(access_token|refresh_token|hf_token|token)"
        r"(\s*[=:]\s*|\s*[\"']:\s*[\"'])"
        r"[^\s,;\"']+"
    ),
    re.compile(r"(?i)https?://[^/\s:@]+:[^@\s/]+@"),
    re.compile(
        r"(?i)[?&](?:api[_-]?key|access[_-]?token|credential|signature|"
        r"x-amz-credential|x-amz-signature)=[^&#\s]+"
    ),
)
_SENSITIVE_KEY_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "PRIVATE_KEY",
    "CREDENTIAL",
    "AUTH",
)


def _sensitive_values(environment: Mapping[str, str]) -> tuple[str, ...]:
    values = {
        str(value)
        for key, value in environment.items()
        if value
        and (
            key.upper().startswith("INCUBUS_")
            or any(marker in key.upper() for marker in _SENSITIVE_KEY_MARKERS)
        )
    }
    return tuple(sorted(values, key=len, reverse=True))


def _redact(value: str, *, environment: Mapping[str, str]) -> str:
    result = value
    for secret in _sensitive_values(environment):
        result = result.replace(secret, _REDACTED)
    for pattern in _TOKEN_PATTERNS:
        result = pattern.sub(_REDACTED, result)
    return result


def _failure_payload(
    exception: Exception,
    *,
    run_id: str,
    code_revision: str,
    phase: str,
    environment: Mapping[str, str],
) -> bytes:
    rendered_traceback = "".join(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    document = {
        "schema_version": 1,
        "run_id": run_id,
        "code_revision": code_revision,
        "phase": phase,
        "exception_type": type(exception).__name__,
        "exception_message": _redact(str(exception), environment=environment),
        "traceback": _redact(rendered_traceback, environment=environment),
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _upload_private_failure(
    exception: Exception,
    *,
    target: RemoteCheckpointTarget,
    run_id: str,
    code_revision: str,
    phase: str,
    environment: Mapping[str, str],
) -> None:
    if target.backend is not CheckpointBackend.HF_PRIVATE_BRANCH or target.branch is None:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=None)
    api.create_repo(
        repo_id=target.location,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    info = api.model_info(repo_id=target.location)
    if info.private is not True:
        raise RuntimeError("failure report destination is not private")
    api.create_branch(repo_id=target.location, branch=target.branch, exist_ok=True)
    api.upload_file(
        repo_id=target.location,
        repo_type="model",
        revision=target.branch,
        path_in_repo=f"failures/{run_id}-latest.json",
        path_or_fileobj=io.BytesIO(
            _failure_payload(
                exception,
                run_id=run_id,
                code_revision=code_revision,
                phase=phase,
                environment=environment,
            )
        ),
        commit_message=f"Record failure for {run_id}",
    )


def run_with_failure_reporting(
    operation: Callable[[], _T],
    *,
    target: RemoteCheckpointTarget,
    run_id: str,
    code_revision: str,
    phase: str,
    environment: Mapping[str, str],
) -> _T:
    """Run an operation and privately persist sanitized failure evidence."""

    try:
        return operation()
    except Exception as exception:
        try:
            _upload_private_failure(
                exception,
                target=target,
                run_id=run_id,
                code_revision=code_revision,
                phase=phase,
                environment=environment,
            )
        except Exception:
            pass
        raise
