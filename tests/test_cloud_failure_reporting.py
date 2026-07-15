from __future__ import annotations

import json
import sys
import types

import pytest

from metaflora_incubus.cloud_failure_reporting import run_with_failure_reporting
from metaflora_incubus.cloud_training import CheckpointBackend, RemoteCheckpointTarget


def _private_target() -> RemoteCheckpointTarget:
    return RemoteCheckpointTarget.create(
        backend=CheckpointBackend.HF_PRIVATE_BRANCH,
        location="private-owner/private-checkpoints",
        branch="incubus-training-v1",
    )


def test_failure_report_is_private_bound_and_redacted(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    uploaded: dict[str, object] = {}

    class FakeHfApi:
        def __init__(self, *, token=None):
            calls.append(("init", {"token": token}))

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def model_info(self, **kwargs):
            calls.append(("model_info", kwargs))
            return types.SimpleNamespace(private=True)

        def create_branch(self, **kwargs):
            calls.append(("create_branch", kwargs))

        def upload_file(self, **kwargs):
            uploaded.update(kwargs)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    environment = {
        "INCUBUS_CHECKPOINT_HMAC_KEY": "checkpoint-super-secret",
        "INCUBUS_SOURCE_REPO": "private/source-name",
        "HF_TOKEN": "hf_environment_token_123456789",
        "WANDB_API_KEY": "wandb-environment-secret",
        "HUGGING_FACE_HUB_TOKEN": "hf_alternative_environment_token",
    }

    def fail() -> None:
        raise RuntimeError(
            "failed with checkpoint-super-secret, private/source-name, "
            "hf_inline_token_123456789 and Bearer bearer-secret-value; "
            "wandb-environment-secret; hf_alternative_environment_token; "
            "https://alice:basic-password@example.invalid/path; "
            "https://example.invalid/file?X-Amz-Credential=credential-value&"
            "X-Amz-Signature=signature-value&api_key=query-secret"
        )

    with pytest.raises(RuntimeError, match="checkpoint-super-secret"):
        run_with_failure_reporting(
            fail,
            target=_private_target(),
            run_id="incubus-v1-run",
            code_revision="a" * 40,
            phase="training-and-build",
            environment=environment,
        )

    assert ("init", {"token": None}) in calls
    assert (
        "create_repo",
        {
            "repo_id": "private-owner/private-checkpoints",
            "repo_type": "model",
            "private": True,
            "exist_ok": True,
        },
    ) in calls
    assert (
        "create_branch",
        {
            "repo_id": "private-owner/private-checkpoints",
            "branch": "incubus-training-v1",
            "exist_ok": True,
        },
    ) in calls
    assert uploaded["revision"] == "incubus-training-v1"
    assert uploaded["path_in_repo"] == "failures/incubus-v1-run-latest.json"
    assert uploaded["repo_type"] == "model"

    payload = json.loads(uploaded["path_or_fileobj"].getvalue())
    rendered = json.dumps(payload, sort_keys=True)
    for secret in (
        "checkpoint-super-secret",
        "private/source-name",
        "hf_environment_token_123456789",
        "hf_inline_token_123456789",
        "bearer-secret-value",
        "wandb-environment-secret",
        "hf_alternative_environment_token",
        "basic-password",
        "credential-value",
        "signature-value",
        "query-secret",
    ):
        assert secret not in rendered
    assert payload["exception_type"] == "RuntimeError"
    assert payload["code_revision"] == "a" * 40
    assert payload["run_id"] == "incubus-v1-run"
    assert payload["phase"] == "training-and-build"
    assert "traceback" in payload
    assert "[REDACTED]" in rendered


def test_reporting_failure_never_masks_original_exception(monkeypatch) -> None:
    class BrokenHfApi:
        def __init__(self, *, token=None):
            pass

        def create_repo(self, **kwargs):
            raise OSError("reporting service unavailable")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=BrokenHfApi))
    original = LookupError("original training failure")

    def fail() -> None:
        raise original

    with pytest.raises(LookupError) as caught:
        run_with_failure_reporting(
            fail,
            target=_private_target(),
            run_id="incubus-v1-run",
            code_revision="b" * 40,
            phase="bootstrap-validation",
            environment={},
        )

    assert caught.value is original


def test_success_returns_result_without_contacting_hub(monkeypatch) -> None:
    class UnexpectedHfApi:
        def __init__(self, *, token=None):
            raise AssertionError("Hub must not be contacted on success")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=UnexpectedHfApi)
    )

    result = run_with_failure_reporting(
        lambda: {"status": "ok"},
        target=_private_target(),
        run_id="incubus-v1-run",
        code_revision="c" * 40,
        phase="training-and-build",
        environment={},
    )

    assert result == {"status": "ok"}


def test_public_destination_blocks_upload_and_preserves_original(monkeypatch) -> None:
    calls: list[str] = []

    class PublicHfApi:
        def __init__(self, *, token=None):
            calls.append("init")

        def create_repo(self, **kwargs):
            calls.append("create_repo")

        def model_info(self, **kwargs):
            calls.append("model_info")
            return types.SimpleNamespace(private=False)

        def create_branch(self, **kwargs):
            calls.append("create_branch")

        def upload_file(self, **kwargs):
            calls.append("upload_file")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=PublicHfApi))
    original = RuntimeError("training failed")

    def fail() -> None:
        raise original

    with pytest.raises(RuntimeError) as caught:
        run_with_failure_reporting(
            fail,
            target=_private_target(),
            run_id="incubus-v1-run",
            code_revision="d" * 40,
            phase="training-and-build",
            environment={},
        )

    assert caught.value is original
    assert calls == ["init", "create_repo", "model_info"]
