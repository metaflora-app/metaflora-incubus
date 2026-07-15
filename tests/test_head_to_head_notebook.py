from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks" / "metaflora-incubus-head-to-head.ipynb"
)


def load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_source() -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in load_notebook()["cells"] if cell["cell_type"] == "code"
    )


def test_head_to_head_notebook_is_clean_and_every_code_cell_compiles() -> None:
    notebook = load_notebook()
    cells = notebook["cells"]

    assert cells[0]["cell_type"] == "markdown"
    assert len(cells) >= 4
    for index, cell in enumerate(cells):
        assert cell["outputs"] == [] if cell["cell_type"] == "code" else True
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_notebook_downloads_only_pinned_required_baselines_and_verifies_bytes() -> None:
    raw = notebook_source()

    assert "head-to-head-v1-baselines.json" in raw
    assert "promotion_required" in raw
    assert "hf_hub_download" in raw
    assert 'revision=pin["artifact_revision"]' in raw
    assert "artifact_sha256" in raw
    assert "artifact_size_bytes" in raw
    assert 'b"GGUF"' in raw
    assert "verified.json" in raw


def test_notebook_restores_every_private_input_without_previous_session_paths() -> None:
    raw = notebook_source()

    assert "INCUBUS_BOOTSTRAP" in raw
    assert "INCUBUS_CODE_REVISION" in raw
    assert "metaflora/incubus-checkpoints" in raw
    assert "incubus-training-v1" in raw
    assert "HfApi" in raw
    assert "checkpoint_head" in raw
    assert "model_info" in raw
    assert 're.fullmatch(r"[0-9a-f]{40}", checkpoint_head)' in raw
    assert "runs/incubus-v1-refine-001/exports/candidate-upload-receipt.json" in raw
    assert "revision=checkpoint_head" in raw
    assert 'candidate_upload_receipt["artifact_revision"]' in raw
    assert 'candidate_upload_receipt["artifact_path"]' in raw
    assert 'candidate_upload_receipt["remote_prefix"]' in raw
    assert "revision=candidate_artifact_revision" in raw
    assert 'f"{candidate_remote_prefix}/candidate-export.json"' in raw
    assert "runs/incubus-v1-run/artifacts/metaflora-incubus-v1.gguf" in raw
    assert "runs/incubus-v1-run/artifacts/artifact-metadata.json" in raw
    assert "runs/incubus-v1-run/artifacts/llama-server" not in raw
    assert "INCUBUS_CANDIDATE_GGUF" not in raw
    assert "INCUBUS_INCUMBENT_GGUF" not in raw
    assert "INCUBUS_LLAMA_SERVER" not in raw
    assert "UserSecretsClient" not in raw
    assert "/kaggle/input/incubus-private-runtime-bootstrap/bootstrap-key.txt" in raw
    assert 'code_revision = "7ec9bcd46001b0ecd8d15e83203835f06dca59ea"' in raw


def test_notebook_builds_a_pinned_local_cuda_server_with_actionable_logs() -> None:
    raw = notebook_source()

    assert 'candidate_receipt["artifact"]["sha256"]' in raw
    assert 'candidate_upload_receipt["artifact_sha256"]' in raw
    assert 'candidate_upload_receipt["artifact_size_bytes"]' in raw
    assert 'incumbent_receipt["artifact_sha256"]' in raw
    assert "chmod(0o700)" in raw
    assert "llama.cpp-head-to-head" in raw
    assert "llama_cpp_revision" in raw
    assert "llama.cpp revision is not pinned" in raw
    assert "llama.cpp checkout verification failed" in raw
    assert '"cmake"' in raw
    assert '"--target", "llama-server"' in raw
    assert '"-DGGML_CUDA=ON"' in raw
    assert "llama-server-build.log" in raw
    assert "llama-server build failed" in raw
    assert 'server_name = "runs/incubus-v1-run/artifacts/llama-server"' not in raw
    assert "restored_server" not in raw


def test_notebook_runs_private_harness_and_writes_hard_case_evidence() -> None:
    raw = notebook_source()

    assert "run_head_to_head_benchmark.py" in raw
    assert "INCUBUS_BENCHMARK_SIGNING_KEY" in raw
    assert "head-to-head-report.json" in raw
    assert "hard-case-failures.jsonl" in raw
    assert "hard-case-failures-manifest.json" in raw
    assert "failure_score_threshold" in raw
    assert "0.75" in raw


def test_notebook_uses_only_hash_locked_dependencies_and_pythonpath() -> None:
    raw = notebook_source()
    requirements = Path("requirements/h2h.in").read_text(encoding="utf-8")
    lock = Path("requirements/h2h-linux.lock").read_text(encoding="utf-8")

    assert "requirements/h2h-linux.lock" in raw
    assert '"--require-hashes"' in raw
    assert '"huggingface_hub>=1.23,<2"' not in raw
    assert '"cryptography>=46,<47"' not in raw
    assert '"-e", str(REPO)' not in raw
    assert 'runtime_environment["PYTHONPATH"]' in raw
    assert "env=runtime_environment" in raw
    assert "cryptography==48.0.1" in requirements
    assert "huggingface-hub==1.23.0" in requirements
    assert "cryptography==48.0.1" in lock
    assert "huggingface-hub==1.23.0" in lock


def test_notebook_does_not_train_recover_or_publish() -> None:
    raw = notebook_source().casefold()

    for prohibited in (
        "run_free_gpu.py",
        "recover_free_gpu.py",
        "kaggle_recover_export.py",
        "publish_to_huggingface",
        "upload_folder",
        "create_repo",
        "git push",
    ):
        assert prohibited not in raw
