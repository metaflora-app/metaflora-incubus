from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "metaflora-incubus-head-to-head.ipynb"
)


def load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_source() -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
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
    assert "revision=pin[\"artifact_revision\"]" in raw
    assert "artifact_sha256" in raw
    assert "artifact_size_bytes" in raw
    assert "b\"GGUF\"" in raw
    assert "verified.json" in raw


def test_notebook_runs_private_harness_and_writes_hard_case_evidence() -> None:
    raw = notebook_source()

    assert "run_head_to_head_benchmark.py" in raw
    assert "INCUBUS_BENCHMARK_SIGNING_KEY" in raw
    assert "head-to-head-report.json" in raw
    assert "hard-case-failures.jsonl" in raw
    assert "hard-case-failures-manifest.json" in raw
    assert "failure_score_threshold" in raw
    assert "0.75" in raw


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
