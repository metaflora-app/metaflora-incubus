from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "metaflora-incubus-bonsai-advisory.ipynb"
)


def load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_source() -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_bonsai_advisory_notebook_is_clean_and_every_code_cell_compiles() -> None:
    notebook = load_notebook()

    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert len(notebook["cells"]) >= 5
    for index, cell in enumerate(notebook["cells"]):
        assert cell["outputs"] == [] if cell["cell_type"] == "code" else True
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_bonsai_notebook_uses_private_bootstrap_without_kaggle_secrets() -> None:
    raw = notebook_source()

    assert "/kaggle/input/incubus-private-runtime-bootstrap/bootstrap-key.txt" in raw
    assert "UserSecretsClient" not in raw
    assert "userdata.get" not in raw
    assert 'code_revision = "7ec9bcd46001b0ecd8d15e83203835f06dca59ea"' in raw
    assert "decrypt_cloud_bootstrap" in raw
    assert "install_cloud_bootstrap" in raw


def test_bonsai_notebook_has_a_strict_disk_guard_and_safe_skip_receipt() -> None:
    raw = notebook_source()

    assert "pin['id'] == 'bonsai-27b'" in raw
    assert "3_803_452_480" in raw
    assert "4_877_194_304" in raw
    assert "shutil.disk_usage(WORK_ROOT).free" in raw
    assert "bonsai-advisory-skip.json" in raw
    assert "'status': 'skipped'" in raw
    assert "'reason': 'insufficient_disk'" in raw
    assert "raise SystemExit(0)" in raw


def test_bonsai_notebook_is_advisory_only_and_keeps_mandatory_h2h_separate() -> None:
    raw = notebook_source()

    assert "bonsai-advisory-output" in raw
    assert "bonsai-advisory-v1.json" in raw
    assert "'role': 'conditional'" in raw
    assert '"promotion_required": false' not in raw
    assert "head-to-head-v1-baselines.json" in raw
    assert "run_head_to_head_benchmark.py" in raw
    assert "publish_to_huggingface" not in raw.casefold()
    assert "upload_folder" not in raw.casefold()
