from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/metaflora-incubus-recover-gguf.ipynb")


def test_recovery_notebook_is_single_cell_and_skips_training() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "".join(code_cells[0]["source"])

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(code_cells) == 1
    assert "scripts/recover_free_gpu.py" in source
    assert 'userdata.get("INCUBUS_BOOTSTRAP")' in source
    assert "SFTTrainer" not in source
    assert "DPOTrainer" not in source
    assert "run_free_gpu.py" not in source
    assert 'trusted_code_revision = "ce3ddbf70a5a827b97e2591a9b6393eb3c99be40"' in source
