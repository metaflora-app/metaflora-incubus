from __future__ import annotations

import json
import re
import subprocess
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
    assert "requirements/recovery-linux.lock" in source
    assert "requirements/cloud-linux.lock" not in source
    assert "pip\", \"uninstall" not in source
    assert "torchvision" not in source
    assert "torchaudio" not in source
    revision = re.search(r'trusted_code_revision = "([0-9a-f]{40})"', source)
    assert revision is not None
    for required_path in (
        "requirements/recovery-linux.lock",
        "scripts/recover_free_gpu.py",
        "src/metaflora_incubus/cloud_training_runtime.py",
    ):
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision.group(1)}:{required_path}"],
            check=True,
        )
