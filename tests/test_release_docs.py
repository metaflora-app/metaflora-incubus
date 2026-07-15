from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_model_card_does_not_promise_an_unshipped_windows_installer() -> None:
    card = (ROOT / "release/templates/huggingface/README.md.tmpl").read_text(encoding="utf-8")

    assert "install.ps1" not in card
    assert "Windows automatic installation is not included in v1" in card
