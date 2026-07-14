from pathlib import Path

import torch

from metaflora_incubus import cli


def test_doctor_returns_success_when_resources_are_available(monkeypatch, capsys) -> None:
    usage = type("Usage", (), {"free": 100_000})()
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda _path: usage)
    monkeypatch.setattr(cli, "_available_ram_bytes", lambda: 100_000)
    monkeypatch.setattr(cli, "_available_vram_bytes", lambda: 100_000)

    exit_code = cli.main(["doctor", "--model-size-gb", "0.00001"])

    assert exit_code == 0
    assert '"ready": true' in capsys.readouterr().out


def test_evaluate_writes_assessments_without_network(monkeypatch, tmp_path: Path) -> None:
    answers = tmp_path / "answers.jsonl"
    answers.write_text('{"prompt":"p","answer":"a"}\n', encoding="utf-8")
    output = tmp_path / "assessment.json"

    assessment = type("Assessment", (), {"__dict__": {"answer_class": "answer", "useful": True}})()
    monkeypatch.setattr(
        "metaflora_incubus.evaluation.judge_answer_semantically",
        lambda **_kwargs: assessment,
    )

    exit_code = cli.main(
        [
            "evaluate",
            "--answers",
            str(answers),
            "--judge-url",
            "http://127.0.0.1:8081",
            "--judge-model",
            "judge",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert '"useful": true' in output.read_text(encoding="utf-8")


def test_targets_and_hash_helpers_work_without_downloading_a_checkpoint(
    monkeypatch, capsys, tmp_path
) -> None:
    class Weight:
        ndim = 2
        shape = (4, 2)

    class Projection:
        weight = Weight()

    class Model:
        def named_modules(self):
            return (("block.o_proj", Projection()),)

        def state_dict(self):
            return {"weight": torch.tensor([1.0, 2.0])}

    import transformers

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: Model(),
    )
    data = tmp_path / "data.txt"
    data.write_text("Incubus", encoding="utf-8")

    assert cli.main(["targets", "--model", "fake/model"]) == 0
    assert "block.o_proj" in capsys.readouterr().out
    assert len(cli._file_sha256(data)) == 64
    assert len(cli._model_fingerprint(Model())) == 64


def test_publish_hf_dry_run_fails_closed_for_empty_bundle(tmp_path: Path, capsys) -> None:
    exit_code = cli.main(
        [
            "publish-hf",
            "--bundle",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "missing_required_file" in capsys.readouterr().out
