"""Command-line surface for building and serving Metaflora Incubus v1."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from hashlib import sha256
from pathlib import Path

from metaflora_incubus.preflight import build_preflight_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="incubus", description="Metaflora Incubus v1 toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check local disk, RAM and accelerator")
    doctor.add_argument("--model-size-gb", type=float, required=True)
    doctor.add_argument("--required-vram-gb", type=float, default=0)

    targets = subparsers.add_parser("targets", help="inspect editable projection matrices")
    targets.add_argument("--model", required=True)

    run = subparsers.add_parser("run", help="build a local Metaflora Incubus v1 candidate")
    run.add_argument("--model", required=True)
    run.add_argument("--calibration", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--strength", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=42)

    serve = subparsers.add_parser("serve", help="serve an exported model over localhost only")
    serve.add_argument("--model", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8080)

    evaluate = subparsers.add_parser(
        "evaluate", help="label response JSONL with a local semantic judge"
    )
    evaluate.add_argument("--answers", type=Path, required=True)
    evaluate.add_argument("--judge-url", required=True)
    evaluate.add_argument("--judge-model", required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    publish_hf = subparsers.add_parser(
        "publish-hf", help="validate and publish a signed Hugging Face release bundle"
    )
    publish_hf.add_argument("--bundle", type=Path, required=True)
    publish_hf.add_argument("--repo-id", default="metaflora/incubus")
    publish_hf.add_argument("--dry-run", action="store_true")
    publish_hf.add_argument("--public-key", type=Path)
    publish_hf.add_argument("--prohibited-identifiers", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "targets":
        return _targets(args)
    if args.command == "run":
        return _run(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "publish-hf":
        return _publish_hf(args)
    raise AssertionError(f"unsupported command: {args.command}")


def _doctor(args: argparse.Namespace) -> int:
    model_bytes = int(args.model_size_gb * 1024**3)
    disk_bytes = shutil.disk_usage(Path.cwd()).free
    ram_bytes = _available_ram_bytes()
    report = build_preflight_report(
        model_bytes=model_bytes,
        available_disk_bytes=disk_bytes,
        available_ram_bytes=ram_bytes,
        available_vram_bytes=_available_vram_bytes(),
        required_vram_bytes=int(args.required_vram_gb * 1024**3),
    )
    print(json.dumps(report.__dict__, indent=2))
    return 0 if report.ready else 2


def _targets(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM

    from metaflora_incubus.adapters import discover_transform_targets

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        weights_only=True,
    )
    targets = discover_transform_targets(model.named_modules())
    print(json.dumps([target.__dict__ for target in targets], indent=2))
    return 0


def _run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from metaflora_incubus.manifest import RunManifest
    from metaflora_incubus.pipeline import (
        build_candidate,
        calibrate_input_directions,
        export_candidate,
        load_calibration_pairs,
    )

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=False,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        weights_only=True,
    )
    if torch.cuda.is_available():
        model.to("cuda")
    pairs = load_calibration_pairs(args.calibration)
    _, directions = calibrate_input_directions(model, tokenizer, pairs)
    if not directions:
        raise SystemExit("no compatible projection inputs were found for this checkpoint")
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    candidate, changed_modules = build_candidate(model, directions, strength=args.strength)
    manifest = RunManifest.create(
        base_model=args.model,
        model_revision=str(getattr(model.config, "_commit_hash", None) or "local-or-unpinned"),
        model_sha256=_model_fingerprint(model),
        dataset_sha256=_file_sha256(args.calibration),
        seed=args.seed,
        strength=args.strength,
        transform_version="0.1.0",
    )
    export_candidate(
        candidate=candidate,
        tokenizer=tokenizer,
        output_directory=args.output,
        manifest=manifest,
        changed_modules=changed_modules,
    )
    print(json.dumps({"output": str(args.output), "changed_modules": changed_modules}, indent=2))
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from metaflora_incubus.server import LocalRuntime, create_app

    runtime = LocalRuntime.load(str(args.model))
    uvicorn.run(create_app(runtime), host="127.0.0.1", port=args.port)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from metaflora_incubus.evaluation import judge_answer_semantically

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    assessments = []
    lines = args.answers.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        prompt = payload.get("prompt")
        answer = payload.get("answer")
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise SystemExit(f"answers line {line_number} needs prompt and answer")
        assessment = judge_answer_semantically(
            endpoint=args.judge_url,
            model=args.judge_model,
            prompt=prompt,
            answer=answer,
        )
        assessments.append({**payload, "assessment": assessment.__dict__})
    args.output.write_text(json.dumps(assessments, indent=2, default=str) + "\n", encoding="utf-8")
    useful = sum(1 for item in assessments if item["assessment"]["useful"])
    print(json.dumps({"responses": len(assessments), "useful": useful}, indent=2))
    return 0


def _publish_hf(args: argparse.Namespace) -> int:
    from metaflora_incubus.huggingface_publication import (
        HuggingFaceHubUploader,
        PublicationPolicy,
        evaluate_publication_bundle,
        publish_to_huggingface,
    )

    base = PublicationPolicy.default()
    prohibited = ()
    if args.prohibited_identifiers is not None:
        prohibited = tuple(
            line.strip()
            for line in args.prohibited_identifiers.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    policy = PublicationPolicy(
        repo_id=args.repo_id,
        min_model_bytes=base.min_model_bytes,
        max_model_bytes=base.max_model_bytes,
        prohibited_identifiers=prohibited,
    )

    def reject_signature(*_args: object) -> bool:
        return False

    verifier = reject_signature
    if args.public_key is not None:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(args.public_key.read_text(encoding="ascii").strip(), validate=True)
        )

        def verify_signature(_purpose: str, payload: bytes, signature: bytes) -> bool:
            try:
                public_key.verify(signature, payload)
            except InvalidSignature:
                return False
            return True

        verifier = verify_signature

    if not args.dry_run:
        if args.public_key is None or args.prohibited_identifiers is None:
            raise SystemExit("live publication requires --public-key and --prohibited-identifiers")
        result = publish_to_huggingface(
            args.bundle,
            policy=policy,
            signature_verifier=verifier,
            uploader=HuggingFaceHubUploader(),
        )
        print(
            json.dumps(
                {
                    "repo_id": result.repo_id,
                    "uploaded": result.uploaded,
                    "blockers": [blocker.__dict__ for blocker in result.decision.blockers],
                },
                indent=2,
            )
        )
        return 0 if result.uploaded else 2
    decision = evaluate_publication_bundle(
        args.bundle,
        policy=policy,
        signature_verifier=verifier,
    )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "approved": decision.approved,
                "blockers": [blocker.__dict__ for blocker in decision.blockers],
                "note": "cryptographic approval runs only in pinned release automation",
            },
            indent=2,
        )
    )
    return 0 if decision.approved else 2


def _available_ram_bytes() -> int:
    try:
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return 0


def _available_vram_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(properties.total_memory)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_fingerprint(model) -> str:
    """Hash parameter values before the build input is copied or changed."""
    digest = sha256()
    for name, parameter in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        tensor = parameter.detach().cpu().contiguous()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
