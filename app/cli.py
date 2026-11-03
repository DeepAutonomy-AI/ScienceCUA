from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import Settings, model_defaults, normalize_output_language, settings
from .evaluation import evaluate_generated_skills, write_evaluation_report
from .orchestration import PipelineRuntime
from .models import ModelRuntimeConfig, StagedInput, TargetProfile
from .runtime import DshAgentRuntime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the evidence-grounded multimodal Skill pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run preprocessing through Skill export.")
    run.add_argument("--input", action="append", required=True, help="PDF or HTML input path. Repeatable.")
    run.add_argument("--software-name", required=True)
    run.add_argument("--software-id", default="")
    run.add_argument("--target-version", required=True)
    run.add_argument("--platform", default="unknown")
    run.add_argument("--ui-locale", default="unknown")
    run.add_argument("--input-version", default="unknown")
    run.add_argument(
        "--api-protocol",
        choices=["chat_completions", "responses"],
        default=None,
        help="OpenAI-compatible request protocol used by the API Agent.",
    )
    run.add_argument("--api-base-url", default=model_defaults.base_url)
    run.add_argument("--api-model", default=model_defaults.model)
    run.add_argument("--api-key-env", default=model_defaults.api_key_env)
    run.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh", "max"], default=model_defaults.reasoning_effort)
    run.add_argument(
        "--websearch",
        action="store_true",
        help="Legacy compatibility flag; rejected when supplied.",
    )
    run.add_argument("--data-dir")
    run.add_argument("--output-dir", help="Root for all audit exports and Codex entries (default: output/).")
    run.add_argument("--summary", action="store_true", help="Print a compact completion payload.")

    evaluate = subparsers.add_parser("evaluate", help="Read-only comparison against an Excel benchmark.")
    evaluate.add_argument("--skills-file", required=True)
    evaluate.add_argument("--workbook", required=True)
    evaluate.add_argument("--sheet", default="KAlgebra")
    evaluate.add_argument("--report")

    return parser


def _runtime_settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if getattr(args, "data_dir", None):
        overrides["data_dir"] = Path(args.data_dir).resolve()
    if getattr(args, "output_dir", None):
        overrides["output_dir"] = Path(args.output_dir).resolve()
    return replace(settings, **overrides) if overrides else settings


async def _run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.websearch:
        raise ValueError(
            "The legacy --websearch flag is disabled. Use the Web UI for automatic manual discovery."
        )
    runtime_settings = _runtime_settings(args)
    runtime = PipelineRuntime(runtime_settings, DshAgentRuntime())
    target = TargetProfile(
        software_name=args.software_name.strip(),
        software_id=args.software_id.strip() or None,
        target_version=args.target_version.strip(),
        platform=args.platform,
        ui_locale=args.ui_locale,
    )
    api_key = os.getenv(args.api_key_env)
    model_config = ModelRuntimeConfig(
        api_protocol=args.api_protocol or "",
        base_url=args.api_base_url,
        model=args.api_model,
        api_key=api_key,
        reasoning_effort=args.reasoning_effort,
        websearch=False,
    )
    if not model_config.api_key or not model_config.model.strip():
        raise ValueError(
            f"The API Agent requires --api-model and a non-empty environment variable {args.api_key_env}."
        )
    run_id = f"run_{uuid.uuid4().hex[:20]}"
    store = runtime.create_store(run_id)
    staged_inputs: list[StagedInput] = []
    for index, raw_path in enumerate(args.input):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        if len(data) > runtime_settings.max_source_bytes:
            raise ValueError(f"{path} exceeds max_source_bytes={runtime_settings.max_source_bytes}")
        input_id = f"IN{index + 1}"
        staged_path = store.stage_upload(input_id, path.name, data)
        staged_inputs.append(
            StagedInput(
                input_id=input_id,
                kind="uploaded_file",
                display_name=path.name,
                user_declared_version=args.input_version,
                staged_path=str(staged_path),
                declared_media_type=None,
            )
        )
    public_inputs = [
        {
            "input_id": item.input_id,
            "kind": item.kind,
            "display_name": item.display_name,
            "user_declared_version": item.user_declared_version,
            "declared_media_type": item.declared_media_type,
        }
        for item in staged_inputs
    ]
    store.initialize(
        {
            "run_id": run_id,
            "target_profile": target.model_dump(mode="json"),
            "inputs": public_inputs,
            "model": model_config.public_dict(),
            "output_language": normalize_output_language(runtime_settings.skill_output_language),
        }
    )
    store.emit(
        "run_registered",
        {
            "target_profile": target.model_dump(mode="json"),
            "inputs": public_inputs,
            "model": model_config.public_dict(),
            "entrypoint": "cli",
        },
    )
    await runtime.execute(run_id, target, staged_inputs, model_config)
    result = store.read_result() or {}
    return {"run_id": run_id, "result": result, "state": store.read_state()}


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "run":
        payload = asyncio.run(_run_pipeline(args))
        if args.summary:
            result = payload["result"]
            print(
                json.dumps(
                    {
                        "run_id": payload["run_id"],
                        "state": payload["state"],
                        "outcome": result.get("outcome"),
                        "evidence_count": result.get("snapshot", {}).get("evidence_count"),
                        "task_count": len(result.get("tasks") or []),
                        "reference_count": len(result.get("references") or []),
                        "trajectory_count": len(result.get("trajectories") or []),
                        "skill_count": len(result.get("skills") or []),
                        "output_directory": result.get("skill_registry", {}).get("output_directory"),
                        "audit_directory": result.get("skill_registry", {}).get("audit_directory"),
                        "agent_skill_directory": result.get("skill_registry", {}).get("agent_skill_directory"),
                        "agent_entry_directory": result.get("skill_registry", {}).get("agent_entry_directory"),
                        "agent_entry_manifest_file": result.get("skill_registry", {}).get("agent_entry_manifest_file"),
                        "agent_entry_count": result.get("skill_registry", {}).get("agent_entry_count"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evaluate":
        skills_file = Path(args.skills_file).expanduser().resolve()
        report = evaluate_generated_skills(
            skills_file,
            Path(args.workbook).expanduser().resolve(),
            sheet_name=args.sheet,
        )
        output = (
            Path(args.report).expanduser().resolve()
            if args.report
            else skills_file.parent / "evaluation.json"
        )
        write_evaluation_report(report, output)
        print(json.dumps({"report": str(output), **report}, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
