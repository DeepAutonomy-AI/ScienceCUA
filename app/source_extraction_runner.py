"""CLI used through DSH's native terminal; no business tool registration."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adaptive_source import SourceContext, canonical_bytes, digest, validate_output, write_json


def execute_script(assignment: Path, script: Path, output: Path, replay: bool) -> None:
    ctx = SourceContext(assignment, replay=replay)
    spec = importlib.util.spec_from_file_location("source_extractor", script)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load extraction script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extract = getattr(module, "extract", None)
    if not callable(extract):
        raise ValueError("extract.py must define extract(ctx) and return the result dictionary")
    result = extract(ctx)
    if hasattr(result, "model_dump"):
        result = result.model_dump(mode="json")
    write_json(output, result)


def run_extraction(assignment: Path, script: Path, output: Path, *, replay: bool = False) -> dict:
    ctx = SourceContext(assignment)
    script = script.resolve()
    output = output.resolve()
    if not script.is_relative_to(ctx.root) or not output.is_relative_to(ctx.root):
        raise ValueError("Script and candidate must be inside this extraction workspace")
    command = [sys.executable, "-B", str(Path(__file__).resolve()),
               "_execute", "--assignment", str(assignment), "--script", str(script), "--output", str(output)]
    if replay:
        command.append("--replay")
    report = {"schema_version": "adaptive-source-validation.v1", "valid": False, "issues": []}
    try:
        environment = dict(os.environ)
        # Extraction does not need provider credentials inherited by the DSH shell.
        for key in list(environment):
            if any(token in key.upper() for token in ("API_KEY", "CAPABILITY_CONFIG", "CAPABILITY_TOKEN")):
                environment.pop(key)
        environment["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(command, cwd=ctx.root, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=ctx.assignment["execution_timeout_seconds"], env=environment)
        logs = ctx.root / "execution"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / ("replay.stdout.log" if replay else "stdout.log")).write_text(completed.stdout[-500_000:], encoding="utf-8")
        (logs / ("replay.stderr.log" if replay else "stderr.log")).write_text(completed.stderr[-500_000:], encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Extraction exited {completed.returncode}: {completed.stderr[-6000:]}")
        if output.stat().st_size > ctx.assignment["max_source_bytes"]:
            raise ValueError("Candidate exceeds the output byte budget")
        raw = json.loads(output.read_text(encoding="utf-8"))
        report, _ = validate_output(SourceContext(assignment, replay=replay), raw)
        report["script_sha256"] = digest(script.read_bytes())
        report["candidate_file"] = str(output)
        report["replay"] = replay
    except Exception as exc:
        report["valid"] = False
        report["issues"].append({"severity": "error", "code": "extraction_execution_failed", "message": str(exc)[:8000]})
    write_json(ctx.root / "validation" / ("replay.json" if replay else "report.json"), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("run", "validate", "_execute"))
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--script", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    ctx = SourceContext(args.assignment)
    script = args.script or ctx.root / "work" / "extract.py"
    output = args.output or ctx.root / "candidate" / "extraction.json"
    if args.operation == "_execute":
        execute_script(args.assignment, script, output, args.replay)
        return 0
    if args.operation == "run":
        report = run_extraction(args.assignment, script, output, replay=args.replay)
    else:
        try:
            raw = json.loads(output.read_text(encoding="utf-8"))
            report, _ = validate_output(SourceContext(args.assignment), raw)
        except Exception as exc:
            report = {"valid": False, "issues": [{"severity": "error", "code": "invalid_candidate", "message": str(exc)}]}
        write_json(ctx.root / "validation" / "report.json", report)
    # The complete report stays on disk; keep terminal observations bounded.
    preview = {**report, "issues": report.get("issues", [])[:20], "issue_count": len(report.get("issues", [])),
               "gaps": report.get("gaps", [])[:10], "warnings": report.get("warnings", [])[:10],
               "report_file": str(ctx.root / "validation" / ("replay.json" if args.replay else "report.json"))}
    print(json.dumps(preview, ensure_ascii=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
