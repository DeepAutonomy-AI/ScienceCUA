from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _tokens(value: str) -> set[str]:
    value = value.lower()
    latin = re.findall(r"[a-z0-9_./+-]+", value)
    cjk = re.findall(r"[\u4e00-\u9fff]", value)
    bigrams = ["".join(cjk[index : index + 2]) for index in range(max(0, len(cjk) - 1))]
    return {item for item in [*latin, *cjk, *bigrams] if len(item) > 1 or item.isalnum()}


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _best_matches(gold: list[str], generated: list[str]) -> list[float]:
    if not gold:
        return []
    return [max((_similarity(item, candidate) for candidate in generated), default=0.0) for item in gold]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_generated_skills(
    generated_skills_path: Path,
    workbook_path: Path,
    *,
    sheet_name: str = "KAlgebra",
) -> dict[str, Any]:
    """Read an Excel benchmark without changing it and emit non-generative comparison metrics.

    This function deliberately has no model client and returns only aggregate scores/row identifiers.
    The generator never receives the workbook or this report as a tool observation.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - exercised by installation guidance
        raise RuntimeError("Excel evaluation requires openpyxl; install the project's dev dependencies.") from exc

    generated = json.loads(generated_skills_path.read_text(encoding="utf-8"))
    workbook = load_workbook(workbook_path, read_only=True, data_only=False)
    if sheet_name not in workbook.sheetnames:
        raise ValueError(f"Workbook does not contain sheet {sheet_name!r}")
    worksheet = workbook[sheet_name]
    try:
        gold_rows = []
        for row in range(2, worksheet.max_row + 1):
            task = str(worksheet.cell(row, 2).value or "").strip()
            skill_id = str(worksheet.cell(row, 12).value or "").strip()
            skill_title = str(worksheet.cell(row, 13).value or "").strip()
            skill_content = str(worksheet.cell(row, 14).value or "").strip()
            reference_id = str(worksheet.cell(row, 15).value or "").strip()
            reference_title = str(worksheet.cell(row, 16).value or "").strip()
            reference_content = str(worksheet.cell(row, 17).value or "").strip()
            if task or skill_id or reference_id:
                gold_rows.append(
                    {
                        "row": row,
                        "task": task,
                        "skill_id": skill_id,
                        "skill_text": "\n".join(item for item in [skill_title, skill_content] if item),
                        "reference_id": reference_id,
                        "reference_text": "\n".join(
                            item for item in [reference_title, reference_content] if item
                        ),
                    }
                )
    finally:
        workbook.close()

    generated_tasks = generated.get("tasks") or []
    generated_references = generated.get("references") or []
    generated_skills = generated.get("skills") or []
    generated_task_text = [
        str(item["task"]) if "task" in item else
        "\n".join(str(item.get(key) or "") for key in ("title", "objective"))
        for item in generated_tasks
    ]
    def reference_text(item: dict[str, Any]) -> str:
        entries = item.get("entries") or []
        entry_text = "\n".join(
            "\n".join(str(entry.get(key) or "") for key in ("title", "content"))
            for entry in entries
            if isinstance(entry, dict)
        )
        return "\n".join(
            value for value in [str(item.get("title") or ""), str(item.get("content") or ""), entry_text] if value
        )

    generated_reference_text = [reference_text(item) for item in generated_references]
    generated_skill_text = [
        "\n".join(
            [
                str(item.get("title") or ""),
                str(item.get("description") or ""),
                "\n".join(str(step.get("instruction") or "") for step in item.get("procedure") or []),
            ]
        )
        for item in generated_skills
    ]
    gold_tasks = [item["task"] for item in gold_rows if item["task"]]
    gold_references = [item["reference_text"] for item in gold_rows if item["reference_text"]]
    gold_skills = [item["skill_text"] for item in gold_rows if item["skill_text"]]
    task_matches = _best_matches(gold_tasks, generated_task_text)
    reference_matches = _best_matches(gold_references, generated_reference_text)
    skill_matches = _best_matches(gold_skills, generated_skill_text)
    cited_skills = sum(
        1 for item in generated_skills
        if any(step.get("citations") for step in item.get("procedure") or [])
    )
    cited_steps = sum(
        1
        for item in generated_skills
        for step in item.get("procedure") or []
        if step.get("citations")
    )
    total_steps = sum(len(item.get("procedure") or []) for item in generated_skills)

    def mean(items: list[float]) -> float:
        return round(sum(items) / len(items), 4) if items else 0.0

    report = {
        "schema_version": "v1",
        "benchmark": {
            "workbook_sha256": _sha256(workbook_path),
            "sheet": sheet_name,
            "gold_task_rows": len(gold_tasks),
            "gold_reference_rows": len(gold_references),
            "gold_skill_rows": len(gold_skills),
        },
        "generated": {
            "skills_file": str(generated_skills_path),
            "task_count": len(generated_tasks),
            "reference_count": len(generated_references),
            "skill_count": len(generated_skills),
            "skills_with_citations_ratio": round(cited_skills / len(generated_skills), 4)
            if generated_skills
            else 0.0,
            "procedure_steps_with_citations_ratio": round(cited_steps / total_steps, 4)
            if total_steps
            else 0.0,
        },
        "lexical_comparison": {
            "task_best_match_mean": mean(task_matches),
            "reference_best_match_mean": mean(reference_matches),
            "skill_best_match_mean": mean(skill_matches),
            "task_matches_at_or_above_0_15": sum(item >= 0.15 for item in task_matches),
            "reference_matches_at_or_above_0_15": sum(item >= 0.15 for item in reference_matches),
            "skill_matches_at_or_above_0_15": sum(item >= 0.15 for item in skill_matches),
        },
        "notes": [
            "The benchmark workbook was opened read-only and never supplied to a generation Agent.",
            "Lexical similarity is a diagnostic for coverage, not a quality or factuality verdict.",
            "Evidence citation validity is enforced by the generation pipeline before export.",
        ],
    }
    return report


def write_evaluation_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return output_path
