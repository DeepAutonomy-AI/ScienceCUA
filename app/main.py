from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import unquote, urlparse

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import __version__
from .config import model_defaults, normalize_output_language, settings
from .generation_coverage import GenerationCoverage, generation_coverage_state, saved_generation_coverage_state
from .orchestration import PipelineRuntime
from .knowledge import load_corpus_snapshot
from .models import ModelRuntimeConfig, StagedInput, TargetProfile, normalize_saved_target_profile_payload
from .runtime import DshAgentRuntime
from .result_preview import read_result_preview, read_source_previews
from .orchestration.source_stage import SourceStageError
from .storage import RunStore, list_run_ids


app = FastAPI(title="Resource-to-Skill DSH Pipeline", version=__version__)
runtime = PipelineRuntime(settings, DshAgentRuntime())
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
PDF_SIZE_LIMIT_MESSAGE = "The PDF is too large. Only PDF files smaller than 50 MB are supported."
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024


class SourceSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1, max_length=100)
    api_key: str | None = Field(default=None, repr=False, max_length=4096)


class SourceSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str | None = Field(default=None, repr=False, max_length=4096)


class ScopeSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outline_artifact_id: str = Field(pattern=r"^art_[0-9a-f]{24}$")
    mode: Literal["full", "selected"]
    section_ids: list[str] = Field(default_factory=list)
    api_key: str | None = Field(default=None, repr=False, max_length=4096)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(
        static_dir / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": app.version,
        "data_dir": str(settings.data_dir),
        "dsh_runtime": "native",
        "scope_selection_enabled": runtime.settings.scope_selection_enabled,
        "html_processing_mode": runtime.settings.html_processing_mode,
    }


@app.get("/api/model-defaults")
async def public_model_defaults() -> dict[str, str | int]:
    return model_defaults.public_dict()


@app.post("/api/runs", status_code=202)
async def create_run(
    software_name: Annotated[str, Form()] = "",
    target_version: Annotated[str, Form()] = "",
    platform: Annotated[str, Form()] = "",
    ui_locale: Annotated[str, Form()] = "unknown",
    source_mode: Annotated[Literal["manual", "discovery"] | None, Form()] = None,
    output_language: Annotated[str | None, Form()] = None,
    generation_coverage: Annotated[GenerationCoverage, Form()] = "medium",
    software_id: Annotated[str, Form()] = "",
    urls_json: Annotated[str, Form()] = "[]",
    file_versions_json: Annotated[str, Form()] = "[]",
    api_protocol: Annotated[str, Form()] = "",
    api_base_url: Annotated[str, Form()] = model_defaults.base_url,
    api_model: Annotated[str, Form()] = model_defaults.model,
    reasoning_effort: Annotated[str, Form()] = model_defaults.reasoning_effort,
    api_key: Annotated[str, Form()] = "",
    websearch: Annotated[bool, Form()] = False,
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> dict[str, Any]:
    if websearch:
        raise HTTPException(
            status_code=422,
            detail="Web Search is disabled for the legacy websearch flag; use source_mode=discovery.",
        )
    try:
        url_inputs = json.loads(urls_json)
        file_versions = json.loads(file_versions_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON form field: {exc}") from exc
    if not isinstance(url_inputs, list) or not isinstance(file_versions, list):
        raise HTTPException(status_code=422, detail="URLs and file versions must be arrays")
    try:
        resolved_language = normalize_output_language(output_language or settings.skill_output_language)
        target = TargetProfile(
            software_name=software_name.strip(),
            software_id=software_id.strip() or None,
            target_version=target_version.strip(),
            platform=platform.strip(),
            ui_locale=ui_locale,
        )
        effective_api_key = _resolve_api_key(api_key, api_base_url)
        model_config = ModelRuntimeConfig(
            api_protocol=api_protocol.strip(),
            base_url=api_base_url,
            model=api_model,
            reasoning_effort=reasoning_effort.strip() or model_defaults.reasoning_effort,
            api_key=effective_api_key,
            websearch=False,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not model_config.api_key or not model_config.model.strip():
        raise HTTPException(status_code=422, detail="API Agent requires both API Key and model name")

    upload_files = files or []
    has_url = any(
        (isinstance(item, str) and item.strip())
        or (isinstance(item, dict) and str(item.get("url") or "").strip())
        for item in url_inputs
    )
    source_count = len(upload_files) + sum(
        bool(item.strip()) if isinstance(item, str) else
        bool(str(item.get("url") or "").strip()) if isinstance(item, dict) else False
        for item in url_inputs
    )
    if source_mode == "manual" and source_count != 1:
        raise HTTPException(status_code=422, detail="Please provide exactly one PDF or HTML source")
    if source_mode == "discovery" and source_count:
        raise HTTPException(status_code=422, detail="Automatic discovery does not accept uploaded sources")
    if not upload_files and not has_url and source_mode != "discovery":
        raise HTTPException(status_code=422, detail="At least one file or URL is required")
    for item in url_inputs:
        candidate = item if isinstance(item, str) else item.get("url") if isinstance(item, dict) else ""
        parsed = urlparse(str(candidate or "").strip())
        if parsed.username or parsed.password:
            raise HTTPException(status_code=422, detail="Source URLs cannot contain embedded credentials")
        hostname = (parsed.hostname or "").casefold()
        if hostname == "github.com" or hostname.endswith(".github.com"):
            raise HTTPException(
                status_code=422,
                detail="This pipeline supports PDF/HTML sources and does not include GitHub repositories.",
            )

    upload_payloads: list[tuple[UploadFile, bytes]] = []
    for upload in upload_files:
        data = await _read_upload_limited(upload, settings.max_source_bytes)
        if len(data) > settings.max_source_bytes:
            raise HTTPException(
                status_code=413,
                detail=_source_size_error_detail(upload.filename, upload.content_type, data),
            )
        upload_payloads.append((upload, data))

    run_id = f"run_{uuid.uuid4().hex[:20]}"
    store = runtime.create_store(run_id)
    staged_inputs: list[StagedInput] = []
    for index, (upload, data) in enumerate(upload_payloads):
        input_id = f"IN{index + 1}"
        path = store.stage_upload(input_id, upload.filename or "upload.bin", data)
        version = str(file_versions[index]) if index < len(file_versions) else "unknown"
        staged_inputs.append(
            StagedInput(
                input_id=input_id,
                kind="uploaded_file",
                display_name=upload.filename or "upload.bin",
                user_declared_version=version or "unknown",
                staged_path=str(path),
                declared_media_type=upload.content_type,
            )
        )
    for item in url_inputs:
        if isinstance(item, str):
            url = item.strip()
            version = "unknown"
        elif isinstance(item, dict):
            url = str(item.get("url") or "").strip()
            version = str(item.get("user_declared_version") or "unknown")
        else:
            continue
        if not url:
            continue
        staged_inputs.append(
            StagedInput(
                input_id=f"IN{len(staged_inputs) + 1}",
                kind="url",
                display_name=_display_name_for_url(url),
                user_declared_version=version,
                url=url,
            )
        )
    if not staged_inputs and source_mode != "discovery":
        raise HTTPException(status_code=422, detail="At least one valid file or URL is required")

    public_inputs = [
        {
            "input_id": item.input_id,
            "kind": item.kind,
            "display_name": item.display_name,
            "user_declared_version": item.user_declared_version,
            "url": item.url,
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
            "source_mode": source_mode or "manual",
            "output_language": resolved_language,
            **generation_coverage_state(generation_coverage),
            "scope_selection_enabled": settings.scope_selection_enabled and len(staged_inputs) <= 1,
        }
    )
    store.emit(
        "run_registered",
        {
            "target_profile": target.model_dump(mode="json"),
            "inputs": public_inputs,
            "model": model_config.public_dict(),
            **generation_coverage_state(generation_coverage),
        },
    )
    runtime.start(run_id, target, staged_inputs, model_config)
    return {"run_id": run_id, "status": "queued"}


@app.get("/api/runs")
async def list_runs(limit: int = Query(default=60, ge=1, le=100)) -> dict[str, Any]:
    runs = []
    for run_id in list_run_ids(settings.data_dir)[:limit]:
        state = RunStore(settings.data_dir, run_id).read_state()
        if "target_profile" in state:
            state["target_profile"] = normalize_saved_target_profile_payload(state["target_profile"])
        runs.append(state)
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    store = _existing_store(run_id)
    runtime.recover_scope_state(run_id)
    state = store.read_state()
    if "target_profile" in state:
        state["target_profile"] = normalize_saved_target_profile_payload(state["target_profile"])
    state["active_task"] = runtime.has_active_task(run_id)
    state["requires_api_key"] = (
        state.get("phase") in {"awaiting_source_selection", "source_selected",
                               "awaiting_scope_selection", "scope_selected"}
        and not runtime.has_model_credentials(run_id)
        and not _resolve_api_key("", str(state.get("model", {}).get("base_url") or ""))
    )
    result = read_result_preview(store, state)
    sources = [] if ((result or {}).get("snapshot") or {}).get("source_results") else read_source_previews(store, state)
    return {"state": state, "result": result, "sources": sources,
            "source_discovery": runtime.source_discovery_payload(run_id),
            "source_scope": runtime.source_scope_payload(run_id)}


def _source_action_key(run_id: str, api_key: str | None) -> str | None:
    if (api_key or "").strip() or not runtime.has_model_credentials(run_id):
        base_url = str(runtime.get_store(run_id).read_state().get("model", {}).get("base_url") or "")
        return _resolve_api_key(api_key or "", base_url)
    return None


@app.post("/api/runs/{run_id}/select-source", status_code=202)
async def select_source(run_id: str, selection: SourceSelectionRequest) -> dict[str, Any]:
    _existing_store(run_id)
    try:
        return runtime.select_source(run_id, selection.candidate_id,
                                     _source_action_key(run_id, selection.api_key))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/search-again", status_code=202)
async def search_again(run_id: str, request: SourceSearchRequest) -> dict[str, Any]:
    _existing_store(run_id)
    try:
        return runtime.search_again(run_id, _source_action_key(run_id, request.api_key))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/select-scope", status_code=202)
async def select_scope(run_id: str, selection: ScopeSelectionRequest) -> dict[str, Any]:
    _existing_store(run_id)
    try:
        return runtime.select_scope(
            run_id, selection.outline_artifact_id, selection.mode, selection.section_ids,
            _source_action_key(run_id, selection.api_key),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/rerun-reference", status_code=202)
async def rerun_reference_from_frozen_t5(
    run_id: str,
    output_language: Annotated[str | None, Form()] = None,
    generation_coverage: Annotated[GenerationCoverage | None, Form()] = None,
    api_protocol: Annotated[str, Form()] = "",
    api_base_url: Annotated[str, Form()] = model_defaults.base_url,
    api_model: Annotated[str, Form()] = model_defaults.model,
    reasoning_effort: Annotated[str, Form()] = model_defaults.reasoning_effort,
    api_key: Annotated[str, Form()] = "",
) -> dict[str, Any]:
    """Create a fresh Run that reuses T5 and executes every downstream stage."""

    source_store = _existing_store(run_id)
    source_state = source_store.read_state()
    corpus_artifact_id = source_state.get("corpus_snapshot_artifact_id")
    if not source_state.get("t5_stage_complete") or not isinstance(
        corpus_artifact_id, str
    ):
        raise HTTPException(
            status_code=409,
            detail="This Run has no frozen T5 CorpusSnapshot to reuse.",
        )
    try:
        resolved_language = normalize_output_language(
            output_language or source_state.get("output_language") or settings.skill_output_language
        )
        frozen = load_corpus_snapshot(source_store, corpus_artifact_id)
        target = frozen.target_profile
        generation_configuration = (
            generation_coverage_state(generation_coverage)
            if generation_coverage is not None
            else saved_generation_coverage_state(settings, source_state)
        )
        effective_api_key = _resolve_api_key(api_key, api_base_url)
        model_config = ModelRuntimeConfig(
            api_protocol=api_protocol.strip(),
            base_url=api_base_url,
            model=api_model,
            reasoning_effort=reasoning_effort.strip() or model_defaults.reasoning_effort,
            api_key=effective_api_key,
            websearch=False,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail=f"The frozen T5 cache is incomplete or invalid: {exc}",
        ) from exc
    if not model_config.api_key or not model_config.model.strip():
        raise HTTPException(status_code=422, detail="API Agent requires both API Key and model name")

    new_run_id = f"run_{uuid.uuid4().hex[:20]}"
    store = runtime.create_store(new_run_id)
    store.initialize(
        {
            "run_id": new_run_id,
            "target_profile": target.model_dump(mode="json"),
            "inputs": [],
            "model": model_config.public_dict(),
            "execution_mode": "full_pipeline_from_frozen_t5",
            "output_language": resolved_language,
            **generation_configuration,
            "t5_cache_source_run_id": run_id,
            "t5_cache_corpus_snapshot_id": frozen.corpus_snapshot_id,
        }
    )
    store.emit(
        "run_registered",
        {
            "target_profile": target.model_dump(mode="json"),
            "inputs": [],
            "model": model_config.public_dict(),
            "execution_mode": "full_pipeline_from_frozen_t5",
            "t5_cache_source_run_id": run_id,
            "t5_cache_corpus_snapshot_id": frozen.corpus_snapshot_id,
            **generation_configuration,
        },
    )
    try:
        imported = await asyncio.to_thread(
            runtime.import_frozen_corpus,
            run_id=new_run_id,
            source_store=source_store,
            corpus_artifact_id=corpus_artifact_id,
        )
    except (FileNotFoundError, ValueError, SourceStageError) as exc:
        raise HTTPException(
            status_code=409,
            detail=f"The frozen T5 cache is incomplete or invalid: {exc}",
        ) from exc
    runtime.start_reference_from_frozen(new_run_id, target, model_config)
    return {
        "run_id": new_run_id,
        "status": "queued",
        "execution_mode": "full_pipeline_from_frozen_t5",
        "t5_cache_source_run_id": run_id,
        "corpus_snapshot_id": imported.corpus_snapshot_id,
        "corpus_snapshot_artifact_id": store.read_state()["corpus_snapshot_artifact_id"],
    }


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, Any]:
    _existing_store(run_id)
    return runtime.cancel(run_id)


@app.get("/api/runs/{run_id}/events")
async def get_events(
    run_id: str,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    store = _existing_store(run_id)
    events, next_offset = store.read_events(after=after, limit=limit)
    return {"events": events, "next_offset": next_offset}


@app.get("/api/runs/{run_id}/artifacts/{artifact_id}")
async def get_artifact(run_id: str, artifact_id: str) -> FileResponse:
    store = _existing_store(run_id)
    try:
        record = store.artifact_record(artifact_id)
        path = store.artifact_path(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    return FileResponse(
        path,
        media_type=record.media_type,
        filename=path.name,
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


def _existing_store(run_id: str) -> RunStore:
    if not run_id.startswith("run_") or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for char in run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    root = settings.data_dir / "runs" / run_id
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return runtime.get_store(run_id)


def _display_name_for_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https", "file"}:
        candidate = Path(unquote(parsed.path)).name
        return candidate or parsed.netloc or url[:160]
    stripped = url.rstrip("/\\")
    candidate = Path(stripped.replace("\\", "/")).name
    return candidate or url[:160]


def _resolve_api_key(submitted_key: str, base_url: str) -> str | None:
    """Select a process-only credential without serializing it into run state.

    A submitted form value takes precedence. A custom credential variable is
    used only for the configured endpoint. Nimabo endpoints keep their isolated
    credential; other endpoints use the standard OpenAI-compatible variable.
    """
    if submitted_key.strip():
        return submitted_key.strip()
    if base_url.strip().rstrip("/") == model_defaults.base_url and model_defaults.api_key_env != "OPENAI_API_KEY":
        return os.getenv(model_defaults.api_key_env)
    hostname = (urlparse(base_url).hostname or "").casefold()
    env_name = (
        "NIMABO_API_KEY"
        if hostname == "nimabo.io" or hostname.endswith(".nimabo.io")
        else "OPENAI_API_KEY"
    )
    return os.getenv(env_name)


async def _read_upload_limited(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(UPLOAD_READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        chunks.append(chunk)
        if total > limit:
            break
    return b"".join(chunks)


def _source_size_error_detail(
    filename: str | None, content_type: str | None, data: bytes
) -> str:
    lower_name = (filename or "").casefold()
    lower_type = (content_type or "").casefold()
    if (
        lower_name.endswith(".pdf")
        or lower_type == "application/pdf"
        or data.startswith(b"%PDF-")
    ):
        return PDF_SIZE_LIMIT_MESSAGE
    return "Uploaded file is too large; the system only supports files under 50MB."


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
