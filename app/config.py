from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .llm_protocol import default_api_protocol


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelDefaults:
    """Process defaults shared by API, CLI, model validation and the web form."""

    base_url: str = os.getenv("PREPROCESS_API_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model: str = os.getenv("PREPROCESS_API_MODEL", "").strip()
    reasoning_effort: str = os.getenv("PREPROCESS_REASONING_EFFORT", "xhigh").strip()
    api_protocol: str = os.getenv("PREPROCESS_API_PROTOCOL", "").strip()
    api_key_env: str = os.getenv("PREPROCESS_API_KEY_ENV", "OPENAI_API_KEY").strip()

    def __post_init__(self) -> None:
        endpoint = urlparse(self.base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
            raise ValueError("PREPROCESS_API_BASE_URL must be an http(s) URL")
        if endpoint.username or endpoint.password:
            raise ValueError("PREPROCESS_API_BASE_URL cannot contain embedded credentials")
        protocol = self.api_protocol or default_api_protocol(self.base_url)
        if protocol not in {"chat_completions", "responses"}:
            raise ValueError("PREPROCESS_API_PROTOCOL must be chat_completions or responses")
        if self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("PREPROCESS_REASONING_EFFORT has an unsupported value")
        object.__setattr__(self, "api_protocol", protocol)

    def protocol_for(self, base_url: str) -> str:
        if base_url.strip().rstrip("/") == self.base_url:
            return self.api_protocol
        return default_api_protocol(base_url)

    def public_dict(self) -> dict[str, str | int]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_protocol": self.api_protocol,
            "reasoning_effort": self.reasoning_effort,
        }


model_defaults = ModelDefaults()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("PREPROCESS_DATA_DIR", BASE_DIR / "data"))
    output_dir: Path = Path(os.getenv("PREPROCESS_OUTPUT_DIR", BASE_DIR / "output"))
    max_source_bytes: int = int(os.getenv("PREPROCESS_MAX_SOURCE_BYTES", 50 * 1024 * 1024))
    # Controlled child-page budget for each user-provided HTML root. The root
    # page is always normalized and does not consume this budget.
    max_html_pages: int = int(os.getenv("PREPROCESS_MAX_HTML_PAGES", 80))
    # Additive HTML acquisition experiment. The established parser/navigation
    # route remains the default and is available without a model planner.
    html_processing_mode: str = os.getenv(
        "PREPROCESS_HTML_PROCESSING_MODE", "legacy"
    ).strip().casefold()
    html_guided_max_rounds: int = int(os.getenv("PREPROCESS_HTML_GUIDED_MAX_ROUNDS", "3"))
    html_guided_link_batch_size: int = int(os.getenv("PREPROCESS_HTML_GUIDED_LINK_BATCH_SIZE", "80"))
    html_guided_fetch_concurrency: int = int(os.getenv("PREPROCESS_HTML_GUIDED_FETCH_CONCURRENCY", "4"))
    html_guided_planner_transport: str = os.getenv(
        "PREPROCESS_HTML_GUIDED_PLANNER_TRANSPORT", "direct"
    ).strip().casefold()
    html_guided_planner_reasoning_effort: str = os.getenv(
        "PREPROCESS_HTML_GUIDED_PLANNER_REASONING_EFFORT", "none"
    ).strip().casefold()
    html_guided_review_reasoning_effort: str = os.getenv(
        "PREPROCESS_HTML_GUIDED_REVIEW_REASONING_EFFORT", "low"
    ).strip().casefold()
    # Retain chapter selection for development, but ship whole-manual runs.
    scope_selection_enabled: bool = os.getenv(
        "PREPROCESS_SCOPE_SELECTION_ENABLED", "0"
    ).strip().casefold() in {"1", "true", "yes", "on"}
    # T5 profiles:
    # - agent: adaptive script extraction with native DSH tools;
    # - agent_architect: fixed-parser/tree review route (default);
    # - agent_planner: the former full EvidencePlan comparison route;
    # - agent_legacy: the original T5 Source Review comparison route;
    # - agent_curated: the later T5.1 six-tool curation experiment;
    # - deterministic: the later parser-only experiment.
    t5_processing_mode: str = (
        os.getenv("PREPROCESS_T5_PROCESSING_MODE", "agent_architect").strip().casefold()
        or "agent_architect"
    )
    source_extraction_timeout_seconds: float = float(
        os.getenv("PREPROCESS_SOURCE_EXTRACTION_TIMEOUT", "300")
    )
    source_extraction_max_turns: int = int(
        os.getenv("PREPROCESS_SOURCE_EXTRACTION_MAX_TURNS", "3")
    )
    # Downstream stage switches. Production defaults keep the complete pipeline
    # enabled; tests may disable a later slice to isolate a stage.
    t6_reference_enabled: bool = os.getenv(
        "PREPROCESS_T6_REFERENCE_ENABLED", "1"
    ).strip().casefold() not in {"0", "false", "no", "off"}
    # Opt-in presentation pass; source spans and the established review stay intact.
    reference_formatting_mode: str = os.getenv(
        "PREPROCESS_REFERENCE_FORMATTING_MODE", "source_preserved"
    ).strip().casefold()
    # Task Discovery and all later stages remain enabled by default.
    t6_task_enabled: bool = os.getenv(
        "PREPROCESS_T6_TASK_ENABLED", "1"
    ).strip().casefold() not in {"0", "false", "no", "off"}
    # New runs use the combined Task Discovery and Trajectory Inference stage.
    audit_inline_chars: int = int(os.getenv("PREPROCESS_AUDIT_INLINE_CHARS", 8_000))
    api_timeout_seconds: float = float(os.getenv("PREPROCESS_API_TIMEOUT", 120))
    api_max_attempts: int = int(os.getenv("PREPROCESS_API_MAX_ATTEMPTS", 10))
    fetch_timeout_seconds: float = float(os.getenv("PREPROCESS_FETCH_TIMEOUT", 30))
    max_knowledge_candidates: int = int(os.getenv("PREPROCESS_MAX_KNOWLEDGE_CANDIDATES", 24))
    max_task_candidates: int = int(os.getenv("PREPROCESS_MAX_TASK_CANDIDATES", 16))
    skill_induction_concurrency: int = int(os.getenv("PREPROCESS_SKILL_INDUCTION_CONCURRENCY", 5))
    max_evidence_read_chars: int = int(os.getenv("PREPROCESS_MAX_EVIDENCE_READ_CHARS", 18_000))
    websearch_html_max_pages: int = int(os.getenv("PREPROCESS_WEBSEARCH_HTML_MAX_PAGES", 80))
    source_discovery_timeout_seconds: float = float(os.getenv("PREPROCESS_DISCOVERY_TIMEOUT", "600"))
    source_parallel_validation: bool = os.getenv("PREPROCESS_DISCOVERY_PARALLEL_VALIDATION", "1").lower() not in {"0", "false", "off"}
    source_pdf_validation_timeout_seconds: float = float(os.getenv("PREPROCESS_DISCOVERY_PDF_TIMEOUT", "45"))
    max_vision_image_bytes: int = int(
        os.getenv("PREPROCESS_MAX_VISION_IMAGE_BYTES", 5 * 1024 * 1024)
    )
    # This controls generated Skill and Reference prose, not the target software's UI locale.
    skill_output_language: str = (
        os.getenv("PREPROCESS_SKILL_OUTPUT_LANGUAGE", "English").strip() or "English"
    )


settings = Settings()


def normalize_output_language(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"chinese", "zh", "zh-cn", "中文"}:
        return "Chinese"
    if normalized in {"english", "en", "en-us"}:
        return "English"
    raise ValueError("Output language must be Chinese or English")
