"""Protocol inference for OpenAI-compatible runtime configuration."""

from urllib.parse import urlparse


def default_api_protocol(base_url: str) -> str:
    """Infer Responses only from its endpoint path; provider names are irrelevant."""
    path = urlparse(base_url.strip()).path.rstrip("/")
    return "responses" if path.endswith("/responses") else "chat_completions"
