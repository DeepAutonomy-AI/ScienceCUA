from __future__ import annotations

import json
from typing import Any

import httpx


class ModelAPIError(RuntimeError):
    """A safe, user-facing description of an API Agent failure."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "api_agent_error",
        status_code: int | None = None,
        provider_message: str = "",
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.provider_message = provider_message

    def public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": type(self).__name__,
            "category": self.category,
            "message": str(self),
        }
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        if self.provider_message:
            payload["provider_message"] = self.provider_message
        return payload


class PipelineError(RuntimeError):
    """A safe, user-facing description of a non-API pipeline failure."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "pipeline_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.details = details or {}

    def public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": type(self).__name__,
            "category": self.category,
            "message": str(self),
        }
        if self.details:
            payload["details"] = self.details
        return payload


class RunAbortedError(PipelineError):
    """Raised inside a worker after another worker has failed or the run was cancelled."""

    def __init__(self) -> None:
        super().__init__(
            "The current run has stopped; no further API requests will be sent.",
            category="run_aborted",
        )


class NoUsableSourcesError(PipelineError):
    """Raised when source acquisition produced no modality the pipeline can process."""

    def __init__(
        self,
        *,
        input_count: int,
        excluded_count: int,
        failed_count: int,
        reason_counts: dict[str, int],
    ) -> None:
        super().__init__(
            "No usable sources are available: all sources were excluded or failed during acquisition. The pipeline has stopped.",
            category="no_usable_sources",
            details={
                "input_count": input_count,
                "excluded_count": excluded_count,
                "failed_count": failed_count,
                "reason_counts": reason_counts,
            },
        )


def is_quota_error(status_code: int, message: str) -> bool:
    if status_code not in {402, 429}:
        return False
    normalized = " ".join(str(message or "").casefold().split())
    return status_code == 402 or any(
        marker in normalized
        for marker in (
            "insufficient_quota",
            "insufficient quota",
            "quota exceeded",
            "exceeded your current quota",
            "billing",
            "credit balance",
            "no credit",
            "not enough credits",
            "balance is insufficient",
            "余额不足",
            "额度不足",
            "没有额度",
            "无额度",
            "欠费",
        )
    )


def as_model_api_error(exc: Exception, *, context: str = "API Agent") -> ModelAPIError:
    if isinstance(exc, ModelAPIError):
        return exc
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        provider_message = _provider_error_message(exc.response)
        if is_quota_error(status_code, provider_message):
            category = "quota_exhausted"
            summary = "The API quota is insufficient or exhausted. Add credit or use another available account, then retry."
        elif status_code == 429:
            category = "rate_limited"
            summary = "The API request rate remains limited after multiple retries."
        elif status_code == 401:
            category = "authentication_error"
            summary = "API authentication failed. Check the API key."
        elif status_code == 403:
            category = "permission_denied"
            summary = "The API denied access. Check account, model, and endpoint permissions."
        elif status_code == 404:
            category = "model_or_endpoint_not_found"
            summary = "The API endpoint or model was not found. Check the Base URL and model name."
        elif status_code in {400, 409, 413, 415, 422}:
            category = "invalid_api_request"
            summary = "The API rejected the request. Check the protocol, model capabilities, and request configuration."
        elif status_code >= 500:
            category = "provider_error"
            summary = "The API provider returned a server error that persisted after multiple retries."
        else:
            category = "api_http_error"
            summary = f"The API request failed (HTTP {status_code})."
        message = f"{context} failed: {summary}"
        if provider_message:
            message += f" Provider details: {provider_message}"
        return ModelAPIError(
            message,
            category=category,
            status_code=status_code,
            provider_message=provider_message,
        )
    if isinstance(exc, httpx.TimeoutException):
        return ModelAPIError(
            f"{context} failed: the API request timed out. Check service status, network connectivity, and timeout settings.",
            category="api_timeout",
            provider_message=str(exc)[:1_000],
        )
    if isinstance(exc, httpx.TransportError):
        detail = str(exc)[:1_000]
        message = f"{context} failed: could not connect to the API service. Check the Base URL, network, and service status."
        if detail:
            message += f" Connection details: {detail}"
        return ModelAPIError(
            message,
            category="api_connection_error",
            provider_message=detail,
        )
    if isinstance(exc, (json.JSONDecodeError, KeyError, TypeError)):
        detail = str(exc)[:1_000]
        return ModelAPIError(
            f"{context} failed: the API returned an unparseable response. Response details: {detail}",
            category="invalid_api_response",
            provider_message=detail,
        )
    detail = str(exc).strip()[:1_000]
    message = f"{context} failed"
    if detail:
        message += f": {detail}"
    return ModelAPIError(message, category="api_agent_error", provider_message=detail)


def public_pipeline_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ModelAPIError):
        return exc.public_dict()
    if isinstance(exc, PipelineError):
        return exc.public_dict()
    return {
        "type": type(exc).__name__,
        "category": "pipeline_error",
        "message": str(exc) or "The pipeline encountered an unrecovered error.",
    }


def _provider_error_message(response: httpx.Response) -> str:
    text = response.text.strip()
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return text[:1_500]
    values: list[str] = []
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            for key in ("type", "code", "message"):
                value = error.get(key)
                if value not in (None, ""):
                    values.append(str(value))
        elif error not in (None, ""):
            values.append(str(error))
        for key in ("type", "code", "message", "detail"):
            value = payload.get(key)
            if value not in (None, "") and str(value) not in values:
                values.append(str(value))
    if values:
        return " | ".join(values)[:1_500]
    return text[:1_500]
