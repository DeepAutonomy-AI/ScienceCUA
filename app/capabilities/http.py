from __future__ import annotations

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

from .gateway import CapabilityGateway, CapabilityGatewayError, CapabilityInvocation


def create_capability_router(gateway: CapabilityGateway) -> APIRouter:
    """Create the private HTTP surface consumed by the DSH proxy."""

    router = APIRouter()

    @router.post("/invoke", include_in_schema=False)
    def invoke(
        request: CapabilityInvocation,
        authorization: str | None = Header(default=None),
    ):
        prefix = "Bearer "
        token = authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
        try:
            result = gateway.invoke(token, request)
        except CapabilityGatewayError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"ok": False, "error": {"code": exc.code, "message": str(exc)}},
            )
        return {"ok": True, "result": result}

    return router
