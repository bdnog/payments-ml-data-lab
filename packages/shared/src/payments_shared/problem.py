"""RFC 7807 ``application/problem+json`` support for the FastAPI services.

Needs FastAPI (``pip install payments-shared[http]``), so it is not imported by the package root.
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from payments_shared.schemas import ProblemDetails

PROBLEM_MEDIA_TYPE = "application/problem+json"


class ProblemJSONResponse(JSONResponse):
    media_type = PROBLEM_MEDIA_TYPE


def problem_response(
    request: Request, status: int, title: str, detail: str | None = None, **extra: Any
) -> ProblemJSONResponse:
    problem = ProblemDetails(
        title=title, status=status, detail=detail, instance=request.url.path, **extra
    )
    return ProblemJSONResponse(status_code=status, content=problem.model_dump(exclude_none=True))


def not_implemented(request: Request) -> ProblemJSONResponse:
    """The answer of every stub route: in the contract, not built yet."""
    detail = (
        f"{request.method} {request.url.path} is part of the contract but is not implemented yet."
    )
    return problem_response(request, 501, "Not Implemented", detail)


def problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """The ``responses=`` entry that documents ``ProblemDetails`` for the given statuses."""
    return {
        status: {"model": ProblemDetails, "description": _TITLES[status]} for status in statuses
    }


_TITLES = {
    400: "Bad request",
    404: "Not found",
    409: "Conflict",
    422: "Validation error",
    501: "Not implemented",
}


def install_problem_handling(app: FastAPI) -> None:
    """Serve validation errors as problems and describe problems with the right media type."""

    @app.exception_handler(RequestValidationError)
    async def _validation_problem(request: Request, exc: RequestValidationError):
        errors = jsonable_encoder(exc.errors(), custom_encoder={Exception: str})
        trimmed = [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in errors]
        return problem_response(
            request, 422, "Validation error", "The request is not valid.", errors=trimmed
        )

    original_openapi = app.openapi

    def openapi() -> dict[str, Any]:
        schema = original_openapi()
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                if not isinstance(operation, dict):
                    continue
                for response in operation.get("responses", {}).values():
                    content = response.get("content", {})
                    json_entry = content.get("application/json")
                    ref = (json_entry or {}).get("schema", {}).get("$ref", "")
                    if ref.endswith("/ProblemDetails"):
                        content[PROBLEM_MEDIA_TYPE] = content.pop("application/json")
        return schema

    app.openapi = openapi  # type: ignore[method-assign]
