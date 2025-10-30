from __future__ import annotations
from typing import Any
from rest_framework.views import exception_handler as drf_exception_handler
from rest_framework.response import Response
from rest_framework import status as http
from rest_framework.exceptions import (
    APIException, ValidationError, NotFound, PermissionDenied,
    AuthenticationFailed, NotAuthenticated
)

def _stringify_detail(detail: Any) -> str:
    if isinstance(detail, (list, tuple)):
        return " ; ".join([_stringify_detail(x) for x in detail])
    if isinstance(detail, dict):
        try:
            k, v = next(iter(detail.items()))
            return f"{k}: {_stringify_detail(v)}"
        except StopIteration:
            return ""
    return str(detail)

def custom_exception_handler(exc, context):
    """
    Wrap all DRF errors into HTTP 200 with a normalized body.
    Adds a compact 'code' for client-side branching.
    """
    resp = drf_exception_handler(exc, context)

    if resp is None:
        return Response(
            {"status": "server_error", "code": "server_error", "message": _stringify_detail(str(exc))},
            status=200
        )

    if isinstance(exc, ValidationError):
        payload = {"status": "bad_request", "code": "validation_error", "message": _stringify_detail(exc.detail)}
        if isinstance(exc.detail, (dict, list)):
            payload["errors"] = exc.detail
        return Response(payload, status=200)

    if isinstance(exc, NotFound):
        return Response({"status": "not_found", "code": "not_found", "message": _stringify_detail(exc.detail)}, status=200)

    if isinstance(exc, (PermissionDenied, NotAuthenticated, AuthenticationFailed)):
        return Response({"status": "permission_denied", "code": "permission_denied", "message": _stringify_detail(exc.detail)}, status=200)

    if isinstance(exc, APIException):
        return Response({"status": "api_error", "code": getattr(exc, "default_code", "api_error"), "message": _stringify_detail(exc.detail)}, status=200)

    return Response({"status": "error", "code": "error", "message": _stringify_detail(exc.detail)}, status=200)
