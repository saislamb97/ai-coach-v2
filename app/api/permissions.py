# api/permissions.py
from __future__ import annotations

import logging
from typing import Optional

from rest_framework.permissions import BasePermission
from rest_framework.request import Request

from api.utils import (
    get_active_auth,
    get_origin_host,
    get_client_ip,
    normalize_allowed_entries,
    host_in_allowed,
    ip_in_allowed,
)

log = logging.getLogger("permissions.api.permissions")


def _redact_api_key(raw: str) -> str:
    """
    Only show last 4 chars of an API key when logging.
    """
    try:
        if raw.startswith("Api-Key "):
            val = raw.split(" ", 1)[1].strip()
            return f"Api-Key ****{val[-4:]}"
        return raw
    except Exception:
        return raw


class HasValidAPIKeyAndAllowedOrigin(BasePermission):
    """
    Permission check for every API/WS request.

    Requires BOTH:
      1. Valid + active ApiAuth key in Authorization header:
           Authorization: Api-Key <key>
      2. One of these allowlist checks to pass:
           - Origin/Referer host is allowed
           - OR Host header is allowed
           - OR client IP is allowed (exact or CIDR)

    The allowlist lives in ApiAuth.allowed_origins.
    """

    message = "Invalid API key or unauthorized origin/IP."

    def has_permission(self, request: Request, view) -> bool:
        origin = request.META.get("HTTP_ORIGIN")
        referer = request.META.get("HTTP_REFERER")
        host_hdr = request.get_host().split(":")[0] if request.get_host() else None
        ip = get_client_ip(request)
        auth_header = request.headers.get("Authorization", "")

        log.info(
            "[auth] method=%s path=%s origin=%s referer=%s host=%s ip=%s auth=%s",
            request.method,
            request.path,
            origin,
            referer,
            host_hdr,
            ip,
            _redact_api_key(auth_header),
        )

        # Step 1: validate API key
        auth_obj, err = get_active_auth(request)
        if not auth_obj:
            self.message = err or "Invalid or inactive API key."
            log.warning(
                "[auth] api_key_error ip=%s path=%s raw_auth=%r",
                ip,
                request.path,
                auth_header,
            )
            return False

        # Step 2: normalize allow list
        allowed = normalize_allowed_entries(auth_obj.allowed_origins)

        # Step 3: Check Origin/Referer host
        origin_host = get_origin_host(request)
        if host_in_allowed(origin_host, allowed):
            log.info("[auth] origin_ok host=%s", origin_host)
            return True

        # Step 4: Check Host header (for backend->backend, curl, etc.)
        if host_in_allowed(host_hdr, allowed):
            log.info("[auth] host_ok host=%s", host_hdr)
            return True

        # Step 5: Check client IP / CIDR
        if ip_in_allowed(ip, allowed):
            log.info("[auth] ip_ok ip=%s", ip)
            return True

        # Nothing matched
        self.message = "Origin/Host/IP not allowed for this API key."
        log.warning(
            "[auth] origin_host_ip_denied ip=%s host=%s allowed=%s",
            ip,
            host_hdr,
            sorted(allowed),
        )
        return False
