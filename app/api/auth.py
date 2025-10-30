# api/auth.py
from __future__ import annotations

import logging
from typing import Optional, Tuple

from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from rest_framework.request import Request

from user.models import ApiAuth

log = logging.getLogger(__name__)


class ApiKeyAuthentication(BaseAuthentication):
    """
    DRF authentication that:
      - extracts 'Authorization: Api-Key <key>'
      - looks up an active ApiAuth row
      - authenticates as the associated user

    Returns (user, auth_obj) so views can access request.auth.allowed_origins, etc.
    """

    keyword = "Api-Key"

    def _extract_key(self, request: Request) -> Optional[str]:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith(f"{self.keyword} "):
            return None
        return auth_header.split(" ", 1)[1].strip()

    def authenticate(self, request: Request) -> Optional[Tuple[object, ApiAuth]]:
        key = self._extract_key(request)
        if not key:
            # No header or wrong prefix → let other authenticators run (or unauth if none).
            return None

        try:
            auth_obj = ApiAuth.objects.get(api_key=key, is_active=True)
        except ApiAuth.DoesNotExist:
            log.warning("[auth] invalid API key attempt from ip=%s path=%s", request.META.get("REMOTE_ADDR"), request.path)
            raise exceptions.AuthenticationFailed("Invalid or inactive API key.")

        # At this point we trust auth_obj.user as the DRF user.
        return (auth_obj.user, auth_obj)
