# security/middleware/security.py
import re, logging
from datetime import datetime, timedelta
from django.core.cache import cache
from django.http import HttpResponseForbidden
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.deprecation import MiddlewareMixin
from django.conf import settings
from django.dispatch import Signal

from security.utils import get_client_ip

unauthorized_access = Signal()

SQLI = [
    r"(%27)|(')|(--)|(%23)|(#)",
    r"\b(SELECT|UPDATE|DELETE|INSERT|ALTER|DROP|CREATE|REPLACE|TRUNCATE)\b",
    r"\b(OR|AND)\b\s+\d+\s*=\s*\d+",
]

BLOCK_THRESHOLD = 100
ATTEMPT_PERIOD = 3600  # seconds
BLOCK_DURATIONS = [24 * 3600, 48 * 3600, 72 * 3600]

log = logging.getLogger(__name__)

class SecurityMiddleware(MiddlewareMixin):
    def __init__(self, get_response):
        self.get_response = get_response
        # Allow override via settings; default to 'core:api-doc'
        self.home_url_name = getattr(settings, "SECURITY_HOME_URL_NAME", "core:api-doc")
        self.home_url = reverse(self.home_url_name)
        super().__init__(get_response)

    def process_request(self, request):
        ip = get_client_ip(request)

        blocked = cache.get(f"blocked_{ip}")
        if blocked:
            _, expiry = blocked
            if datetime.now() < expiry:
                return HttpResponseForbidden("Your IP is temporarily blocked due to suspicious activity.")

        if self._suspicious(request):
            self._track(ip)
            if self._should_block(ip):
                self._block(ip)
                return HttpResponseForbidden("Your IP is temporarily blocked due to suspicious activity.")
            unauthorized_access.send(sender=self.__class__, request=request, reason="SQL injection attempt")
        return None

    def process_response(self, request, response):
        if response.status_code in (403, 404) and request.path != self.home_url:
            reason = "Forbidden" if response.status_code == 403 else "Not Found"
            unauthorized_access.send(sender=self.__class__, request=request, reason=reason)
            return redirect(self.home_url)
        return response

    def _suspicious(self, request):
        path = request.get_full_path()
        return any(re.search(p, path, re.IGNORECASE) for p in SQLI)

    def _track(self, ip):
        count, last = cache.get(f"attempts_{ip}", (0, None))
        if last and (datetime.now() - last).total_seconds() > ATTEMPT_PERIOD:
            count = 0
        cache.set(f"attempts_{ip}", (count + 1, datetime.now()), timeout=ATTEMPT_PERIOD)

    def _should_block(self, ip):
        count, _ = cache.get(f"attempts_{ip}", (0, None))
        return count >= BLOCK_THRESHOLD

    def _block(self, ip):
        data = cache.get(f"blocked_{ip}")
        if not isinstance(data, tuple) or len(data) != 2:
            data = (0, datetime.now())
        num, _ = data
        dur = BLOCK_DURATIONS[min(num, len(BLOCK_DURATIONS) - 1)]
        expiry = datetime.now() + timedelta(seconds=dur)
        cache.set(f"blocked_{ip}", (num + 1, expiry), timeout=dur)
        cache.delete(f"attempts_{ip}")
        log.warning(f"Blocked IP {ip} for {dur} seconds (count={num + 1}).")
