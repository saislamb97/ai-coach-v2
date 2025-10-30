# api/utils.py
from __future__ import annotations

from typing import Optional, Tuple, Set
from urllib.parse import urlparse
from ipaddress import ip_address, ip_network

from django.http import HttpRequest

from user.models import ApiAuth


def get_client_ip(request: HttpRequest) -> Optional[str]:
    """
    Try to get the original client IP.
    - Honors X-Forwarded-For (takes the left-most)
    - Falls back to REMOTE_ADDR
    """
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def get_origin_host(request: HttpRequest) -> Optional[str]:
    """
    Extract the host (no port) from Origin or Referer header, lowercased.
    ex: "https://app.example.com/foo" -> "app.example.com"
    """
    raw = request.META.get("HTTP_ORIGIN") or request.META.get("HTTP_REFERER")
    if not raw:
        return None
    try:
        return urlparse(raw).netloc.split(":")[0].lower()
    except Exception:
        return None


def extract_api_key(request: HttpRequest) -> Optional[str]:
    """
    Read Authorization: Api-Key <key>
    Returns just <key> or None.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Api-Key "):
        return None
    return auth.split(" ", 1)[1].strip()


def get_active_auth(request: HttpRequest) -> Tuple[Optional[ApiAuth], Optional[str]]:
    """
    Given a request:
    - Extract the API key from Authorization
    - Look up an active ApiAuth row
    Returns (ApiAuth or None, error_message or None)
    """
    key = extract_api_key(request)
    if not key:
        return None, "Missing Authorization header. Use: 'Api-Key <key>'."

    try:
        auth_obj = ApiAuth.objects.get(api_key=key, is_active=True)
        return auth_obj, None
    except ApiAuth.DoesNotExist:
        return None, "Invalid or inactive API key."


def normalize_allowed_entries(entries) -> Set[str]:
    """
    Normalize allowed_origins entries to a set of comparable tokens.

    We accept:
    - "*" wildcard
    - Plain hostnames ("app.example.com", ".example.com")
    - Full URLs ("https://sub.foo.com") -> extracts host
    - Exact IPs
    - CIDR blocks ("203.0.113.0/24")
    """
    out: Set[str] = set()
    for e in entries or []:
        if not e:
            continue
        e = e.strip()
        if e == "*":
            out.add("*")
            continue
        if "://" in e:
            # looks like a URL
            try:
                host = urlparse(e).netloc.split(":")[0].lower()
                if host:
                    out.add(host)
            except Exception:
                continue
        else:
            # raw host or IP or CIDR
            out.add(e.split(":")[0].lower())
    return out


def host_in_allowed(host: Optional[str], allowed: Set[str]) -> bool:
    """
    Check whether a given host is allowed.
    Supports:
    - exact match
    - wildcard "*"
    - subdomain match for entries that start with "."
    """
    if not host:
        return False
    h = host.lower()

    if "*" in allowed:
        return True
    if h in allowed:
        return True

    # Subdomain wildcard: ".example.com" allows "api.example.com" and "example.com"
    for a in allowed:
        if a.startswith(".") and (h.endswith(a) or h == a.lstrip(".")):
            return True

    return False


def ip_in_allowed(ip: Optional[str], allowed: Set[str]) -> bool:
    """
    Check whether an IP string is allowed directly or via CIDR.
    """
    if not ip:
        return False

    # Exact IP match
    if ip in allowed:
        return True

    # CIDR ranges
    for a in allowed:
        if "/" in a:
            try:
                if ip_address(ip) in ip_network(a, strict=False):
                    return True
            except Exception:
                continue

    return False
