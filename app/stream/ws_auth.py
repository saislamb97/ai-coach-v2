# socket/ws_auth.py
from __future__ import annotations
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs
from ipaddress import ip_address, ip_network

from django.utils.encoding import force_str
from user.models import ApiAuth

# ---- header helpers from scope ----
def _get_header(scope, key: bytes) -> Optional[str]:
    for k, v in scope.get("headers", []):
        if k.lower() == key.lower():
            return force_str(v)
    return None

def get_origin_host_from_scope(scope) -> Optional[str]:
    origin = _get_header(scope, b"origin") or _get_header(scope, b"referer")
    if not origin:
        return None
    try:
        return urlparse(origin).netloc.split(":")[0].lower()
    except Exception:
        return None

def get_host_header_from_scope(scope) -> Optional[str]:
    host = _get_header(scope, b"host")
    if not host:
        return None
    return host.split(":")[0].lower()

def get_client_ip_from_scope(scope) -> Optional[str]:
    # honor x-forwarded-for if present (left-most)
    xff = _get_header(scope, b"x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    client = scope.get("client")
    if isinstance(client, (list, tuple)) and client:
        return str(client[0])
    return None

# ---- query/cookie helpers ----
def _get_query_param(scope, key: str) -> Optional[str]:
    try:
        raw = scope.get("query_string") or b""
        qs = parse_qs(force_str(raw), keep_blank_values=False)
        vals = qs.get(key)
        if vals:
            v = (vals[0] or "").strip()
            return v or None
    except Exception:
        pass
    return None

def _get_cookie(scope, name: str) -> Optional[str]:
    cookie = _get_header(scope, b"cookie") or ""
    if not cookie:
        return None
    # Simple parse; avoids importing cookies lib
    try:
        parts = [p.strip() for p in cookie.split(";")]
        for p in parts:
            if not p or "=" not in p:
                continue
            k, v = p.split("=", 1)
            if k.strip() == name:
                return v.strip() or None
    except Exception:
        return None
    return None

def extract_api_key_from_scope(scope) -> Optional[str]:
    """
    Order of precedence:
      1) Authorization: Api-Key <key>
      2) query string:  ?api_key=<key>
      3) cookie:        api_key=<key>
    """
    # 1) Authorization header
    auth = _get_header(scope, b"authorization") or ""
    if auth.startswith("Api-Key "):
        key = auth.split(" ", 1)[1].strip()
        if key:
            return key

    # 2) Query string (WS-friendly)
    key = _get_query_param(scope, "api_key") or _get_query_param(scope, "apikey")
    if key:
        return key

    # 3) Cookie fallback (also WS-friendly)
    key = _get_cookie(scope, "api_key")
    if key:
        return key

    return None

# ---- allowlist utils (same behavior as your DRF permission) ----
def _normalize_allowed(entries) -> set[str]:
    out: set[str] = set()
    for e in entries or []:
        e = (e or "").strip()
        if not e:
            continue
        if e == "*":
            out.add("*"); continue
        if "://" in e:
            try:
                host = urlparse(e).netloc.split(":")[0].lower()
                if host: out.add(host)
            except Exception:
                continue
        else:
            out.add(e.split(":")[0].lower())
    return out

def _host_allowed(host: Optional[str], allowed: set[str]) -> bool:
    if not host: return False
    h = host.lower()
    if "*" in allowed: return True
    if h in allowed: return True
    for a in allowed:
        if a.startswith(".") and (h.endswith(a) or h == a.lstrip(".")):
            return True
    return False

def _ip_allowed(ip: Optional[str], allowed: set[str]) -> bool:
    if not ip: return False
    if ip in allowed: return True
    for a in allowed:
        if "/" in a:
            try:
                if ip_address(ip) in ip_network(a, strict=False):
                    return True
            except Exception:
                continue
    return False

# ---- main validator for WS connect ----
def validate_ws_api_key_and_origin(scope) -> Tuple[Optional[ApiAuth], Optional[str]]:
    """
    Returns (ApiAuth, None) if ok; (None, error_msg) if denied.
    Accepts API key via:
      - Authorization: Api-Key <key>
      - ?api_key=<key> (WS query param)
      - cookie: api_key=<key>
    """
    key = extract_api_key_from_scope(scope)
    if not key:
        return None, (
            "Missing API key. Provide it as 'Authorization: Api-Key <key>', "
            "or in the WebSocket URL '?api_key=<key>', or a 'api_key' cookie."
        )

    try:
        auth = ApiAuth.objects.get(api_key=key, is_active=True)
    except ApiAuth.DoesNotExist:
        return None, "Invalid or inactive API key."

    allowed = _normalize_allowed(auth.allowed_origins)
    origin_host = get_origin_host_from_scope(scope)
    host_hdr    = get_host_header_from_scope(scope)
    ip          = get_client_ip_from_scope(scope)

    if _host_allowed(origin_host, allowed): return auth, None
    if _host_allowed(host_hdr, allowed):    return auth, None
    if _ip_allowed(ip, allowed):            return auth, None

    return None, "Origin/Host/IP not allowed for this API key."
