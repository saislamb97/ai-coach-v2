import json
import mimetypes
import re
from urllib.parse import urljoin
from django.conf import settings


class SmartMediaMiddleware:
    """
    📤 Response:
        - For API/GraphQL/WS responses with JSON:
            → Convert relative media paths to absolute URLs.
    """

    _api_prefixes = ("/api", "/graphql", "/ws")

    def __init__(self, get_response):
        self.get_response = get_response
        self.media_url = (getattr(settings, "MEDIA_URL", "/media/") or "/media/").rstrip("/") + "/"

    def __call__(self, request):
        path = request.path or ""

        # Normal request handling
        response = self.get_response(request)

        # Only post-process JSON responses for API/GraphQL/WS
        if not any(path.startswith(p) for p in self._api_prefixes):
            return response
        if "application/json" not in (response.get("Content-Type") or ""):
            return response

        try:
            payload = json.loads(response.content)
        except Exception:
            return response

        def looks_like_file(value: str) -> bool:
            if "/" not in value:
                return False
            mime, _ = mimetypes.guess_type(value)
            return mime is not None

        def to_absolute_url(p: str) -> str:
            if p.startswith(("http://", "https://")):
                return p
            rel_path = p[len(self.media_url):] if p.startswith(self.media_url) else p.lstrip("/")
            if self.media_url.startswith(("http://", "https://")):
                return urljoin(self.media_url, rel_path)
            return request.build_absolute_uri(urljoin(self.media_url, rel_path))

        def transform(obj):
            if isinstance(obj, dict):
                return {k: transform(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [transform(v) for v in obj]
            if isinstance(obj, str) and looks_like_file(obj):
                return to_absolute_url(obj)
            return obj

        updated = transform(payload)

        response.content = json.dumps(updated).encode("utf-8")
        response["Content-Length"] = str(len(response.content))
        return response
