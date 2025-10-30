# aicoach/asgi.py
import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aicoach.settings")
django.setup()

from django.core.cache import cache
from django.conf import settings
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from channels.auth import AuthMiddlewareStack
from django.urls import path

if settings.DEBUG:
    cache.clear()

django_asgi_app = get_asgi_application()

from stream.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            URLRouter([
                path("ws/", URLRouter(websocket_urlpatterns)),  # <-- mount prefix
            ])
        )
    ),
})
