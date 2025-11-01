# settings.py — organized, environment-driven, and production-hardened
from __future__ import annotations

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple
from celery.schedules import crontab
from kombu import Queue
from corsheaders.defaults import default_headers
from dotenv import load_dotenv

# =============================================================================
# Base paths & env helpers
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", BASE_DIR.parent))

# Load .env from project root (if present)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

def env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def env_list(key: str, default: str = "", sep: str = ",") -> List[str]:
    return [item.strip() for item in os.getenv(key, default).split(sep) if item and item.strip()]


# =============================================================================
# Runtime flags & site basics
# =============================================================================
DEBUG = env_bool("DEBUG", True)
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "insecure-dev-key")
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8002").rstrip("/")
SITE_ID = env_int("SITE_ID", 1)

# Hosts / proxies
ALLOWED_HOSTS = env_list("ALLOWED_HOSTS", "*")
USE_X_FORWARDED_HOST = True  # honor reverse proxy headers
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Global cache TTL used across app
CACHE_TTL = env_int("CACHE_TTL", 600)

# =============================================================================
# Installed apps
# =============================================================================
INSTALLED_APPS = [
    # ASGI / server
    "daphne",
    # Django admin Theme
    'simpleui',

    # Django core
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "django.contrib.sites",
    "django.contrib.postgres",

    "corsheaders",

    "rest_framework",
    "django_filters",
    "drf_spectacular",

    "storages",

    # Apps
    "user",
    "core",
    "engine",
    "agent",
    "memory",
    "api",
    "stream",
    "security",
]

# =============================================================================
# Middleware (Base)
# =============================================================================
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",

    # Custom security middlewares
    "security.middleware.security.SecurityMiddleware",
    "security.middleware.media_handler.SmartMediaMiddleware",
]

ROOT_URLCONF = "aicoach.urls"
ASGI_APPLICATION = "aicoach.asgi.application"

APPEND_SLASH = True
# X_FRAME_OPTIONS = "DENY"

# =============================================================================
# Templates
# =============================================================================
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    },
]

# =============================================================================
# CORS / CSRF
# =============================================================================
CORS_ALLOW_ALL_ORIGINS = env_bool("CORS_ALLOW_ALL_ORIGINS", True)
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = list(default_headers)

CSRF_TRUSTED_ORIGINS: List[str] = [
    "https://*.nudgyt.com",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:3000",
]
if BASE_URL.startswith(("http://", "https://")):
    CSRF_TRUSTED_ORIGINS.append(BASE_URL)

# =============================================================================
# Auth / Accounts (Allauth + SAML)
# =============================================================================
AUTH_USER_MODEL = "user.User"

AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",            # admin + username/password
]

# =============================================================================
# Database (PostgreSQL)
# =============================================================================
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("DATABASE_NAME", "aicoach"),
        "USER": os.getenv("DATABASE_USER", "postgres"),
        "PASSWORD": os.getenv("DATABASE_PASSWORD", "postgres"),
        "HOST": os.getenv("DATABASE_HOST", "localhost"),
        "PORT": os.getenv("DATABASE_PORT", "5432"),
        "OPTIONS": {
            "connect_timeout": 10,
            # NEW: honor DB_SSLMODE env; Render should be "require"
            "sslmode": os.getenv("DB_SSLMODE", "prefer"),
        },
    }
}

# =============================================================================
# Redis endpoints (split per subsystem; override via env)
# =============================================================================
REDIS_URL_CACHE    = os.getenv("REDIS_URL_CACHE",    "redis://localhost:6379/1")
REDIS_URL_CHANNELS = os.getenv("REDIS_URL_CHANNELS", "redis://localhost:6379/2")

# =============================================================================
# Cache (Django)
# =============================================================================
CACHES: Dict[str, Any] = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": REDIS_URL_CACHE,
        # Optional: small key prefix if you share Redis across projects
        "KEY_PREFIX": os.getenv("CACHE_KEY_PREFIX", "aicoach"),
    }
}

# =============================================================================
# Channels (Django Channels)
# =============================================================================
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {"hosts": [REDIS_URL_CHANNELS]},
    }
}

# =============================================================================
# DRF / Swagger
# =============================================================================
# settings.py
REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_AUTHENTICATION_CLASSES": ["api.auth.ApiKeyAuthentication"],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
        "api.permissions.HasValidAPIKeyAndAllowedOrigin",
    ],
    "DEFAULT_PAGINATION_CLASS": "api.pagination.DefaultPageNumberPagination",
    "PAGE_SIZE": 10,
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.SearchFilter",
        "rest_framework.filters.OrderingFilter",
    ],
    "EXCEPTION_HANDLER": "api.exceptions.custom_exception_handler",
}

SPECTACULAR_SETTINGS = {
    "TITLE": "aicoach API",
    "DESCRIPTION": "Welcome to aicoach Agent Portal.",
    "VERSION": "1.0.0",
    "SCHEMA_PATH_PREFIX": r"/api/",
    "CONTACT": {"name": "aicoach AI Agent Team", "email": "support@nudgyt.com", "url": "https://aicoach.nudgyt.com"},
    "SECURITY_SCHEMES": {
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "Authorization", "description": "Format: `Api-Key <key>`"}
    },
    "SECURITY": [{"ApiKeyAuth": []}],
}

# =============================================================================
# Passwords
# =============================================================================
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 12}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
    {"NAME": "security.validators.ComplexPasswordValidator"},
]

# =============================================================================
# i18n / tz
# =============================================================================
LANGUAGE_CODE = "en-us"
TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Kuala_Lumpur")
USE_I18N = True
USE_TZ = True

# =============================================================================
# Static / Media
# =============================================================================
# Always fine to keep extra static asset dirs
STATICFILES_DIRS = [BASE_DIR / "assets"]  # optional

# ---- Local development / no S3 ----
STATIC_URL  = "/static/"
# where collectstatic will put files
STATIC_ROOT = Path(os.getenv("STATIC_ROOT", PROJECT_ROOT / "static")).resolve()

MEDIA_URL   = "/media/"
MEDIA_ROOT  = Path(os.getenv("MEDIA_ROOT", PROJECT_ROOT / "media")).resolve()
    

# =============================================================================
# Uploads / Forms
# =============================================================================
DATA_UPLOAD_MAX_MEMORY_SIZE = 150 * 1024 * 1024  # 150MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 150 * 1024 * 1024  # 150MB
DATA_UPLOAD_MAX_NUMBER_FIELDS = 10_000
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
DATA_UPLOAD_MAX_NUMBER_FILES = 150

# =============================================================================
# Defaults
# =============================================================================
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# === SimpleUI Settings for aicoach ===

# Theme and branding
SIMPLEUI_DEFAULT_THEME = 'admin.lte.css'
SIMPLEUI_HOME_TITLE = 'aicoach AI System'
SIMPLEUI_LOGO = '/static/logo.png'
SIMPLEUI_FAVICON = '/static/favicon.ico'
SIMPLEUI_HOME_ICON = 'fa fa-home'

# Home page modules
SIMPLEUI_HOME_INFO = False
SIMPLEUI_HOME_QUICK = True
SIMPLEUI_HOME_ACTION = True

# Offline mode & analytics
SIMPLEUI_STATIC_OFFLINE = True
SIMPLEUI_ANALYSIS = True
SIMPLEUI_DEFAULT_ICON = True
SIMPLEUI_LOADING = True

# Optional: keep Django’s system apps in the menu
SIMPLEUI_CONFIG = {
    'system_keep': True,

    # === Custom Menu Layout ===
    'menus': [
        {
            'name': 'API',
            'icon': 'fas fa-plug',
            'models': [
                {
                    'name': 'Swagger UI',
                    'icon': 'fa fa-link',
                    'url': '/api/schema/swagger-ui/',
                    'newTab': True
                },
                {
                    'name': 'ReDoc',
                    'icon': 'fa fa-book',
                    'url': '/api/schema/redoc/',
                    'newTab': True
                },
                {
                    'name': 'Raw Schema',
                    'icon': 'fa fa-code',
                    'url': '/api/schema/',
                    'newTab': True
                },
                {
                    'name': 'Test Console',
                    'icon': 'fa fa-vial',
                    'url': '/api/test/',
                    'newTab': True
                }
            ]
        }
    ]
}

# =============================================================================
# Temp / Logs
# =============================================================================
LOG_DIR = Path(os.getenv("LOG_DIR", PROJECT_ROOT / "temp" / "logs"))
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", PROJECT_ROOT / "temp" / "audio"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "terminal_like": {
            "format": "[{levelname}] {asctime} {module}.{name}: {message}",
            "style": "{",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "terminal_like"},
        "rotating_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "server.log"),
            "maxBytes": 1024 * 1024,
            "backupCount": 10,
            "encoding": "utf-8",
            "formatter": "terminal_like",
        },
    },
    "root": {"handlers": ["console", "rotating_file"], "level": "INFO"},
}