from django.apps import AppConfig


class MemoryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = "memory"

    def ready(self):
        print("⚡ MemoryConfig.ready() called — signals loaded")
        from . import signals  # noqa

