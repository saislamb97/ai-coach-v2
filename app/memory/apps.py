from django.apps import AppConfig


class MemoryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = "memory"

    def ready(self):
        # Import signals so the receivers register on app load
        import memory.signals  # noqa: F401
