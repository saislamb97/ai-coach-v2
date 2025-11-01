# api/schema.py
from drf_spectacular.extensions import OpenApiAuthenticationExtension

class ApiKeyScheme(OpenApiAuthenticationExtension):
    # must point to your actual auth class
    target_class = 'api.auth.ApiKeyAuthentication'
    name = 'ApiKeyAuth'  # referenced by SPECTACULAR_SETTINGS.SECURITY/SECURITY_SCHEMES

    def get_security_definition(self, auto_schema):
        return {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': (
                "Provide your API key in the Authorization header.\n\n"
                "Format: `Api-Key <your_key>`"
            ),
        }
