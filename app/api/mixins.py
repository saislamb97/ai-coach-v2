# api/mixins.py
from __future__ import annotations
from typing import Optional
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AbstractBaseUser
from rest_framework.exceptions import PermissionDenied
from api.utils import get_active_auth   # ← use your existing helper

UserModel = get_user_model()

class TenantScopedQuerysetMixin:
    """
    Default scope = authenticated Django user resolved from API key.
    If user_id is provided AND the requester is staff/superuser,
    we scope to that user instead.

    Views must set `user_lookup_field`, e.g.:
      - "user"              (Agent, Session)
      - "session__user"     (Chat, Slides)
    """
    user_lookup_field = "user"

    def _resolve_auth_user(self) -> Optional[AbstractBaseUser]:
        """
        Robustly resolve a real Django User from:
          1) request.user (if it's a Django user)
          2) request.auth.user (if your auth object carries a FK to user)
          3) active API key via get_active_auth(request)
        """
        req = self.request

        u = getattr(req, "user", None)
        if isinstance(u, AbstractBaseUser):
            return u

        auth_obj = getattr(req, "auth", None)
        if getattr(auth_obj, "user", None) and isinstance(auth_obj.user, AbstractBaseUser):
            return auth_obj.user

        api_auth, _ = get_active_auth(req)
        if api_auth and getattr(api_auth, "user", None):
            return api_auth.user  # must be a Django user FK on your ApiAuth

        return None

    def _resolve_target_user(self) -> Optional[AbstractBaseUser]:
        request = self.request
        params = request.query_params

        # staff override
        user_id = params.get("user_id")
        if user_id:
            # only staff/superuser may cross-scope
            if not (request.user and (getattr(request.user, "is_staff", False) or getattr(request.user, "is_superuser", False))):
                raise PermissionDenied("Cross-user queries require staff privileges.")
            try:
                return UserModel.objects.get(pk=user_id)
            except UserModel.DoesNotExist:
                return None

        # default: resolve from API key / auth chain
        return self._resolve_auth_user()

    def scope_to_tenant(self, qs):
        user = self._resolve_target_user()
        if user is None:
            return qs.none()
        return qs.filter(**{f"{self.user_lookup_field}_id": user.pk})
