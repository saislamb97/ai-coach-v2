# api/mixins.py
from __future__ import annotations
from typing import Optional
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AbstractBaseUser
from rest_framework.exceptions import PermissionDenied

UserModel = get_user_model()  # runtime model; don't use in annotations

class TenantScopedQuerysetMixin:
    """
    Default scope = request.user.
    If user_id / username / email is provided AND the requester is staff/superuser,
    we scope to that user instead.

    Views must set `user_lookup_field`, e.g.:
      - "user"              (Agent, Session)
      - "session__user"     (Chat, Slides)
    """
    user_lookup_field = "user"

    def _resolve_target_user(self) -> Optional[AbstractBaseUser]:
        request = self.request
        params = request.query_params

        u: Optional[AbstractBaseUser] = None

        user_id = params.get("user_id")
        username = params.get("username")
        email = params.get("email")

        if user_id or username or email:
            if not (request.user and (request.user.is_staff or request.user.is_superuser)):
                raise PermissionDenied("Cross-user queries require staff privileges.")

            q = {}
            if user_id:
                q["id"] = user_id
            if username:
                q["username__iexact"] = username
            if email:
                q["email__iexact"] = email

            try:
                u = UserModel.objects.get(**q)
            except UserModel.DoesNotExist:
                return None
        else:
            # when authenticated user isn’t a Django user instance (rare),
            # just return None and let caller produce qs.none()
            u = request.user if isinstance(getattr(request, "user", None), AbstractBaseUser) else None

        return u

    def scope_to_tenant(self, qs):
        user = self._resolve_target_user()
        if user is None:
            return qs.none()
        # use pk instead of id to stay generic
        return qs.filter(**{f"{self.user_lookup_field}_id": user.pk})
