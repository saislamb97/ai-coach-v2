# agent/selectors.py
from __future__ import annotations

from typing import Optional, List

from django.conf import settings
from django.core.cache import cache
from django.db.models import QuerySet

from .models import Agent

CACHE_TTL = int(getattr(settings, "CACHE_TTL", 600))


def get_agent(bot_id: str | None = None, agent_id: int | None = None) -> Optional[Agent]:
    """
    Return a single active Agent, looked up by bot_id (UUID) or database id.
    Uses cache for speed. Returns None if not found or inactive.
    """
    if not bot_id and not agent_id:
        return None

    cache_key = f"agent:{bot_id or agent_id}"
    agent: Optional[Agent] = cache.get(cache_key)
    if agent:
        return agent if agent.is_active else None

    qs: QuerySet[Agent] = Agent.objects.select_related("voice", "user")
    try:
        agent = qs.get(bot_id=bot_id, is_active=True) if bot_id else qs.get(pk=agent_id, is_active=True)

        # Write to cache using both identifiers so future lookups are cheap
        cache.set(f"agent:{agent.bot_id}", agent, timeout=CACHE_TTL)
        cache.set(f"agent:{agent.pk}", agent, timeout=CACHE_TTL)
        return agent
    except Agent.DoesNotExist:
        return None


def get_agents_by_user(user_id: int) -> List[Agent]:
    """
    Return all active Agents owned by a given user.
    Result is cached per user.
    """
    cache_key = f"agent_list:user:{user_id}"
    cached_list = cache.get(cache_key)
    if cached_list is not None:
        # Filter out any inactive objects that might have been deactivated after caching
        return [a for a in cached_list if a.is_active]

    qs: QuerySet[Agent] = (
        Agent.objects.filter(user_id=user_id, is_active=True)
        .select_related("voice", "user")
        .order_by("-created_at")
    )

    agents = list(qs)
    cache.set(cache_key, agents, timeout=CACHE_TTL)
    return agents
