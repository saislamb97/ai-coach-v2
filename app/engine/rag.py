# engine/rag.py
from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional
import asyncio

from .nodes import QAState, n_prepare, n_run, n_finalize_and_persist

log = logging.getLogger(__name__)

class Rag:
    """
    Thin façade used by WS/HTTP layers.
    New pipeline (parallelized):
      1) n_prepare
      2) n_run                 -> streams text tokens + emotion + TTS; tools run in parallel (slide deltas stream)
      3) n_finalize_and_persist
    """

    def __init__(
        self,
        *,
        queue: Optional[asyncio.Queue] = None,
        limits=None,
        model: Optional[str] = None,
        temperature: float = 0.3,  # maintained for parity
    ):
        self.queue = queue
        self.limits = limits
        self.model = model
        self.temperature = temperature

    async def run(self, *, query: str, bot_id: str, thread_id: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        log.info("[rag] start bot=%s thread=%s q_len=%d model=%s",
                 bot_id, thread_id, len(query or ""), self.model or "(default)")

        state: QAState = {
            "bot_id": bot_id,
            "thread_id": thread_id,
            "query": (query or "").strip(),
            "queue": self.queue,
            "limits": self.limits,
            "model": (self.model or ""),
            "timings": {},
        }

        state = await n_prepare(state)
        state = await n_run(state)
        state = await n_finalize_and_persist(state)

        state["timings"]["total_ms"] = int((time.perf_counter() - t0) * 1000)
        log.info("[rag] done bot=%s thread=%s total_ms=%d resp_len=%d",
                 bot_id, thread_id, state["timings"]["total_ms"], len(state.get("response") or ""))
        return {"response": state.get("response", ""), "timings": state["timings"]}

    async def cancel(self):
        # Provided for API symmetry; no mid-run cancellation in this agent.
        log.debug("[rag] cancel (noop)")
        return
