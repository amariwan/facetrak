"""Soft resource guard: warn + GC when RSS exceeds the configured cap.

Never kills the process — the OOM killer / systemd / Docker limits own
hard enforcement. Linux-first: uses /proc, degrades gracefully elsewhere.
"""

import gc
import logging

from facetrak.platform.metrics import self_rss_mb

logger = logging.getLogger(__name__)


class ResourceGuard:
    def __init__(self, max_memory_mb: float = 0):
        self.max_memory_mb = float(max_memory_mb or 0)
        self._warned = False

    @property
    def enabled(self) -> bool:
        return self.max_memory_mb > 0

    def check(self) -> dict:
        """Returns status; logs (once per breach) when over budget."""
        rss = self_rss_mb()
        status = {
            "enabled": self.enabled,
            "rss_mb": rss,
            "max_mb": self.max_memory_mb or None,
            "over": False,
        }
        if not self.enabled or rss is None:
            return status
        if rss > self.max_memory_mb:
            status["over"] = True
            gc.collect()
            if not self._warned:
                logger.warning(
                    "RSS %.0f MB exceeds budget %.0f MB (soft limit, no action taken)",
                    rss,
                    self.max_memory_mb,
                )
                self._warned = True
        elif rss < 0.9 * self.max_memory_mb:
            self._warned = False
        return status
