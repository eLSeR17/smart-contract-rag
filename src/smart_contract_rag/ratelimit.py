"""Token-bucket rate limiter (in-memory, per key)."""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """Simple thread-safe token bucket for one key.

    Parameters
    ----------
    capacity:
        Max tokens (burst).  For per-minute limits this equals the
        per-minute rate.
    refill_rate:
        Tokens added per second.
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = float(capacity)
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, n: int = 1) -> bool:
        """Try to consume *n* tokens.  Returns ``False`` if not enough."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.refill_rate)
            self._last = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    @property
    def remaining(self) -> int:
        with self._lock:
            now = time.monotonic()
            tokens = min(self.capacity, self._tokens + (now - self._last) * self.refill_rate)
            return max(0, int(tokens))


class RateLimiter:
    """Per-key rate limiter.

    Parameters
    ----------
    default_per_minute:
        Default requests allowed per minute per key.
    """

    def __init__(self, default_per_minute: int = 60) -> None:
        self.default_per_minute = max(1, default_per_minute)
        self._buckets: dict[str, TokenBucket] = {}
        self._limits: dict[str, int] = {}
        self._lock = threading.Lock()

    def check(self, key_id: str, per_minute: int | None = None) -> tuple[bool, int]:
        """Attempt to consume one request for *key_id*.

        Returns
        -------
        tuple[bool, int]
            ``(allowed, remaining)`` — whether the request passes and how
            many tokens remain in the window.
        """
        limit = per_minute or self.default_per_minute
        with self._lock:
            bucket = self._buckets.get(key_id)
            # Recreate the bucket when the limit differs (e.g. a key's
            # plan changed) so the new limit applies immediately.
            if bucket is None or self._limits.get(key_id) != limit:
                bucket = TokenBucket(capacity=limit, refill_rate=limit / 60.0)
                self._buckets[key_id] = bucket
                self._limits[key_id] = limit
        allowed = bucket.consume()
        return allowed, bucket.remaining
