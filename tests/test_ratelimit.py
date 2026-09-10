"""Unit tests for the token-bucket rate limiter."""

from __future__ import annotations

import time

from smart_contract_rag.ratelimit import RateLimiter, TokenBucket


class TestTokenBucket:
    def test_full_capacity_allows_burst(self) -> None:
        bucket = TokenBucket(capacity=5, refill_rate=0.0)
        assert all(bucket.consume() for _ in range(5))
        assert not bucket.consume()  # exhausted

    def test_remaining_reports_correctly(self) -> None:
        bucket = TokenBucket(capacity=10, refill_rate=0.0)
        bucket.consume()
        assert bucket.remaining == 9

    def test_refill_over_time(self) -> None:
        bucket = TokenBucket(capacity=1, refill_rate=60.0)  # 1 token/sec
        assert bucket.consume()
        assert not bucket.consume()
        time.sleep(0.02)  # 20 ms → ~1.2 tokens
        assert bucket.consume()


class TestRateLimiter:
    def test_limits_per_key(self) -> None:
        limiter = RateLimiter(default_per_minute=3)
        key = "scrag_test"
        allowed = [limiter.check(key)[0] for _ in range(4)]
        assert allowed == [True, True, True, False]

    def test_independent_keys(self) -> None:
        limiter = RateLimiter(default_per_minute=2)
        assert limiter.check("k1")[0] is True
        assert limiter.check("k1")[0] is True
        assert limiter.check("k1")[0] is False
        assert limiter.check("k2")[0] is True  # unaffected

    def test_limit_change_recreates_bucket(self) -> None:
        limiter = RateLimiter(default_per_minute=1)
        assert limiter.check("k")[0] is True
        assert limiter.check("k")[0] is False
        # Raise the key's limit; the new budget applies immediately.
        assert limiter.check("k", per_minute=5)[0] is True

    def test_remaining_decreases(self) -> None:
        limiter = RateLimiter(default_per_minute=5)
        _, rem0 = limiter.check("k")
        _, rem1 = limiter.check("k")
        assert rem0 == 4
        assert rem1 == 3
