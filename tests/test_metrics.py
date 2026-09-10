"""Unit tests for the Prometheus-format metrics registry."""

from __future__ import annotations

from smart_contract_rag.metrics import Metrics


class TestMetrics:
    def test_counter_increments(self) -> None:
        m = Metrics()
        m.inc("http_requests_total")
        m.inc("http_requests_total")
        out = m.render()
        assert "smart_contract_rag_http_requests_total 2" in out

    def test_histogram_quantiles(self) -> None:
        m = Metrics()
        for i in range(1, 101):
            m.observe("request_duration_ms", float(i))
        out = m.render()
        assert 'smart_contract_rag_request_duration_ms_quantile{quantile="0.5"}' in out
        assert 'smart_contract_rag_request_duration_ms_quantile{quantile="0.9"}' in out
        assert "request_duration_ms_count 100" in out
        assert "request_duration_ms_sum" in out

    def test_empty_registry_renders_empty(self) -> None:
        m = Metrics()
        assert m.render().strip() == ""

    def test_timer_context_manager(self) -> None:
        import time as _time

        m = Metrics()
        with m.time("query_latency_s"):
            _time.sleep(0.005)
        out = m.render()
        assert "query_latency_s_count 1" in out
