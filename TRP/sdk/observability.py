from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Tuple


LabelTuple = Tuple[Tuple[str, str], ...]


def _labels(**kwargs: Any) -> LabelTuple:
    return tuple(sorted((str(k), str(v)) for k, v in kwargs.items()))


class TRPMetrics:
    """
    Lightweight thread-safe metrics collector (Prometheus text exposition).
    Focuses on critical counters and latency totals without extra dependencies.
    """

    def __init__(self, *, backend: str):
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._backend = backend
        self._counters: Counter[tuple[str, LabelTuple]] = Counter()
        self._gauges: Dict[tuple[str, LabelTuple], float] = {}
        self._set_gauge("trp_router_info", 1.0, backend=backend)

    def observe_http(self, *, path: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._counters[("trp_http_requests_total", _labels(path=path, status_code=status_code))] += 1
            self._counters[("trp_http_request_latency_ms_count", _labels(path=path))] += 1
            self._counters[("trp_http_request_latency_ms_sum", _labels(path=path))] += float(latency_ms)

    def observe_frame(
        self,
        *,
        request_frame_type: str,
        response_frame_type: str,
        latency_ms: float,
        error_class: str | None = None,
        error_code: str | None = None,
    ) -> None:
        with self._lock:
            self._counters[(
                "trp_frames_total",
                _labels(request_frame_type=request_frame_type, response_frame_type=response_frame_type),
            )] += 1
            self._counters[("trp_frame_latency_ms_count", _labels(request_frame_type=request_frame_type))] += 1
            self._counters[("trp_frame_latency_ms_sum", _labels(request_frame_type=request_frame_type))] += float(latency_ms)
            if error_class or error_code:
                self._counters[(
                    "trp_nacks_total",
                    _labels(error_class=error_class or "UNKNOWN", error_code=error_code or "UNKNOWN"),
                )] += 1

    def set_readiness(self, *, ready: bool) -> None:
        self._set_gauge("trp_router_ready", 1.0 if ready else 0.0)

    def observe_audit_event(self, *, event_name: str) -> None:
        with self._lock:
            self._counters[("trp_audit_events_total", _labels(event_name=event_name))] += 1
            lease_action = _lease_action_from_event_name(event_name)
            if lease_action is not None:
                self._counters[("trp_async_lease_events_total", _labels(action=lease_action))] += 1

    def render_prometheus(self) -> str:
        with self._lock:
            lines: list[str] = []
            uptime = max(0.0, time.time() - self._started_at)
            gauges = dict(self._gauges)
            gauges[("trp_router_uptime_seconds", tuple())] = uptime

            counters = dict(self._counters)

        # HELP/TYPE (key metrics)
        lines.extend(
            [
                "# HELP trp_router_info Static router info labels",
                "# TYPE trp_router_info gauge",
                "# HELP trp_router_ready Readiness state (1=ready,0=not ready)",
                "# TYPE trp_router_ready gauge",
                "# HELP trp_router_uptime_seconds Router process uptime in seconds",
                "# TYPE trp_router_uptime_seconds gauge",
                "# HELP trp_http_requests_total Total HTTP requests handled",
                "# TYPE trp_http_requests_total counter",
                "# HELP trp_http_request_latency_ms_count HTTP request latency observations (count)",
                "# TYPE trp_http_request_latency_ms_count counter",
                "# HELP trp_http_request_latency_ms_sum HTTP request latency observations (sum ms)",
                "# TYPE trp_http_request_latency_ms_sum counter",
                "# HELP trp_frames_total TRP frame responses by request/response frame type",
                "# TYPE trp_frames_total counter",
                "# HELP trp_frame_latency_ms_count TRP frame latency observations (count)",
                "# TYPE trp_frame_latency_ms_count counter",
                "# HELP trp_frame_latency_ms_sum TRP frame latency observations (sum ms)",
                "# TYPE trp_frame_latency_ms_sum counter",
                "# HELP trp_nacks_total Total NACKs by error class/code",
                "# TYPE trp_nacks_total counter",
                "# HELP trp_audit_events_total Total audit events by event name",
                "# TYPE trp_audit_events_total counter",
                "# HELP trp_async_lease_events_total Async execution lease events by action",
                "# TYPE trp_async_lease_events_total counter",
            ]
        )

        for (name, labels), value in sorted(gauges.items(), key=lambda x: (x[0][0], x[0][1])):
            lines.append(self._metric_line(name, labels, value))
        for (name, labels), value in sorted(counters.items(), key=lambda x: (x[0][0], x[0][1])):
            lines.append(self._metric_line(name, labels, value))
        return "\n".join(lines) + "\n"

    def _set_gauge(self, name: str, value: float, **labels_kwargs: Any) -> None:
        with self._lock:
            self._gauges[(name, _labels(**labels_kwargs))] = float(value)

    @staticmethod
    def _metric_line(name: str, labels: LabelTuple, value: float) -> str:
        if labels:
            rendered = ",".join(f'{k}="{_escape_label(v)}"' for k, v in labels)
            return f"{name}{{{rendered}}} {float(value)}"
        return f"{name} {float(value)}"


def _escape_label(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _lease_action_from_event_name(event_name: str) -> Optional[str]:
    mapping = {
        "call.async_lease_renewed": "renewed",
        "call.async_lease_lost": "lost",
        "call.async_lease_renew_error": "renew_error",
    }
    return mapping.get(event_name)


class MetricsAuditLogger:
    def __init__(self, metrics: TRPMetrics):
        self._metrics = metrics

    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        self._metrics.observe_audit_event(event_name=event_name)


class JsonStdoutAuditLogger:
    """
    Structured stdout logging (single-line JSON) for easy local/container collection.
    """

    def __init__(self, *, enabled: bool = True):
        self._enabled = bool(enabled)

    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        rec = {
            "timestamp_ms": int(time.time() * 1000),
            "event_name": event_name,
            "payload": payload,
        }
        try:
            line = json.dumps(rec, ensure_ascii=False, default=str, separators=(",", ":"))
        except Exception:
            line = json.dumps(
                {
                    "timestamp_ms": rec["timestamp_ms"],
                    "event_name": event_name,
                    "payload": str(payload),
                },
                ensure_ascii=False,
            )
        print(line, file=sys.stdout, flush=False)


class CompositeAuditLogger:
    def __init__(self, *loggers: Any):
        self._loggers = [x for x in loggers if x is not None]

    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        for logger in self._loggers:
            try:
                logger.log_event(event_name, payload)
            except Exception:
                # Audit failures must not break the main flow.
                continue
