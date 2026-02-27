from __future__ import annotations

import os
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from sdk.in_memory_impl import (
    InMemoryCapabilityRegistry,
    InMemorySessionManager,
    InMemoryIdempotencyStore,
)
from sdk.redis_impl import (
    RedisRESPClient,
    RedisSessionManager,
    RedisIdempotencyStore,
    RedisRuntimeStateStore,
)
from sdk.basic_adapter_executor import (
    BasicAdapterRegistry,
    BasicExecutor,
    BasicResultShaper,
)
from sdk.basic_policy import BasicPolicyEngine
from sdk.frame_validation import TRPFrameValidationError, validate_trp_frame
from sdk.observability import TRPMetrics, MetricsAuditLogger, JsonStdoutAuditLogger, CompositeAuditLogger
from sdk.router_service import RouterService

app = FastAPI(title="TRP Router v0.1")


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(min_value, v)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, *, min_value: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return max(min_value, v)


def _schema_nack(raw_frame: Any, message: str, error_code: str = "TRP_2000") -> Dict[str, Any]:
    req = raw_frame if isinstance(raw_frame, dict) else {}
    return {
        "trp_version": req.get("trp_version", "0.1"),
        "frame_type": "NACK",
        "session_id": req.get("session_id"),
        "frame_id": f"res_{req.get('frame_id', 'unknown')}",
        "trace_id": req.get("trace_id"),
        "timestamp_ms": int(time.time() * 1000),
        "catalog_epoch": req.get("catalog_epoch"),
        "seq": req.get("seq"),
        "payload": {
            "nack_of_frame_id": req.get("frame_id"),
            "nack_of_call_id": None,
            "error_class": "SCHEMA_MISMATCH",
            "error_code": error_code,
            "message": message,
            "retryable": False,
            "retry_hint": {},
        },
    }

# Composition root / dependency wiring
registry = InMemoryCapabilityRegistry()
state_backend = str(os.getenv("TRP_STATE_BACKEND", "memory")).strip().lower()
redis_url = os.getenv("TRP_REDIS_URL", "redis://127.0.0.1:6379/0")
redis_prefix = str(os.getenv("TRP_REDIS_PREFIX", "trp")).strip() or "trp"
redis_client = None
metrics = TRPMetrics(backend=state_backend)
audit = CompositeAuditLogger(
    JsonStdoutAuditLogger(enabled=_env_bool("TRP_AUDIT_STDOUT", True)),
    MetricsAuditLogger(metrics),
)

if state_backend == "redis":
    redis_client = RedisRESPClient(redis_url, timeout_sec=float(os.getenv("TRP_REDIS_TIMEOUT_SEC", "2")))
    # Fail fast at startup to avoid assuming persistence is enabled when Redis is unreachable.
    redis_client.ping()
    sessions = RedisSessionManager(
        redis=redis_client,
        catalog_epoch_provider=lambda: registry.catalog_epoch,
        default_retry_budget=_env_int("TRP_RETRY_BUDGET", 3),
        session_ttl_sec=_env_int("TRP_SESSION_TTL_SEC", 86400),
        key_prefix=redis_prefix,
    )
    idempotency = RedisIdempotencyStore(redis=redis_client, key_prefix=redis_prefix)
    runtime_state = RedisRuntimeStateStore(redis=redis_client, key_prefix=redis_prefix)
else:
    sessions = InMemorySessionManager(
        catalog_epoch_provider=lambda: registry.catalog_epoch,
        default_retry_budget=_env_int("TRP_RETRY_BUDGET", 3),
    )
    idempotency = InMemoryIdempotencyStore()
    runtime_state = None

router_service = RouterService(
    sessions=sessions,
    registry=registry,
    policy=BasicPolicyEngine(
        allow_critical_with_token=_env_bool("TRP_ALLOW_CRITICAL_WITH_TOKEN", True),
        approval_hmac_secret=os.getenv("TRP_APPROVAL_HMAC_SECRET") or None,
        approval_require_signed=_env_bool("TRP_APPROVAL_REQUIRE_SIGNED", False),
        approval_allow_legacy_prefix=_env_bool("TRP_APPROVAL_ALLOW_LEGACY_PREFIX", True),
        approval_clock_skew_sec=_env_int("TRP_APPROVAL_CLOCK_SKEW_SEC", 30),
    ),
    idempotency=idempotency,
    adapters=BasicAdapterRegistry(),
    executor=BasicExecutor(base_sleep_sec=_env_int("TRP_EXECUTOR_BASE_SLEEP_MS", 10, min_value=0) / 1000.0),
    shaper=BasicResultShaper(),
    audit=audit,
    runtime_state=runtime_state,
    call_record_ttl_sec=_env_int("TRP_CALL_RECORD_TTL_SEC", 86400),
    async_result_ttl_sec=_env_int("TRP_ASYNC_RESULT_TTL_SEC", 600),
    async_execution_lease_sec=_env_int("TRP_ASYNC_EXECUTION_LEASE_SEC", 30),
    async_execution_heartbeat_sec=_env_float("TRP_ASYNC_EXECUTION_HEARTBEAT_SEC", 0.0) or None,
    async_event_limit=_env_int("TRP_ASYNC_EVENT_LIMIT", 256),
    async_cleanup_interval_sec=_env_int("TRP_ASYNC_CLEANUP_INTERVAL_SEC", 30),
    worker_id=os.getenv("TRP_ROUTER_WORKER_ID") or None,
)


@app.post("/trp/frame")
async def trp_frame(request: Request) -> Dict[str, Any]:
    raw_frame: Any = None
    t0 = time.time()
    response_frame_type = "UNKNOWN"
    err_cls = None
    err_code = None
    http_status_code = 200
    try:
        raw_frame = await request.json()
        if _env_bool("TRP_STRICT_VALIDATION", True):
            frame = validate_trp_frame(raw_frame)
        else:
            frame = raw_frame
        # Optional: inject auth_context from header/session.
        # frame["auth_context"] = {"role": "ops", "can_delete": True}
        res = router_service.handle_frame(frame)
        response_frame_type = str(res.get("frame_type", "UNKNOWN"))
        if response_frame_type == "NACK":
            p = res.get("payload", {}) if isinstance(res.get("payload"), dict) else {}
            err_cls = p.get("error_class")
            err_code = p.get("error_code")
        return res
    except TRPFrameValidationError as e:
        res = _schema_nack(raw_frame, str(e))
        response_frame_type = "NACK"
        err_cls = "SCHEMA_MISMATCH"
        err_code = "TRP_2000"
        return res
    except ValueError as e:
        # Includes request.json() JSON decode errors.
        if raw_frame is None:
            res = _schema_nack(None, f"invalid JSON body: {e}")
            response_frame_type = "NACK"
            err_cls = "SCHEMA_MISMATCH"
            err_code = "TRP_2000"
            return res
        http_status_code = 500
        raise HTTPException(status_code=500, detail=f"router error: {e}")
    except Exception as e:
        http_status_code = 500
        raise HTTPException(status_code=500, detail=f"router error: {e}")
    finally:
        latency_ms = (time.time() - t0) * 1000.0
        request_frame_type = None
        if isinstance(raw_frame, dict):
            request_frame_type = raw_frame.get("frame_type")
        if request_frame_type:
            metrics.observe_frame(
                request_frame_type=str(request_frame_type),
                response_frame_type=str(response_frame_type),
                latency_ms=latency_ms,
                error_class=str(err_cls) if err_cls is not None else None,
                error_code=str(err_code) if err_code is not None else None,
            )
        metrics.observe_http(path="/trp/frame", status_code=http_status_code, latency_ms=latency_ms)


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    metrics.observe_http(path="/healthz", status_code=200, latency_ms=0.0)
    return {
        "ok": True,
        "backend": state_backend,
        "timestamp_ms": int(time.time() * 1000),
    }


@app.get("/readyz")
async def readyz() -> JSONResponse:
    ready = True
    detail: Dict[str, Any] = {"backend": state_backend}
    if state_backend == "redis":
        try:
            ready = bool(redis_client and redis_client.ping())
            detail["redis_ping"] = "PONG" if ready else "NO_PONG"
        except Exception as e:
            ready = False
            detail["redis_error"] = str(e)
    metrics.set_readiness(ready=ready)
    code = 200 if ready else 503
    metrics.observe_http(path="/readyz", status_code=code, latency_ms=0.0)
    return JSONResponse(status_code=code, content={"ok": ready, **detail, "timestamp_ms": int(time.time() * 1000)})


@app.get("/metrics")
async def metrics_endpoint() -> PlainTextResponse:
    # Refresh readiness gauge before export (especially in Redis mode).
    if state_backend == "redis":
        try:
            metrics.set_readiness(ready=bool(redis_client and redis_client.ping()))
        except Exception:
            metrics.set_readiness(ready=False)
    else:
        metrics.set_readiness(ready=True)
    body = metrics.render_prometheus()
    metrics.observe_http(path="/metrics", status_code=200, latency_ms=0.0)
    return PlainTextResponse(content=body, media_type="text/plain; version=0.0.4")
