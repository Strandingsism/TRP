from __future__ import annotations

import os
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request

from sdk.in_memory_impl import (
    InMemoryCapabilityRegistry,
    InMemorySessionManager,
    InMemoryIdempotencyStore,
    PrintAuditLogger,
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

# 组装（Composition Root / 依赖注入）
registry = InMemoryCapabilityRegistry()
audit = PrintAuditLogger()

state_backend = str(os.getenv("TRP_STATE_BACKEND", "memory")).strip().lower()
redis_url = os.getenv("TRP_REDIS_URL", "redis://127.0.0.1:6379/0")
redis_prefix = str(os.getenv("TRP_REDIS_PREFIX", "trp")).strip() or "trp"

if state_backend == "redis":
    redis_client = RedisRESPClient(redis_url, timeout_sec=float(os.getenv("TRP_REDIS_TIMEOUT_SEC", "2")))
    # 启动时 fail-fast，避免误以为已启用持久化
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
    executor=BasicExecutor(),
    shaper=BasicResultShaper(),
    audit=audit,
    runtime_state=runtime_state,
    call_record_ttl_sec=_env_int("TRP_CALL_RECORD_TTL_SEC", 86400),
    async_result_ttl_sec=_env_int("TRP_ASYNC_RESULT_TTL_SEC", 600),
    async_execution_lease_sec=_env_int("TRP_ASYNC_EXECUTION_LEASE_SEC", 30),
    async_event_limit=_env_int("TRP_ASYNC_EVENT_LIMIT", 256),
    async_cleanup_interval_sec=_env_int("TRP_ASYNC_CLEANUP_INTERVAL_SEC", 30),
    worker_id=os.getenv("TRP_ROUTER_WORKER_ID") or None,
)


@app.post("/trp/frame")
async def trp_frame(request: Request) -> Dict[str, Any]:
    raw_frame: Any = None
    try:
        raw_frame = await request.json()
        if _env_bool("TRP_STRICT_VALIDATION", True):
            frame = validate_trp_frame(raw_frame)
        else:
            frame = raw_frame
        # 可选：从 header / session 注入 auth_context
        # frame["auth_context"] = {"role": "ops", "can_delete": True}
        return router_service.handle_frame(frame)
    except TRPFrameValidationError as e:
        return _schema_nack(raw_frame, str(e))
    except ValueError as e:
        # 包含 request.json() 的 JSON decode 错误
        if raw_frame is None:
            return _schema_nack(None, f"invalid JSON body: {e}")
        raise HTTPException(status_code=500, detail=f"router error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"router error: {e}")
