from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from sdk.approval_tokens import mint_approval_token
from sdk.basic_adapter_executor import BasicAdapterRegistry, BasicExecutor, BasicResultShaper
from sdk.basic_policy import BasicPolicyEngine
from sdk.frame_validation import TRPFrameValidationError, validate_trp_frame
from sdk.in_memory_impl import (
    InMemoryCapabilityRegistry,
    InMemoryIdempotencyStore,
    InMemorySessionManager,
)
from sdk.redis_impl import (
    RedisIdempotencyStore,
    RedisRESPClient,
    RedisRuntimeStateStore,
    RedisSessionManager,
)
from sdk.router_service import RouterService


class NullAuditLogger:
    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        return None


@dataclass
class FrameCtx:
    session_id: str
    catalog_epoch: int
    seq: int = 1
    trace_id: str = ""

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = f"trc_{uuid.uuid4().hex[:8]}"

    def next_frame(
        self,
        *,
        frame_type: str,
        payload: Dict[str, Any],
        frame_id: Optional[str] = None,
        auth_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        frame = {
            "trp_version": "0.1",
            "frame_type": frame_type,
            "session_id": self.session_id,
            "frame_id": frame_id or f"frm_{uuid.uuid4().hex[:8]}",
            "trace_id": self.trace_id,
            "timestamp_ms": int(time.time() * 1000),
            "catalog_epoch": self.catalog_epoch,
            "seq": self.seq,
            "payload": payload,
        }
        if auth_context is not None:
            frame["auth_context"] = auth_context
        self.seq += 1
        return frame


def _build_memory_service(*, policy: Optional[BasicPolicyEngine] = None) -> RouterService:
    registry = InMemoryCapabilityRegistry()
    return RouterService(
        sessions=InMemorySessionManager(catalog_epoch_provider=lambda: registry.catalog_epoch),
        registry=registry,
        policy=policy or BasicPolicyEngine(),
        idempotency=InMemoryIdempotencyStore(),
        adapters=BasicAdapterRegistry(),
        executor=BasicExecutor(),
        shaper=BasicResultShaper(),
        audit=NullAuditLogger(),
    )


def _build_redis_service(*, prefix: str, approval_secret: Optional[str] = None) -> RouterService:
    redis = RedisRESPClient("redis://127.0.0.1:6379/0")
    registry = InMemoryCapabilityRegistry()
    return RouterService(
        sessions=RedisSessionManager(
            redis=redis,
            catalog_epoch_provider=lambda: registry.catalog_epoch,
            key_prefix=prefix,
        ),
        registry=registry,
        policy=BasicPolicyEngine(
            approval_hmac_secret=approval_secret,
            approval_require_signed=bool(approval_secret),
            approval_allow_legacy_prefix=not bool(approval_secret),
        ),
        idempotency=RedisIdempotencyStore(redis=redis, key_prefix=prefix),
        adapters=BasicAdapterRegistry(),
        executor=BasicExecutor(),
        shaper=BasicResultShaper(),
        audit=NullAuditLogger(),
        runtime_state=RedisRuntimeStateStore(redis=redis, key_prefix=prefix),
    )


def _hello_and_sync(service: RouterService, *, agent_id: str = "pytest") -> tuple[Dict[str, Any], FrameCtx]:
    hello = {
        "trp_version": "0.1",
        "frame_type": "HELLO_REQ",
        "session_id": None,
        "frame_id": f"frm_hello_{uuid.uuid4().hex[:8]}",
        "trace_id": f"trc_{uuid.uuid4().hex[:8]}",
        "timestamp_ms": int(time.time() * 1000),
        "catalog_epoch": None,
        "seq": None,
        "payload": {"agent_id": agent_id, "supported_versions": ["0.1"], "resume_session_id": None},
    }
    res_h = service.handle_frame(hello)
    assert res_h["frame_type"] == "HELLO_RES", res_h

    ctx = FrameCtx(
        session_id=res_h["payload"]["session_id"],
        catalog_epoch=res_h["payload"]["catalog_epoch"],
    )
    res_s = service.handle_frame(
        ctx.next_frame(
            frame_type="CATALOG_SYNC_REQ",
            payload={"mode": "FULL", "known_epoch": ctx.catalog_epoch},
        )
    )
    assert res_s["frame_type"] == "CATALOG_SYNC_RES", res_s
    return res_h, ctx


def _wait_async_final(service: RouterService, ctx: FrameCtx, *, call_id: str, timeout_sec: float = 3.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_partial = None
    while time.time() < deadline:
        res = service.handle_frame(
            ctx.next_frame(
                frame_type="RESULT_QUERY_REQ",
                payload={"call_id": call_id, "after_event_id": 0, "include_partials": True},
            )
        )
        if res["frame_type"] == "PARTIAL_RESULT":
            last_partial = res
            if res["payload"].get("terminal") and res["payload"].get("final_available"):
                final = service.handle_frame(
                    ctx.next_frame(frame_type="RESULT_QUERY_REQ", payload={"call_id": call_id})
                )
                assert final["frame_type"] == "RESULT_QUERY_RES", final
                return {"partial": last_partial, "final": final}
        elif res["frame_type"] == "RESULT_QUERY_RES":
            status = res["payload"]["status"]
            if status in {"SUCCESS", "FAILED", "NOT_FOUND"}:
                return {"partial": last_partial, "final": res}
        time.sleep(0.05)
    raise AssertionError(f"async call {call_id} did not finish in time")


def _redis_available() -> bool:
    try:
        return RedisRESPClient("redis://127.0.0.1:6379/0").ping()
    except Exception:
        return False


@pytest.mark.parametrize(
    "bad_frame, expected_msg_substr",
    [
        (
            {
                "trp_version": "0.1",
                "frame_type": "HELLO_REQ",
                "session_id": None,
                "frame_id": "frm_bad_1",
                "trace_id": "trc_bad_1",
                "timestamp_ms": 1,
                "catalog_epoch": None,
                "seq": None,
                "payload": {
                    "agent_id": "x",
                    "supported_versions": ["0.1"],
                    "resume_session_id": None,
                    "extra_field": 1,
                },
            },
            "payload.extra_field",
        ),
        (
            {
                "trp_version": "0.1",
                "frame_type": "CALL_REQ",
                "session_id": "sess_x",
                "frame_id": "frm_bad_2",
                "trace_id": "trc_bad_2",
                "timestamp_ms": 1,
                "catalog_epoch": 1,
                "seq": "2",
                "payload": {
                    "call_id": "c1",
                    "idempotency_key": None,
                    "idx": 0,
                    "cap_id": "cap.search.web.v1",
                    "depends_on": [],
                    "attempt": 1,
                    "timeout_ms": 8000,
                    "approval_token": None,
                    "execution_mode": "SYNC",
                    "args": {"query": "x"},
                },
            },
            "seq",
        ),
    ],
)
def test_strict_frame_validation_rejects_invalid_frames(bad_frame: Dict[str, Any], expected_msg_substr: str) -> None:
    with pytest.raises(TRPFrameValidationError) as exc:
        validate_trp_frame(bad_frame)
    assert expected_msg_substr in str(exc.value)


def test_memory_async_partial_result_flow() -> None:
    service = _build_memory_service()
    _, ctx = _hello_and_sync(service, agent_id="pytest_async")

    call_id = "call_async_test"
    ack = service.handle_frame(
        ctx.next_frame(
            frame_type="CALL_REQ",
            payload={
                "call_id": call_id,
                "idempotency_key": None,
                "idx": 0,
                "cap_id": "cap.search.web.v1",
                "depends_on": [],
                "attempt": 1,
                "timeout_ms": 8000,
                "approval_token": None,
                "execution_mode": "ASYNC",
                "args": {"query": "pytest async partial", "top_k": 5},
            },
        )
    )
    assert ack["frame_type"] == "ACK", ack
    done = _wait_async_final(service, ctx, call_id=call_id)
    assert done["final"]["payload"]["status"] == "SUCCESS", done["final"]
    assert done["partial"] is not None
    kinds = [e.get("kind") for e in done["partial"]["payload"].get("events", [])]
    assert any(k in {"SUMMARY", "DATA_CHUNK", "STATE"} for k in kinds)


def test_signed_approval_tokens_high_and_critical() -> None:
    secret = "pytest-secret"
    policy = BasicPolicyEngine(
        approval_hmac_secret=secret,
        approval_require_signed=True,
        approval_allow_legacy_prefix=False,
    )
    service = _build_memory_service(policy=policy)
    _, ctx = _hello_and_sync(service, agent_id="pytest_approval")

    # HIGH success
    high_args = {"title": "ok", "body": "desc", "priority": "high"}
    high_token = mint_approval_token(secret=secret, cap_id="cap.ticket.create.v1", args=high_args, ttl_sec=60)
    res_high = service.handle_frame(
        ctx.next_frame(
            frame_type="CALL_REQ",
            payload={
                "call_id": "call_high_ok",
                "idempotency_key": "idem_high_ok",
                "idx": 3,
                "cap_id": "cap.ticket.create.v1",
                "depends_on": [],
                "attempt": 1,
                "timeout_ms": 8000,
                "approval_token": high_token,
                "execution_mode": "SYNC",
                "args": high_args,
            },
        )
    )
    assert res_high["frame_type"] == "RESULT", res_high

    # HIGH args mismatch -> approval required
    tampered_args = dict(high_args)
    tampered_args["body"] = "tampered"
    res_bad = service.handle_frame(
        ctx.next_frame(
            frame_type="CALL_REQ",
            payload={
                "call_id": "call_high_bad",
                "idempotency_key": "idem_high_bad",
                "idx": 3,
                "cap_id": "cap.ticket.create.v1",
                "depends_on": [],
                "attempt": 1,
                "timeout_ms": 8000,
                "approval_token": high_token,
                "execution_mode": "SYNC",
                "args": tampered_args,
            },
        )
    )
    assert res_bad["frame_type"] == "NACK", res_bad
    assert res_bad["payload"]["error_class"] == "APPROVAL_REQUIRED", res_bad

    # CRITICAL no can_delete -> policy denied
    crit_args = {"resource_type": "doc", "resource_id": "123", "reason": "cleanup"}
    crit_token = mint_approval_token(secret=secret, cap_id="cap.record.delete.v1", args=crit_args, ttl_sec=60)
    res_crit_deny = service.handle_frame(
        ctx.next_frame(
            frame_type="CALL_REQ",
            auth_context={"role": "ops"},
            payload={
                "call_id": "call_crit_deny",
                "idempotency_key": "idem_crit_deny",
                "idx": 4,
                "cap_id": "cap.record.delete.v1",
                "depends_on": [],
                "attempt": 1,
                "timeout_ms": 8000,
                "approval_token": crit_token,
                "execution_mode": "SYNC",
                "args": crit_args,
            },
        )
    )
    assert res_crit_deny["frame_type"] == "NACK", res_crit_deny
    assert res_crit_deny["payload"]["error_class"] == "POLICY_DENIED", res_crit_deny

    # CRITICAL with permission -> success
    res_crit_ok = service.handle_frame(
        ctx.next_frame(
            frame_type="CALL_REQ",
            auth_context={"role": "ops", "can_delete": True},
            payload={
                "call_id": "call_crit_ok",
                "idempotency_key": "idem_crit_ok",
                "idx": 4,
                "cap_id": "cap.record.delete.v1",
                "depends_on": [],
                "attempt": 1,
                "timeout_ms": 8000,
                "approval_token": crit_token,
                "execution_mode": "SYNC",
                "args": crit_args,
            },
        )
    )
    assert res_crit_ok["frame_type"] == "RESULT", res_crit_ok


@pytest.mark.integration
def test_redis_persistence_for_call_records_async_state_and_idempotency() -> None:
    if not _redis_available():
        pytest.skip("Redis not available on 127.0.0.1:6379")

    prefix = f"trp_pytest_{uuid.uuid4().hex[:8]}"

    # Phase 1: seed records
    service1 = _build_redis_service(prefix=prefix)
    _, ctx1 = _hello_and_sync(service1, agent_id="pytest_redis")

    dep_call_id = "call_dep_seed"
    res_dep = service1.handle_frame(
        ctx1.next_frame(
            frame_type="CALL_REQ",
            payload={
                "call_id": dep_call_id,
                "idempotency_key": None,
                "idx": 0,
                "cap_id": "cap.search.web.v1",
                "depends_on": [],
                "attempt": 1,
                "timeout_ms": 8000,
                "approval_token": None,
                "execution_mode": "SYNC",
                "args": {"query": "seed dep record"},
            },
        )
    )
    assert res_dep["frame_type"] == "RESULT", res_dep

    async_call_id = "call_async_seed"
    res_async_ack = service1.handle_frame(
        ctx1.next_frame(
            frame_type="CALL_REQ",
            payload={
                "call_id": async_call_id,
                "idempotency_key": None,
                "idx": 0,
                "cap_id": "cap.search.web.v1",
                "depends_on": [],
                "attempt": 1,
                "timeout_ms": 8000,
                "approval_token": None,
                "execution_mode": "ASYNC",
                "args": {"query": "seed async", "top_k": 5},
            },
        )
    )
    assert res_async_ack["frame_type"] == "ACK", res_async_ack
    async_done = _wait_async_final(service1, ctx1, call_id=async_call_id)
    assert async_done["final"]["payload"]["status"] == "SUCCESS", async_done

    write_args = {"title": "seed write", "body": "phase1", "priority": "medium"}
    write_call_id = "call_write_seed"
    idem_key = "idem_write_seed"
    res_write = service1.handle_frame(
        ctx1.next_frame(
            frame_type="CALL_REQ",
            payload={
                "call_id": write_call_id,
                "idempotency_key": idem_key,
                "idx": 3,
                "cap_id": "cap.ticket.create.v1",
                "depends_on": [],
                "attempt": 1,
                "timeout_ms": 8000,
                "approval_token": "appr_legacy_ok",
                "execution_mode": "SYNC",
                "args": write_args,
            },
        )
    )
    assert res_write["frame_type"] == "RESULT", res_write
    seed_ticket_id = res_write["payload"]["result"]["data"]["ticket_id"]

    # Phase 2: simulate router restart by creating a new RouterService instance
    service2 = _build_redis_service(prefix=prefix)
    ctx2 = FrameCtx(session_id=ctx1.session_id, catalog_epoch=ctx1.catalog_epoch, seq=ctx1.seq, trace_id=ctx1.trace_id)

    # depends_on should resolve via persisted call_records
    res_dep2 = service2.handle_frame(
        ctx2.next_frame(
            frame_type="CALL_REQ",
            payload={
                "call_id": "call_dep_after_restart",
                "idempotency_key": None,
                "idx": 0,
                "cap_id": "cap.search.web.v1",
                "depends_on": [dep_call_id],
                "attempt": 1,
                "timeout_ms": 8000,
                "approval_token": None,
                "execution_mode": "SYNC",
                "args": {"query": "depends_on after restart"},
            },
        )
    )
    assert res_dep2["frame_type"] == "RESULT", res_dep2

    # async state + partial events should be queryable after restart
    res_async_q = service2.handle_frame(
        ctx2.next_frame(
            frame_type="RESULT_QUERY_REQ",
            payload={"call_id": async_call_id, "after_event_id": 0, "include_partials": True},
        )
    )
    assert res_async_q["frame_type"] in {"PARTIAL_RESULT", "RESULT_QUERY_RES"}, res_async_q
    if res_async_q["frame_type"] == "PARTIAL_RESULT":
        assert res_async_q["payload"]["terminal"] is True, res_async_q
        assert len(res_async_q["payload"].get("events", [])) > 0, res_async_q
    else:
        assert res_async_q["payload"]["status"] == "SUCCESS", res_async_q

    # idempotency replay should return same ticket_id after restart
    res_write_replay = service2.handle_frame(
        ctx2.next_frame(
            frame_type="CALL_REQ",
            payload={
                "call_id": write_call_id,
                "idempotency_key": idem_key,
                "idx": 3,
                "cap_id": "cap.ticket.create.v1",
                "depends_on": [],
                "attempt": 1,
                "timeout_ms": 8000,
                "approval_token": "appr_legacy_ok",
                "execution_mode": "SYNC",
                "args": {"title": "seed write", "body": "phase2 replay", "priority": "medium"},
            },
        )
    )
    assert res_write_replay["frame_type"] == "RESULT", res_write_replay
    replay_ticket_id = res_write_replay["payload"]["result"]["data"]["ticket_id"]
    assert replay_ticket_id == seed_ticket_id
