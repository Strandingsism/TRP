from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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
from sdk.observability import TRPMetrics
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


class CountingSearchExecutor(BasicExecutor):
    def __init__(self) -> None:
        super().__init__()
        import threading

        self._lock = threading.Lock()
        self.execute_count = 0

    def execute(self, cap, native_args, timeout_ms):  # type: ignore[override]
        if cap.cap_id == "cap.search.web.v1":
            with self._lock:
                self.execute_count += 1
            # widen race window to exercise cross-instance claim
            time.sleep(0.05)
        return super().execute(cap, native_args, timeout_ms)


class SlowCountingSearchExecutor(CountingSearchExecutor):
    def __init__(self, *, sleep_sec: float) -> None:
        super().__init__()
        self._sleep_sec = float(sleep_sec)

    def execute(self, cap, native_args, timeout_ms):  # type: ignore[override]
        if cap.cap_id == "cap.search.web.v1":
            with self._lock:
                self.execute_count += 1
            time.sleep(self._sleep_sec)
            return BasicExecutor.execute(self, cap, native_args, timeout_ms)
        return BasicExecutor.execute(self, cap, native_args, timeout_ms)


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


def _build_redis_service(
    *,
    prefix: str,
    approval_secret: Optional[str] = None,
    executor: Optional[Any] = None,
    worker_id: Optional[str] = None,
    async_execution_lease_sec: int = 30,
    async_execution_heartbeat_sec: Optional[float] = None,
) -> RouterService:
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
        executor=executor or BasicExecutor(),
        shaper=BasicResultShaper(),
        audit=NullAuditLogger(),
        runtime_state=RedisRuntimeStateStore(redis=redis, key_prefix=prefix),
        async_execution_lease_sec=async_execution_lease_sec,
        async_execution_heartbeat_sec=async_execution_heartbeat_sec,
        worker_id=worker_id,
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


def test_metrics_collector_renders_prometheus_text() -> None:
    m = TRPMetrics(backend="memory")
    m.set_readiness(ready=True)
    m.observe_http(path="/trp/frame", status_code=200, latency_ms=12.5)
    m.observe_frame(
        request_frame_type="CALL_REQ",
        response_frame_type="NACK",
        latency_ms=8.0,
        error_class="SCHEMA_MISMATCH",
        error_code="TRP_2000",
    )
    out = m.render_prometheus()
    assert "trp_router_info" in out
    assert 'trp_router_ready ' in out
    assert 'trp_http_requests_total{path="/trp/frame",status_code="200"}' in out
    assert 'trp_frames_total{request_frame_type="CALL_REQ",response_frame_type="NACK"}' in out
    assert 'trp_nacks_total{error_class="SCHEMA_MISMATCH",error_code="TRP_2000"}' in out


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


@pytest.mark.integration
def test_redis_async_event_append_is_atomic_under_concurrency() -> None:
    if not _redis_available():
        pytest.skip("Redis not available on 127.0.0.1:6379")

    prefix = f"trp_atomic_{uuid.uuid4().hex[:8]}"
    store_a = RedisRuntimeStateStore(redis=RedisRESPClient("redis://127.0.0.1:6379/0"), key_prefix=prefix)
    store_b = RedisRuntimeStateStore(redis=RedisRESPClient("redis://127.0.0.1:6379/0"), key_prefix=prefix)
    session_id = "sess_atomic"
    call_id = "call_atomic"

    # Seed async state
    seeded = store_a.merge_async_call_state(
        session_id,
        call_id,
        {
            "status": "RUNNING",
            "result_frame_type": None,
            "result_payload": None,
            "events": [],
            "next_event_id": 1,
            "updated_at": int(time.time() * 1000),
        },
        ttl_sec=300,
    )
    assert seeded.get("next_event_id") == 1

    workers = 8
    per_worker = 25
    total = workers * per_worker

    def push_many(worker_id: int) -> None:
        store = store_a if worker_id % 2 == 0 else store_b
        for i in range(per_worker):
            store.append_async_event(
                session_id,
                call_id,
                {"kind": "STATE", "worker": worker_id, "i": i},
                event_limit=2000,
                ttl_sec=300,
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(push_many, w) for w in range(workers)]
        for fut in futs:
            fut.result()

    final_state = store_a.get_async_call_state(session_id, call_id)
    assert final_state is not None
    events = final_state.get("events", [])
    assert isinstance(events, list)
    assert len(events) == total, len(events)
    next_event_id = int(final_state.get("next_event_id", 0))
    assert next_event_id == total + 1, next_event_id

    event_ids = [int(e["event_id"]) for e in events]
    assert len(set(event_ids)) == total
    assert sorted(event_ids) == list(range(1, total + 1))


@pytest.mark.integration
def test_redis_async_execution_claim_prevents_duplicate_worker_execution() -> None:
    if not _redis_available():
        pytest.skip("Redis not available on 127.0.0.1:6379")

    prefix = f"trp_claim_{uuid.uuid4().hex[:8]}"
    shared_executor = CountingSearchExecutor()
    service_a = _build_redis_service(prefix=prefix, executor=shared_executor, worker_id="worker_a")
    service_b = _build_redis_service(prefix=prefix, executor=shared_executor, worker_id="worker_b")

    _, ctx = _hello_and_sync(service_a, agent_id="pytest_claim")
    call_id = "call_claim_once"

    # Seed async state as if ACK path accepted the call and queued background work.
    service_a._set_async_call_state(  # type: ignore[attr-defined]
        ctx.session_id,
        call_id,
        {
            "status": "QUEUED",
            "result_frame_type": None,
            "result_payload": None,
            "events": [],
            "next_event_id": 1,
            "updated_at": int(time.time() * 1000),
        },
    )
    service_a._append_async_event(  # type: ignore[attr-defined]
        ctx.session_id,
        call_id,
        {"kind": "STATE", "stage": "QUEUED", "message": "seeded for claim test"},
    )

    frame = {
        "trp_version": "0.1",
        "frame_type": "CALL_REQ",
        "session_id": ctx.session_id,
        "frame_id": f"frm_async_seed_{uuid.uuid4().hex[:8]}",
        "trace_id": ctx.trace_id,
        "timestamp_ms": int(time.time() * 1000),
        "catalog_epoch": ctx.catalog_epoch,
        "seq": ctx.seq,  # not used in _run_async_call
        "payload": {
            "call_id": call_id,
            "idempotency_key": None,
            "idx": 0,
            "cap_id": "cap.search.web.v1",
            "depends_on": [],
            "attempt": 1,
            "timeout_ms": 8000,
            "approval_token": None,
            "execution_mode": "ASYNC",
            "args": {"query": "claim once", "top_k": 5},
        },
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(service_a._run_async_call, dict(frame))  # type: ignore[attr-defined]
        fut_b = pool.submit(service_b._run_async_call, dict(frame))  # type: ignore[attr-defined]
        fut_a.result()
        fut_b.result()

    # Exactly one underlying executor execution should have happened.
    assert shared_executor.execute_count == 1, shared_executor.execute_count

    # Final state should be terminal success and queryable.
    q = service_a.handle_frame(
        ctx.next_frame(
            frame_type="RESULT_QUERY_REQ",
            payload={"call_id": call_id},
        )
    )
    assert q["frame_type"] == "RESULT_QUERY_RES", q
    assert q["payload"]["status"] == "SUCCESS", q


@pytest.mark.integration
def test_redis_async_execution_lease_renew_prevents_steal_after_lease_window() -> None:
    if not _redis_available():
        pytest.skip("Redis not available on 127.0.0.1:6379")

    prefix = f"trp_lease_renew_{uuid.uuid4().hex[:8]}"
    shared_executor = SlowCountingSearchExecutor(sleep_sec=2.4)
    service_a = _build_redis_service(
        prefix=prefix,
        executor=shared_executor,
        worker_id="worker_a",
        async_execution_lease_sec=1,
        async_execution_heartbeat_sec=0.2,
    )
    service_b = _build_redis_service(
        prefix=prefix,
        executor=shared_executor,
        worker_id="worker_b",
        async_execution_lease_sec=1,
        async_execution_heartbeat_sec=0.2,
    )

    _, ctx = _hello_and_sync(service_a, agent_id="pytest_lease_renew")
    call_id = "call_lease_renew"

    service_a._set_async_call_state(  # type: ignore[attr-defined]
        ctx.session_id,
        call_id,
        {
            "status": "QUEUED",
            "result_frame_type": None,
            "result_payload": None,
            "events": [],
            "next_event_id": 1,
            "updated_at": int(time.time() * 1000),
        },
    )
    service_a._append_async_event(  # type: ignore[attr-defined]
        ctx.session_id,
        call_id,
        {"kind": "STATE", "stage": "QUEUED", "message": "seeded for lease renew test"},
    )

    frame = {
        "trp_version": "0.1",
        "frame_type": "CALL_REQ",
        "session_id": ctx.session_id,
        "frame_id": f"frm_async_seed_{uuid.uuid4().hex[:8]}",
        "trace_id": ctx.trace_id,
        "timestamp_ms": int(time.time() * 1000),
        "catalog_epoch": ctx.catalog_epoch,
        "seq": ctx.seq,
        "payload": {
            "call_id": call_id,
            "idempotency_key": None,
            "idx": 0,
            "cap_id": "cap.search.web.v1",
            "depends_on": [],
            "attempt": 1,
            "timeout_ms": 8000,
            "approval_token": None,
            "execution_mode": "ASYNC",
            "args": {"query": "lease renew", "top_k": 5},
        },
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(service_a._run_async_call, dict(frame))  # type: ignore[attr-defined]
        time.sleep(1.35)  # > initial 1s lease window; heartbeat should have renewed
        fut_b = pool.submit(service_b._run_async_call, dict(frame))  # type: ignore[attr-defined]
        fut_b.result()
        fut_a.result()

    assert shared_executor.execute_count == 1, shared_executor.execute_count
    q = service_a.handle_frame(ctx.next_frame(frame_type="RESULT_QUERY_REQ", payload={"call_id": call_id}))
    assert q["frame_type"] == "RESULT_QUERY_RES", q
    assert q["payload"]["status"] == "SUCCESS", q
