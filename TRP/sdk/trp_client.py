from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .trp_types import (
    TRP_VERSION, FrameType, ErrorClass, CapabilityBrief,
    FrameEnvelope, CallSpec
)


# =========================
# Exception definitions
# =========================

class TRPError(Exception):
    def __init__(self, message: str, error_class: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_class = error_class
        self.error_code = error_code


class CatalogMismatchError(TRPError):
    pass


class OrderViolationError(TRPError):
    def __init__(self, message: str, expected_seq: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.expected_seq = expected_seq


class ApprovalRequiredError(TRPError):
    pass


class PolicyDeniedError(TRPError):
    pass


class SchemaMismatchError(TRPError):
    pass


class RetryableTRPError(TRPError):
    def __init__(self, message: str, backoff_ms: int = 200, **kwargs):
        super().__init__(message, **kwargs)
        self.backoff_ms = backoff_ms


# =========================
# Transport abstraction (HTTP / MCP / IPC all supported)
# =========================

class TRPTransport(Protocol):
    def send_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send one TRP frame and return the Router response frame (dict).
        """
        ...


# =========================
# Client state
# =========================

@dataclass
class SessionState:
    session_id: Optional[str] = None
    catalog_epoch: Optional[int] = None
    seq: int = 0
    retry_budget: int = 3
    alias_table: Dict[int, CapabilityBrief] = field(default_factory=dict)  # idx -> cap
    cap_index: Dict[str, int] = field(default_factory=dict)                # cap_id -> idx


# =========================
# Main client
# =========================

class RouterClient:
    """
    The single object exposed to LLM/PTC (can be bound as `router`).
    """
    def __init__(self, transport: TRPTransport, agent_id: str = "agent_claude_ptc"):
        self.transport = transport
        self.agent_id = agent_id
        self.state = SessionState()

    # ---------- Public API ----------

    def hello(self, resume_session_id: Optional[str] = None) -> Dict[str, Any]:
        frame = self._mk_frame(
            frame_type=FrameType.HELLO_REQ,
            seq=None,
            payload={
                "agent_id": self.agent_id,
                "supported_versions": [TRP_VERSION],
                "resume_session_id": resume_session_id,
            }
        )
        res = self._send(frame)
        self._expect_type(res, FrameType.HELLO_RES)

        payload = res["payload"]
        self.state.session_id = payload["session_id"]
        self.state.catalog_epoch = payload["catalog_epoch"]
        self.state.retry_budget = payload.get("retry_budget", 3)
        self.state.seq = payload.get("seq_start", 1) - 1
        return payload

    def sync_catalog(self, mode: str = "FULL") -> List[CapabilityBrief]:
        # sync_catalog initializes session if needed; it should not require alias_table beforehand.
        if not self.state.session_id:
            self.hello()
        frame = self._mk_frame(
            frame_type=FrameType.CATALOG_SYNC_REQ,
            seq=self._next_seq(),
            payload={"mode": mode, "known_epoch": self.state.catalog_epoch}
        )
        res = self._send(frame)
        self._expect_type(res, FrameType.CATALOG_SYNC_RES)

        payload = res["payload"]
        self.state.catalog_epoch = payload["catalog_epoch"]
        self.state.alias_table.clear()
        self.state.cap_index.clear()

        caps: List[CapabilityBrief] = []
        for item in payload["alias_table"]:
            cap = CapabilityBrief(
                idx=item["idx"],
                cap_id=item["cap_id"],
                name=item["name"],
                desc=item["desc"],
                risk_tier=item["risk_tier"],
                io_class=item["io_class"],
                arg_template=item.get("arg_template", {}),
                schema_digest=item.get("schema_digest", ""),
            )
            caps.append(cap)
            self.state.alias_table[cap.idx] = cap
            self.state.cap_index[cap.cap_id] = cap.idx

        return caps

    def call(
        self,
        *,
        idx: int,
        cap_id: str,
        args: Dict[str, Any],
        call_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        timeout_ms: int = 15000,
        approval_token: Optional[str] = None,
        max_attempts: Optional[int] = None,
        execution_mode: str = "SYNC",
    ) -> Dict[str, Any]:
        """
        Single call (SDK automatically handles partial retries).
        """
        self._ensure_session()
        call_id = call_id or self._new_id("call")
        max_attempts = max_attempts or self.state.retry_budget

        attempt = 1
        while True:
            frame = self._mk_frame(
                frame_type=FrameType.CALL_REQ,
                seq=self._next_seq(),
                payload={
                    "call_id": call_id,
                    "idempotency_key": idempotency_key,
                    "idx": idx,
                    "cap_id": cap_id,
                    "depends_on": [],
                    "attempt": attempt,
                    "timeout_ms": timeout_ms,
                    "approval_token": approval_token,
                    "execution_mode": execution_mode,
                    "args": args,
                }
            )

            try:
                res = self._send(frame)
                ftype = res["frame_type"]
                if ftype == FrameType.RESULT:
                    return res["payload"]
                if ftype == FrameType.ACK:
                    # v0.1 defaults to sync execution; if made async later, polling can be added here.
                    return res["payload"]
                if ftype == FrameType.NACK:
                    self._raise_from_nack(res["payload"])
                raise TRPError(f"unexpected frame_type: {ftype}")

            except RetryableTRPError as e:
                if attempt >= max_attempts:
                    raise
                time.sleep(max(e.backoff_ms, 50) / 1000.0)
                attempt += 1
                continue

    def call_async(
        self,
        *,
        idx: int,
        cap_id: str,
        args: Dict[str, Any],
        call_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        timeout_ms: int = 15000,
        approval_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Async call: returns ACK immediately (ACCEPTED/IN_PROGRESS/DUPLICATE).
        """
        return self.call(
            idx=idx,
            cap_id=cap_id,
            args=args,
            call_id=call_id,
            idempotency_key=idempotency_key,
            timeout_ms=timeout_ms,
            approval_token=approval_token,
            max_attempts=1,
            execution_mode="ASYNC",
        )

    def call_by_cap(
        self,
        *,
        cap_id: str,
        args: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Convenience API: call by cap_id, SDK resolves idx.
        """
        self._ensure_session()
        if cap_id not in self.state.cap_index:
            self.sync_catalog()
        idx = self.state.cap_index[cap_id]
        return self.call(idx=idx, cap_id=cap_id, args=args, **kwargs)

    def cap_query(
        self,
        *,
        idx: int,
        cap_id: str,
        include_examples: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch detailed capability schema/policy hints on demand to avoid oversized catalogs.
        """
        self._ensure_session()
        frame = self._mk_frame(
            frame_type=FrameType.CAP_QUERY_REQ,
            seq=self._next_seq(),
            payload={
                "idx": idx,
                "cap_id": cap_id,
                "include_examples": include_examples,
            },
        )
        res = self._send(frame)
        if res["frame_type"] == FrameType.CAP_QUERY_RES:
            return res["payload"]
        if res["frame_type"] == FrameType.NACK:
            self._raise_from_nack(res["payload"])
        raise TRPError(f"unexpected frame_type: {res['frame_type']}")

    def query_result_frame(
        self,
        *,
        call_id: str,
        after_event_id: Optional[int] = None,
        include_partials: bool = True,
    ) -> Dict[str, Any]:
        """
        Query async call result and return the full frame.
        When after_event_id is provided, Router may return PARTIAL_RESULT.
        """
        self._ensure_session()
        payload: Dict[str, Any] = {"call_id": call_id}
        if after_event_id is not None:
            payload["after_event_id"] = int(after_event_id)
            payload["include_partials"] = bool(include_partials)
        frame = self._mk_frame(
            frame_type=FrameType.RESULT_QUERY_REQ,
            seq=self._next_seq(),
            payload=payload,
        )
        res = self._send(frame)
        if res["frame_type"] in (FrameType.RESULT_QUERY_RES, FrameType.PARTIAL_RESULT):
            return res
        if res["frame_type"] == FrameType.NACK:
            self._raise_from_nack(res["payload"])
        raise TRPError(f"unexpected frame_type: {res['frame_type']}")

    def query_result(
        self,
        *,
        call_id: str,
        after_event_id: Optional[int] = None,
        include_partials: bool = False,
    ) -> Dict[str, Any]:
        """
        Query async call result.
        By default, return RESULT_QUERY_RES.payload (backward compatible behavior).
        If after_event_id is provided and Router returns PARTIAL_RESULT, return
        PARTIAL_RESULT.payload and include `_frame_type`.
        """
        res = self.query_result_frame(
            call_id=call_id,
            after_event_id=after_event_id,
            include_partials=include_partials,
        )
        payload = dict(res["payload"])
        payload["_frame_type"] = res["frame_type"]
        return payload

    def wait_result(
        self,
        *,
        call_id: str,
        timeout_ms: int = 30000,
        poll_interval_ms: int = 100,
        on_partial: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Poll until async call result is available and return query_result payload.
        If on_partial is provided, consume PARTIAL_RESULT events and invoke callback.
        """
        deadline = time.time() + max(timeout_ms, 1) / 1000.0
        poll_sec = max(poll_interval_ms, 10) / 1000.0
        partial_cursor: Optional[int] = 0 if on_partial is not None else None
        while True:
            if partial_cursor is not None:
                frame = self.query_result_frame(
                    call_id=call_id,
                    after_event_id=partial_cursor,
                    include_partials=True,
                )
                if frame["frame_type"] == FrameType.PARTIAL_RESULT:
                    p = frame["payload"]
                    for ev in p.get("events", []) or []:
                        try:
                            on_partial(ev)  # type: ignore[misc]
                        except Exception:
                            # Callback errors must not break the wait loop.
                            pass
                    nxt = p.get("next_after_event_id")
                    if isinstance(nxt, int) and nxt >= 0:
                        partial_cursor = nxt

                    if p.get("terminal") and p.get("final_available"):
                        # Fetch final result without partials to avoid receiving the same partial batch again.
                        res = self.query_result(call_id=call_id, include_partials=False)
                    else:
                        res = p
                else:
                    res = dict(frame["payload"])
                    res["_frame_type"] = frame["frame_type"]
            else:
                res = self.query_result(call_id=call_id, include_partials=False)
            status = str(res.get("status", "UNKNOWN")).upper()
            if status in {"SUCCESS", "FAILED", "NOT_FOUND"}:
                return res
            if time.time() >= deadline:
                raise RetryableTRPError(
                    f"wait_result timeout for call_id={call_id}",
                    backoff_ms=poll_interval_ms,
                    error_class="TRANSIENT",
                    error_code="TRP_3003",
                )
            time.sleep(poll_sec)

    def batch(
        self,
        calls: List[CallSpec],
        mode: str = "PARALLEL",
        max_concurrency: int = 4,
    ) -> Dict[str, Any]:
        self._ensure_session()

        frame = self._mk_frame(
            frame_type=FrameType.CALL_BATCH_REQ,
            seq=self._next_seq(),
            payload={
                "batch_id": self._new_id("batch"),
                "mode": mode,
                "max_concurrency": max_concurrency,
                "calls": [
                    {
                        "call_id": c.call_id,
                        "idempotency_key": c.idempotency_key,
                        "idx": c.idx,
                        "cap_id": c.cap_id,
                        "attempt": c.attempt,
                        "timeout_ms": c.timeout_ms,
                        "args": c.args,
                        "depends_on": c.depends_on,
                        "approval_token": c.approval_token,
                    }
                    for c in calls
                ]
            }
        )

        res = self._send(frame)
        if res["frame_type"] == FrameType.CALL_BATCH_RES:
            return res["payload"]
        if res["frame_type"] == FrameType.NACK:
            self._raise_from_nack(res["payload"])
        raise TRPError(f"unexpected frame_type: {res['frame_type']}")

    # ---------- Internal helpers ----------

    def _send(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        raw = self.transport.send_frame(frame)
        # Minimal response validation.
        if "frame_type" not in raw or "payload" not in raw:
            raise TRPError("invalid router response")
        return raw

    def _mk_frame(self, frame_type: FrameType, seq: Optional[int], payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "trp_version": TRP_VERSION,
            "frame_type": frame_type,
            "session_id": self.state.session_id,
            "frame_id": self._new_id("frm"),
            "trace_id": self._new_id("trc"),
            "timestamp_ms": int(time.time() * 1000),
            "catalog_epoch": self.state.catalog_epoch,
            "seq": seq,
            "payload": payload,
        }

    def _next_seq(self) -> int:
        self.state.seq += 1
        return self.state.seq

    def _ensure_session(self) -> None:
        if not self.state.session_id:
            self.hello()
        if self.state.catalog_epoch is None or not self.state.alias_table:
            self.sync_catalog()

    def _expect_type(self, frame: Dict[str, Any], expected: FrameType) -> None:
        got = frame["frame_type"]
        if got != expected:
            if got == FrameType.NACK:
                self._raise_from_nack(frame["payload"])
            raise TRPError(f"expected {expected}, got {got}")

    def _raise_from_nack(self, payload: Dict[str, Any]) -> None:
        err_cls = payload.get("error_class")
        code = payload.get("error_code")
        msg = payload.get("message", "router nack")
        retryable = payload.get("retryable", False)
        retry_hint = payload.get("retry_hint", {}) or {}

        if err_cls == ErrorClass.CATALOG_MISMATCH:
            # Auto-sync catalog, then raise a retryable error.
            self.sync_catalog()
            raise RetryableTRPError(
                msg, backoff_ms=retry_hint.get("backoff_ms", 50),
                error_class=str(err_cls), error_code=code
            )
        if err_cls == ErrorClass.ORDER_VIOLATION:
            expected_seq = retry_hint.get("expected_seq")
            if expected_seq is not None:
                # Align local seq to expected_seq-1 so next _next_seq() matches.
                self.state.seq = max(0, int(expected_seq) - 1)
            raise RetryableTRPError(
                msg, backoff_ms=retry_hint.get("backoff_ms", 50),
                error_class=str(err_cls), error_code=code
            )
        if retryable:
            raise RetryableTRPError(
                msg, backoff_ms=retry_hint.get("backoff_ms", 200),
                error_class=str(err_cls), error_code=code
            )

        if err_cls == ErrorClass.APPROVAL_REQUIRED:
            raise ApprovalRequiredError(msg, error_class=str(err_cls), error_code=code)
        if err_cls == ErrorClass.POLICY_DENIED:
            raise PolicyDeniedError(msg, error_class=str(err_cls), error_code=code)
        if err_cls == ErrorClass.SCHEMA_MISMATCH:
            raise SchemaMismatchError(msg, error_class=str(err_cls), error_code=code)

        raise TRPError(msg, error_class=str(err_cls), error_code=code)

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"
