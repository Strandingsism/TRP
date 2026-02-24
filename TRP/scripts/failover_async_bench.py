from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import requests


def _now_ms() -> int:
    return int(time.time() * 1000)


class SessionCtx:
    def __init__(self, session_id: str, catalog_epoch: int, trace_id: Optional[str] = None) -> None:
        self.session_id = session_id
        self.catalog_epoch = catalog_epoch
        self.seq = 1
        self.trace_id = trace_id or f"trc_failover_{uuid.uuid4().hex[:8]}"

    def next_frame(self, *, frame_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        seq = self.seq
        self.seq += 1
        return {
            "trp_version": "0.1",
            "frame_type": frame_type,
            "session_id": self.session_id,
            "frame_id": f"frm_{uuid.uuid4().hex[:12]}",
            "trace_id": self.trace_id,
            "timestamp_ms": _now_ms(),
            "catalog_epoch": self.catalog_epoch,
            "seq": seq,
            "payload": payload,
        }


def post_frame(http: requests.Session, base_url: str, frame: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
    r = http.post(f"{base_url.rstrip('/')}/trp/frame", json=frame, timeout=timeout_sec)
    r.raise_for_status()
    return r.json()


def post_frame_with_recovery(
    http: requests.Session,
    base_url: str,
    ctx: SessionCtx,
    *,
    frame_type: str,
    payload: Dict[str, Any],
    timeout_sec: float,
    max_retries: int = 4,
) -> Dict[str, Any]:
    for _ in range(max_retries):
        frame = ctx.next_frame(frame_type=frame_type, payload=payload)
        res = post_frame(http, base_url, frame, timeout_sec)
        if res.get("frame_type") != "NACK":
            return res
        nack = res.get("payload", {}) if isinstance(res.get("payload"), dict) else {}
        err_cls = str(nack.get("error_class", ""))
        err_code = str(nack.get("error_code", ""))
        retry_hint = nack.get("retry_hint", {}) if isinstance(nack.get("retry_hint"), dict) else {}
        if err_cls == "ORDER_VIOLATION" and err_code in {"TRP_1002", "TRP_1004"}:
            expected = retry_hint.get("expected_seq")
            if isinstance(expected, int) and expected > 0:
                ctx.seq = expected
                continue
        if err_cls == "CATALOG_MISMATCH" or err_code == "TRP_1003":
            sync = post_frame(
                http,
                base_url,
                ctx.next_frame(
                    frame_type="CATALOG_SYNC_REQ",
                    payload={"mode": "FULL", "known_epoch": ctx.catalog_epoch},
                ),
                timeout_sec,
            )
            if sync.get("frame_type") == "CATALOG_SYNC_RES":
                new_epoch = sync.get("payload", {}).get("catalog_epoch")
                if isinstance(new_epoch, int):
                    ctx.catalog_epoch = new_epoch
                continue
        return res
    return res


def hello_and_sync(http: requests.Session, base_url: str, timeout_sec: float, agent_id: str) -> SessionCtx:
    hello = {
        "trp_version": "0.1",
        "frame_type": "HELLO_REQ",
        "session_id": None,
        "frame_id": f"frm_hello_{uuid.uuid4().hex[:12]}",
        "trace_id": f"trc_{uuid.uuid4().hex[:8]}",
        "timestamp_ms": _now_ms(),
        "catalog_epoch": None,
        "seq": None,
        "payload": {"agent_id": agent_id, "supported_versions": ["0.1"], "resume_session_id": None},
    }
    h = post_frame(http, base_url, hello, timeout_sec)
    if h.get("frame_type") != "HELLO_RES":
        raise RuntimeError(f"HELLO failed: {h}")
    ctx = SessionCtx(h["payload"]["session_id"], h["payload"]["catalog_epoch"], trace_id=hello["trace_id"])
    s = post_frame(
        http,
        base_url,
        ctx.next_frame(frame_type="CATALOG_SYNC_REQ", payload={"mode": "FULL", "known_epoch": ctx.catalog_epoch}),
        timeout_sec,
    )
    if s.get("frame_type") != "CATALOG_SYNC_RES":
        raise RuntimeError(f"CATALOG_SYNC failed: {s}")
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser(description="Async failover benchmark via duplicate CALL_REQ(ASYNC)+RESULT_QUERY")
    ap.add_argument("--primary-url", default="http://127.0.0.1:8001")
    ap.add_argument("--secondary-url", default="http://127.0.0.1:8002")
    ap.add_argument("--num-calls", type=int, default=64)
    ap.add_argument("--timeout-sec", type=float, default=8.0)
    ap.add_argument("--call-timeout-ms", type=int, default=10000)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--kill-window-sec", type=float, default=2.0, help="Pause after primaries are queued so operator can kill primary")
    ap.add_argument("--recovery-timeout-sec", type=float, default=60.0)
    ap.add_argument("--poll-interval-sec", type=float, default=0.05)
    args = ap.parse_args()

    http_primary = requests.Session()
    http_secondary = requests.Session()
    t0 = time.perf_counter()
    ctx = hello_and_sync(http_primary, args.primary_url, args.timeout_sec, "failover_bench")

    calls: Dict[str, Dict[str, Any]] = {}
    primary_ack_ok = 0
    primary_ack_err = 0
    for i in range(args.num_calls):
        call_id = f"fail_{i}_{uuid.uuid4().hex[:6]}"
        call_args = {"query": f"failover bench {i}", "top_k": args.top_k}
        calls[call_id] = {
            "args": call_args,
            "status": "PENDING",
            "resend_attempts": 0,
            "secondary_accepted": 0,
            "secondary_in_progress": 0,
            "secondary_duplicate": 0,
        }
        res = post_frame_with_recovery(
            http_primary,
            args.primary_url,
            ctx,
            frame_type="CALL_REQ",
            payload={
                "call_id": call_id,
                "idempotency_key": None,
                "idx": 0,
                "cap_id": "cap.search.web.v1",
                "depends_on": [],
                "attempt": 1,
                "timeout_ms": args.call_timeout_ms,
                "approval_token": None,
                "execution_mode": "ASYNC",
                "args": call_args,
            },
            timeout_sec=args.timeout_sec,
        )
        if res.get("frame_type") == "ACK":
            primary_ack_ok += 1
        else:
            primary_ack_err += 1
            calls[call_id]["status"] = "FAILED_TO_QUEUE"

    print(json.dumps({"phase": "queued", "primary_ack_ok": primary_ack_ok, "primary_ack_err": primary_ack_err}, ensure_ascii=False), flush=True)
    print(json.dumps({"phase": "kill_window", "kill_window_sec": args.kill_window_sec, "message": "kill primary now"}, ensure_ascii=False), flush=True)
    time.sleep(max(0.0, args.kill_window_sec))

    deadline = time.time() + max(1.0, args.recovery_timeout_sec)
    final_success = 0
    final_failed = 0
    result_query_errors = 0
    resend_nacks = 0
    resend_http_errors = 0

    pending = [cid for cid, rec in calls.items() if rec["status"] == "PENDING"]
    while pending and time.time() < deadline:
        next_pending: List[str] = []
        for call_id in pending:
            rec = calls[call_id]
            rec["resend_attempts"] += 1

            # Duplicate async call to secondary; stale lease path will requeue if needed.
            try:
                ack = post_frame_with_recovery(
                    http_secondary,
                    args.secondary_url,
                    ctx,
                    frame_type="CALL_REQ",
                    payload={
                        "call_id": call_id,
                        "idempotency_key": None,
                        "idx": 0,
                        "cap_id": "cap.search.web.v1",
                        "depends_on": [],
                        "attempt": 1,
                        "timeout_ms": args.call_timeout_ms,
                        "approval_token": None,
                        "execution_mode": "ASYNC",
                        "args": rec["args"],
                    },
                    timeout_sec=args.timeout_sec,
                )
                if ack.get("frame_type") == "ACK":
                    status = str(ack.get("payload", {}).get("status"))
                    if status == "ACCEPTED":
                        rec["secondary_accepted"] += 1
                    elif status == "IN_PROGRESS":
                        rec["secondary_in_progress"] += 1
                    elif status == "DUPLICATE":
                        rec["secondary_duplicate"] += 1
                elif ack.get("frame_type") == "NACK":
                    resend_nacks += 1
            except Exception:
                resend_http_errors += 1
                next_pending.append(call_id)
                continue

            try:
                q = post_frame_with_recovery(
                    http_secondary,
                    args.secondary_url,
                    ctx,
                    frame_type="RESULT_QUERY_REQ",
                    payload={"call_id": call_id, "after_event_id": 0, "include_partials": True},
                    timeout_sec=args.timeout_sec,
                )
            except Exception:
                result_query_errors += 1
                next_pending.append(call_id)
                continue

            if q.get("frame_type") == "RESULT_QUERY_RES":
                status = str(q.get("payload", {}).get("status"))
                if status == "SUCCESS":
                    rec["status"] = "SUCCESS"
                    final_success += 1
                    continue
                if status == "FAILED":
                    rec["status"] = "FAILED"
                    final_failed += 1
                    continue
            elif q.get("frame_type") == "PARTIAL_RESULT":
                qp = q.get("payload", {}) if isinstance(q.get("payload"), dict) else {}
                if bool(qp.get("terminal")) and bool(qp.get("final_available")):
                    qf = post_frame_with_recovery(
                        http_secondary,
                        args.secondary_url,
                        ctx,
                        frame_type="RESULT_QUERY_REQ",
                        payload={"call_id": call_id},
                        timeout_sec=args.timeout_sec,
                    )
                    if qf.get("frame_type") == "RESULT_QUERY_RES":
                        status = str(qf.get("payload", {}).get("status"))
                        if status == "SUCCESS":
                            rec["status"] = "SUCCESS"
                            final_success += 1
                            continue
                        if status == "FAILED":
                            rec["status"] = "FAILED"
                            final_failed += 1
                            continue
            next_pending.append(call_id)

        pending = next_pending
        if pending:
            time.sleep(max(0.0, args.poll_interval_sec))

    timed_out = 0
    for call_id in pending:
        calls[call_id]["status"] = "TIMEOUT"
        timed_out += 1

    elapsed_sec = time.perf_counter() - t0
    accepted_takeovers = sum(int(c["secondary_accepted"]) > 0 for c in calls.values())
    summary = {
        "primary_url": args.primary_url,
        "secondary_url": args.secondary_url,
        "num_calls": args.num_calls,
        "primary_ack_ok": primary_ack_ok,
        "primary_ack_err": primary_ack_err,
        "final_success": final_success,
        "final_failed": final_failed,
        "timed_out": timed_out,
        "accepted_takeovers": accepted_takeovers,
        "resend_nacks": resend_nacks,
        "resend_http_errors": resend_http_errors,
        "result_query_errors": result_query_errors,
        "elapsed_sec": round(elapsed_sec, 3),
        "sample_calls": {
            cid: {
                "status": rec["status"],
                "resend_attempts": rec["resend_attempts"],
                "secondary_accepted": rec["secondary_accepted"],
                "secondary_in_progress": rec["secondary_in_progress"],
                "secondary_duplicate": rec["secondary_duplicate"],
            }
            for cid, rec in list(calls.items())[:5]
        },
    }
    print(json.dumps({"phase": "done", **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
