from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests


def _now_ms() -> int:
    return int(time.time() * 1000)


class SessionCtx:
    def __init__(self, session_id: str, catalog_epoch: int, trace_id: Optional[str] = None) -> None:
        self.session_id = session_id
        self.catalog_epoch = catalog_epoch
        self.seq = 1
        self.trace_id = trace_id or f"trc_bench_{uuid.uuid4().hex[:8]}"
        self._lock = threading.Lock()

    def next_frame(self, *, frame_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
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


def _wait_async_final(
    http: requests.Session,
    base_url: str,
    ctx: SessionCtx,
    call_id: str,
    *,
    timeout_sec: float,
    poll_interval_sec: float,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    after_event_id = 0
    while time.time() < deadline:
        q = post_frame(
            http,
            base_url,
            ctx.next_frame(
                frame_type="RESULT_QUERY_REQ",
                payload={"call_id": call_id, "after_event_id": after_event_id, "include_partials": True},
            ),
            timeout_sec,
        )
        if q.get("frame_type") == "PARTIAL_RESULT":
            p = q.get("payload", {})
            try:
                after_event_id = int(p.get("next_after_event_id", after_event_id))
            except Exception:
                pass
            if p.get("terminal") and p.get("final_available"):
                final = post_frame(
                    http,
                    base_url,
                    ctx.next_frame(frame_type="RESULT_QUERY_REQ", payload={"call_id": call_id}),
                    timeout_sec,
                )
                return final
        elif q.get("frame_type") == "RESULT_QUERY_RES":
            status = str(q.get("payload", {}).get("status"))
            if status in {"SUCCESS", "FAILED", "NOT_FOUND"}:
                return q
        time.sleep(poll_interval_sec)
    raise TimeoutError(f"async call {call_id} did not finish before timeout")


def run_worker(
    *,
    worker_idx: int,
    base_url: str,
    calls_per_worker: int,
    timeout_sec: float,
    mode: str,
    top_k: int,
    poll_interval_sec: float,
) -> Dict[str, Any]:
    http = requests.Session()
    ctx = hello_and_sync(http, base_url, timeout_sec, agent_id=f"bench_w{worker_idx}")
    latencies_ms: List[float] = []
    success = 0
    failed = 0
    nacks = 0
    errors: List[str] = []

    for i in range(calls_per_worker):
        call_id = f"w{worker_idx}_c{i}_{uuid.uuid4().hex[:6]}"
        query = f"bench worker {worker_idx} call {i}"
        frame = ctx.next_frame(
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
                "execution_mode": "ASYNC" if mode == "async" else "SYNC",
                "args": {"query": query, "top_k": top_k},
            },
        )
        t0 = time.perf_counter()
        try:
            res = post_frame(http, base_url, frame, timeout_sec)
            if mode == "async":
                if res.get("frame_type") != "ACK":
                    if res.get("frame_type") == "NACK":
                        nacks += 1
                    failed += 1
                    errors.append(f"ACK:{res.get('frame_type')}")
                    continue
                res = _wait_async_final(
                    http,
                    base_url,
                    ctx,
                    call_id,
                    timeout_sec=timeout_sec,
                    poll_interval_sec=poll_interval_sec,
                )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency_ms)
            if res.get("frame_type") == "RESULT":
                success += 1
            elif res.get("frame_type") == "RESULT_QUERY_RES" and res.get("payload", {}).get("status") == "SUCCESS":
                success += 1
            elif res.get("frame_type") == "NACK":
                nacks += 1
                failed += 1
            else:
                failed += 1
                errors.append(str(res.get("frame_type")))
        except Exception as e:
            failed += 1
            errors.append(type(e).__name__)
    http.close()
    return {
        "worker": worker_idx,
        "latencies_ms": latencies_ms,
        "success": success,
        "failed": failed,
        "nacks": nacks,
        "errors": errors[:10],
    }


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def main() -> None:
    ap = argparse.ArgumentParser(description="TRP HTTP benchmark (sync/async)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--calls-per-worker", type=int, default=50)
    ap.add_argument("--mode", choices=["sync", "async"], default="sync")
    ap.add_argument("--timeout-sec", type=float, default=10.0)
    ap.add_argument("--poll-interval-sec", type=float, default=0.03)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    started = time.perf_counter()
    worker_results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [
            pool.submit(
                run_worker,
                worker_idx=w,
                base_url=args.base_url,
                calls_per_worker=args.calls_per_worker,
                timeout_sec=args.timeout_sec,
                mode=args.mode,
                top_k=args.top_k,
                poll_interval_sec=args.poll_interval_sec,
            )
            for w in range(args.workers)
        ]
        for fut in as_completed(futs):
            worker_results.append(fut.result())
    elapsed_sec = max(1e-9, time.perf_counter() - started)

    latencies = [x for wr in worker_results for x in wr["latencies_ms"]]
    success = sum(int(wr["success"]) for wr in worker_results)
    failed = sum(int(wr["failed"]) for wr in worker_results)
    nacks = sum(int(wr["nacks"]) for wr in worker_results)
    total_attempts = args.workers * args.calls_per_worker

    summary = {
        "mode": args.mode,
        "base_url": args.base_url,
        "workers": args.workers,
        "calls_per_worker": args.calls_per_worker,
        "attempted_calls": total_attempts,
        "completed_measurements": len(latencies),
        "success": success,
        "failed": failed,
        "nacks": nacks,
        "success_rate": round((success / total_attempts) if total_attempts else 0.0, 4),
        "elapsed_sec": round(elapsed_sec, 4),
        "throughput_rps": round((success / elapsed_sec) if elapsed_sec else 0.0, 2),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 2) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 2) if latencies else 0.0,
            "p99": round(percentile(latencies, 0.99), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "sample_errors": [e for wr in worker_results for e in wr.get("errors", [])][:20],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
