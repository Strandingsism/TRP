from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional

from .router_interfaces import (
    CapabilityRegistry, SessionManager, PolicyEngine, IdempotencyStore,
    AdapterRegistry, Executor, ResultShaper, AuditLogger, RuntimeStateStore
)


class RouterService:
    def __init__(
        self,
        *,
        sessions: SessionManager,
        registry: CapabilityRegistry,
        policy: PolicyEngine,
        idempotency: IdempotencyStore,
        adapters: AdapterRegistry,
        executor: Executor,
        shaper: ResultShaper,
        audit: AuditLogger,
        runtime_state: Optional[RuntimeStateStore] = None,
        call_record_ttl_sec: int = 86400,
        async_result_ttl_sec: int = 600,
        async_event_limit: int = 256,
        async_cleanup_interval_sec: int = 30,
    ):
        self.sessions = sessions
        self.registry = registry
        self.policy = policy
        self.idempotency = idempotency
        self.adapters = adapters
        self.executor = executor
        self.shaper = shaper
        self.audit = audit
        self.runtime_state = runtime_state
        self._call_record_ttl_sec = max(1, int(call_record_ttl_sec))
        self._call_records_lock = threading.Lock()
        # session_id -> call_id -> {"status": "SUCCESS"|"FAILED", "retryable": bool}
        self._call_records: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._async_calls_lock = threading.Lock()
        # session_id -> call_id -> async state
        self._async_calls: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._async_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="trp-async")
        self._async_result_ttl_ms = max(1000, int(async_result_ttl_sec * 1000))
        self._async_event_limit = max(1, int(async_event_limit))
        self._async_cleanup_interval_ms = max(1000, int(async_cleanup_interval_sec * 1000))
        self._async_last_cleanup_ms = 0

    def handle_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        ftype = frame.get("frame_type")
        try:
            if ftype == "HELLO_REQ":
                return self._hello(frame)
            if ftype == "CATALOG_SYNC_REQ":
                return self._catalog_sync(frame)
            if ftype == "CAP_QUERY_REQ":
                return self._cap_query(frame)
            if ftype == "CALL_REQ":
                return self._call(frame)
            if ftype == "CALL_BATCH_REQ":
                return self._call_batch(frame)
            if ftype == "RESULT_QUERY_REQ":
                return self._result_query(frame)
            return self._nack(frame, None, "INTERNAL_ERROR", "TRP_5000", f"unsupported frame_type: {ftype}", False)
        except Exception as e:
            # 兜底（理论上顺序类错误不会再走到这里）
            self.audit.log_event("router.internal_error", {"error": repr(e), "frame": self._safe_frame(frame)})
            return self._nack(frame, None, "INTERNAL_ERROR", "TRP_5001", "internal error", False)

    # ---------- handlers ----------

    def _hello(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        p = frame["payload"]
        info = self.sessions.hello(
            agent_id=p["agent_id"],
            resume_session_id=p.get("resume_session_id")
        )
        return self._res(
            frame,
            "HELLO_RES",
            info,
            seq=None,
            catalog_epoch=info["catalog_epoch"],
            session_id=info["session_id"],
        )

    def _catalog_sync(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        session_id = frame["session_id"]

        seq_nack = self._check_seq(frame)
        if seq_nack is not None:
            return seq_nack

        epoch, alias_table = self.registry.get_catalog(session_id)
        payload = {
            "catalog_epoch": epoch,
            "alias_table": alias_table,
            "ttl_sec": 600,
        }
        return self._res(
            frame,
            "CATALOG_SYNC_RES",
            payload,
            seq=frame["seq"],
            catalog_epoch=epoch,
            session_id=session_id,
        )

    def _call(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        execution_mode = str(frame.get("payload", {}).get("execution_mode", "SYNC")).upper()
        if execution_mode == "ASYNC":
            return self._call_async(frame)
        return self._execute_call_core(frame, check_seq=True)

    def _call_async(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        session_id = frame["session_id"]

        seq_nack = self._check_seq(frame)
        if seq_nack is not None:
            return seq_nack

        p = frame["payload"]
        call_id = p["call_id"]

        existing = self._get_async_call_state(session_id, call_id)
        if existing is not None:
            status = existing.get("status")
            if status in {"QUEUED", "RUNNING"}:
                return self._ack(frame, call_id, status="IN_PROGRESS")
            if status in {"SUCCESS", "FAILED"}:
                return self._res(
                    frame,
                    "ACK",
                    {
                        "ack_of_frame_id": frame.get("frame_id"),
                        "ack_of_call_id": call_id,
                        "status": "DUPLICATE",
                        "expected_seq_next": None,
                    },
                    seq=frame.get("seq"),
                    catalog_epoch=frame.get("catalog_epoch"),
                    session_id=session_id,
                )

        self._set_async_call_state(
            session_id,
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
        self._append_async_event(
            session_id,
            call_id,
            {
                "kind": "STATE",
                "stage": "QUEUED",
                "message": "accepted for async execution",
            },
        )

        worker_frame = self._clone_frame_for_async(frame)
        self._async_pool.submit(self._run_async_call, worker_frame)
        return self._ack(frame, call_id, status="ACCEPTED")

    def _result_query(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        session_id = frame["session_id"]
        seq_nack = self._check_seq(frame)
        if seq_nack is not None:
            return seq_nack

        p = frame["payload"]
        call_id = p.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return self._nack(frame, None, "SCHEMA_MISMATCH", "TRP_2005", "call_id is required", False)
        after_event_id_raw = p.get("after_event_id")
        after_event_id: Optional[int] = None
        if after_event_id_raw is not None:
            try:
                after_event_id = int(after_event_id_raw)
            except (TypeError, ValueError):
                return self._nack(frame, None, "SCHEMA_MISMATCH", "TRP_2006", "after_event_id must be an integer", False)
            if after_event_id < 0:
                return self._nack(frame, None, "SCHEMA_MISMATCH", "TRP_2006", "after_event_id must be >= 0", False)
        include_partials = bool(p.get("include_partials", True))

        state = self._get_async_call_state(session_id, call_id)
        if state is None:
            payload = {
                "call_id": call_id,
                "status": "NOT_FOUND",
                "result": None,
                "error": None,
                "last_event_id": 0,
            }
            return self._res(
                frame,
                "RESULT_QUERY_RES",
                payload,
                seq=frame.get("seq"),
                catalog_epoch=frame.get("catalog_epoch"),
                session_id=session_id,
            )

        st = state.get("status")
        last_event_id = max(0, int(state.get("next_event_id", 1)) - 1)
        if include_partials and after_event_id is not None:
            events_page = self._get_async_events_after(session_id, call_id, after_event_id)
            events = events_page["events"]
            if events:
                payload = {
                    "call_id": call_id,
                    "status": "IN_PROGRESS" if st in {"QUEUED", "RUNNING"} else st,
                    "events": events,
                    "next_after_event_id": events[-1]["event_id"],
                    "events_truncated": events_page["events_truncated"],
                    "first_retained_event_id": events_page["first_retained_event_id"],
                    "last_event_id": last_event_id,
                    "terminal": st in {"SUCCESS", "FAILED"},
                    "final_available": st in {"SUCCESS", "FAILED"},
                }
                return self._res(
                    frame,
                    "PARTIAL_RESULT",
                    payload,
                    seq=frame.get("seq"),
                    catalog_epoch=frame.get("catalog_epoch"),
                    session_id=session_id,
                )

        if st in {"QUEUED", "RUNNING"}:
            payload = {
                "call_id": call_id,
                "status": "IN_PROGRESS",
                "result": None,
                "error": None,
                "last_event_id": last_event_id,
            }
        elif st == "SUCCESS":
            payload = {
                "call_id": call_id,
                "status": "SUCCESS",
                "result": state.get("result_payload"),
                "error": None,
                "last_event_id": last_event_id,
            }
        elif st == "FAILED":
            payload = {
                "call_id": call_id,
                "status": "FAILED",
                "result": None,
                "error": state.get("result_payload"),
                "last_event_id": last_event_id,
            }
        else:
            payload = {
                "call_id": call_id,
                "status": "UNKNOWN",
                "result": None,
                "error": None,
                "last_event_id": last_event_id,
            }

        return self._res(
            frame,
            "RESULT_QUERY_RES",
            payload,
            seq=frame.get("seq"),
            catalog_epoch=frame.get("catalog_epoch"),
            session_id=session_id,
        )

    def _cap_query(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        session_id = frame["session_id"]

        seq_nack = self._check_seq(frame)
        if seq_nack is not None:
            return seq_nack

        p = frame["payload"]
        idx = p["idx"]
        cap_id = p["cap_id"]
        include_examples = bool(p.get("include_examples", False))

        try:
            cap = self.registry.resolve(
                session_id=session_id,
                catalog_epoch=frame["catalog_epoch"],
                idx=idx,
                cap_id=cap_id,
            )
        except Exception:
            return self._nack(
                frame,
                None,
                "CATALOG_MISMATCH",
                "TRP_1003",
                "catalog mismatch, sync catalog first",
                True,
                retry_hint={"action": "SYNC_CATALOG", "backoff_ms": 50},
            )

        payload = {
            "idx": idx,
            "cap_id": cap.cap_id,
            "canonical_schema": self._build_canonical_schema(cap.arg_template),
            "policy_hints": {
                "requires_approval": cap.risk_tier in {"HIGH", "CRITICAL"},
                "idempotency_required": cap.io_class == "WRITE",
            },
            "examples": (
                [{"args": self._build_example_args(cap.arg_template)}]
                if include_examples else []
            ),
        }
        return self._res(
            frame,
            "CAP_QUERY_RES",
            payload,
            seq=frame["seq"],
            catalog_epoch=frame["catalog_epoch"],
            session_id=session_id,
        )

    def _call_batch(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        seq_nack = self._check_seq(frame)
        if seq_nack is not None:
            return seq_nack

        p = frame["payload"]
        batch_id = p["batch_id"]
        mode = str(p.get("mode", "PARALLEL")).upper()
        try:
            max_concurrency = max(1, int(p.get("max_concurrency", 4)))
        except (TypeError, ValueError):
            return self._nack(frame, None, "SCHEMA_MISMATCH", "TRP_2004", "max_concurrency must be an integer", False)
        calls = p.get("calls", [])
        results_map = (
            self._execute_batch_parallel(frame, calls, max_concurrency=max_concurrency)
            if mode == "PARALLEL" and len(calls) > 1
            else self._execute_batch_sequential(frame, calls)
        )
        results = [results_map[c["call_id"]] for c in calls]
        any_fail = any(r["status"] == "FAILED" for r in results)

        status = "PARTIAL_SUCCESS" if any_fail and len(results) > 0 else "SUCCESS"
        if any_fail and all(r["status"] == "FAILED" for r in results):
            status = "FAILED"

        payload = {
            "batch_id": batch_id,
            "status": status,
            "results": results,
        }
        return self._res(
            frame,
            "CALL_BATCH_RES",
            payload,
            seq=frame["seq"],
            catalog_epoch=frame["catalog_epoch"],
            session_id=frame["session_id"],
        )

    def _execute_batch_sequential(
        self,
        frame: Dict[str, Any],
        calls: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        results_map: Dict[str, Dict[str, Any]] = {}
        for c in calls:
            sub_frame = dict(frame)
            sub_frame["frame_id"] = f'{frame["frame_id"]}:{c["call_id"]}'
            sub_frame["payload"] = c
            # batch 内部子调用不再推进 seq（seq 已由 batch 占用）
            res = self._execute_call_core(sub_frame, check_seq=False)
            results_map[c["call_id"]] = self._batch_result_item(c["call_id"], res)
        return results_map

    def _execute_batch_parallel(
        self,
        frame: Dict[str, Any],
        calls: List[Dict[str, Any]],
        *,
        max_concurrency: int,
    ) -> Dict[str, Dict[str, Any]]:
        session_id = frame["session_id"]
        pending: Dict[str, Dict[str, Any]] = {c["call_id"]: c for c in calls}
        running: Dict[Future[Dict[str, Any]], str] = {}
        results_map: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            while pending or running:
                # 先处理明显无法满足的依赖（依赖已失败）
                for call_id in list(pending.keys()):
                    c = pending[call_id]
                    dep_failure = self._find_failed_dependency(
                        session_id=session_id,
                        depends_on=c.get("depends_on", []),
                        batch_results=results_map,
                    )
                    if dep_failure is None:
                        continue
                    results_map[call_id] = {
                        "call_id": call_id,
                        "status": "FAILED",
                        "error_class": "ORDER_VIOLATION",
                        "error_code": "TRP_1006",
                        "retryable": False,
                        "message": f"dependency failed: {dep_failure}",
                    }
                    self._record_call_result(session_id, call_id, success=False, retryable=False)
                    pending.pop(call_id, None)

                # 提交 ready 任务（考虑依赖和并发上限）
                slots = max_concurrency - len(running)
                if slots > 0:
                    ready_ids = []
                    for call_id, c in pending.items():
                        if self._deps_ready(
                            session_id=session_id,
                            depends_on=c.get("depends_on", []),
                            batch_results=results_map,
                        ):
                            ready_ids.append(call_id)
                    for call_id in ready_ids[:slots]:
                        c = pending.pop(call_id)
                        sub_frame = dict(frame)
                        sub_frame["frame_id"] = f'{frame["frame_id"]}:{call_id}'
                        sub_frame["payload"] = c
                        fut = pool.submit(self._execute_call_core, sub_frame, check_seq=False)
                        running[fut] = call_id

                if not running:
                    # 无可执行任务但还有 pending：依赖缺失/循环依赖
                    for call_id in list(pending.keys()):
                        results_map[call_id] = {
                            "call_id": call_id,
                            "status": "FAILED",
                            "error_class": "ORDER_VIOLATION",
                            "error_code": "TRP_1005",
                            "retryable": True,
                            "message": "depends_on not satisfied or cyclic dependency",
                        }
                        self._record_call_result(session_id, call_id, success=False, retryable=True)
                        pending.pop(call_id, None)
                    break

                done, _ = wait(list(running.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    call_id = running.pop(fut)
                    res = fut.result()
                    results_map[call_id] = self._batch_result_item(call_id, res)

        return results_map

    def _batch_result_item(self, call_id: str, res: Dict[str, Any]) -> Dict[str, Any]:
        if res["frame_type"] == "RESULT":
            return {
                "call_id": call_id,
                "status": "SUCCESS",
                "result": res["payload"]["result"],
            }
        nack = res["payload"]
        return {
            "call_id": call_id,
            "status": "FAILED",
            "error_class": nack["error_class"],
            "error_code": nack["error_code"],
            "retryable": nack["retryable"],
            "message": nack["message"],
        }

    # ---------- core call pipeline ----------

    def _execute_call_core(self, frame: Dict[str, Any], *, check_seq: bool) -> Dict[str, Any]:
        session_id = frame["session_id"]

        if check_seq:
            seq_nack = self._check_seq(frame)
            if seq_nack is not None:
                return seq_nack

        p = frame["payload"]
        call_id = p["call_id"]
        idx = p["idx"]
        cap_id = p["cap_id"]
        depends_on = p.get("depends_on", []) or []
        args = p.get("args", {})
        try:
            timeout_ms = int(p.get("timeout_ms", 15000))
        except (TypeError, ValueError):
            self._record_call_result(session_id, call_id, success=False, retryable=False)
            return self._nack(frame, call_id, "SCHEMA_MISMATCH", "TRP_2002", "timeout_ms must be an integer", False)
        if timeout_ms <= 0:
            self._record_call_result(session_id, call_id, success=False, retryable=False)
            return self._nack(frame, call_id, "SCHEMA_MISMATCH", "TRP_2002", "timeout_ms must be > 0", False)
        idem_key = p.get("idempotency_key")
        approval_token = p.get("approval_token")
        auth_context = frame.get("auth_context")

        # 0) 依赖检查（v0.1：要求依赖调用已完成且成功）
        if not isinstance(depends_on, list) or any(not isinstance(x, str) for x in depends_on):
            self._record_call_result(session_id, call_id, success=False, retryable=False)
            return self._nack(frame, call_id, "SCHEMA_MISMATCH", "TRP_2003", "depends_on must be a list of call_id strings", False)
        dep_check = self._check_dependencies(session_id=session_id, depends_on=depends_on)
        if dep_check is not None:
            return self._nack(
                frame,
                call_id,
                "ORDER_VIOLATION",
                dep_check["error_code"],
                dep_check["message"],
                dep_check["retryable"],
                retry_hint=dep_check.get("retry_hint"),
            )

        # 1) 目录解析与校验（idx + cap_id + epoch）
        try:
            cap = self.registry.resolve(
                session_id=session_id,
                catalog_epoch=frame["catalog_epoch"],
                idx=idx,
                cap_id=cap_id,
            )
        except Exception:
            return self._nack(
                frame,
                call_id,
                "CATALOG_MISMATCH",
                "TRP_1003",
                "catalog mismatch, sync catalog first",
                True,
                retry_hint={"action": "SYNC_CATALOG", "backoff_ms": 50},
            )

        # 2) 高副作用幂等检查
        if cap.io_class == "WRITE" and not idem_key:
            self._record_call_result(session_id, call_id, success=False, retryable=False)
            return self._nack(
                frame,
                call_id,
                "NON_IDEMPOTENT_BLOCKED",
                "TRP_4003",
                "idempotency_key required for WRITE capability",
                False,
            )

        if idem_key:
            cached = self.idempotency.get(cap.cap_id, idem_key)
            if cached is not None:
                return self._res(
                    frame,
                    "RESULT",
                    cached,
                    seq=frame.get("seq"),
                    catalog_epoch=frame.get("catalog_epoch"),
                    session_id=session_id,
                )

        # 3) 参数校验与语义映射
        adapter = self.adapters.get(cap.adapter_key)
        try:
            adapter.validate_canonical_args(cap, args)
            native_args = adapter.to_native_args(cap, args)
        except Exception as e:
            self._record_call_result(session_id, call_id, success=False, retryable=False)
            return self._nack(frame, call_id, "SCHEMA_MISMATCH", "TRP_2001", str(e), False)

        # 4) 策略判断（权限/审批）
        decision = self.policy.evaluate(
            auth_context=auth_context,
            cap=cap,
            args=args,
            idempotency_key=idem_key,
            approval_token=approval_token,
        )
        if not decision.allowed and decision.requires_approval:
            self._record_call_result(session_id, call_id, success=False, retryable=False)
            return self._nack(
                frame,
                call_id,
                "APPROVAL_REQUIRED",
                "TRP_4002",
                decision.reason or "approval required",
                False,
            )
        if not decision.allowed:
            self._record_call_result(session_id, call_id, success=False, retryable=False)
            return self._nack(
                frame,
                call_id,
                "POLICY_DENIED",
                "TRP_4001",
                decision.reason or "denied",
                False,
            )

        # 5) 执行
        t0 = time.time()
        try:
            raw = self.executor.execute(cap, native_args, timeout_ms=timeout_ms)
            shaped = self.shaper.shape_success(cap, raw)
            payload = {
                "call_id": call_id,
                "idx": idx,
                "cap_id": cap.cap_id,
                "status": "SUCCESS",
                "result": shaped,
                "usage": {
                    "router_ms": int((time.time() - t0) * 1000),
                }
            }
            if idem_key:
                self.idempotency.put(cap.cap_id, idem_key, payload)
            self._record_call_result(session_id, call_id, success=True, retryable=False)

            self.audit.log_event("call.succeeded", {
                "session_id": session_id,
                "call_id": call_id,
                "idx": idx,
                "cap_id": cap.cap_id,
            })

            return self._res(
                frame,
                "RESULT",
                payload,
                seq=frame.get("seq"),
                catalog_epoch=frame.get("catalog_epoch"),
                session_id=session_id,
            )

        except TimeoutError:
            self._record_call_result(session_id, call_id, success=False, retryable=True)
            return self._nack(
                frame,
                call_id,
                "TRANSIENT",
                "TRP_3001",
                "executor timeout",
                True,
                retry_hint={"backoff_ms": 200},
            )
        except Exception as e:
            self._record_call_result(session_id, call_id, success=False, retryable=False)
            self.audit.log_event("call.failed", {
                "session_id": session_id,
                "call_id": call_id,
                "idx": idx,
                "cap_id": cap.cap_id,
                "error": repr(e),
            })
            return self._nack(frame, call_id, "EXECUTOR_ERROR", "TRP_3002", str(e), False)

    # ---------- seq / ordering ----------

    def _check_seq(self, frame: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        成功返回 None；失败返回 NACK frame。
        不把顺序类异常抛到 handle_frame 顶层。
        """
        try:
            r = self.sessions.check_and_advance_seq(
                session_id=frame["session_id"],
                seq=frame["seq"],
                frame_id=frame["frame_id"],
            )
            self.audit.log_event("frame.accepted", {
                "session_id": frame["session_id"],
                "frame_id": frame["frame_id"],
                "seq": frame["seq"],
                "expected_seq_next": r.get("expected_seq_next"),
            })
            return None

        except Exception as e:
            ename = e.__class__.__name__

            if ename == "OrderViolationError":
                expected_seq = getattr(e, "expected_seq", None)
                return self._nack(
                    frame,
                    None,
                    "ORDER_VIOLATION",
                    "TRP_1002",
                    str(e),
                    True,
                    retry_hint={
                        "expected_seq": expected_seq,
                        "backoff_ms": 50,
                    },
                )

            if ename == "DuplicateFrameError":
                # MVP 先返回 NACK；后面可升级成 ACK DUPLICATE
                return self._nack(
                    frame,
                    None,
                    "ORDER_VIOLATION",
                    "TRP_1004",
                    str(e),
                    False,
                    retry_hint={},
                )

            if isinstance(e, ValueError) and "unknown session_id" in str(e):
                return self._nack(
                    frame,
                    None,
                    "INTERNAL_ERROR",
                    "TRP_5002",
                    "unknown session_id",
                    False,
                )

            raise

    # ---------- response helpers ----------

    def _res(
        self,
        req: Dict[str, Any],
        frame_type: str,
        payload: Dict[str, Any],
        *,
        seq: Optional[int],
        catalog_epoch: Optional[int],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "trp_version": req.get("trp_version", "0.1"),
            "frame_type": frame_type,
            "session_id": session_id,
            "frame_id": f'res_{req.get("frame_id", "unknown")}',
            "trace_id": req.get("trace_id"),
            "timestamp_ms": int(time.time() * 1000),
            "catalog_epoch": catalog_epoch,
            "seq": seq,
            "payload": payload,
        }

    def _nack(
        self,
        req: Dict[str, Any],
        call_id: Optional[str],
        error_class: str,
        error_code: str,
        message: str,
        retryable: bool,
        retry_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._res(
            req,
            "NACK",
            {
                "nack_of_frame_id": req.get("frame_id"),
                "nack_of_call_id": call_id,
                "error_class": error_class,
                "error_code": error_code,
                "message": message,
                "retryable": retryable,
                "retry_hint": retry_hint or {},
            },
            seq=req.get("seq"),
            catalog_epoch=req.get("catalog_epoch"),
            session_id=req.get("session_id"),
        )

    def _ack(
        self,
        req: Dict[str, Any],
        call_id: Optional[str],
        *,
        status: str = "ACCEPTED",
        expected_seq_next: Optional[int] = None,
    ) -> Dict[str, Any]:
        if expected_seq_next is None and req.get("seq") is not None:
            try:
                expected_seq_next = int(req["seq"]) + 1
            except Exception:
                expected_seq_next = None
        return self._res(
            req,
            "ACK",
            {
                "ack_of_frame_id": req.get("frame_id"),
                "ack_of_call_id": call_id,
                "status": status,
                "expected_seq_next": expected_seq_next,
            },
            seq=req.get("seq"),
            catalog_epoch=req.get("catalog_epoch"),
            session_id=req.get("session_id"),
        )

    @staticmethod
    def _safe_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
        x = dict(frame)
        if "payload" in x and isinstance(x["payload"], dict):
            p = dict(x["payload"])
            if "args" in p:
                p["args"] = "<redacted>"
            x["payload"] = p
        return x

    def _clone_frame_for_async(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        cloned = dict(frame)
        if isinstance(frame.get("payload"), dict):
            cloned["payload"] = dict(frame["payload"])
        return cloned

    def _run_async_call(self, frame: Dict[str, Any]) -> None:
        session_id = frame.get("session_id")
        payload = frame.get("payload", {}) if isinstance(frame.get("payload"), dict) else {}
        call_id = payload.get("call_id")
        if not isinstance(session_id, str) or not isinstance(call_id, str):
            return

        self._set_async_call_state(
            session_id,
            call_id,
            {
                "status": "RUNNING",
                "result_frame_type": None,
                "result_payload": None,
                "updated_at": int(time.time() * 1000),
            },
        )
        self._append_async_event(
            session_id,
            call_id,
            {
                "kind": "STATE",
                "stage": "RUNNING",
                "message": "async execution started",
            },
        )

        try:
            res = self._execute_call_core(frame, check_seq=False)
        except Exception as e:
            self.audit.log_event("call.async_failed", {
                "session_id": session_id,
                "call_id": call_id,
                "error": repr(e),
            })
            res = self._nack(frame, call_id, "INTERNAL_ERROR", "TRP_5001", "internal error", False)

        if res["frame_type"] == "RESULT":
            self._publish_async_success_partials(session_id, call_id, res["payload"])
            self._set_async_call_state(
                session_id,
                call_id,
                {
                    "status": "SUCCESS",
                    "result_frame_type": "RESULT",
                    "result_payload": res["payload"],
                    "updated_at": int(time.time() * 1000),
                },
            )
            self._append_async_event(
                session_id,
                call_id,
                {
                    "kind": "STATE",
                    "stage": "COMPLETED",
                    "message": "async execution completed",
                },
            )
            return

        if res["frame_type"] == "NACK":
            self._append_async_event(
                session_id,
                call_id,
                {
                    "kind": "ERROR",
                    "error": {
                        "error_class": res["payload"].get("error_class"),
                        "error_code": res["payload"].get("error_code"),
                        "message": res["payload"].get("message"),
                        "retryable": res["payload"].get("retryable"),
                    },
                },
            )
            self._set_async_call_state(
                session_id,
                call_id,
                {
                    "status": "FAILED",
                    "result_frame_type": "NACK",
                    "result_payload": res["payload"],
                    "updated_at": int(time.time() * 1000),
                },
            )
            return

        self._set_async_call_state(
            session_id,
            call_id,
            {
                "status": "FAILED",
                "result_frame_type": str(res.get("frame_type")),
                "result_payload": {
                    "error_class": "INTERNAL_ERROR",
                    "error_code": "TRP_5003",
                    "message": f"unexpected async result frame_type: {res.get('frame_type')}",
                    "retryable": False,
                },
                "updated_at": int(time.time() * 1000),
            },
        )

    def _set_async_call_state(self, session_id: str, call_id: str, state: Dict[str, Any]) -> None:
        if self.runtime_state is not None:
            patch = dict(state)
            updated_at = int(patch.get("updated_at", int(time.time() * 1000)))
            patch["updated_at"] = updated_at
            patch["expires_at"] = updated_at + self._async_result_ttl_ms
            self.runtime_state.merge_async_call_state(
                session_id,
                call_id,
                patch,
                ttl_sec=max(1, self._async_result_ttl_ms // 1000),
            )
            return
        with self._async_calls_lock:
            s = self._async_calls.setdefault(session_id, {})
            existing = s.get(call_id, {})
            merged = dict(existing)
            merged.update(state)
            if "events" in existing and "events" not in state:
                merged["events"] = existing["events"]
            if "next_event_id" in existing and "next_event_id" not in state:
                merged["next_event_id"] = existing["next_event_id"]
            updated_at = int(merged.get("updated_at", int(time.time() * 1000)))
            merged["expires_at"] = updated_at + self._async_result_ttl_ms
            s[call_id] = merged
            self._maybe_prune_async_states_locked(updated_at)

    def _get_async_call_state(self, session_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        if self.runtime_state is not None:
            rec = self.runtime_state.get_async_call_state(session_id, call_id)
            if rec is None:
                return None
            now_ms = int(time.time() * 1000)
            try:
                expires_at = int(rec.get("expires_at", now_ms + self._async_result_ttl_ms))
            except Exception:
                expires_at = now_ms + self._async_result_ttl_ms
            if expires_at <= now_ms:
                return None
            return dict(rec)
        with self._async_calls_lock:
            now_ms = int(time.time() * 1000)
            self._maybe_prune_async_states_locked(now_ms)
            s = self._async_calls.get(session_id)
            if s is None:
                return None
            rec = s.get(call_id)
            if rec is None:
                return None
            expires_at = rec.get("expires_at")
            if isinstance(expires_at, int) and expires_at <= now_ms:
                s.pop(call_id, None)
                if not s:
                    self._async_calls.pop(session_id, None)
                return None
            return dict(rec)

    def _append_async_event(self, session_id: str, call_id: str, event: Dict[str, Any]) -> None:
        now_ms = int(time.time() * 1000)
        if self.runtime_state is not None:
            event_rec = dict(event)
            event_rec.setdefault("timestamp_ms", now_ms)
            self.runtime_state.append_async_event(
                session_id,
                call_id,
                event_rec,
                event_limit=self._async_event_limit,
                ttl_sec=max(1, self._async_result_ttl_ms // 1000),
            )
            return
        with self._async_calls_lock:
            s = self._async_calls.setdefault(session_id, {})
            rec = s.setdefault(call_id, {})
            events = rec.setdefault("events", [])
            next_event_id = int(rec.get("next_event_id", 1))
            event_rec = dict(event)
            event_rec["event_id"] = next_event_id
            event_rec.setdefault("timestamp_ms", now_ms)
            events.append(event_rec)
            if len(events) > self._async_event_limit:
                overflow = len(events) - self._async_event_limit
                del events[:overflow]
                rec["dropped_event_count"] = int(rec.get("dropped_event_count", 0)) + overflow
            rec["next_event_id"] = next_event_id + 1
            rec["updated_at"] = now_ms
            rec["expires_at"] = now_ms + self._async_result_ttl_ms
            self._maybe_prune_async_states_locked(now_ms)

    def _get_async_events_after(self, session_id: str, call_id: str, after_event_id: int) -> Dict[str, Any]:
        if self.runtime_state is not None:
            rec = self.runtime_state.get_async_call_state(session_id, call_id)
            if rec is None:
                return {
                    "events": [],
                    "events_truncated": False,
                    "first_retained_event_id": 0,
                }
            events = rec.get("events", [])
            first_retained_event_id = 0
            if isinstance(events, list) and events:
                try:
                    first_retained_event_id = int(events[0].get("event_id", 0))
                except Exception:
                    first_retained_event_id = 0
            else:
                events = []
            events_truncated = bool(events) and after_event_id < (first_retained_event_id - 1)
            out: List[Dict[str, Any]] = []
            for e in events:
                try:
                    eid = int(e.get("event_id", 0))
                except Exception:
                    eid = 0
                if eid > after_event_id and isinstance(e, dict):
                    out.append(dict(e))
            return {
                "events": out,
                "events_truncated": events_truncated,
                "first_retained_event_id": first_retained_event_id,
            }
        with self._async_calls_lock:
            now_ms = int(time.time() * 1000)
            self._maybe_prune_async_states_locked(now_ms)
            s = self._async_calls.get(session_id)
            if s is None:
                return {
                    "events": [],
                    "events_truncated": False,
                    "first_retained_event_id": 0,
                }
            rec = s.get(call_id)
            if rec is None:
                return {
                    "events": [],
                    "events_truncated": False,
                    "first_retained_event_id": 0,
                }
            events = rec.get("events", [])
            first_retained_event_id = 0
            if events:
                try:
                    first_retained_event_id = int(events[0].get("event_id", 0))
                except Exception:
                    first_retained_event_id = 0
            events_truncated = bool(events) and after_event_id < (first_retained_event_id - 1)
            out: List[Dict[str, Any]] = []
            for e in events:
                try:
                    eid = int(e.get("event_id", 0))
                except Exception:
                    eid = 0
                if eid > after_event_id:
                    out.append(dict(e))
            return {
                "events": out,
                "events_truncated": events_truncated,
                "first_retained_event_id": first_retained_event_id,
            }

    def _maybe_prune_async_states_locked(self, now_ms: int) -> None:
        if self.runtime_state is not None:
            return
        if now_ms - int(self._async_last_cleanup_ms) < self._async_cleanup_interval_ms:
            return
        self._async_last_cleanup_ms = now_ms
        expired_sessions: List[str] = []
        for session_id, calls in list(self._async_calls.items()):
            expired_call_ids: List[str] = []
            for call_id, rec in list(calls.items()):
                expires_at = rec.get("expires_at")
                try:
                    exp_ms = int(expires_at)
                except Exception:
                    exp_ms = now_ms + self._async_result_ttl_ms
                    rec["expires_at"] = exp_ms
                if exp_ms <= now_ms:
                    expired_call_ids.append(call_id)
            for call_id in expired_call_ids:
                calls.pop(call_id, None)
            if not calls:
                expired_sessions.append(session_id)
        for session_id in expired_sessions:
            self._async_calls.pop(session_id, None)

    def _publish_async_success_partials(self, session_id: str, call_id: str, result_payload: Dict[str, Any]) -> None:
        result_obj = result_payload.get("result", {})
        if not isinstance(result_obj, dict):
            return

        summary = result_obj.get("summary")
        if isinstance(summary, str) and summary:
            self._append_async_event(
                session_id,
                call_id,
                {
                    "kind": "SUMMARY",
                    "summary": summary,
                },
            )

        data = result_obj.get("data", {})
        if not isinstance(data, dict):
            return

        # 常见大字段分段：搜索 items / SQL rows
        for path_key in ("items", "rows"):
            items = data.get(path_key)
            if not isinstance(items, list):
                continue
            chunk_size = 2
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                self._append_async_event(
                    session_id,
                    call_id,
                    {
                        "kind": "DATA_CHUNK",
                        "path": f"result.data.{path_key}",
                        "chunk_index": i // chunk_size,
                        "chunk_count": (len(items) + chunk_size - 1) // chunk_size,
                        "items": chunk,
                    },
                )
            return

        # 非列表结果也给一个轻量 data 摘要事件，避免只有最终 RESULT_QUERY 可见
        preview_keys = [k for k in list(data.keys())[:5]]
        if preview_keys:
            self._append_async_event(
                session_id,
                call_id,
                {
                    "kind": "DATA_PREVIEW",
                    "keys": preview_keys,
                },
            )

    def _check_dependencies(self, *, session_id: str, depends_on: List[str]) -> Optional[Dict[str, Any]]:
        for dep_call_id in depends_on:
            rec = self._get_call_record(session_id, dep_call_id)
            if rec is None:
                return {
                    "error_code": "TRP_1005",
                    "message": f"dependency not completed: {dep_call_id}",
                    "retryable": True,
                    "retry_hint": {"backoff_ms": 50},
                }
            if rec.get("status") != "SUCCESS":
                retryable = bool(rec.get("retryable", False))
                return {
                    "error_code": "TRP_1006",
                    "message": f"dependency failed: {dep_call_id}",
                    "retryable": retryable,
                    "retry_hint": {"backoff_ms": 50} if retryable else {},
                }
        return None

    def _deps_ready(
        self,
        *,
        session_id: str,
        depends_on: List[str],
        batch_results: Dict[str, Dict[str, Any]],
    ) -> bool:
        for dep_call_id in depends_on:
            if dep_call_id in batch_results:
                if batch_results[dep_call_id]["status"] != "SUCCESS":
                    return False
                continue
            rec = self._get_call_record(session_id, dep_call_id)
            if rec is None or rec.get("status") != "SUCCESS":
                return False
        return True

    def _find_failed_dependency(
        self,
        *,
        session_id: str,
        depends_on: List[str],
        batch_results: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        for dep_call_id in depends_on:
            if dep_call_id in batch_results and batch_results[dep_call_id]["status"] == "FAILED":
                return dep_call_id
            rec = self._get_call_record(session_id, dep_call_id)
            if rec is not None and rec.get("status") != "SUCCESS":
                return dep_call_id
        return None

    def _record_call_result(self, session_id: str, call_id: str, *, success: bool, retryable: bool) -> None:
        if self.runtime_state is not None:
            self.runtime_state.put_call_record(
                session_id,
                call_id,
                {
                    "status": "SUCCESS" if success else "FAILED",
                    "retryable": retryable,
                    "updated_at": int(time.time() * 1000),
                },
                ttl_sec=self._call_record_ttl_sec,
            )
            return
        with self._call_records_lock:
            s = self._call_records.setdefault(session_id, {})
            s[call_id] = {
                "status": "SUCCESS" if success else "FAILED",
                "retryable": retryable,
                "updated_at": int(time.time() * 1000),
            }

    def _get_call_record(self, session_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        if self.runtime_state is not None:
            rec = self.runtime_state.get_call_record(session_id, call_id)
            return dict(rec) if rec is not None else None
        with self._call_records_lock:
            s = self._call_records.get(session_id)
            if s is None:
                return None
            rec = s.get(call_id)
            return dict(rec) if rec is not None else None

    @staticmethod
    def _build_canonical_schema(arg_template: Dict[str, str]) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for name, spec in arg_template.items():
            optional = spec.endswith("?")
            base = spec[:-1] if optional else spec
            schema: Dict[str, Any] = {}

            if base == "string":
                schema = {"type": "string"}
            elif base == "email":
                schema = {"type": "string", "format": "email"}
            elif base == "int":
                schema = {"type": "integer"}
            elif base == "bool":
                schema = {"type": "boolean"}
            else:
                # MVP：未知模板类型保守降级为字符串
                schema = {"type": "string", "x-trp-template": base}

            properties[name] = schema
            if not optional:
                required.append(name)

        return {
            "type": "object",
            "required": required,
            "properties": properties,
        }

    @staticmethod
    def _build_example_args(arg_template: Dict[str, str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, spec in arg_template.items():
            base = spec[:-1] if spec.endswith("?") else spec
            if base == "string":
                out[name] = f"example_{name}"
            elif base == "email":
                out[name] = "user@example.com"
            elif base == "int":
                out[name] = 1
            elif base == "bool":
                out[name] = True
            else:
                out[name] = f"example_{name}"
        return out
