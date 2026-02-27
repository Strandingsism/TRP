from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Callable

from .router_interfaces import (
    CapabilityMeta,
    CapabilityRegistry,
    SessionManager,
    IdempotencyStore,
    AuditLogger,
)


# =========================
# Custom exceptions (for fine-grained RouterService mapping)
# =========================

class OrderViolationError(Exception):
    def __init__(self, expected_seq: int, got_seq: int):
        super().__init__(f"seq out of order: expected={expected_seq}, got={got_seq}")
        self.expected_seq = expected_seq
        self.got_seq = got_seq


class DuplicateFrameError(Exception):
    def __init__(self, frame_id: str):
        super().__init__(f"duplicate frame: {frame_id}")
        self.frame_id = frame_id


class CatalogMismatchError(Exception):
    pass


# =========================
# Default capability catalog (static)
# =========================

def _digest(arg_template: Dict[str, str]) -> str:
    payload = json.dumps(arg_template, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_default_capabilities() -> List[CapabilityMeta]:
    caps: List[CapabilityMeta] = [
        CapabilityMeta(
            cap_id="cap.search.web.v1",
            name="web_search",
            desc="Search public web content by query",
            risk_tier="LOW",
            io_class="READ",
            schema_digest=_digest({"query": "string", "top_k": "int?"}),
            arg_template={"query": "string", "top_k": "int?"},
            adapter_key="search",
        ),
        CapabilityMeta(
            cap_id="cap.fetch.doc.v1",
            name="doc_fetch",
            desc="Fetch a document by document_id",
            risk_tier="LOW",
            io_class="READ",
            schema_digest=_digest({"document_id": "string"}),
            arg_template={"document_id": "string"},
            adapter_key="fetch",
        ),
        CapabilityMeta(
            cap_id="cap.query.sql_read.v1",
            name="sql_read_query",
            desc="Run a read-only SQL query on an analytics dataset",
            risk_tier="MEDIUM",
            io_class="READ",
            schema_digest=_digest({"sql": "string", "limit": "int?"}),
            arg_template={"sql": "string", "limit": "int?"},
            adapter_key="sql_read",
        ),
        CapabilityMeta(
            cap_id="cap.ticket.create.v1",
            name="ticket_create",
            desc="Create a ticket in issue tracker",
            risk_tier="HIGH",
            io_class="WRITE",
            schema_digest=_digest({"title": "string", "body": "string", "priority": "string?"}),
            arg_template={"title": "string", "body": "string", "priority": "string?"},
            adapter_key="ticket_create",
        ),
        CapabilityMeta(
            cap_id="cap.record.delete.v1",
            name="record_delete",
            desc="Delete a record by type and id",
            risk_tier="CRITICAL",
            io_class="WRITE",
            schema_digest=_digest({"resource_type": "string", "resource_id": "string", "reason": "string?"}),
            arg_template={"resource_type": "string", "resource_id": "string", "reason": "string?"},
            adapter_key="record_delete",
        ),
    ]
    return caps


# =========================
# In-memory CapabilityRegistry
# =========================

class InMemoryCapabilityRegistry(CapabilityRegistry):
    """
    - Maintain catalog_epoch
    - Provide stable idx->cap_id mapping per session (globally consistent in current implementation)
    """
    def __init__(self, capabilities: Optional[List[CapabilityMeta]] = None):
        self._lock = threading.Lock()
        self._epoch = 1
        self._caps: List[CapabilityMeta] = capabilities or build_default_capabilities()
        self._idx_map: Dict[int, CapabilityMeta] = {i: c for i, c in enumerate(self._caps)}
        self._cap_map: Dict[str, CapabilityMeta] = {c.cap_id: c for c in self._caps}

    @property
    def catalog_epoch(self) -> int:
        return self._epoch

    def bump_epoch(self) -> None:
        with self._lock:
            self._epoch += 1

    def replace_capabilities(self, capabilities: List[CapabilityMeta], bump_epoch: bool = True) -> None:
        with self._lock:
            self._caps = capabilities
            self._idx_map = {i: c for i, c in enumerate(self._caps)}
            self._cap_map = {c.cap_id: c for c in self._caps}
            if bump_epoch:
                self._epoch += 1

    def get_catalog(self, session_id: str) -> Tuple[int, List[Dict[str, Any]]]:
        with self._lock:
            alias_table: List[Dict[str, Any]] = []
            for idx, cap in self._idx_map.items():
                alias_table.append({
                    "idx": idx,
                    "cap_id": cap.cap_id,
                    "name": cap.name,
                    "desc": cap.desc,
                    "risk_tier": cap.risk_tier,
                    "io_class": cap.io_class,
                    "arg_template": cap.arg_template,
                    "schema_digest": cap.schema_digest,
                })
            return self._epoch, alias_table

    def resolve(self, session_id: str, catalog_epoch: int, idx: int, cap_id: str) -> CapabilityMeta:
        with self._lock:
            if catalog_epoch != self._epoch:
                raise CatalogMismatchError(f"epoch mismatch: got={catalog_epoch}, current={self._epoch}")
            cap = self._idx_map.get(idx)
            if cap is None:
                raise CatalogMismatchError(f"idx not found: {idx}")
            if cap.cap_id != cap_id:
                raise CatalogMismatchError(f"idx/cap_id mismatch: idx={idx}, expected={cap.cap_id}, got={cap_id}")
            return cap


# =========================
# In-memory SessionManager
# =========================

class InMemorySessionManager(SessionManager):
    """
    Maintains:
    - expected_seq
    - seen_frame_ids (replay/duplicate protection)
    """
    def __init__(self, catalog_epoch_provider: Callable[[], int], default_retry_budget: int = 3):
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._catalog_epoch_provider = catalog_epoch_provider
        self._default_retry_budget = default_retry_budget

    def hello(self, agent_id: str, resume_session_id: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            if resume_session_id and resume_session_id in self._sessions:
                s = self._sessions[resume_session_id]
                return {
                    "session_id": resume_session_id,
                    "catalog_epoch": self._catalog_epoch_provider(),
                    "retry_budget": s["retry_budget"],
                    "seq_start": s["expected_seq"],
                    "features": ["CATALOG_SYNC", "CAP_QUERY", "CALL", "CALL_BATCH", "RESULT_QUERY", "PARTIAL_RESULT", "ASYNC_CALL", "APPROVAL"],
                }

            sid = f"sess_{uuid.uuid4().hex[:12]}"
            self._sessions[sid] = {
                "agent_id": agent_id,
                "expected_seq": 1,
                "retry_budget": self._default_retry_budget,
                "seen_frame_ids": set(),  # type: ignore[var-annotated]
                "created_at": time.time(),
            }
            return {
                "session_id": sid,
                "catalog_epoch": self._catalog_epoch_provider(),
                "retry_budget": self._default_retry_budget,
                "seq_start": 1,
                "features": ["CATALOG_SYNC", "CAP_QUERY", "CALL", "CALL_BATCH", "RESULT_QUERY", "PARTIAL_RESULT", "ASYNC_CALL", "APPROVAL"],
            }

    def check_and_advance_seq(self, session_id: str, seq: int, frame_id: str) -> Dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"unknown session_id: {session_id}")
            s = self._sessions[session_id]
            seen: set = s["seen_frame_ids"]

            # frame_id dedupe (replay protection)
            if frame_id in seen:
                raise DuplicateFrameError(frame_id)

            expected = int(s["expected_seq"])
            if seq != expected:
                raise OrderViolationError(expected_seq=expected, got_seq=seq)

            # Accept
            seen.add(frame_id)
            s["expected_seq"] = expected + 1
            return {"ok": True, "expected_seq_next": s["expected_seq"]}


# =========================
# In-memory IdempotencyStore
# =========================

class InMemoryIdempotencyStore(IdempotencyStore):
    def __init__(self):
        self._lock = threading.Lock()
        # key: (cap_id, idempotency_key) -> {"expires_at": ..., "payload": ...}
        self._store: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get(self, cap_id: str, idempotency_key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        k = (cap_id, idempotency_key)
        with self._lock:
            rec = self._store.get(k)
            if rec is None:
                return None
            if rec["expires_at"] < now:
                self._store.pop(k, None)
                return None
            return rec["payload"]

    def put(self, cap_id: str, idempotency_key: str, result_payload: Dict[str, Any], ttl_sec: int = 86400) -> None:
        k = (cap_id, idempotency_key)
        with self._lock:
            self._store[k] = {
                "expires_at": time.time() + ttl_sec,
                "payload": result_payload,
            }


# =========================
# Simple audit logger (stdout)
# =========================

class PrintAuditLogger(AuditLogger):
    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        # In production, this can be replaced by structlog / OpenTelemetry / Kafka.
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            body = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            body = str(payload)
        print(f"[{ts}] {event_name} {body}")
