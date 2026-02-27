from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple


# ========= Capability Registry and Catalog =========

@dataclass
class CapabilityMeta:
    cap_id: str
    name: str
    desc: str
    risk_tier: str
    io_class: str
    schema_digest: str
    arg_template: Dict[str, str]
    adapter_key: str  # Which adapter handles this capability.


class CapabilityRegistry(Protocol):
    def get_catalog(self, session_id: str) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Return (catalog_epoch, alias_table).
        Each alias_table item should at least include:
        idx/cap_id/name/desc/risk/io/schema_digest.
        """
        ...

    def resolve(self, session_id: str, catalog_epoch: int, idx: int, cap_id: str) -> CapabilityMeta:
        """
        Validate (catalog_epoch, idx, cap_id) and return CapabilityMeta.
        Should raise a CatalogMismatch-like exception on mismatch.
        """
        ...


# ========= Session and Ordering =========

class SessionManager(Protocol):
    def hello(self, agent_id: str, resume_session_id: Optional[str]) -> Dict[str, Any]:
        """
        Create/resume a session and return session_id, catalog_epoch,
        seq_start, retry_budget, etc.
        """
        ...

    def check_and_advance_seq(self, session_id: str, seq: int, frame_id: str) -> Dict[str, Any]:
        """
        Validate sequence order.
        Return {"ok": True, "expected_seq_next": ...}
        or raise OrderViolation / Duplicate.
        """
        ...


# ========= Policy and Approval =========

@dataclass
class PolicyDecision:
    allowed: bool
    requires_approval: bool = False
    reason: Optional[str] = None


class PolicyEngine(Protocol):
    def evaluate(
        self,
        *,
        auth_context: Optional[Dict[str, Any]],
        cap: CapabilityMeta,
        args: Dict[str, Any],
        idempotency_key: Optional[str],
        approval_token: Optional[str],
    ) -> PolicyDecision:
        ...


# ========= Idempotency =========

class IdempotencyStore(Protocol):
    def get(self, cap_id: str, idempotency_key: str) -> Optional[Dict[str, Any]]:
        ...

    def put(self, cap_id: str, idempotency_key: str, result_payload: Dict[str, Any], ttl_sec: int = 86400) -> None:
        ...


# ========= Router Runtime State (optional persistence) =========

class RuntimeStateStore(Protocol):
    def get_call_record(self, session_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        ...

    def put_call_record(self, session_id: str, call_id: str, record: Dict[str, Any], ttl_sec: int) -> None:
        ...

    def get_async_call_state(self, session_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        ...

    def put_async_call_state(self, session_id: str, call_id: str, state: Dict[str, Any], ttl_sec: int) -> None:
        ...

    def merge_async_call_state(self, session_id: str, call_id: str, patch: Dict[str, Any], ttl_sec: int) -> Dict[str, Any]:
        """
        Atomically merge async state patch and return the full merged state.
        In multi-instance deployments, avoid get-modify-set lost updates.
        """
        ...

    def append_async_event(
        self,
        session_id: str,
        call_id: str,
        event: Dict[str, Any],
        *,
        event_limit: int,
        ttl_sec: int,
    ) -> Dict[str, Any]:
        """
        Atomically append async event and maintain event_id / trim / ttl.
        Return the updated full state.
        """
        ...

    def claim_async_execution(
        self,
        session_id: str,
        call_id: str,
        *,
        worker_id: str,
        lease_ttl_sec: int,
        state_ttl_sec: int,
    ) -> Dict[str, Any]:
        """
        Atomically attempt to claim async execution ownership (claim/lease).
        Return {"claimed": bool, "reason": "...", "state": {...}|None}.
        """
        ...

    def renew_async_execution_lease(
        self,
        session_id: str,
        call_id: str,
        *,
        worker_id: str,
        lease_ttl_sec: int,
        state_ttl_sec: int,
    ) -> Dict[str, Any]:
        """
        Atomically renew async execution lease (only current owner can renew).
        Return {"renewed": bool, "reason": "...", "state": {...}|None}.
        """
        ...


# ========= Argument Semantics Mapping =========

class Adapter(Protocol):
    def validate_canonical_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> None:
        """
        Validate schema + semantics + safety checks (when needed).
        """
        ...

    def to_native_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        canonical args -> tool-native args
        """
        ...


class AdapterRegistry(Protocol):
    def get(self, adapter_key: str) -> Adapter:
        ...


# ========= Execution and Result Shaping =========

class Executor(Protocol):
    def execute(self, cap: CapabilityMeta, native_args: Dict[str, Any], timeout_ms: int) -> Dict[str, Any]:
        """
        Execute real tools (HTTP/DB/CLI/MCP).
        Return raw result (dict).
        """
        ...


class ResultShaper(Protocol):
    def shape_success(self, cap: CapabilityMeta, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize result format: summary/data/warnings/artifacts.
        """
        ...

    def shape_error(self, cap: Optional[CapabilityMeta], exc: Exception) -> Dict[str, Any]:
        ...


# ========= Audit =========

class AuditLogger(Protocol):
    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        ...
