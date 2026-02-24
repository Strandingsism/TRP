from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple


# ========= 能力注册与目录 =========

@dataclass
class CapabilityMeta:
    cap_id: str
    name: str
    desc: str
    risk_tier: str
    io_class: str
    schema_digest: str
    arg_template: Dict[str, str]
    adapter_key: str  # 指向哪个 adapter


class CapabilityRegistry(Protocol):
    def get_catalog(self, session_id: str) -> Tuple[int, List[Dict[str, Any]]]:
        """
        返回 (catalog_epoch, alias_table)
        alias_table 内每项至少包含 idx/cap_id/name/desc/risk/io/schema_digest
        """
        ...

    def resolve(self, session_id: str, catalog_epoch: int, idx: int, cap_id: str) -> CapabilityMeta:
        """
        校验 (catalog_epoch, idx, cap_id) 并返回 CapabilityMeta
        若失配应抛 CatalogMismatch 异常
        """
        ...


# ========= 会话与顺序 =========

class SessionManager(Protocol):
    def hello(self, agent_id: str, resume_session_id: Optional[str]) -> Dict[str, Any]:
        """
        创建/恢复会话，返回 session_id、catalog_epoch、seq_start、retry_budget ...
        """
        ...

    def check_and_advance_seq(self, session_id: str, seq: int, frame_id: str) -> Dict[str, Any]:
        """
        校验顺序号。
        返回 {"ok": True, "expected_seq_next": ...}
        或抛 OrderViolation / Duplicate
        """
        ...


# ========= 策略与审批 =========

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


# ========= 幂等 =========

class IdempotencyStore(Protocol):
    def get(self, cap_id: str, idempotency_key: str) -> Optional[Dict[str, Any]]:
        ...

    def put(self, cap_id: str, idempotency_key: str, result_payload: Dict[str, Any], ttl_sec: int = 86400) -> None:
        ...


# ========= Router 运行时状态（可选持久化） =========

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
        原子合并 async state patch，返回合并后的完整 state。
        多实例场景下应避免 get-modify-set 丢更新。
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
        原子追加 async event，并维护 event_id / trim / ttl，返回更新后的完整 state。
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
        原子尝试获取 async 执行权（claim/lease）。
        返回 {"claimed": bool, "reason": "...", "state": {...}|None}
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
        原子续租 async 执行权（仅当前 owner 可续租）。
        返回 {"renewed": bool, "reason": "...", "state": {...}|None}
        """
        ...


# ========= 参数语义映射 =========

class Adapter(Protocol):
    def validate_canonical_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> None:
        """
        schema + 语义 + 安全校验（必要时）
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


# ========= 执行与结果整形 =========

class Executor(Protocol):
    def execute(self, cap: CapabilityMeta, native_args: Dict[str, Any], timeout_ms: int) -> Dict[str, Any]:
        """
        调用真实工具（HTTP/DB/CLI/MCP）
        返回原始结果（dict）
        """
        ...


class ResultShaper(Protocol):
    def shape_success(self, cap: CapabilityMeta, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一 result 格式：summary/data/warnings/artifacts
        """
        ...

    def shape_error(self, cap: Optional[CapabilityMeta], exc: Exception) -> Dict[str, Any]:
        ...


# ========= 审计 =========

class AuditLogger(Protocol):
    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        ...
