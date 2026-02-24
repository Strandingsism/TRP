from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Literal


TRP_VERSION = "0.1"


class FrameType(str, Enum):
    HELLO_REQ = "HELLO_REQ"
    HELLO_RES = "HELLO_RES"
    CATALOG_SYNC_REQ = "CATALOG_SYNC_REQ"
    CATALOG_SYNC_RES = "CATALOG_SYNC_RES"
    CAP_QUERY_REQ = "CAP_QUERY_REQ"
    CAP_QUERY_RES = "CAP_QUERY_RES"
    CALL_REQ = "CALL_REQ"
    CALL_BATCH_REQ = "CALL_BATCH_REQ"
    RESULT_QUERY_REQ = "RESULT_QUERY_REQ"
    RESULT_QUERY_RES = "RESULT_QUERY_RES"
    PARTIAL_RESULT = "PARTIAL_RESULT"
    ACK = "ACK"
    NACK = "NACK"
    RESULT = "RESULT"
    CALL_BATCH_RES = "CALL_BATCH_RES"


class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IOClass(str, Enum):
    READ = "READ"
    WRITE = "WRITE"


class ErrorClass(str, Enum):
    TRANSIENT = "TRANSIENT"
    ORDER_VIOLATION = "ORDER_VIOLATION"
    CATALOG_MISMATCH = "CATALOG_MISMATCH"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    POLICY_DENIED = "POLICY_DENIED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    NON_IDEMPOTENT_BLOCKED = "NON_IDEMPOTENT_BLOCKED"
    EXECUTOR_ERROR = "EXECUTOR_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class RetryHint:
    expected_seq: Optional[int] = None
    backoff_ms: Optional[int] = None
    max_attempts: Optional[int] = None
    action: Optional[str] = None  # e.g. "SYNC_CATALOG"


@dataclass
class NackPayload:
    nack_of_frame_id: str
    nack_of_call_id: Optional[str]
    error_class: ErrorClass
    error_code: str
    message: str
    retryable: bool
    retry_hint: RetryHint = field(default_factory=RetryHint)


@dataclass
class AckPayload:
    ack_of_frame_id: str
    ack_of_call_id: Optional[str]
    status: Literal["ACCEPTED", "IN_PROGRESS", "DUPLICATE"] = "ACCEPTED"
    expected_seq_next: Optional[int] = None


@dataclass
class CapabilityBrief:
    idx: int
    cap_id: str
    name: str
    desc: str
    risk_tier: RiskTier
    io_class: IOClass
    arg_template: Dict[str, str]
    schema_digest: str


@dataclass
class FrameEnvelope:
    trp_version: str
    frame_type: FrameType
    session_id: Optional[str]
    frame_id: str
    trace_id: str
    timestamp_ms: int
    catalog_epoch: Optional[int]
    seq: Optional[int]
    payload: Dict[str, Any]


@dataclass
class CallSpec:
    call_id: str
    idempotency_key: Optional[str]
    idx: int
    cap_id: str
    attempt: int
    timeout_ms: int
    args: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    approval_token: Optional[str] = None
