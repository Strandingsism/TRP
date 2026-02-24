from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError


TRP_VERSION_LITERAL = "0.1"


class TRPFrameValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class HelloPayload(_StrictModel):
    agent_id: StrictStr = Field(min_length=1)
    supported_versions: List[StrictStr] = Field(min_length=1)
    resume_session_id: Optional[StrictStr] = None


class CatalogSyncPayload(_StrictModel):
    mode: Literal["FULL", "DELTA"] = "FULL"
    known_epoch: Optional[StrictInt] = None


class CapQueryPayload(_StrictModel):
    idx: StrictInt = Field(ge=0)
    cap_id: StrictStr = Field(min_length=1)
    include_examples: StrictBool = False


class CallPayload(_StrictModel):
    call_id: StrictStr = Field(min_length=1)
    idempotency_key: Optional[StrictStr] = None
    idx: StrictInt = Field(ge=0)
    cap_id: StrictStr = Field(min_length=1)
    depends_on: List[StrictStr] = Field(default_factory=list)
    attempt: StrictInt = Field(ge=1)
    timeout_ms: StrictInt = Field(ge=1)
    approval_token: Optional[StrictStr] = None
    execution_mode: Literal["SYNC", "ASYNC"] = "SYNC"
    args: Dict[str, Any]


class BatchCallItem(_StrictModel):
    call_id: StrictStr = Field(min_length=1)
    idempotency_key: Optional[StrictStr] = None
    idx: StrictInt = Field(ge=0)
    cap_id: StrictStr = Field(min_length=1)
    attempt: StrictInt = Field(ge=1)
    timeout_ms: StrictInt = Field(ge=1)
    args: Dict[str, Any]
    depends_on: List[StrictStr] = Field(default_factory=list)
    approval_token: Optional[StrictStr] = None


class CallBatchPayload(_StrictModel):
    batch_id: StrictStr = Field(min_length=1)
    mode: Literal["PARALLEL", "SEQUENTIAL"] = "PARALLEL"
    max_concurrency: StrictInt = Field(ge=1)
    calls: List[BatchCallItem]


class ResultQueryPayload(_StrictModel):
    call_id: StrictStr = Field(min_length=1)
    after_event_id: Optional[StrictInt] = Field(default=None, ge=0)
    include_partials: StrictBool = True


class Envelope(_StrictModel):
    trp_version: Literal["0.1"]
    frame_type: Literal[
        "HELLO_REQ",
        "CATALOG_SYNC_REQ",
        "CAP_QUERY_REQ",
        "CALL_REQ",
        "CALL_BATCH_REQ",
        "RESULT_QUERY_REQ",
    ]
    session_id: Optional[StrictStr]
    frame_id: StrictStr = Field(min_length=1)
    trace_id: StrictStr = Field(min_length=1)
    timestamp_ms: StrictInt = Field(ge=0)
    catalog_epoch: Optional[StrictInt]
    seq: Optional[StrictInt]
    payload: Dict[str, Any]
    auth_context: Optional[Dict[str, Any]] = None


_PAYLOAD_MODELS = {
    "HELLO_REQ": HelloPayload,
    "CATALOG_SYNC_REQ": CatalogSyncPayload,
    "CAP_QUERY_REQ": CapQueryPayload,
    "CALL_REQ": CallPayload,
    "CALL_BATCH_REQ": CallBatchPayload,
    "RESULT_QUERY_REQ": ResultQueryPayload,
}


def validate_trp_frame(frame: Any) -> Dict[str, Any]:
    if not isinstance(frame, dict):
        raise TRPFrameValidationError("frame must be a JSON object")

    try:
        env = Envelope.model_validate(frame)
    except ValidationError as e:
        raise TRPFrameValidationError(_fmt_validation_error(e)) from e

    _validate_envelope_invariants(env)
    payload_model = _PAYLOAD_MODELS[env.frame_type]

    try:
        payload = payload_model.model_validate(env.payload)
    except ValidationError as e:
        raise TRPFrameValidationError(_fmt_validation_error(e, prefix="payload")) from e

    out = env.model_dump(mode="python")
    out["payload"] = payload.model_dump(mode="python")
    return out


def _validate_envelope_invariants(env: Envelope) -> None:
    if env.frame_type == "HELLO_REQ":
        if env.session_id is not None:
            raise TRPFrameValidationError("HELLO_REQ.session_id must be null")
        if env.catalog_epoch is not None:
            raise TRPFrameValidationError("HELLO_REQ.catalog_epoch must be null")
        if env.seq is not None:
            raise TRPFrameValidationError("HELLO_REQ.seq must be null")
        return

    if not isinstance(env.session_id, str) or not env.session_id:
        raise TRPFrameValidationError(f"{env.frame_type}.session_id is required")
    if env.catalog_epoch is None:
        raise TRPFrameValidationError(f"{env.frame_type}.catalog_epoch is required")
    if env.seq is None:
        raise TRPFrameValidationError(f"{env.frame_type}.seq is required")
    if env.seq <= 0:
        raise TRPFrameValidationError(f"{env.frame_type}.seq must be > 0")


def _fmt_validation_error(e: ValidationError, *, prefix: Optional[str] = None) -> str:
    first = e.errors()[0] if e.errors() else {"loc": (), "msg": str(e)}
    loc_parts = [str(x) for x in first.get("loc", ())]
    loc = ".".join(loc_parts)
    if prefix:
        loc = f"{prefix}.{loc}" if loc else prefix
    if not loc:
        loc = "frame"
    msg = first.get("msg", "invalid frame")
    return f"{loc}: {msg}"
