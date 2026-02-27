from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Dict, Optional, Tuple


TOKEN_PREFIX = "appr1"


class ApprovalTokenError(Exception):
    pass


def canonical_args_digest(args: Dict[str, Any]) -> str:
    """
    Build a stable digest for canonical args, used to bind approval tokens.
    """
    if not isinstance(args, dict):
        raise ApprovalTokenError("args must be a dict")
    body = json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def mint_approval_token(
    *,
    secret: str,
    cap_id: str,
    args: Dict[str, Any],
    ttl_sec: int = 300,
    subject: Optional[str] = None,
    issued_at: Optional[int] = None,
    token_id: Optional[str] = None,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    if not isinstance(secret, str) or not secret:
        raise ApprovalTokenError("secret is required")
    if not isinstance(cap_id, str) or not cap_id:
        raise ApprovalTokenError("cap_id is required")
    now = int(issued_at if issued_at is not None else time.time())
    ttl_sec = max(1, int(ttl_sec))
    payload: Dict[str, Any] = {
        "v": 1,
        "cap_id": cap_id,
        "args_digest": canonical_args_digest(args),
        "iat": now,
        "exp": now + ttl_sec,
        "jti": token_id or f"appr_{uuid.uuid4().hex[:12]}",
    }
    if subject:
        payload["sub"] = subject
    if extra_claims:
        payload.update(extra_claims)

    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload_json)
    sig_b64 = _sign(secret, payload_b64)
    return f"{TOKEN_PREFIX}.{payload_b64}.{sig_b64}"


def validate_approval_token(
    *,
    token: str,
    secret: str,
    cap_id: str,
    args: Dict[str, Any],
    now_ts: Optional[int] = None,
    clock_skew_sec: int = 30,
) -> Tuple[bool, str]:
    if not isinstance(token, str) or not token:
        return False, "approval_token is required"
    if not isinstance(secret, str) or not secret:
        return False, "approval HMAC secret is not configured"

    try:
        prefix, payload_b64, sig_b64 = token.split(".", 2)
    except ValueError:
        return False, "invalid approval_token format"
    if prefix != TOKEN_PREFIX:
        return False, "invalid approval_token format"

    expected_sig = _sign(secret, payload_b64)
    if not hmac.compare_digest(sig_b64, expected_sig):
        return False, "invalid approval_token signature"

    try:
        payload_raw = _b64url_decode(payload_b64)
        payload = json.loads(payload_raw.decode("utf-8"))
    except Exception:
        return False, "invalid approval_token payload"

    if not isinstance(payload, dict):
        return False, "invalid approval_token payload"

    if int(payload.get("v", 0)) != 1:
        return False, "unsupported approval_token version"

    token_cap_id = payload.get("cap_id")
    if token_cap_id != cap_id:
        return False, "approval_token cap_id mismatch"

    try:
        exp = int(payload.get("exp"))
        iat = int(payload.get("iat", 0))
    except Exception:
        return False, "approval_token missing exp/iat"

    now = int(now_ts if now_ts is not None else time.time())
    skew = max(0, int(clock_skew_sec))
    if now > exp + skew:
        return False, "approval_token expired"
    if iat > now + skew:
        return False, "approval_token not yet valid"

    args_digest = payload.get("args_digest")
    if args_digest != canonical_args_digest(args):
        return False, "approval_token args mismatch"

    return True, "ok"


def _sign(secret: str, payload_b64: str) -> str:
    mac = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return _b64url_encode(mac)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))
