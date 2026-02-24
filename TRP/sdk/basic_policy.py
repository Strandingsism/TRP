from __future__ import annotations

from typing import Any, Dict, Optional

from .router_interfaces import (
    CapabilityMeta,
    PolicyDecision,
    PolicyEngine,
)
from .approval_tokens import validate_approval_token


class BasicPolicyEngine(PolicyEngine):
    """
    基础策略（适合 MVP）：
    1) LOW / MEDIUM 默认放行
    2) HIGH / CRITICAL 默认要求 approval_token
    3) 可通过 auth_context 做简单权限控制
    """

    def __init__(
        self,
        allow_critical_with_token: bool = True,
        *,
        approval_hmac_secret: Optional[str] = None,
        approval_require_signed: bool = False,
        approval_allow_legacy_prefix: bool = True,
        approval_clock_skew_sec: int = 30,
    ):
        self.allow_critical_with_token = allow_critical_with_token
        self.approval_hmac_secret = approval_hmac_secret or None
        self.approval_require_signed = bool(approval_require_signed)
        self.approval_allow_legacy_prefix = bool(approval_allow_legacy_prefix)
        self.approval_clock_skew_sec = max(0, int(approval_clock_skew_sec))

    def evaluate(
        self,
        *,
        auth_context: Optional[Dict[str, Any]],
        cap: CapabilityMeta,
        args: Dict[str, Any],
        idempotency_key: Optional[str],
        approval_token: Optional[str],
    ) -> PolicyDecision:
        auth_context = auth_context or {}

        # ---- 可选：按角色限制能力 ----
        # auth_context 示例：
        # {"role": "analyst"} 或 {"role": "ops", "can_delete": True}
        role = auth_context.get("role")
        if role == "readonly" and cap.io_class == "WRITE":
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                reason="readonly role cannot invoke WRITE capabilities",
            )

        # ---- MEDIUM: 可加额外限制（示例）----
        if cap.risk_tier == "MEDIUM":
            # 例：限制 sql 查询规模（这里仅示意）
            if cap.cap_id == "cap.query.sql_read.v1":
                limit = args.get("limit", 50)
                if isinstance(limit, int) and limit > 200:
                    return PolicyDecision(
                        allowed=False,
                        requires_approval=False,
                        reason="limit > 200 is blocked by policy",
                    )
            return PolicyDecision(allowed=True)

        # ---- LOW: 放行 ----
        if cap.risk_tier == "LOW":
            return PolicyDecision(allowed=True)

        # ---- HIGH: 要审批 ----
        if cap.risk_tier == "HIGH":
            if not approval_token:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=True,
                    reason="HIGH risk capability requires approval_token",
                )
            ok, reason = self._validate_approval_token(
                approval_token=approval_token,
                cap=cap,
                args=args,
            )
            if not ok:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=True,
                    reason=reason,
                )
            return PolicyDecision(allowed=True)

        # ---- CRITICAL: 更严格 ----
        if cap.risk_tier == "CRITICAL":
            if not self.allow_critical_with_token:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason="CRITICAL capabilities are disabled",
                )
            if not approval_token:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=True,
                    reason="CRITICAL capability requires approval_token",
                )
            ok, reason = self._validate_approval_token(
                approval_token=approval_token,
                cap=cap,
                args=args,
            )
            if not ok:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=True,
                    reason=reason,
                )

            # 二次权限开关（比如只有 ops 才能删）
            if not auth_context.get("can_delete", False):
                return PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason="missing can_delete permission in auth_context",
                )
            return PolicyDecision(allowed=True)

        # 未知风险等级：保守拒绝
        return PolicyDecision(
            allowed=False,
            requires_approval=False,
            reason=f"unknown risk_tier: {cap.risk_tier}",
        )

    def _validate_approval_token(
        self,
        *,
        approval_token: str,
        cap: CapabilityMeta,
        args: Dict[str, Any],
    ) -> tuple[bool, str]:
        token = str(approval_token)

        if self.approval_hmac_secret:
            if token.startswith("appr1."):
                return validate_approval_token(
                    token=token,
                    secret=self.approval_hmac_secret,
                    cap_id=cap.cap_id,
                    args=args,
                    clock_skew_sec=self.approval_clock_skew_sec,
                )
            if self.approval_require_signed:
                return False, "signed approval_token required"
            if self.approval_allow_legacy_prefix and token.startswith("appr_"):
                return True, "ok"
            return False, "invalid approval_token format"

        # 未配置签名校验：兼容旧前缀规则
        if token.startswith("appr_"):
            return True, "ok"
        return False, "invalid approval_token format"
