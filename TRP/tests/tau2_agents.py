from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import Any, Optional

from pydantic import BaseModel, Field

from tau2.agent.base import is_valid_agent_history_message
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate


def _safe_json_loads(text: Optional[str]) -> Any:
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _router_function_schema_stub(
    op: str,
    name: Optional[str] = None,
    arguments: Optional[dict] = None,
    calls: Optional[list[dict]] = None,
) -> str:
    """TRP-style router for agent tool use.

    Args:
        op: Router operation. Use "call" for one tool call or "batch" for multiple independent tool calls.
        name: Tool name for op="call". Must exactly match a catalog tool name.
        arguments: JSON object arguments for op="call".
        calls: For op="batch", list of tool call specs in order. Each item must be an object like
            {"name": "<tool_name>", "arguments": {...}}.

    Returns:
        Router returns a JSON object with summary and results. This function is schema-only for the LLM.
    """
    raise RuntimeError("router schema stub should never be executed directly")


class PendingEnvCall(BaseModel):
    env_tool_call_id: str
    name: str
    arguments: dict


class PendingRouterCall(BaseModel):
    router_tool_call_id: str
    op: str
    env_calls: list[PendingEnvCall]
    llm_visible_tool_calls: int = 1


class TRPRouterAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    pending_router_call: Optional[PendingRouterCall] = None
    router_error_retries_left: int = 2


class LLMParallelHintAgent(LLMAgent):
    """A direct-tools baseline agent with stronger guidance for multi-tool turns."""

    @property
    def system_prompt(self) -> str:
        base = super().system_prompt
        extra = """
<tool_calling_guidance>
- You may call multiple tools in the same assistant turn when they are independent.
- Prefer grouping independent lookups/checks into one assistant message with multiple tool calls.
- If later steps depend on earlier tool outputs, call tools sequentially across turns.
</tool_calling_guidance>
""".strip()
        return f"{base}\n{extra}"


class TRPRouterAgent(LLMAgent):
    """Expose a single router tool to the LLM and translate router calls to environment tool calls."""

    ROUTER_TOOL_NAME = "router"
    ROUTER_VERSION = "tau2-trp-v1"

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.env_tools = list(tools)
        self.env_tools_by_name = {tool.name: tool for tool in self.env_tools}
        self.router_tool = as_tool(_router_function_schema_stub)
        self.tools = [self.router_tool]
        self._catalog_text = self._build_catalog_text(self.env_tools)

    @classmethod
    def _build_catalog_text(cls, tools: list[Tool]) -> str:
        lines: list[str] = []
        for tool in tools:
            schema = tool.openai_schema.get("function", {})
            params = schema.get("parameters", {}) or {}
            props = params.get("properties", {}) or {}
            required = set(params.get("required", []) or [])
            param_parts = []
            for name, prop in props.items():
                t = prop.get("type", "any")
                suffix = " (required)" if name in required else ""
                param_parts.append(f"{name}:{t}{suffix}")
            params_str = ", ".join(param_parts) if param_parts else "(no params)"
            desc = (schema.get("description") or "").strip().splitlines()[0] if schema.get("description") else ""
            lines.append(f"- {tool.name}({params_str}) :: {desc}")
        return "\n".join(lines)

    @property
    def system_prompt(self) -> str:
        # Keep Tau2 base instruction style, but replace direct tool calling with router usage.
        return (
            "<instructions>\n"
            "You are a customer service agent that helps the user according to the <policy>.\n"
            "In each turn you can either:\n"
            "- Send a message to the user.\n"
            "- Make exactly one call to the `router` tool.\n"
            "You cannot do both at the same time.\n\n"
            "Router semantics:\n"
            "- op='call': one tool call.\n"
            "- op='batch': multiple independent tool calls in one router call.\n"
            "- Prefer op='batch' when multiple calls are independent and can be grouped.\n"
            "- Use exact tool names from the catalog.\n"
            "- Wait for router results before making dependent calls.\n"
            "Always make sure you generate valid JSON only.\n"
            "</instructions>\n"
            f"<policy>\n{self.domain_policy}\n</policy>\n"
            f"<router_version>{self.ROUTER_VERSION}</router_version>\n"
            "<catalog>\n"
            f"{self._catalog_text}\n"
            "</catalog>"
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> TRPRouterAgentState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return TRPRouterAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),  # will be router-facing history
            pending_router_call=None,
            router_error_retries_left=2,
        )

    def generate_next_message(
        self, message: Message, state: TRPRouterAgentState
    ) -> tuple[AssistantMessage, TRPRouterAgentState]:
        self._append_router_facing_input(message, state)

        while True:
            messages = state.system_messages + state.messages
            router_facing_assistant = generate(
                model=self.llm,
                tools=self.tools,
                messages=messages,
                **self.llm_args,
            )
            state.messages.append(router_facing_assistant)

            if not router_facing_assistant.is_tool_call():
                state.router_error_retries_left = 2
                return router_facing_assistant, state

            ok, env_assistant_or_none = self._translate_router_message(
                router_facing_assistant, state
            )
            if ok and env_assistant_or_none is not None:
                state.router_error_retries_left = 2
                return env_assistant_or_none, state

            # Invalid router call payload; append synthetic tool error(s) and let the LLM repair.
            if state.router_error_retries_left <= 0:
                fallback = AssistantMessage(
                    role="assistant",
                    content=(
                        "I encountered an internal router-format issue and cannot continue safely. "
                        "Please let me try again."
                    ),
                    tool_calls=None,
                    cost=router_facing_assistant.cost,
                    usage=router_facing_assistant.usage,
                    raw_data=router_facing_assistant.raw_data,
                )
                return fallback, state
            state.router_error_retries_left -= 1

    def _append_router_facing_input(
        self, message: Message, state: TRPRouterAgentState
    ) -> None:
        if isinstance(message, MultiToolMessage):
            if state.pending_router_call is not None:
                state.messages.append(self._build_router_tool_message(message.tool_messages, state))
                state.pending_router_call = None
            else:
                state.messages.extend(message.tool_messages)
            return
        if isinstance(message, ToolMessage) and state.pending_router_call is not None:
            state.messages.append(self._build_router_tool_message([message], state))
            state.pending_router_call = None
            return
        if isinstance(message, (AssistantMessage, ToolMessage, SystemMessage)) or (
            hasattr(message, "role")
        ):
            # UserMessage and normal ToolMessage path
            state.messages.append(message)  # type: ignore[arg-type]
            return
        state.messages.append(message)  # type: ignore[arg-type]

    def _translate_router_message(
        self, router_msg: AssistantMessage, state: TRPRouterAgentState
    ) -> tuple[bool, Optional[AssistantMessage]]:
        if router_msg.tool_calls is None or len(router_msg.tool_calls) == 0:
            self._append_router_errors(
                router_msg,
                ["Router tool call message was empty."],
                state,
            )
            return False, None
        if len(router_msg.tool_calls) != 1:
            self._append_router_errors(
                router_msg,
                [f"Exactly one router tool call is allowed per turn. Got {len(router_msg.tool_calls)}."],
                state,
            )
            return False, None

        router_tc = router_msg.tool_calls[0]
        if router_tc.name != self.ROUTER_TOOL_NAME:
            self._append_router_errors(
                router_msg,
                [f"Unknown tool '{router_tc.name}'. Only '{self.ROUTER_TOOL_NAME}' is available."],
                state,
            )
            return False, None

        parse_res = self._parse_router_request(router_tc)
        if parse_res["errors"]:
            self._append_router_errors(router_msg, parse_res["errors"], state)
            return False, None

        env_calls: list[PendingEnvCall] = []
        tool_calls: list[ToolCall] = []
        for idx, call in enumerate(parse_res["calls"]):
            env_tc_id = f"trp_env_{uuid.uuid4().hex[:12]}_{idx}"
            name = call["name"]
            arguments = call["arguments"]
            env_calls.append(
                PendingEnvCall(
                    env_tool_call_id=env_tc_id,
                    name=name,
                    arguments=arguments,
                )
            )
            tool_calls.append(
                ToolCall(
                    id=env_tc_id,
                    name=name,
                    arguments=arguments,
                    requestor="assistant",
                )
            )

        if not router_tc.id:
            router_tc.id = f"trp_router_{uuid.uuid4().hex[:12]}"

        state.pending_router_call = PendingRouterCall(
            router_tool_call_id=router_tc.id,
            op=parse_res["op"],
            env_calls=env_calls,
            llm_visible_tool_calls=1,
        )

        raw = deepcopy(router_msg.raw_data) if isinstance(router_msg.raw_data, dict) else {}
        raw["trp_router_meta"] = {
            "router_version": self.ROUTER_VERSION,
            "op": parse_res["op"],
            "llm_visible_tool_calls": 1,
            "env_tool_call_count": len(tool_calls),
            "env_tool_names": [tc.name for tc in tool_calls],
        }
        env_msg = AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=tool_calls,
            timestamp=router_msg.timestamp,
            cost=router_msg.cost,
            usage=router_msg.usage,
            raw_data=raw or None,
        )
        return True, env_msg

    def _parse_router_request(self, router_tc: ToolCall) -> dict[str, Any]:
        args = router_tc.arguments or {}
        op = args.get("op") or args.get("action")
        if isinstance(op, str):
            op = op.strip().lower()
        errors: list[str] = []
        calls: list[dict[str, Any]] = []

        if op not in {"call", "batch"}:
            errors.append("router.op must be 'call' or 'batch'.")
            return {"op": op, "calls": calls, "errors": errors}

        if op == "call":
            name = args.get("name") or args.get("tool_name")
            arguments = args.get("arguments")
            if arguments is None:
                arguments = args.get("args")
            if not isinstance(name, str) or not name.strip():
                errors.append("router call requires non-empty string field 'name'.")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                errors.append("router call field 'arguments' must be a JSON object.")
            if not errors:
                calls.append({"name": name.strip(), "arguments": arguments})
        else:
            batch_calls = args.get("calls") or args.get("requests") or []
            if not isinstance(batch_calls, list) or len(batch_calls) == 0:
                errors.append("router batch requires non-empty list field 'calls'.")
            else:
                for i, item in enumerate(batch_calls):
                    if not isinstance(item, dict):
                        errors.append(f"router.calls[{i}] must be an object.")
                        continue
                    name = item.get("name") or item.get("tool_name")
                    arguments = item.get("arguments")
                    if arguments is None:
                        arguments = item.get("args")
                    if arguments is None:
                        arguments = {}
                    if not isinstance(name, str) or not name.strip():
                        errors.append(f"router.calls[{i}].name must be a non-empty string.")
                        continue
                    if not isinstance(arguments, dict):
                        errors.append(f"router.calls[{i}].arguments must be an object.")
                        continue
                    calls.append({"name": name.strip(), "arguments": arguments})

        if not errors:
            unknown = [c["name"] for c in calls if c["name"] not in self.env_tools_by_name]
            if unknown:
                errors.append(
                    "Unknown tool name(s): " + ", ".join(sorted(set(unknown)))
                )

        return {"op": op, "calls": calls, "errors": errors}

    def _append_router_errors(
        self, router_msg: AssistantMessage, errors: list[str], state: TRPRouterAgentState
    ) -> None:
        payload = {
            "status": "ERROR",
            "error_class": "ROUTER_SCHEMA_ERROR",
            "errors": errors,
        }
        for tc in router_msg.tool_calls or []:
            # OpenAI-compatible providers require every assistant tool_call to be
            # followed by a tool message with the same tool_call_id, even when the
            # tool name/payload is invalid and we're returning a synthetic schema error.
            if not tc.id:
                tc.id = f"trp_router_err_{uuid.uuid4().hex[:10]}"
            state.messages.append(
                ToolMessage(
                    id=tc.id,
                    role="tool",
                    requestor="assistant",
                    content=json.dumps(payload, ensure_ascii=False),
                    error=True,
                )
            )

    def _build_router_tool_message(
        self, tool_messages: list[ToolMessage], state: TRPRouterAgentState
    ) -> ToolMessage:
        pending = state.pending_router_call
        if pending is None:
            # Should not happen; fallback to pass-through synthetic wrapper.
            payload = {
                "status": "ERROR",
                "error_class": "ROUTER_INTERNAL_ERROR",
                "errors": ["Received tool result but no pending router call exists."],
                "results": [],
            }
            return ToolMessage(
                id=f"trp_router_orphan_{uuid.uuid4().hex[:10]}",
                role="tool",
                requestor="assistant",
                content=json.dumps(payload, ensure_ascii=False),
                error=True,
            )

        by_id = {tm.id: tm for tm in tool_messages}
        results = []
        errors = []
        for idx, env_call in enumerate(pending.env_calls):
            tm = by_id.get(env_call.env_tool_call_id)
            if tm is None:
                errors.append(f"Missing tool response for id={env_call.env_tool_call_id}")
                continue
            parsed_content = _safe_json_loads(tm.content)
            results.append(
                {
                    "index": idx,
                    "name": env_call.name,
                    "arguments": env_call.arguments,
                    "ok": not tm.error,
                    "result": parsed_content,
                }
            )
            if tm.error:
                errors.append(f"{env_call.name} failed")

        status = "SUCCESS" if not errors else ("PARTIAL_SUCCESS" if results else "FAILED")
        payload = {
            "router_version": self.ROUTER_VERSION,
            "op": pending.op,
            "status": status,
            "summary": f"Executed {len(results)} call(s); errors={len(errors)}.",
            "results": results,
        }
        if errors:
            payload["errors"] = errors

        return ToolMessage(
            id=pending.router_tool_call_id,
            role="tool",
            requestor="assistant",
            content=json.dumps(payload, ensure_ascii=False),
            error=bool(errors),
        )


def register_tau2_custom_agents() -> None:
    from tau2.registry import registry

    existing = set(registry.get_agents())
    if "trp_router_agent" not in existing:
        registry.register_agent(TRPRouterAgent, "trp_router_agent")
    if "llm_agent_parallel_hint" not in existing:
        registry.register_agent(LLMParallelHintAgent, "llm_agent_parallel_hint")
