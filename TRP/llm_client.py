#!/usr/bin/env python3
"""
LLM client adapter: use an OpenAI-compatible API as an Anthropic replacement.
Keeps compatibility with existing client.messages.create() and response.content / response.stop_reason usage.
"""
import json
import os
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# Prefer generic TRP vars, fallback to commonly used OpenAI-compatible names.
API_KEY = (
    os.getenv("TRP_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
if not API_KEY:
    raise ValueError("Set TRP_API_KEY (or OPENAI_API_KEY)")

BASE_URL = (
    os.getenv("TRP_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
)
if not BASE_URL:
    raise ValueError("Set TRP_BASE_URL (or OPENAI_BASE_URL)")

DEFAULT_MODEL = os.getenv("TRP_MODEL_ID") or os.getenv("MODEL_ID")
if not DEFAULT_MODEL:
    raise ValueError("Set TRP_MODEL_ID (or MODEL_ID)")

_openai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _anthropic_tools_to_openai(tools: list) -> list:
    """Convert Anthropic-style tools to OpenAI-style function tools."""
    out = []
    for t in tools or []:
        name = t.get("name", "")
        desc = t.get("description", "")
        schema = t.get("input_schema") or {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": schema,
            },
        })
    return out


def _messages_anthropic_to_openai(messages: list, system: str | None) -> list:
    """Convert Anthropic-style messages used in this project to OpenAI format."""
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            if isinstance(content, list) and all(
                isinstance(p, dict) and p.get("type") == "tool_result" for p in content
            ):
                # Multiple tool_result entries -> multiple role=tool messages (OpenAI requirement).
                for part in content:
                    out.append({
                        "role": "tool",
                        "tool_call_id": part.get("tool_use_id", ""),
                        "content": part.get("content", ""),
                    })
            else:
                if isinstance(content, list):
                    content = "\n".join(str(c) for c in content)
                out.append({"role": "user", "content": content or ""})
        elif role == "assistant":
            if isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if isinstance(block, (dict, SimpleNamespace)):
                        typ = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                        if typ == "text":
                            text_parts.append(getattr(block, "text", None) or (block.get("text", "") if isinstance(block, dict) else ""))
                        elif typ == "tool_use":
                            bid = getattr(block, "id", None) or (block.get("id", "") if isinstance(block, dict) else "")
                            bname = getattr(block, "name", None) or (block.get("name", "") if isinstance(block, dict) else "")
                            binp = getattr(block, "input", None) or (block.get("input", {}) if isinstance(block, dict) else {})
                            tool_calls.append({
                                "id": bid,
                                "type": "function",
                                "function": {"name": bname, "arguments": json.dumps(binp, ensure_ascii=False)},
                            })
                    else:
                        if hasattr(block, "type"):
                            if block.type == "text":
                                text_parts.append(getattr(block, "text", ""))
                            elif block.type == "tool_use":
                                tool_calls.append({
                                    "id": block.id,
                                    "type": "function",
                                    "function": {"name": block.name, "arguments": json.dumps(block.input, ensure_ascii=False)},
                                })
                body = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
                if tool_calls:
                    body["tool_calls"] = tool_calls
                out.append(body)
            else:
                out.append({"role": "assistant", "content": content or ""})
        # skip "tool" role here; we already expanded user content into tool messages
    return out


def _block_from_text(text: str):
    b = SimpleNamespace()
    b.type = "text"
    b.text = text
    return b


def _block_from_tool_call(tc: dict):
    b = SimpleNamespace()
    b.type = "tool_use"
    b.id = tc.get("id", "")
    b.name = (tc.get("function") or {}).get("name", "")
    try:
        b.input = json.loads((tc.get("function") or {}).get("arguments") or "{}")
    except Exception:
        b.input = {}
    return b


def _openai_response_to_anthropic(choice) -> tuple[list, str]:
    """Convert one OpenAI choice into (content_blocks, stop_reason)."""
    msg = choice.message if hasattr(choice, "message") else choice.get("message", {})
    content = getattr(msg, "content", None) or msg.get("content") or ""
    tool_calls = getattr(msg, "tool_calls", None) or msg.get("tool_calls") or []
    blocks = []
    if content:
        blocks.append(_block_from_text(content))
    for tc in tool_calls:
        blocks.append(_block_from_tool_call(tc))
    finish = getattr(choice, "finish_reason", None) or choice.get("finish_reason") or "stop"
    stop_reason = "tool_use" if tool_calls else "end_turn"
    return blocks, stop_reason


class _Response:
    def __init__(self, content: list, stop_reason: str):
        self.content = content
        self.stop_reason = stop_reason


class _MessagesAPI:
    def create(self, *, model: str = None, system: str = None, messages: list = None, tools: list = None, max_tokens: int = 8000):
        model = model or DEFAULT_MODEL
        openai_tools = _anthropic_tools_to_openai(tools) if tools else None
        openai_messages = _messages_anthropic_to_openai(messages or [], system)
        kwargs = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"
        resp = _openai_client.chat.completions.create(**kwargs)
        choice = resp.choices[0] if resp.choices else None
        if not choice:
            return _Response([_block_from_text("")], "end_turn")
        blocks, stop_reason = _openai_response_to_anthropic(choice)
        return _Response(blocks, stop_reason)


class _Client:
    def __init__(self, base_url: str = None):
        # Existing project code may pass base_url (e.g. ANTHROPIC_BASE_URL);
        # this adapter always uses the configured Qwen endpoint and ignores base_url.
        pass

    @property
    def messages(self):
        return _MessagesAPI()


# Public API: replace Anthropic(base_url=...) while keeping client.messages.create(...) usage compatible.
def get_client(base_url: str = None):
    return _Client(base_url=base_url)


# Default singleton for drop-in replacement of `client = Anthropic(...)`.
client = _Client()
