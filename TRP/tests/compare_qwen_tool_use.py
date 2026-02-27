from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def _find_trp_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "sdk").exists() and (p / "app").exists():
            return p
    raise RuntimeError(f"Unable to locate TRP root from {start}")


TRP_ROOT = _find_trp_root(Path(__file__).resolve())

if str(TRP_ROOT) not in sys.path:
    sys.path.insert(0, str(TRP_ROOT))


# ---- result models ----


@dataclass
class ToolCallRecord:
    tool_name: str
    latency_ms: float
    ok: bool
    output_preview: str
    error: Optional[str] = None


@dataclass
class RunRecord:
    runner: str
    task_id: str
    success: bool
    latency_ms: float
    llm_turns: int
    tool_calls: int
    final_text: str
    tool_call_records: List[ToolCallRecord]
    validator_errors: List[str]
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


@dataclass
class TaskCase:
    task_id: str
    prompt: str
    must_contain: List[str]
    group: str = "basic"
    prompt_trp: Optional[str] = None
    prompt_traditional: Optional[str] = None

    def prompt_for(self, runner: str) -> str:
        if runner == "trp_router_single_tool" and self.prompt_trp:
            return self.prompt_trp
        if runner == "traditional_direct_tools" and self.prompt_traditional:
            return self.prompt_traditional
        return self.prompt


DEFAULT_TASKS: List[TaskCase] = [
    TaskCase(
        task_id="search_titles",
        prompt=(
            "Use tools to search the web for query 'TRP protocol design' with top_k=3, "
            "then answer with the top 2 titles and total count. "
            "Include the exact title strings from the tool output."
        ),
        must_contain=["Result 1 for TRP protocol design", "Result 2 for TRP protocol design"],
        group="basic",
    ),
    TaskCase(
        task_id="search_titles_ptc",
        prompt=(
            "Use tools to search the web for query 'PTC programmatic tool calling' with top_k=4. "
            "Return the first 3 titles and the total count, and include exact title strings."
        ),
        must_contain=[
            "Result 1 for PTC programmatic tool calling",
            "Result 2 for PTC programmatic tool calling",
            "Result 3 for PTC programmatic tool calling",
        ],
        group="basic",
    ),
    TaskCase(
        task_id="search_single_hit",
        prompt=(
            "Use tools to search the web for query 'Anthropic advanced tool use' with top_k=1. "
            "Return the exact first title."
        ),
        must_contain=["Result 1 for Anthropic advanced tool use"],
        group="basic",
    ),
    TaskCase(
        task_id="doc_fetch",
        prompt=(
            "Use tools to fetch document_id 'DOC-42' and return its title plus updated_at. "
            "Include exact strings."
        ),
        must_contain=["Document DOC-42", "2026-02-24T00:00:00Z"],
        group="basic",
    ),
    TaskCase(
        task_id="doc_fetch_alt",
        prompt=(
            "Use tools to fetch document_id 'DOC-99'. Return the title and a short sentence quoting the exact updated_at."
        ),
        must_contain=["Document DOC-99", "2026-02-24T00:00:00Z"],
        group="basic",
    ),
    TaskCase(
        task_id="sql_read",
        prompt=(
            "Use tools to run a read-only SQL query 'SELECT id,name,value FROM metrics' with limit=2. "
            "Return row_count and the names in the rows. Include exact names."
        ),
        must_contain=["alpha", "beta"],
        group="basic",
    ),
    TaskCase(
        task_id="sql_read_variant",
        prompt=(
            "Use tools to run a read-only SQL query 'SELECT name,value FROM metrics' with limit=2. "
            "Return the row_count and list both names exactly."
        ),
        must_contain=["alpha", "beta"],
        group="basic",
    ),
    TaskCase(
        task_id="combo_search_then_doc",
        prompt=(
            "Use tools for two steps: (1) search the web for query 'router protocol framing' with top_k=2, "
            "(2) fetch document_id 'DOC-7'. "
            "Then answer with the first search title and the fetched document title. Include exact strings."
        ),
        must_contain=["Result 1 for router protocol framing", "Document DOC-7"],
        group="basic",
    ),
    TaskCase(
        task_id="combo_sql_then_search",
        prompt=(
            "Use tools for two steps: (1) run read-only SQL 'SELECT id,name FROM metrics' with limit=2, "
            "(2) search the web for query 'TRP vs traditional tool use' with top_k=2. "
            "Return the SQL names and the first search title, using exact strings from tool outputs."
        ),
        must_contain=["alpha", "beta", "Result 1 for TRP vs traditional tool use"],
        group="basic",
    ),
    TaskCase(
        task_id="combo_three_reads",
        prompt=(
            "Use three read tools: search web query 'catalog epoch index cap_id' with top_k=2, "
            "fetch document_id 'DOC-123', and run SQL 'SELECT name FROM metrics' with limit=2. "
            "Return one line containing the first search title, fetched document title, and both SQL names. "
            "Use exact strings."
        ),
        must_contain=["Result 1 for catalog epoch index cap_id", "Document DOC-123", "alpha", "beta"],
        group="basic",
    ),
    TaskCase(
        task_id="batch_search_three_queries",
        group="batch",
        prompt=(
            "Search these 3 queries and return the first title for each: "
            "'router protocol design', 'tool routing architecture', 'catalog epoch consistency'. "
            "If your tools provide a batch search capability, use one batch search call. "
            "Use exact title strings."
        ),
        prompt_trp=(
            "Search these 3 queries and return the first title for each: "
            "'router protocol design', 'tool routing architecture', 'catalog epoch consistency'. "
            "Use exactly one router batch_search call (not multiple router call ops). "
            "Use exact title strings."
        ),
        must_contain=[
            "Result 1 for router protocol design",
            "Result 1 for tool routing architecture",
            "Result 1 for catalog epoch consistency",
        ],
    ),
    TaskCase(
        task_id="batch_search_four_queries",
        group="batch",
        prompt=(
            "Search these 4 queries and return the first title of each plus a short list: "
            "'TRP protocol', 'PTC tool call', 'router batch call', 'idempotency key'. "
            "If a batch search tool exists, prefer one batch call."
        ),
        prompt_trp=(
            "Use one router batch_search call for 4 queries: "
            "'TRP protocol', 'PTC tool call', 'router batch call', 'idempotency key'. "
            "Return the first title for each query using exact strings."
        ),
        must_contain=[
            "Result 1 for TRP protocol",
            "Result 1 for PTC tool call",
            "Result 1 for router batch call",
            "Result 1 for idempotency key",
        ],
    ),
    TaskCase(
        task_id="batch_search_compare_queries",
        group="batch",
        prompt=(
            "Find the first title for each query: "
            "'approval token signing', 'partial result protocol', 'async result query', 'catalog sync'. "
            "If available, use batch search once."
        ),
        prompt_trp=(
            "Use one router batch_search call for queries "
            "'approval token signing', 'partial result protocol', 'async result query', 'catalog sync'. "
            "Return the first title for each query exactly."
        ),
        must_contain=[
            "Result 1 for approval token signing",
            "Result 1 for partial result protocol",
            "Result 1 for async result query",
            "Result 1 for catalog sync",
        ],
    ),
    TaskCase(
        task_id="batch_search_then_doc",
        group="batch",
        prompt=(
            "Do two steps: (1) search three queries "
            "'router lease renewal', 'stale lease requeue', 'async failover benchmark' and get the first title for each "
            "(use batch search if available), (2) fetch document_id 'DOC-555'. "
            "Return the three first titles and the fetched document title with exact strings."
        ),
        prompt_trp=(
            "Do two steps: (1) use one router batch_search call for queries "
            "'router lease renewal', 'stale lease requeue', 'async failover benchmark', "
            "(2) fetch document_id 'DOC-555' with router call. "
            "Return the three first titles and the fetched document title with exact strings."
        ),
        must_contain=[
            "Result 1 for router lease renewal",
            "Result 1 for stale lease requeue",
            "Result 1 for async failover benchmark",
            "Document DOC-555",
        ],
    ),
    TaskCase(
        task_id="batch_search_five_queries",
        group="batch",
        prompt=(
            "Return the first title for each of these 5 queries: "
            "'cap query schema', 'router metrics prometheus', 'redis runtime state', 'tool audit logs', 'batch dependency graph'. "
            "If you have a batch search capability, use one batch call."
        ),
        prompt_trp=(
            "Use one router batch_search call for 5 queries: "
            "'cap query schema', 'router metrics prometheus', 'redis runtime state', 'tool audit logs', 'batch dependency graph'. "
            "Return the first title for each query exactly."
        ),
        must_contain=[
            "Result 1 for cap query schema",
            "Result 1 for router metrics prometheus",
            "Result 1 for redis runtime state",
            "Result 1 for tool audit logs",
            "Result 1 for batch dependency graph",
        ],
    ),
]


def _make_batch_task(
    *,
    task_id: str,
    queries: List[str],
    group: str = "batch",
    include_doc_id: Optional[str] = None,
) -> TaskCase:
    q_literals = ", ".join([f"'{q}'" for q in queries])
    must = [f"Result 1 for {q}" for q in queries]
    if include_doc_id:
        must.append(f"Document {include_doc_id}")
        prompt = (
            f"Do two steps: (1) search these queries {q_literals} and return the first title for each "
            "(use batch search if available), (2) fetch document_id "
            f"'{include_doc_id}'. Return all first titles and the fetched document title with exact strings."
        )
        prompt_trp = (
            f"Use one router batch_search call for queries {q_literals}, then one router call to fetch "
            f"document_id '{include_doc_id}'. Return all first titles and the fetched document title with exact strings."
        )
    else:
        prompt = (
            f"Search these queries {q_literals} and return the first title for each. "
            "If a batch search tool exists, prefer one batch search call. Use exact title strings."
        )
        prompt_trp = (
            f"Use one router batch_search call for queries {q_literals}. "
            "Return the first title for each query using exact strings."
        )
    return TaskCase(
        task_id=task_id,
        group=group,
        prompt=prompt,
        prompt_trp=prompt_trp,
        must_contain=must,
    )


EXTRA_SHOWCASE_BATCH_TASKS: List[TaskCase] = [
    _make_batch_task(
        task_id="batch_search_six_queries_proto",
        queries=[
            "tool router protocol",
            "catalog alias table",
            "router batch request",
            "approval token policy",
            "partial result event",
            "redis lease renew",
        ],
    ),
    _make_batch_task(
        task_id="batch_search_six_queries_eval",
        queries=[
            "agent benchmark workload",
            "web agent realistic tasks",
            "assistantbench navigation",
            "webarena tasks",
            "gaia tool use benchmark",
            "agent observability traces",
        ],
    ),
    _make_batch_task(
        task_id="batch_search_doc_combo_1",
        queries=[
            "router ack async call",
            "result query polling",
            "idempotency replay result",
            "catalog mismatch retry",
        ],
        include_doc_id="DOC-901",
    ),
    _make_batch_task(
        task_id="batch_search_doc_combo_2",
        queries=[
            "batch dependency scheduling",
            "order violation retry hint",
            "schema mismatch validation",
            "policy denied critical delete",
        ],
        include_doc_id="DOC-902",
    ),
    _make_batch_task(
        task_id="batch_search_doc_combo_3",
        queries=[
            "prometheus metrics router",
            "json audit logger events",
            "lease lost event metric",
            "readiness healthz readyz",
        ],
        include_doc_id="DOC-903",
    ),
    _make_batch_task(
        task_id="batch_search_doc_combo_4",
        queries=[
            "search adapter canonical args",
            "sql read adapter limit",
            "document fetch capability",
            "router result shaper summary",
        ],
        include_doc_id="DOC-904",
    ),
    _make_batch_task(
        task_id="batch_search_five_queries_ops",
        queries=[
            "async failover takeover",
            "stale lease requeue",
            "runtime state store redis",
            "lua atomic append event",
            "claim async execution",
        ],
    ),
    _make_batch_task(
        task_id="batch_search_five_queries_ctx",
        queries=[
            "tool context efficiency",
            "single router tool interface",
            "canonical args adapter",
            "cap query schema hints",
            "catalog epoch drift",
        ],
    ),
    _make_batch_task(
        task_id="batch_search_four_queries_safety",
        queries=[
            "approval token expiration",
            "approval args binding",
            "critical action permission",
            "non idempotent blocked write",
        ],
    ),
]


def _task_profiles() -> Dict[str, List[TaskCase]]:
    basic = [t for t in DEFAULT_TASKS if t.group == "basic"]
    batch = [t for t in DEFAULT_TASKS if t.group == "batch"]
    showcase24 = basic + batch + EXTRA_SHOWCASE_BATCH_TASKS
    return {
        "balanced15": list(DEFAULT_TASKS),
        "batch5": list(batch),
        "showcase24": showcase24,
    }


TASK_PROFILE_NOTES: Dict[str, str] = {
    "balanced15": (
        "Balanced local smoke benchmark: 10 basic read tasks + 5 batch/fan-out tasks."
    ),
    "batch5": (
        "Batch-only stress slice to highlight fan-out aggregation and reduced LLM-visible tool calls."
    ),
    "showcase24": (
        "Benchmark-inspired local agent workload: 24 tasks with mixed read tasks and fan-out/batch tasks. "
        "Designed for personal reproducible comparisons (not full benchmark scale) while preserving multi-step tool-use structure."
    ),
}


def _json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _preview(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_text_from_blocks(blocks: List[Any]) -> str:
    parts: List[str] = []
    for b in blocks or []:
        if hasattr(b, "type") and getattr(b, "type", None) == "text":
            parts.append(str(getattr(b, "text", "")))
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(str(b.get("text", "")))
    return "\n".join([p for p in parts if p]).strip()


def _validate_text(task: TaskCase, text: str) -> List[str]:
    errs: List[str] = []
    for kw in task.must_contain:
        if kw not in text:
            errs.append(f"missing required substring: {kw}")
    return errs


def _load_qwen_client():
    # Lazy import so `--help` does not require API key setup.
    import llm_client  # type: ignore

    _ensure_llm_client_usage_patch(llm_client)
    return llm_client.client


def _usage_to_dict(usage_obj: Any) -> Optional[Dict[str, Any]]:
    if usage_obj is None:
        return None
    out: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage_obj, key, None)
        if v is None and isinstance(usage_obj, dict):
            v = usage_obj.get(key)
        if v is not None:
            try:
                out[key] = int(v)
            except Exception:
                out[key] = v
    if not out:
        return None
    return out


def _ensure_llm_client_usage_patch(llm_client_mod: Any) -> None:
    if getattr(llm_client_mod, "_trp_usage_patch_installed", False):
        return

    llm_client_mod._trp_last_usage = None  # type: ignore[attr-defined]

    orig_create_completion = llm_client_mod._openai_client.chat.completions.create

    def wrapped_create_completion(*args: Any, **kwargs: Any) -> Any:
        resp = orig_create_completion(*args, **kwargs)
        llm_client_mod._trp_last_usage = _usage_to_dict(getattr(resp, "usage", None))  # type: ignore[attr-defined]
        return resp

    llm_client_mod._openai_client.chat.completions.create = wrapped_create_completion

    def _safe_openai_response_to_anthropic(choice: Any) -> Tuple[List[Any], str]:
        msg = getattr(choice, "message", None)
        if msg is None and isinstance(choice, dict):
            msg = choice.get("message", {})

        def _field(obj: Any, key: str, default: Any = None) -> Any:
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        content = _field(msg, "content", "") or ""
        tool_calls = _field(msg, "tool_calls", None) or []

        blocks: List[Any] = []
        if content:
            blocks.append(llm_client_mod._block_from_text(content))
        for tc in tool_calls:
            if isinstance(tc, dict):
                blocks.append(llm_client_mod._block_from_tool_call(tc))
                continue
            b = type("ToolUseBlock", (), {})()
            b.type = "tool_use"
            b.id = getattr(tc, "id", "")
            fn = getattr(tc, "function", None)
            b.name = getattr(fn, "name", "") if fn is not None else ""
            raw_args = getattr(fn, "arguments", "{}") if fn is not None else "{}"
            try:
                b.input = json.loads(raw_args or "{}")
            except Exception:
                b.input = {}
            blocks.append(b)

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return blocks, stop_reason

    llm_client_mod._openai_response_to_anthropic = _safe_openai_response_to_anthropic

    orig_messages_create = llm_client_mod._MessagesAPI.create

    def wrapped_messages_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        r = orig_messages_create(self, *args, **kwargs)
        try:
            setattr(r, "usage", getattr(llm_client_mod, "_trp_last_usage", None))
        except Exception:
            pass
        return r

    llm_client_mod._MessagesAPI.create = wrapped_messages_create
    llm_client_mod._trp_usage_patch_installed = True  # type: ignore[attr-defined]


# ---- Qwen agent loop (s02-style compatible) ----


class QwenToolLoop:
    def __init__(
        self,
        *,
        system_prompt: str,
        tools: List[Dict[str, Any]],
        handlers: Dict[str, Callable[..., str]],
        model: Optional[str] = None,
    ) -> None:
        self.client = _load_qwen_client()
        self.system_prompt = system_prompt
        self.tools = tools
        self.handlers = handlers
        self.model = model or os.getenv("TRP_MODEL_ID") or os.getenv("MODEL_ID") or ""
        if not self.model:
            raise ValueError("Set TRP_MODEL_ID (or MODEL_ID), or pass model explicitly.")

    def run(self, user_prompt: str, *, max_steps: int = 12) -> RunRecord:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        tool_records: List[ToolCallRecord] = []
        llm_turns = 0
        t0 = time.perf_counter()
        final_text = ""
        error: Optional[str] = None
        usage_total: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage_seen = False

        for _ in range(max_steps):
            llm_turns += 1
            resp = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=messages,
                tools=self.tools,
                max_tokens=4000,
            )
            resp_usage = getattr(resp, "usage", None)
            if isinstance(resp_usage, dict):
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    try:
                        usage_total[k] += int(resp_usage.get(k, 0) or 0)
                        usage_seen = True
                    except Exception:
                        pass
            messages.append({"role": "assistant", "content": resp.content})
            final_text = _extract_text_from_blocks(resp.content)
            if resp.stop_reason != "tool_use":
                break

            tool_results = []
            for block in resp.content:
                if not (hasattr(block, "type") and getattr(block, "type", None) == "tool_use"):
                    continue
                tool_name = str(getattr(block, "name", ""))
                tool_input = getattr(block, "input", {}) or {}
                handler = self.handlers.get(tool_name)
                tc_t0 = time.perf_counter()
                ok = True
                err = None
                if handler is None:
                    ok = False
                    out = f"Error: unknown tool {tool_name}"
                    err = out
                else:
                    try:
                        out = handler(**tool_input)
                    except Exception as e:
                        ok = False
                        err = f"{type(e).__name__}: {e}"
                        out = f"Error: {err}"
                tc_ms = (time.perf_counter() - tc_t0) * 1000.0
                tool_records.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        latency_ms=tc_ms,
                        ok=ok,
                        output_preview=_preview(str(out)),
                        error=err,
                    )
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": getattr(block, "id", ""),
                        "content": str(out),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
        else:
            error = f"max_steps exceeded ({max_steps})"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RunRecord(
            runner="unknown",
            task_id="unknown",
            success=False,
            latency_ms=latency_ms,
            llm_turns=llm_turns,
            tool_calls=len(tool_records),
            final_text=final_text,
            tool_call_records=tool_records,
            validator_errors=[],
            error=error,
            usage=(usage_total if usage_seen else None),
        )


# ---- direct multi-tool runner (s02 style, same capability set as TRP) ----


class DirectToolsRunner:
    def __init__(self) -> None:
        from sdk.in_memory_impl import InMemoryCapabilityRegistry
        from sdk.basic_adapter_executor import BasicAdapterRegistry, BasicExecutor, BasicResultShaper

        self._registry = InMemoryCapabilityRegistry()
        self._adapters = BasicAdapterRegistry()
        self._executor = BasicExecutor()
        self._shaper = BasicResultShaper()
        self._cap_by_id = {c.cap_id: c for c in self._registry._caps}  # type: ignore[attr-defined]

        self.system_prompt = (
            "You are an agent with direct tools. Use tools to solve the task. "
            "Prefer concise final answers. Do not invent tool outputs. "
            "Available direct tools map to TRP capabilities: "
            "web_search, doc_fetch, sql_read_query."
        )
        self.tools = [
            {
                "name": "web_search",
                "description": "Search public web content by query.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                    "required": ["query"],
                },
            },
            {
                "name": "doc_fetch",
                "description": "Fetch a document by document_id.",
                "input_schema": {
                    "type": "object",
                    "properties": {"document_id": {"type": "string"}},
                    "required": ["document_id"],
                },
            },
            {
                "name": "sql_read_query",
                "description": "Run a read-only SQL query on an analytics dataset.",
                "input_schema": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["sql"],
                },
            },
        ]
        self._loop = QwenToolLoop(
            system_prompt=self.system_prompt,
            tools=self.tools,
            handlers={
                "web_search": lambda **kw: self._run_cap("cap.search.web.v1", kw),
                "doc_fetch": lambda **kw: self._run_cap("cap.fetch.doc.v1", kw),
                "sql_read_query": lambda **kw: self._run_cap("cap.query.sql_read.v1", kw),
            },
        )

    def _run_cap(self, cap_id: str, args: Dict[str, Any]) -> str:
        cap = self._cap_by_id[cap_id]
        adapter = self._adapters.get(cap.adapter_key)
        adapter.validate_canonical_args(cap, args)
        native = adapter.to_native_args(cap, args)
        raw = self._executor.execute(cap, native, 8000)
        shaped = self._shaper.shape_success(cap, raw)
        return _json_dumps(shaped)

    def run_task(self, task: TaskCase) -> RunRecord:
        rec = self._loop.run(task.prompt_for("traditional_direct_tools"))
        rec.runner = "traditional_direct_tools"
        rec.task_id = task.task_id
        rec.validator_errors = _validate_text(task, rec.final_text)
        rec.success = (rec.error is None) and (len(rec.validator_errors) == 0)
        return rec


# ---- TRP single-router-tool runner ----


class TRPRouterRunner:
    def __init__(self, *, router_url: str) -> None:
        from sdk.trp_client import RouterClient
        from sdk.trp_transport_http import HttpTRPTransport
        from sdk.trp_types import CallSpec

        self.router_url = router_url
        self._RouterClient = RouterClient
        self._HttpTRPTransport = HttpTRPTransport
        self._CallSpec = CallSpec

        self._client = self._RouterClient(
            self._HttpTRPTransport(self.router_url),
            agent_id=f"cmp_shared_{uuid.uuid4().hex[:6]}",
        )
        # Production usage is closer to reusing session+catalog than cold-starting each task.
        self._client.hello()
        self._client.sync_catalog()

        self.system_prompt = (
            "You are an agent with exactly one tool: router. "
            "The router session is already initialized and catalog is already synced for this benchmark. "
            "Use router op='call' with BOTH idx and cap_id. "
            "Use this fixed capability index map for the entire benchmark (do not re-discover catalog): "
            "idx=0 -> cap.search.web.v1, idx=1 -> cap.fetch.doc.v1, idx=2 -> cap.query.sql_read.v1. "
            "If multiple independent web searches are needed, prefer one router op='batch_search'. "
            "Use only read capabilities for this benchmark. "
            "Return a concise final answer based strictly on tool outputs."
        )
        self.tools = [
            {
                "name": "router",
                "description": (
                    "Single TRP router tool. Operations: call, batch_search. "
                    "For call: provide idx, cap_id, args. "
                    "For batch_search: provide queries[] and optional top_k."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["call", "batch_search"]},
                        "idx": {"type": "integer"},
                        "cap_id": {"type": "string"},
                        "args": {"type": "object"},
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Used by batch_search",
                        },
                        "top_k": {"type": "integer"},
                    },
                    "required": ["op"],
                },
            }
        ]
        self._loop = self._make_loop()

    def _make_loop(self) -> QwenToolLoop:
        def router_tool(**kw: Any) -> str:
            op = str(kw.get("op", "")).strip()
            if op == "sync_catalog":
                caps = self._client.sync_catalog()
                compact = [
                    {
                        "idx": c.idx,
                        "cap_id": c.cap_id,
                        "name": c.name,
                        "desc": c.desc,
                        "risk_tier": c.risk_tier,
                        "io_class": c.io_class,
                        "arg_template": c.arg_template,
                    }
                    for c in caps
                ]
                return _json_dumps({"catalog": compact})
            if op == "cap_query":
                idx = int(kw["idx"])
                cap_id = str(kw["cap_id"])
                include_examples = bool(kw.get("include_examples", False))
                return _json_dumps(self._client.cap_query(idx=idx, cap_id=cap_id, include_examples=include_examples))
            if op == "call":
                idx = int(kw["idx"])
                cap_id = str(kw["cap_id"])
                args = kw.get("args", {}) or {}
                res = self._client.call(idx=idx, cap_id=cap_id, args=args, timeout_ms=8000)
                return _json_dumps(res)
            if op == "batch_search":
                queries = kw.get("queries", []) or []
                if not isinstance(queries, list) or not queries:
                    raise ValueError("batch_search requires non-empty queries[]")
                top_k = int(kw.get("top_k", 2))
                calls = [
                    self._CallSpec(
                        call_id=f"batchq_{uuid.uuid4().hex[:8]}",
                        idempotency_key=None,
                        idx=0,
                        cap_id="cap.search.web.v1",
                        attempt=1,
                        timeout_ms=8000,
                        args={"query": str(q), "top_k": top_k},
                    )
                    for q in queries
                ]
                batch_res = self._client.batch(calls, mode="PARALLEL", max_concurrency=min(8, len(calls)))
                return _json_dumps(batch_res)
            raise ValueError(f"unsupported router op: {op}")

        return QwenToolLoop(
            system_prompt=self.system_prompt,
            tools=self.tools,
            handlers={"router": router_tool},
        )

    def run_task(self, task: TaskCase) -> RunRecord:
        rec = self._loop.run(task.prompt_for("trp_router_single_tool"))
        rec.runner = "trp_router_single_tool"
        rec.task_id = task.task_id
        rec.validator_errors = _validate_text(task, rec.final_text)
        rec.success = (rec.error is None) and (len(rec.validator_errors) == 0)
        return rec


# ---- harness ----


def _serialise_run(rec: RunRecord) -> Dict[str, Any]:
    out = asdict(rec)
    return out


def _summary_table(records: List[RunRecord]) -> Dict[str, Any]:
    by_runner: Dict[str, List[RunRecord]] = {}
    for r in records:
        by_runner.setdefault(r.runner, []).append(r)
    out: Dict[str, Any] = {}
    for runner, rows in by_runner.items():
        n = len(rows)
        success_n = sum(1 for r in rows if r.success)
        lat = [r.latency_ms for r in rows]
        tool_calls = [r.tool_calls for r in rows]
        llm_turns = [r.llm_turns for r in rows]
        usage_rows = [r.usage for r in rows if isinstance(r.usage, dict)]
        usage_summary: Optional[Dict[str, Any]] = None
        if usage_rows:
            usage_summary = {}
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                vals = [int(u.get(k, 0) or 0) for u in usage_rows]
                usage_summary[f"sum_{k}"] = int(sum(vals))
                usage_summary[f"avg_{k}"] = round(sum(vals) / len(usage_rows), 2) if usage_rows else 0.0
        out[runner] = {
            "tasks": n,
            "success": success_n,
            "success_rate": round(success_n / n, 4) if n else 0.0,
            "avg_latency_ms": round(sum(lat) / n, 2) if n else 0.0,
            "avg_tool_calls": round(sum(tool_calls) / n, 2) if n else 0.0,
            "avg_llm_turns": round(sum(llm_turns) / n, 2) if n else 0.0,
            "usage": usage_summary,
        }
    return out


def _summary_table_by_group(records: List[RunRecord], tasks: List[TaskCase]) -> Dict[str, Any]:
    task_group = {t.task_id: t.group for t in tasks}
    groups = sorted(set(task_group.values()))
    by_group: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        subset = [r for r in records if task_group.get(r.task_id) == g]
        by_group[g] = _summary_table(subset)
    return by_group


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare TRP single-router-tool vs s02-style direct multi-tool"
    )
    ap.add_argument("--mode", choices=["trp", "traditional", "both"], default="both")
    ap.add_argument("--router-url", default="http://127.0.0.1:8000", help="TRP router base URL")
    ap.add_argument(
        "--task-profile",
        choices=sorted(_task_profiles().keys()),
        default="showcase24",
        help="Task set profile. showcase24 is the recommended benchmark-inspired local comparison.",
    )
    ap.add_argument("--out", default="", help="Optional JSON output file path")
    args = ap.parse_args()

    profiles = _task_profiles()
    tasks = list(profiles[args.task_profile])
    runners: List[tuple[str, Callable[[TaskCase], RunRecord]]] = []

    if args.mode in {"traditional", "both"}:
        direct_runner = DirectToolsRunner()
        runners.append(("traditional_direct_tools", direct_runner.run_task))
    if args.mode in {"trp", "both"}:
        trp_runner = TRPRouterRunner(router_url=args.router_url)
        runners.append(("trp_router_single_tool", trp_runner.run_task))

    started = time.perf_counter()
    records: List[RunRecord] = []
    for task in tasks:
        for runner_name, fn in runners:
            print(f"[RUN] {runner_name} :: {task.task_id}", flush=True)
            try:
                rec = fn(task)
            except Exception as e:
                rec = RunRecord(
                    runner=runner_name,
                    task_id=task.task_id,
                    success=False,
                    latency_ms=0.0,
                    llm_turns=0,
                    tool_calls=0,
                    final_text="",
                    tool_call_records=[],
                    validator_errors=[],
                    error=f"{type(e).__name__}: {e}",
                    usage=None,
                )
            records.append(rec)
            status = "OK" if rec.success else "FAIL"
            print(
                f"  -> {status} latency={rec.latency_ms:.1f}ms turns={rec.llm_turns} tools={rec.tool_calls}",
                flush=True,
            )
            if rec.validator_errors:
                print(f"     validator_errors: {rec.validator_errors}", flush=True)
            if rec.error:
                print(f"     error: {rec.error}", flush=True)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    payload = {
        "meta": {
            "mode": args.mode,
            "router_url": args.router_url,
            "model_id": os.getenv("TRP_MODEL_ID") or os.getenv("MODEL_ID") or "",
            "task_profile": args.task_profile,
            "task_profile_note": TASK_PROFILE_NOTES.get(args.task_profile, ""),
            "elapsed_ms": round(elapsed_ms, 2),
            "task_count": len(tasks),
            "runner_count": len(runners),
        },
        "summary": _summary_table(records),
        "summary_by_group": _summary_table_by_group(records, tasks),
        "records": [_serialise_run(r) for r in records],
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = TRP_ROOT / "tests" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"compare_qwen_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print("\n=== Summary By Group ===")
    print(json.dumps(payload["summary_by_group"], ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
