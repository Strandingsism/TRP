from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional


SCRIPT_PATH = Path(__file__).resolve()
TRP_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = TRP_ROOT.parent
DEFAULT_TAU2_ROOT = REPO_ROOT / "_vendor" / "tau2-bench"
DEFAULT_RESULTS_DIR = TRP_ROOT / "tests" / "results" / "tau2"


def load_simple_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def mask_secret(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(math.ceil(len(s) * pct)) - 1))
    return s[idx]


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    if num_trials < k:
        raise ValueError(f"num_trials {num_trials} < k {k}")
    return math.comb(success_count, k) / math.comb(num_trials, k)


@dataclass
class Pricing:
    input_rmb_per_mtoken: float
    output_rmb_per_mtoken: float

    def cost_rmb(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            (prompt_tokens / 1_000_000.0) * self.input_rmb_per_mtoken
            + (completion_tokens / 1_000_000.0) * self.output_rmb_per_mtoken
        )


def ensure_tau2_importable(tau2_root: Path) -> None:
    tau2_src = tau2_root / "src"
    if not tau2_src.exists():
        raise FileNotFoundError(f"Tau2 source directory not found: {tau2_src}")
    if str(tau2_src) not in sys.path:
        sys.path.insert(0, str(tau2_src))
    tests_dir = TRP_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))


def register_custom_agents() -> None:
    from tau2_agents import register_tau2_custom_agents

    register_tau2_custom_agents()


def patch_tau2_deepseek_reasoner_compat() -> None:
    """
    Patch Tau2's litellm message conversion to preserve reasoning_content for
    OpenAI-compatible DeepSeek reasoner providers that require it in assistant history.
    This keeps Tau2 evaluation logic unchanged and only patches request serialization.
    """
    import tau2.utils.llm_utils as llm_utils
    from tau2.data_model.message import (
        AssistantMessage,
        SystemMessage,
        ToolMessage,
        UserMessage,
    )

    if getattr(llm_utils, "_trp_reasoner_compat_patched", False):
        return
    llm_utils._trp_reasoner_require_field_always = True

    def _extract_reasoning_content(msg: AssistantMessage) -> Optional[str]:
        raw = getattr(msg, "raw_data", None)
        if not isinstance(raw, dict):
            return None
        # Common shapes from LiteLLM/OpenAI-compatible clients:
        # 1) choice.to_dict() -> {"message": {..., "reasoning_content": "..."}}
        # 2) direct -> {"reasoning_content": "..."}
        node = raw.get("message") if isinstance(raw.get("message"), dict) else raw
        if not isinstance(node, dict):
            return None
        value = node.get("reasoning_content")
        if isinstance(value, str) and value.strip():
            return value
        return None

    def to_litellm_messages_with_reasoning(messages):
        litellm_messages = []
        for message in messages:
            if isinstance(message, UserMessage):
                litellm_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AssistantMessage):
                tool_calls = None
                if message.is_tool_call():
                    tool_calls = [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                            "type": "function",
                        }
                        for tc in message.tool_calls
                    ]
                out = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
                reasoning_content = _extract_reasoning_content(message)
                if reasoning_content is not None:
                    out["reasoning_content"] = reasoning_content
                elif getattr(llm_utils, "_trp_reasoner_require_field_always", False):
                    # Some OpenAI-compatible DeepSeek endpoints require the field to
                    # exist on every assistant history message, even when the provider
                    # omitted it in a prior response.
                    out["reasoning_content"] = "[no reasoning_content returned by provider]"
                litellm_messages.append(out)
            elif isinstance(message, ToolMessage):
                litellm_messages.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": message.id,
                    }
                )
            elif isinstance(message, SystemMessage):
                litellm_messages.append({"role": "system", "content": message.content})
        return litellm_messages

    llm_utils.to_litellm_messages = to_litellm_messages_with_reasoning
    llm_utils._trp_reasoner_compat_patched = True


def patch_tau2_empty_assistant_output_compat() -> None:
    """Retry when an OpenAI-compatible provider returns an empty assistant message.

    Some third-party endpoints for reasoning models occasionally return an assistant
    message with empty content and no tool calls. Tau2 validates assistant messages
    strictly and raises before the simulation can continue. This patch retries the
    same request a few times and, as a last resort, returns a minimal non-empty
    assistant message so the run doesn't crash.
    """

    import tau2.utils.llm_utils as llm_utils
    from tau2.data_model.message import AssistantMessage

    if getattr(llm_utils.generate, "_trp_empty_assistant_patch", False):
        return

    original_generate = llm_utils.generate

    def _is_invalid_empty_assistant(msg: Any) -> bool:
        if not isinstance(msg, AssistantMessage):
            return False
        if msg.tool_calls:
            return False
        content = msg.content
        if content is None:
            return True
        if isinstance(content, str) and not content.strip():
            return True
        return False

    def generate_with_empty_retry(*args: Any, **kwargs: Any) -> Any:
        last_msg = None
        max_attempts = 3
        last_exc: Optional[Exception] = None
        for _ in range(max_attempts):
            try:
                msg = original_generate(*args, **kwargs)
            except json.JSONDecodeError as e:
                # Some providers occasionally return malformed tool_call.function.arguments JSON.
                # Retry the request; often the next sample is valid.
                last_exc = e
                continue
            except Exception as e:
                if "JSONDecodeError" in str(type(e)) or "Expecting ',' delimiter" in str(e):
                    last_exc = e
                    continue
                raise
            if not _is_invalid_empty_assistant(msg):
                return msg
            last_msg = msg
        if last_exc is not None and last_msg is None:
            raise last_exc
        # Last-resort fallback to avoid crashing the whole benchmark run on provider glitches.
        if isinstance(last_msg, AssistantMessage):
            raw = deepcopy(last_msg.raw_data) if isinstance(last_msg.raw_data, dict) else {}
            if isinstance(raw, dict):
                msg_node = raw.get("message")
                if not isinstance(msg_node, dict):
                    msg_node = {}
                msg_node.setdefault(
                    "reasoning_content",
                    "[synthetic fallback after empty assistant reply]",
                )
                raw["message"] = msg_node
            return AssistantMessage(
                role="assistant",
                content="I need a moment to process that. Could you repeat your last request?",
                tool_calls=None,
                cost=last_msg.cost,
                usage=last_msg.usage,
                raw_data=raw or last_msg.raw_data,
            )
        return last_msg

    generate_with_empty_retry._trp_empty_assistant_patch = True  # type: ignore[attr-defined]
    llm_utils.generate = generate_with_empty_retry

    # Tau2 modules import `generate` directly; rebind their local references as well.
    module_names = [
        "tau2.agent.llm_agent",
        "tau2.user.user_simulator",
        "tau2.environment.utils.interface_agent",
        "tau2.evaluator.evaluator_nl_assertions",
        "tau2_agents",
    ]
    for module_name in module_names:
        try:
            mod = __import__(module_name, fromlist=["generate"])
            if hasattr(mod, "generate"):
                setattr(mod, "generate", generate_with_empty_retry)
        except Exception:
            # Best-effort patching: modules may not be imported/available yet.
            pass


def _estimate_from_tau2_official_samples(
    tau2_root: Path, domains: list[str], num_trials: int, methods_count: int
) -> Optional[dict[str, Any]]:
    sample_map = {
        "airline": tau2_root
        / "data"
        / "tau2"
        / "results"
        / "final"
        / "gpt-4.1-mini-2025-04-14_airline_base_gpt-4.1-2025-04-14_4trials.json",
        "retail": tau2_root
        / "data"
        / "tau2"
        / "results"
        / "final"
        / "gpt-4.1-mini-2025-04-14_retail_base_gpt-4.1-2025-04-14_4trials.json",
    }
    if any(d not in sample_map for d in domains):
        return None

    totals = Counter()
    domain_samples: dict[str, Any] = {}
    for domain in domains:
        path = sample_map[domain]
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        sims = data["simulations"]
        if not sims:
            return None
        sums = Counter()
        for sim in sims:
            for msg in sim.get("messages", []):
                usage = msg.get("usage") or {}
                pt = int(usage.get("prompt_tokens", 0) or 0)
                ct = int(usage.get("completion_tokens", 0) or 0)
                sums["prompt_tokens"] += pt
                sums["completion_tokens"] += ct
                role = msg.get("role")
                if role == "assistant":
                    sums["agent_prompt_tokens"] += pt
                    sums["agent_completion_tokens"] += ct
                elif role == "user":
                    sums["user_prompt_tokens"] += pt
                    sums["user_completion_tokens"] += ct
        sims_count = len(sims)
        tasks = len(data["tasks"])
        per_sim = {k: sums[k] / sims_count for k in sums}
        domain_samples[domain] = {
            "sample_file": str(path),
            "tasks": tasks,
            "sample_num_trials": data.get("info", {}).get("num_trials"),
            "per_sim_avg": per_sim,
        }
        sims_target = tasks * num_trials * methods_count
        totals["tasks"] += tasks
        for k, v in per_sim.items():
            totals[k] += v * sims_target

    totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return {
        "domains": domains,
        "num_trials": num_trials,
        "methods_count": methods_count,
        "estimated_tasks_total": int(totals["tasks"]),
        "estimated_simulations_total": int(totals["tasks"] * num_trials * methods_count),
        "estimated_tokens": {
            "prompt_tokens": round(totals["prompt_tokens"]),
            "completion_tokens": round(totals["completion_tokens"]),
            "total_tokens": round(totals["total_tokens"]),
            "agent_prompt_tokens": round(totals["agent_prompt_tokens"]),
            "agent_completion_tokens": round(totals["agent_completion_tokens"]),
            "user_prompt_tokens": round(totals["user_prompt_tokens"]),
            "user_completion_tokens": round(totals["user_completion_tokens"]),
        },
        "domain_sample_details": domain_samples,
        "note": (
            "Estimate extrapolated from Tau2 official airline/retail published trajectories. "
            "Actual deepseek-reasoner completion tokens may differ."
        ),
    }


def _make_litellm_model_and_args_from_env(
    env: dict[str, str],
    model_id: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    model_id = (model_id or env.get("DEEPSEEK_MODEL_ID") or "deepseek-reasoner").strip()
    base_url = (base_url or env.get("DEEPSEEK_BASE_URL") or "").strip() or None
    api_key = (api_key or env.get("DEEPSEEK_API_KEY") or "").strip() or None

    litellm_model = model_id
    if base_url and "/" not in model_id:
        litellm_model = f"openai/{model_id}"

    llm_args: dict[str, Any] = {"temperature": temperature}
    if base_url:
        llm_args["api_base"] = base_url.rstrip("/")
    if api_key:
        llm_args["api_key"] = api_key
    return litellm_model, llm_args


def _task_ids(tasks: list[Any]) -> list[str]:
    return [t.id for t in tasks]


@contextmanager
def _tau2_auto_resume_yes_prompt(enabled: bool = True):
    """Auto-answer 'y' to Tau2 resume prompts for non-interactive reruns."""
    if not enabled:
        yield
        return

    try:
        from tau2.utils.display import ConsoleDisplay
    except Exception:
        yield
        return

    console = ConsoleDisplay.console
    original_input = console.input

    def auto_input(*args: Any, **kwargs: Any):
        prompt = ""
        if args:
            prompt = str(args[0])
        elif "prompt" in kwargs:
            prompt = str(kwargs["prompt"])
        if "Do you want to resume the run? (y/n)" in prompt:
            print("[tau2-auto-resume] answering 'y' to resume prompt")
            return "y"
        return original_input(*args, **kwargs)

    console.input = auto_input  # type: ignore[assignment]
    try:
        yield
    finally:
        console.input = original_input  # type: ignore[assignment]


def _is_recoverable_tau2_run_exception(exc: Exception) -> bool:
    msg = str(exc)
    patterns = [
        "Missing `reasoning_content` field in the assistant message",
        "AssistantMessage must have either content or tool calls",
        "must be followed by tool messages responding to each 'tool_call_id'",
        "insufficient tool messages following tool_calls message",
        "JSONDecodeError",
        "Expecting ',' delimiter",
        "Rate limit",
        "rate limit",
        "429",
        "timeout",
        "timed out",
        "APIConnectionError",
        "Service Unavailable",
        "503",
        "504",
        "502",
        "Connection reset",
    ]
    return any(p in msg for p in patterns)


def _run_tau2_domain(
    *,
    domain: str,
    split: str,
    agent_name: str,
    user_name: str,
    llm_agent_model: str,
    llm_agent_args: dict[str, Any],
    llm_user_model: str,
    llm_user_args: dict[str, Any],
    num_trials: int,
    max_steps: int,
    max_errors: int,
    max_concurrency: int,
    seed: int,
    save_path: Path,
    log_level: str,
) -> tuple[Any, list[str]]:
    from tau2.run import get_tasks, run_tasks

    tasks = get_tasks(task_set_name=domain, task_split_name=split)
    task_ids = _task_ids(tasks)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    max_auto_resume_retries = 20
    attempt = 0
    while True:
        attempt += 1
        try:
            with _tau2_auto_resume_yes_prompt(enabled=True):
                results = run_tasks(
                    domain=domain,
                    tasks=deepcopy(tasks),
                    agent=agent_name,
                    user=user_name,
                    llm_agent=llm_agent_model,
                    llm_args_agent=deepcopy(llm_agent_args),
                    llm_user=llm_user_model,
                    llm_args_user=deepcopy(llm_user_args),
                    num_trials=num_trials,
                    max_steps=max_steps,
                    max_errors=max_errors,
                    save_to=save_path,
                    console_display=False,
                    max_concurrency=max_concurrency,
                    seed=seed,
                    log_level=log_level,
                )
            return results, task_ids
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt >= max_auto_resume_retries or not _is_recoverable_tau2_run_exception(e):
                raise
            wait_sec = min(60.0, 2.0 * attempt)
            print(
                f"[tau2-auto-resume] recoverable error on {domain}/{agent_name} "
                f"(attempt {attempt}/{max_auto_resume_retries}): {e}"
            )
            print(
                f"[tau2-auto-resume] retrying and resuming from {save_path.name} after {wait_sec:.1f}s..."
            )
            time.sleep(wait_sec)


def _iter_sim_rows(domain: str, results: Any) -> Iterable[dict[str, Any]]:
    for sim in results.simulations:
        reward = None
        if getattr(sim, "reward_info", None) is not None:
            reward = sim.reward_info.reward
        yield {
            "domain": domain,
            "simulation": sim,
            "task_id": sim.task_id,
            "trial": sim.trial,
            "reward": reward,
            "success": reward is not None and abs(reward - 1.0) <= 1e-6,
        }


def _aggregate_metrics(
    domain_results: list[tuple[str, Any]],
    pricing: Pricing,
    budget_tokens: Optional[int] = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for domain, results in domain_results:
        rows.extend(list(_iter_sim_rows(domain, results)))
    if not rows:
        raise ValueError("No simulation rows to aggregate")

    num_sims = len(rows)
    rewards = [r["reward"] for r in rows if r["reward"] is not None]
    durations = [float(r["simulation"].duration) for r in rows]

    task_trial_successes: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        task_key = f'{r["domain"]}:{r["task_id"]}'
        task_trial_successes[task_key].append(bool(r["success"]))
    min_trials = min(len(v) for v in task_trial_successes.values())
    pass_hat = {}
    for k in range(1, min_trials + 1):
        vals = [pass_hat_k(len(v), sum(v), k) for v in task_trial_successes.values()]
        pass_hat[k] = mean(vals) if vals else None

    term_counts = Counter(str(r["simulation"].termination_reason) for r in rows)

    token_totals = Counter()
    msg_counts = Counter()
    tool_stats = Counter()
    token_per_sim_total: list[int] = []
    token_per_sim_agent: list[int] = []
    token_per_sim_user: list[int] = []
    prompt_single_request_values: list[int] = []

    for r in rows:
        sim = r["simulation"]
        sim_prompt = sim_completion = 0
        sim_agent = sim_user = 0

        for msg in sim.messages:
            role = getattr(msg, "role", None)
            msg_counts[f"messages_{role}_total"] += 1

            if role in ("assistant", "user"):
                tool_calls = getattr(msg, "tool_calls", None)
                has_tool_calls = tool_calls is not None

                if role == "assistant":
                    msg_counts["assistant_messages_total"] += 1
                    if has_tool_calls:
                        msg_counts["assistant_tool_turns_total"] += 1
                        n_calls = len(tool_calls)
                        tool_stats["assistant_env_tool_calls_total"] += n_calls
                        if n_calls > 1:
                            tool_stats["assistant_multitool_turns_total"] += 1
                        raw = getattr(msg, "raw_data", None)
                        if isinstance(raw, dict):
                            meta = raw.get("trp_router_meta")
                            if isinstance(meta, dict):
                                tool_stats["trp_router_turns_total"] += 1
                                tool_stats["trp_llm_visible_tool_calls_total"] += int(
                                    meta.get("llm_visible_tool_calls", 1) or 1
                                )
                                tool_stats["trp_env_tool_calls_total"] += int(
                                    meta.get("env_tool_call_count", n_calls) or n_calls
                                )
                                if str(meta.get("op")) == "batch":
                                    tool_stats["trp_batch_turns_total"] += 1
                                    tool_stats["trp_batch_env_tool_calls_acc"] += int(
                                        meta.get("env_tool_call_count", n_calls) or n_calls
                                    )
                    else:
                        msg_counts["assistant_text_turns_total"] += 1
                elif role == "user":
                    msg_counts["user_messages_total"] += 1
                    if has_tool_calls:
                        msg_counts["user_tool_turns_total"] += 1
                        tool_stats["user_tool_calls_total"] += len(tool_calls)
                    else:
                        msg_counts["user_text_turns_total"] += 1

                usage = getattr(msg, "usage", None) or {}
                pt = int(usage.get("prompt_tokens", 0) or 0)
                ct = int(usage.get("completion_tokens", 0) or 0)
                prompt_single_request_values.append(pt)

                token_totals["prompt_tokens_total"] += pt
                token_totals["completion_tokens_total"] += ct
                sim_prompt += pt
                sim_completion += ct

                if role == "assistant":
                    token_totals["agent_prompt_tokens_total"] += pt
                    token_totals["agent_completion_tokens_total"] += ct
                    sim_agent += pt + ct
                else:
                    token_totals["user_prompt_tokens_total"] += pt
                    token_totals["user_completion_tokens_total"] += ct
                    sim_user += pt + ct

            elif role == "tool":
                msg_counts["tool_messages_total"] += 1
                if getattr(msg, "error", False):
                    tool_stats["tool_error_messages_total"] += 1

        token_per_sim_total.append(sim_prompt + sim_completion)
        token_per_sim_agent.append(sim_agent)
        token_per_sim_user.append(sim_user)

    if tool_stats["trp_router_turns_total"] == 0:
        tool_stats["llm_visible_tool_calls_total"] = tool_stats["assistant_env_tool_calls_total"]
    else:
        tool_stats["llm_visible_tool_calls_total"] = tool_stats["trp_llm_visible_tool_calls_total"]

    reported_agent_costs = [
        float(r["simulation"].agent_cost) for r in rows if r["simulation"].agent_cost is not None
    ]
    reported_user_costs = [
        float(r["simulation"].user_cost) for r in rows if r["simulation"].user_cost is not None
    ]

    agent_prompt = int(token_totals["agent_prompt_tokens_total"])
    agent_completion = int(token_totals["agent_completion_tokens_total"])
    user_prompt = int(token_totals["user_prompt_tokens_total"])
    user_completion = int(token_totals["user_completion_tokens_total"])
    agent_cost_rmb = pricing.cost_rmb(agent_prompt, agent_completion)
    user_cost_rmb = pricing.cost_rmb(user_prompt, user_completion)
    total_tokens = int(token_totals["prompt_tokens_total"] + token_totals["completion_tokens_total"])

    return {
        "simulations": {
            "count": num_sims,
            "successful_count": sum(1 for r in rows if r["success"]),
            "failed_count": sum(1 for r in rows if not r["success"]),
            "success_rate": sum(1 for r in rows if r["success"]) / num_sims if num_sims else None,
            "avg_reward": mean(rewards) if rewards else None,
            "termination_reason_counts": dict(term_counts),
        },
        "pass_hat_k": {str(k): v for k, v in pass_hat.items()},
        "durations_sec": {
            "sum": sum(durations),
            "avg": mean(durations),
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
            "max": max(durations) if durations else None,
        },
        "tokens": {
            "prompt_tokens_total": int(token_totals["prompt_tokens_total"]),
            "completion_tokens_total": int(token_totals["completion_tokens_total"]),
            "total_tokens": total_tokens,
            "agent_prompt_tokens_total": agent_prompt,
            "agent_completion_tokens_total": agent_completion,
            "agent_total_tokens": agent_prompt + agent_completion,
            "user_prompt_tokens_total": user_prompt,
            "user_completion_tokens_total": user_completion,
            "user_total_tokens": user_prompt + user_completion,
            "avg_total_tokens_per_sim": mean(token_per_sim_total) if token_per_sim_total else None,
            "p50_total_tokens_per_sim": percentile(token_per_sim_total, 0.50),
            "p95_total_tokens_per_sim": percentile(token_per_sim_total, 0.95),
            "p99_total_tokens_per_sim": percentile(token_per_sim_total, 0.99),
            "avg_agent_tokens_per_sim": mean(token_per_sim_agent) if token_per_sim_agent else None,
            "avg_user_tokens_per_sim": mean(token_per_sim_user) if token_per_sim_user else None,
            "max_prompt_tokens_single_request": max(prompt_single_request_values) if prompt_single_request_values else None,
            "budget_usage_ratio": (total_tokens / budget_tokens) if budget_tokens else None,
            "budget_tokens": budget_tokens,
        },
        "costs": {
            "tau2_reported": {
                "agent_total": sum(reported_agent_costs) if reported_agent_costs else None,
                "user_total": sum(reported_user_costs) if reported_user_costs else None,
                "agent_avg_per_sim": mean(reported_agent_costs) if reported_agent_costs else None,
                "user_avg_per_sim": mean(reported_user_costs) if reported_user_costs else None,
            },
            "derived_rmb_from_tokens": {
                "pricing": {
                    "input_rmb_per_mtoken": pricing.input_rmb_per_mtoken,
                    "output_rmb_per_mtoken": pricing.output_rmb_per_mtoken,
                },
                "agent_total_rmb": agent_cost_rmb,
                "user_total_rmb": user_cost_rmb,
                "combined_total_rmb": agent_cost_rmb + user_cost_rmb,
                "avg_rmb_per_sim": (agent_cost_rmb + user_cost_rmb) / num_sims if num_sims else None,
            },
        },
        "messages": {
            **{k: int(v) for k, v in msg_counts.items()},
            "avg_messages_per_sim": (
                sum(int(v) for v in msg_counts.values()) / num_sims if num_sims else None
            ),
        },
        "tool_use": {
            **{k: int(v) for k, v in tool_stats.items()},
            "avg_assistant_env_tool_calls_per_sim": (
                tool_stats["assistant_env_tool_calls_total"] / num_sims if num_sims else None
            ),
            "avg_llm_visible_tool_calls_per_sim": (
                tool_stats["llm_visible_tool_calls_total"] / num_sims if num_sims else None
            ),
            "avg_assistant_env_tool_calls_per_tool_turn": (
                tool_stats["assistant_env_tool_calls_total"] / msg_counts["assistant_tool_turns_total"]
                if msg_counts["assistant_tool_turns_total"]
                else None
            ),
            "assistant_multitool_turn_rate": (
                tool_stats["assistant_multitool_turns_total"] / msg_counts["assistant_tool_turns_total"]
                if msg_counts["assistant_tool_turns_total"]
                else None
            ),
            "trp_avg_batch_size_on_batch_turns": (
                tool_stats["trp_batch_env_tool_calls_acc"] / tool_stats["trp_batch_turns_total"]
                if tool_stats["trp_batch_turns_total"]
                else None
            ),
        },
    }


def _compare_method_metrics(baseline: dict[str, Any], trp: dict[str, Any]) -> dict[str, Any]:
    def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
        if new is None or old in (None, 0):
            return None
        return (new - old) / old

    comp = {
        "success_rate_delta": pct_change(
            trp["simulations"]["success_rate"], baseline["simulations"]["success_rate"]
        ),
        "avg_reward_delta": pct_change(
            trp["simulations"]["avg_reward"], baseline["simulations"]["avg_reward"]
        ),
        "duration_avg_delta": pct_change(
            trp["durations_sec"]["avg"], baseline["durations_sec"]["avg"]
        ),
        "duration_p95_delta": pct_change(
            trp["durations_sec"]["p95"], baseline["durations_sec"]["p95"]
        ),
        "total_tokens_delta": pct_change(
            trp["tokens"]["total_tokens"], baseline["tokens"]["total_tokens"]
        ),
        "avg_total_tokens_per_sim_delta": pct_change(
            trp["tokens"]["avg_total_tokens_per_sim"], baseline["tokens"]["avg_total_tokens_per_sim"]
        ),
        "avg_env_tool_calls_per_sim_delta": pct_change(
            trp["tool_use"]["avg_assistant_env_tool_calls_per_sim"],
            baseline["tool_use"]["avg_assistant_env_tool_calls_per_sim"],
        ),
        "avg_llm_visible_tool_calls_per_sim_delta": pct_change(
            trp["tool_use"]["avg_llm_visible_tool_calls_per_sim"],
            baseline["tool_use"]["avg_llm_visible_tool_calls_per_sim"],
        ),
        "derived_cost_total_rmb_delta": pct_change(
            trp["costs"]["derived_rmb_from_tokens"]["combined_total_rmb"],
            baseline["costs"]["derived_rmb_from_tokens"]["combined_total_rmb"],
        ),
        "pass_hat_k_delta": {},
    }
    for k in sorted(set(baseline["pass_hat_k"]) & set(trp["pass_hat_k"]), key=int):
        comp["pass_hat_k_delta"][k] = pct_change(trp["pass_hat_k"][k], baseline["pass_hat_k"][k])
    return comp


def _print_run_summary(summary: dict[str, Any]) -> None:
    print("\n=== Tau2 TRP vs Traditional Summary ===")
    print(
        f"Domains: {', '.join(summary['meta']['domains'])} | split={summary['meta']['split']} | "
        f"num_trials={summary['meta']['num_trials']}"
    )
    print(f"Agent model: {summary['meta']['agent_model']} | User model: {summary['meta']['user_model']}")
    est = summary.get("estimate")
    if est and est.get("estimated_tokens"):
        print(f"Estimated total tokens (sample-based): {est['estimated_tokens']['total_tokens']:,}")

    for method_key in ("traditional", "trp"):
        m = summary["methods"][method_key]["overall"]
        print(f"\n[{method_key}]")
        print(
            f"success={m['simulations']['successful_count']}/{m['simulations']['count']} "
            f"({m['simulations']['success_rate']:.2%}) avg_reward={m['simulations']['avg_reward']:.4f}"
        )
        print(
            f"duration avg/p95={m['durations_sec']['avg']:.2f}s/{m['durations_sec']['p95']:.2f}s | "
            f"tokens total={m['tokens']['total_tokens']:,} avg/sim={m['tokens']['avg_total_tokens_per_sim']:.0f}"
        )
        print(
            f"env_tool_calls avg/sim={m['tool_use']['avg_assistant_env_tool_calls_per_sim']:.2f} | "
            f"llm_visible avg/sim={m['tool_use']['avg_llm_visible_tool_calls_per_sim']:.2f}"
        )
        print(f"derived_cost~RMB {m['costs']['derived_rmb_from_tokens']['combined_total_rmb']:.2f}")
        if m["pass_hat_k"]:
            print(
                ", ".join(
                    f"pass^{k}={v:.4f}"
                    for k, v in sorted(m["pass_hat_k"].items(), key=lambda kv: int(kv[0]))
                )
            )

    comp = summary["comparison"]["overall_trp_vs_traditional"]
    print("\n[delta TRP vs traditional]")
    for key in (
        "success_rate_delta",
        "duration_avg_delta",
        "duration_p95_delta",
        "total_tokens_delta",
        "avg_llm_visible_tool_calls_per_sim_delta",
        "derived_cost_total_rmb_delta",
    ):
        v = comp.get(key)
        if v is not None:
            print(f"{key}: {v:+.2%}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a fair Tau2 comparison: traditional LLM agent vs TRP-style router agent."
    )
    p.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    p.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    p.add_argument("--domains", nargs="+", default=["airline", "retail"], choices=["airline", "retail"])
    p.add_argument("--split", type=str, default="base")
    p.add_argument("--num-trials", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--max-errors", type=int, default=10)
    p.add_argument("--max-concurrency", type=int, default=1)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--log-level", type=str, default="ERROR")
    p.add_argument(
        "--baseline-agent",
        type=str,
        default="llm_agent",
        choices=["llm_agent", "llm_agent_parallel_hint"],
        help="Traditional baseline agent.",
    )
    p.add_argument("--trp-agent", type=str, default="trp_router_agent")
    p.add_argument("--user-impl", type=str, default="user_simulator")
    p.add_argument("--agent-model", type=str, default=None)
    p.add_argument("--user-model", type=str, default=None)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--price-input-rmb-per-mtoken", type=float, default=2.0)
    p.add_argument("--price-output-rmb-per-mtoken", type=float, default=3.0)
    p.add_argument("--token-budget-limit", type=int, default=120_000_000)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--estimate-only", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.domains = list(dict.fromkeys(args.domains))

    env = load_simple_env(args.env_file)
    agent_model, agent_llm_args = _make_litellm_model_and_args_from_env(
        env=env,
        model_id=args.agent_model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    user_model, user_llm_args = _make_litellm_model_and_args_from_env(
        env=env,
        model_id=(args.user_model or args.agent_model or env.get("DEEPSEEK_MODEL_ID")),
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    pricing = Pricing(args.price_input_rmb_per_mtoken, args.price_output_rmb_per_mtoken)

    estimate = _estimate_from_tau2_official_samples(
        tau2_root=args.tau2_root,
        domains=args.domains,
        num_trials=args.num_trials,
        methods_count=2,
    )
    if estimate is not None:
        est_prompt = estimate["estimated_tokens"]["prompt_tokens"]
        est_completion = estimate["estimated_tokens"]["completion_tokens"]
        est_cost = pricing.cost_rmb(est_prompt, est_completion)
        estimate["estimated_cost_rmb"] = {
            "derived_total_rmb": est_cost,
            "derived_total_rmb_plus_30pct_margin": est_cost * 1.3,
            "pricing": {
                "input_rmb_per_mtoken": pricing.input_rmb_per_mtoken,
                "output_rmb_per_mtoken": pricing.output_rmb_per_mtoken,
            },
        }
        estimate["budget_check"] = {
            "token_budget_limit": args.token_budget_limit,
            "estimated_total_tokens": estimate["estimated_tokens"]["total_tokens"],
            "fits_budget": estimate["estimated_tokens"]["total_tokens"] <= args.token_budget_limit,
            "usage_ratio": (
                estimate["estimated_tokens"]["total_tokens"] / args.token_budget_limit
                if args.token_budget_limit
                else None
            ),
        }
        print(
            f"Sample-based estimate: total_tokens~{estimate['estimated_tokens']['total_tokens']:,}, "
            f"cost~RMB {est_cost:.2f}, budget_usage~{estimate['budget_check']['usage_ratio']:.1%}"
        )
    else:
        print("Sample-based estimate unavailable.")

    if args.estimate_only:
        print(
            json.dumps(
                {
                    "meta": {
                        "domains": args.domains,
                        "split": args.split,
                        "num_trials": args.num_trials,
                        "agent_model": agent_model,
                        "user_model": user_model,
                        "base_url": agent_llm_args.get("api_base"),
                        "api_key_masked": mask_secret(agent_llm_args.get("api_key")),
                    },
                    "estimate": estimate,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    ensure_tau2_importable(args.tau2_root)
    patch_tau2_deepseek_reasoner_compat()
    patch_tau2_empty_assistant_output_compat()
    register_custom_agents()

    from tau2.data_model.simulation import Results

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = args.results_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Running Tau2 benchmark comparison -> {run_root}")
    print(f"Agent model: {agent_model}")
    print(f"User model:  {user_model}")
    print(f"API base:    {agent_llm_args.get('api_base')}")
    print(f"API key:     {mask_secret(agent_llm_args.get('api_key'))}")
    if "deepseek-reasoner" in agent_model:
        print("Reasoner compat patch: ENABLED (preserve assistant reasoning_content in history)")
        print("Empty assistant retry patch: ENABLED (retry provider empty replies)")
    print(
        "Fairness config: "
        f"domains={args.domains}, split={args.split}, num_trials={args.num_trials}, "
        f"seed={args.seed}, max_steps={args.max_steps}, max_errors={args.max_errors}, "
        f"max_concurrency={args.max_concurrency}"
    )

    methods_cfg = {
        "traditional": {"agent_name": args.baseline_agent},
        "trp": {"agent_name": args.trp_agent},
    }
    method_domain_results: dict[str, list[tuple[str, Any]]] = {"traditional": [], "trp": []}
    task_ids_by_domain_and_method: dict[str, dict[str, list[str]]] = defaultdict(dict)
    raw_result_paths: dict[str, dict[str, str]] = defaultdict(dict)

    for method_key, cfg in methods_cfg.items():
        for domain in args.domains:
            save_path = run_root / f"{method_key}_{domain}_{args.split}_trials{args.num_trials}.json"
            print(f"\n[{method_key}] domain={domain} -> {save_path.name}")
            results, task_ids = _run_tau2_domain(
                domain=domain,
                split=args.split,
                agent_name=cfg["agent_name"],
                user_name=args.user_impl,
                llm_agent_model=agent_model,
                llm_agent_args=agent_llm_args,
                llm_user_model=user_model,
                llm_user_args=user_llm_args,
                num_trials=args.num_trials,
                max_steps=args.max_steps,
                max_errors=args.max_errors,
                max_concurrency=args.max_concurrency,
                seed=args.seed,
                save_path=save_path,
                log_level=args.log_level,
            )
            if not isinstance(results, Results):
                print(f"Warning: unexpected results type for {method_key}/{domain}: {type(results)}")
            method_domain_results[method_key].append((domain, results))
            task_ids_by_domain_and_method[domain][method_key] = task_ids
            raw_result_paths[method_key][domain] = str(save_path)

    fairness_checks = {}
    for domain in args.domains:
        t_ids = task_ids_by_domain_and_method[domain].get("traditional", [])
        r_ids = task_ids_by_domain_and_method[domain].get("trp", [])
        fairness_checks[domain] = {
            "same_task_ids": t_ids == r_ids,
            "task_count": len(t_ids),
        }
        if t_ids != r_ids:
            raise RuntimeError(f"Task list mismatch for domain={domain}")

    methods_summary: dict[str, Any] = {}
    for method_key in ("traditional", "trp"):
        per_domain = {}
        for domain, results in method_domain_results[method_key]:
            per_domain[domain] = _aggregate_metrics(
                [(domain, results)],
                pricing=pricing,
                budget_tokens=args.token_budget_limit,
            )
        overall = _aggregate_metrics(
            method_domain_results[method_key],
            pricing=pricing,
            budget_tokens=args.token_budget_limit,
        )
        methods_summary[method_key] = {
            "agent_name": methods_cfg[method_key]["agent_name"],
            "result_paths": raw_result_paths[method_key],
            "domains": per_domain,
            "overall": overall,
        }

    summary = {
        "meta": {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "domains": args.domains,
            "split": args.split,
            "num_trials": args.num_trials,
            "max_steps": args.max_steps,
            "max_errors": args.max_errors,
            "max_concurrency": args.max_concurrency,
            "seed": args.seed,
            "baseline_agent": args.baseline_agent,
            "trp_agent": args.trp_agent,
            "user_impl": args.user_impl,
            "agent_model": agent_model,
            "user_model": user_model,
            "llm_args_agent": {**agent_llm_args, "api_key": mask_secret(agent_llm_args.get("api_key"))},
            "llm_args_user": {**user_llm_args, "api_key": mask_secret(user_llm_args.get("api_key"))},
            "pricing": {
                "input_rmb_per_mtoken": pricing.input_rmb_per_mtoken,
                "output_rmb_per_mtoken": pricing.output_rmb_per_mtoken,
            },
            "token_budget_limit": args.token_budget_limit,
            "tau2_root": str(args.tau2_root),
            "results_dir": str(run_root),
        },
        "estimate": estimate,
        "fairness": {
            "same_domain_task_lists_between_methods": fairness_checks,
            "same_model_and_user_model": True,
            "same_seed": True,
            "same_num_trials": True,
            "same_max_steps": True,
            "same_max_errors": True,
            "same_task_split": True,
            "note": (
                "Both methods run through Tau2 official run_tasks/evaluation with identical domains, tasks, "
                "num_trials, seed, limits, user simulator, and LLM parameters. Only agent implementation differs."
            ),
        },
        "methods": methods_summary,
        "comparison": {
            "overall_trp_vs_traditional": _compare_method_metrics(
                methods_summary["traditional"]["overall"],
                methods_summary["trp"]["overall"],
            ),
            "by_domain_trp_vs_traditional": {
                domain: _compare_method_metrics(
                    methods_summary["traditional"]["domains"][domain],
                    methods_summary["trp"]["domains"][domain],
                )
                for domain in args.domains
            },
        },
    }

    summary_path = run_root / "summary_compare_tau2_trp_vs_traditional.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_run_summary(summary)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
