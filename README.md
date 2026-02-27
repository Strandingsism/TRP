# TRP (Tool Routing Protocol) v2.0

TRP routes all tool use through a single stable interface (`router`). The router handles capability orchestration, policy checks, idempotency, and batch execution so the model does not need to manage a large and unstable tool surface directly.

This `TRP` directory is self-contained for runtime use. It no longer depends on `learn_claude_code` for the Qwen comparison path.

## 1. Benchmark Results

### 1.1 Tau2 (airline + retail, base split, num_trials=4)

Result files:
- `tests/results/tau2/summary_compare_tau2_trp_vs_traditional.json`
- `tests/results/tau2/traditional_airline_base_trials4.json`
- `tests/results/tau2/traditional_retail_base_trials4.json`
- `tests/results/tau2/trp_airline_base_trials4.json`
- `tests/results/tau2/trp_retail_base_trials4.json`

Key metrics (same model, same seed, same task set, only agent style differs):

| Domain | Method | Pass@1 | Avg Tokens / Sim | Avg Duration / Sim |
|---|---:|---:|---:|---:|
| airline | traditional | 0.6200 | 134,635.85 | 229.87s |
| airline | TRP | 0.6300 | 82,801.04 | 255.10s |
| retail | traditional | 0.7697 | 98,501.87 | 169.41s |
| retail | TRP | 0.7829 | 70,065.34 | 203.23s |

Overall (airline + retail, 656 simulations):
- Success rate: traditional `475/656 = 72.41%`, TRP `483/656 = 73.63%` (+1.22pp)
- Total tokens: traditional `71,844,021`, TRP `48,510,003` (about `-32.5%`)
- LLM-visible tool calls: traditional `5,598`, TRP `3,730` (about `-33.4%`)
- End-to-end runtime: TRP is higher in this setup (mainly due to router-side orchestration and batch/async control paths)

### 1.2 Showcase24 (local reproducible workload)

Result files:
- `tests/results/showcase24_full.json`
- `tests/results/showcase24_batch5_smoke.json`

Run configuration (local run on February 27, 2026):
- `task_profile=showcase24`
- `mode=both` (traditional vs TRP)
- `model=qwen-plus`
- `router_url=http://127.0.0.1:8000`

Summary:
- Traditional: `22/24`, success_rate `0.9167`
- TRP: `24/24`, success_rate `1.0000`
- Avg latency: traditional `4919.54ms`, TRP `3282.72ms` (TRP faster here)
- Avg LLM-visible tool calls: traditional `3.17`, TRP `1.58`
- Avg tokens / task: traditional `2006.50`, TRP `1762.42`

Failed tasks (traditional):
- `batch_search_six_queries_eval`
- `batch_search_five_queries_ops`

## 2. Showcase24 Task List (24)

Defined in `tests/compare_qwen_tool_use.py`.

### basic (10)
- `search_titles`
- `search_titles_ptc`
- `search_single_hit`
- `doc_fetch`
- `doc_fetch_alt`
- `sql_read`
- `sql_read_variant`
- `combo_search_then_doc`
- `combo_sql_then_search`
- `combo_three_reads`

### batch (5)
- `batch_search_three_queries`
- `batch_search_four_queries`
- `batch_search_compare_queries`
- `batch_search_then_doc`
- `batch_search_five_queries`

### extra batch (9)
- `batch_search_six_queries_proto`
- `batch_search_six_queries_eval`
- `batch_search_doc_combo_1`
- `batch_search_doc_combo_2`
- `batch_search_doc_combo_3`
- `batch_search_doc_combo_4`
- `batch_search_five_queries_ops`
- `batch_search_five_queries_ctx`
- `batch_search_four_queries_safety`

## 3. Protocol and Architecture

### 3.1 Protocol goals
- Move tool selection, retries, approvals, and idempotency from prompt logic to router logic
- Keep a single stable model-facing interface
- Support sync calls, async calls, batching, and partial result polling

### 3.2 Frame model (TRP v0.1)

Strict validation is implemented in `sdk/frame_validation.py`.

Request frames:
- `HELLO_REQ`
- `CATALOG_SYNC_REQ`
- `CAP_QUERY_REQ`
- `CALL_REQ`
- `CALL_BATCH_REQ`
- `RESULT_QUERY_REQ`

Core fields:
- `trp_version`, `frame_type`, `session_id`, `frame_id`, `trace_id`, `timestamp_ms`, `catalog_epoch`, `seq`, `payload`

Validation invariants:
- `HELLO_REQ` must have `session_id/catalog_epoch/seq = null`
- All other request frames require valid `session_id + catalog_epoch + seq`
- Pydantic strict validation with `extra=forbid` prevents silent field drift

### 3.3 Router workflow

Main implementation: `sdk/router_service.py`  
Service entrypoint: `app/app.py`

Simplified flow:
1. `HELLO_REQ` creates or resumes a session and returns `session_id / catalog_epoch / seq_start / features`
2. `CATALOG_SYNC_REQ` returns capability aliases (`idx <-> cap_id`)
3. `CALL_REQ` / `CALL_BATCH_REQ` executes capability calls
4. Policy, adapter, executor, and result shaping run in sequence
5. For async calls, router returns ACK and later serves terminal or partial results via `RESULT_QUERY_REQ`

Core components:
- Capability registry: `sdk/in_memory_impl.py`
- Policy engine: `sdk/basic_policy.py`
- Adapter/executor/shaper: `sdk/basic_adapter_executor.py`
- Redis-backed persistence: `sdk/redis_impl.py`

### 3.4 Core reliability mechanisms
- Ordering guard: `seq` check, out-of-order requests return NACK
- Replay protection: dedupe by `frame_id`
- Idempotency: cache by `idempotency_key`
- Catalog consistency: detect stale `catalog_epoch` and force resync
- Approval policy: `HIGH/CRITICAL` capabilities can require `approval_token`
- Batch execution: `CALL_BATCH_REQ` supports parallel and sequential modes
- Async continuation: `CALL_REQ(execution_mode=ASYNC)` + `RESULT_QUERY_REQ`

## 4. Reproducibility

### 4.1 Environment setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` from template:
```bash
cp .env.example .env
```

3. Prepare `.env` (minimum):
```bash
TRP_API_KEY=...
TRP_BASE_URL=...
TRP_MODEL_ID=...
```

4. On Windows PowerShell, force UTF-8 output:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

### 4.2 Start TRP router

```bash
uvicorn app.app:app --host 127.0.0.1 --port 8000
```

Health checks:
```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

### 4.3 Run with Docker

TRP does not require Docker to run, but Docker is useful for consistent startup and Redis persistence.

1. Router only (in-memory state):
```bash
docker run --rm -it -p 8000:8000 \
  -v ${PWD}:/app -w /app \
  python:3.12-slim \
  sh -lc "pip install fastapi uvicorn pydantic requests python-dotenv openai && uvicorn app.app:app --host 0.0.0.0 --port 8000"
```

Windows PowerShell equivalent:
```powershell
docker run --rm -it -p 8000:8000 `
  -v ${PWD}:/app -w /app `
  python:3.12-slim `
  sh -lc "pip install fastapi uvicorn pydantic requests python-dotenv openai && uvicorn app.app:app --host 0.0.0.0 --port 8000"
```

2. Start Redis (for persistent runtime state):
```bash
docker run -d --name trp-redis -p 6379:6379 redis:7-alpine
```

3. Router with Redis backend:
```bash
TRP_STATE_BACKEND=redis TRP_REDIS_URL=redis://127.0.0.1:6379/0 \
uvicorn app.app:app --host 127.0.0.1 --port 8000
```

4. Cleanup:
```bash
docker rm -f trp-redis
```

### 4.4 Run Showcase24 comparison

```bash
python tests/compare_qwen_tool_use.py \
  --task-profile showcase24 \
  --mode both \
  --router-url http://127.0.0.1:8000 \
  --out tests/results/showcase24_full.json
```

### 4.5 Run Tau2 (airline + retail)

Default local layout expected by the script:
- Tau2 source: `TRP/data/tau2-bench`
- Env file: `TRP/.env`

```bash
python tests/compare_tau2_benchmark.py \
  --domains airline retail \
  --split base \
  --num-trials 4 \
  --max-concurrency 2 \
  --baseline-agent llm_agent
```

If your Tau2 folder is elsewhere, pass `--tau2-root` explicitly.

Output summary:
- `tests/results/tau2/summary_compare_tau2_trp_vs_traditional.json`

## 5. Fairness Notes for Comparison

For Tau2 comparisons, keep these fixed:
- Same task list (`domain/split`)
- Same `num_trials` and `seed`
- Same user simulator
- Same model and temperature
- Same max steps and max errors

The only variable should be the agent interface:
- traditional: LLM directly calls multiple tools
- TRP: LLM calls a single router tool
