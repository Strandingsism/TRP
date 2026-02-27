# TRP (Tool Routing Protocol) v2.0

TRP 的核心思路是让 LLM 只调用一个稳定入口（`router`），由 Router 侧负责能力编排、策略校验、幂等与批处理执行，降低模型侧工具选择与参数漂移带来的不稳定性。

## 1. 实测结果（先看结论）

### 1.1 Tau2（airline + retail, base, num_trials=4）

结果文件：
- `results/tau2/summary_compare_tau2_trp_vs_traditional.json`
- `results/tau2/traditional_airline_base_trials4.json`
- `results/tau2/traditional_retail_base_trials4.json`
- `results/tau2/trp_airline_base_trials4.json`
- `results/tau2/trp_retail_base_trials4.json`

关键指标（同模型、同 seed、同任务集，仅 agent 实现不同）：

| Domain | Method | Pass@1 | Avg Tokens / Sim | Avg Duration / Sim |
|---|---:|---:|---:|---:|
| airline | traditional | 0.6200 | 134,635.85 | 229.87s |
| airline | TRP | 0.6300 | 82,801.04 | 255.10s |
| retail | traditional | 0.7697 | 98,501.87 | 169.41s |
| retail | TRP | 0.7829 | 70,065.34 | 203.23s |

总览（airline+retail，共 656 simulations）：
- Success rate: traditional `475/656 = 72.41%`，TRP `483/656 = 73.63%`（+1.22pp）
- Total tokens: traditional `71,844,021`，TRP `48,510,003`（约 `-32.5%`）
- LLM-visible tool calls: traditional `5,598`，TRP `3,730`（约 `-33.4%`）
- 总时长：TRP 更高（主要受路由层编排与 batch/async 策略影响）

### 1.2 Showcase24（本地可复现实验集）

结果文件：
- `results/showcase24_full.json`
- `results/showcase24_batch5_smoke.json`

本次运行配置（2026-02-27，本地）：
- `task_profile=showcase24`
- `mode=both`（traditional vs trp）
- `model=qwen-plus`
- `router_url=http://127.0.0.1:8000`

汇总：
- Traditional: `22/24`，success_rate `0.9167`
- TRP: `24/24`，success_rate `1.0000`
- 平均延迟：traditional `4919.54ms`，TRP `3282.72ms`（TRP 更快）
- 平均 LLM 可见工具调用：traditional `3.17`，TRP `1.58`
- 平均 token / task：traditional `2006.50`，TRP `1762.42`

失败任务（traditional）：
- `batch_search_six_queries_eval`
- `batch_search_five_queries_ops`

## 2. Showcase24 任务构成（24 个）

定义位置：`TRP/tests/compare_qwen_tool_use.py`

### basic（10）
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

### batch（5）
- `batch_search_three_queries`
- `batch_search_four_queries`
- `batch_search_compare_queries`
- `batch_search_then_doc`
- `batch_search_five_queries`

### extra batch（9）
- `batch_search_six_queries_proto`
- `batch_search_six_queries_eval`
- `batch_search_doc_combo_1`
- `batch_search_doc_combo_2`
- `batch_search_doc_combo_3`
- `batch_search_doc_combo_4`
- `batch_search_five_queries_ops`
- `batch_search_five_queries_ctx`
- `batch_search_four_queries_safety`

## 3. TRP 协议与实现

## 3.1 协议目标
- 把“工具发现/选择/重试/审批/幂等”从 LLM prompt 侧下沉到 Router
- 让 LLM 只记一个入口，减少多工具 schema 漂移对行为的影响
- 支持同步 + 异步 + 批处理 + 部分结果查询

## 3.2 帧模型（TRP v0.1）

严格校验定义：`TRP/sdk/frame_validation.py`

请求帧：
- `HELLO_REQ`
- `CATALOG_SYNC_REQ`
- `CAP_QUERY_REQ`
- `CALL_REQ`
- `CALL_BATCH_REQ`
- `RESULT_QUERY_REQ`

关键字段：
- `trp_version`, `frame_type`, `session_id`, `frame_id`, `trace_id`, `timestamp_ms`, `catalog_epoch`, `seq`, `payload`

说明：
- `HELLO_REQ` 必须 `session_id/catalog_epoch/seq` 为空
- 其他请求必须携带有效 `session_id + catalog_epoch + seq`
- Pydantic strict + `extra=forbid`，防止静默吞字段

## 3.3 Router 工作流

主实现：`TRP/sdk/router_service.py`  
服务入口：`TRP/app/app.py`

流程（简化）：
1. `HELLO_REQ` 建立或恢复 session，返回 `session_id / catalog_epoch / seq_start / features`
2. `CATALOG_SYNC_REQ` 同步能力目录（idx <-> cap_id）
3. `CALL_REQ` / `CALL_BATCH_REQ` 执行调用
4. 通过 Policy/Adapter/Executor/Shaper 链路完成校验与执行
5. 若异步执行，返回 ACK，后续用 `RESULT_QUERY_REQ` 拉取最终结果或 `PARTIAL_RESULT`

链路组件：
- Capability registry: `TRP/sdk/in_memory_impl.py`
- Policy engine: `TRP/sdk/basic_policy.py`
- Adapter/Executor/Shaper: `TRP/sdk/basic_adapter_executor.py`
- Redis 持久化实现: `TRP/sdk/redis_impl.py`

## 3.4 核心机制
- 顺序保障：`seq` 检查，乱序返回 NACK
- 重放保护：`frame_id` 去重
- 幂等保障：`idempotency_key` 缓存结果
- 目录一致性：`catalog_epoch` 不一致触发重同步
- 策略审批：`HIGH/CRITICAL` 风险能力可要求 `approval_token`
- 批处理：`CALL_BATCH_REQ` 支持并行/串行执行
- 异步续跑：`CALL_REQ(execution_mode=ASYNC)` + `RESULT_QUERY_REQ`

## 4. 如何复现

## 4.1 环境准备

1. 安装依赖（示例）：
```bash
pip install -r requirements.txt
```

2. 准备 `.env`（至少）：
```bash
QWEN_API_KEY=...
MODEL_ID=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

3. Windows PowerShell 避免中文乱码：
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

## 4.2 启动 TRP Router

```bash
cd TRP
uvicorn app.app:app --host 127.0.0.1 --port 8000
```

健康检查：
```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

## 4.3 跑 Showcase24 对比

```bash
python TRP/tests/compare_qwen_tool_use.py \
  --task-profile showcase24 \
  --mode both \
  --router-url http://127.0.0.1:8000 \
  --out results/showcase24_full.json
```

## 4.4 跑 Tau2（airline + retail）

```bash
python TRP/tests/compare_tau2_benchmark.py \
  --domains airline retail \
  --split base \
  --num-trials 4 \
  --max-concurrency 2 \
  --baseline-agent llm_agent
```

输出汇总：
- `results/tau2/summary_compare_tau2_trp_vs_traditional.json`

## 5. 复现公平性说明

在 Tau2 对比中，以下配置应保持一致：
- 同一 task list（domain/split）
- 同一 `num_trials`/`seed`
- 同一 user simulator
- 同一模型与 temperature
- 同一最大步数与错误阈值

唯一变量应是 agent 形态：
- traditional：LLM 直接多工具调用
- TRP：LLM 通过单 router tool 调用

