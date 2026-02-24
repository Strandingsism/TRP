---

# TRP v0.1 协议草案

## 1. 设计目标

### 1.1 核心目标

1. **单一入口**：LLM 侧只暴露一个工具/函数（`router`）
2. **能力可见**：LLM 能看到 router 后面的能力目录（不是黑盒）
3. **短地址调用**：用 `idx` 降低 token 和歧义
4. **安全可控**：参数校验、策略审批、审计统一在 router 层
5. **可重试可恢复**：支持乱序/临时失败重试（类似链路层重传思想）
6. **可与 PTC 融合**：LLM 在 Python 中通过 `router` 调用 TRP

### 1.2 非目标（v0.1 不做）

1. 不定义跨 Router 的路由（只做单 Router 域）
2. 不定义复杂分布式一致性（只做单会话一致性）
3. 不定义工具侧传输协议（HTTP/gRPC/MCP 都可）
4. 不做多租户权限系统细节（只留字段和钩子）

---

## 2. 术语定义

### 2.1 参与方

* **Agent**：LLM 执行体（通常在 PTC 的 Python 环境里）
* **Router**：TRP 协议终端，负责能力目录、参数对齐、策略和转发
* **Registry**：能力注册表（Router 内部组件）
* **Policy Engine**：策略引擎（权限/风险/审批）
* **Adapter**：能力适配器（canonical args → tool-native args）
* **Executor**：真实工具执行器（HTTP API / DB / CLI / MCP）

### 2.2 标识

* **`cap_id`**：能力全局唯一 ID（稳定，不随会话变化）
* **`idx`**：会话内短地址（短编号，可随 `catalog_epoch` 更新）
* **`catalog_epoch`**：能力目录版本号（会话内目录一致性锚点）
* **`session_id`**：会话标识
* **`call_id`**：一次调用唯一标识
* **`idempotency_key`**：幂等键（用于安全重试）

---

## 3. 架构模型（SDN 对照）

### 3.1 控制平面（Control Plane）

负责“知道有哪些能力、能不能用、怎么映射”：

* 能力注册与更新（Registry）
* 目录同步（Catalog Sync）
* 参数语义模板（Canonical Schema）
* 策略与审批（Policy Engine）
* 能力别名分配（`idx` 分配）

### 3.2 数据平面（Data Plane）

负责“把调用跑起来”：

* 调用验证
* 顺序检查
* 幂等处理
* 下游执行
* 重试与超时
* 结果整形与返回

---

## 4. TRP 通信模型（v0.1）

TRP 使用 **请求-响应帧（frame）** 模型。
在 Agent 侧，通常通过一个 Python SDK 包装，例如：

```python
router.sync_catalog()
router.call(idx=3, cap_id="cap.search.web.v1", args={...}, seq=12, call_id="c12", idempotency_key="...")
router.batch([...])
```

底层这些都会编码成 TRP frame 发给 Router。

---

## 5. 通用帧格式（Envelope）

所有 TRP 帧共享同一个外层结构。

```json
{
  "trp_version": "0.1",
  "frame_type": "CALL_REQ",
  "session_id": "sess_01H...",
  "frame_id": "frm_01H...",
  "trace_id": "trc_01H...",
  "timestamp_ms": 1760000000000,
  "catalog_epoch": 42,
  "seq": 12,
  "payload": { ... }
}
```

### 5.1 通用字段说明

* `trp_version`：协议版本（固定 `"0.1"`）
* `frame_type`：帧类型（见第 6 节）
* `session_id`：会话 ID（由 Router 在 HELLO 阶段分配或复用）
* `frame_id`：帧唯一 ID（用于去重/日志）
* `trace_id`：链路追踪 ID（同一任务可共享）
* `timestamp_ms`：客户端时间戳（毫秒）
* `catalog_epoch`：客户端当前目录版本
* `seq`：会话内单调递增序号（用于顺序控制）
* `payload`：帧具体内容

### 5.2 保留字段（推荐）

可选但建议预留：

* `auth_context`：身份上下文（用户/角色/租户）
* `sdk_version`：Agent SDK 版本
* `compression`：是否压缩 payload（未来扩展）
* `signature`：完整性校验（未来扩展）

---

## 6. 帧类型（Frame Types）

TRP v0.1 分两类：

## 6.1 控制帧（Control Frames）

### 6.1.1 `HELLO_REQ` / `HELLO_RES`

用于建立会话、协商协议版本、拿到初始 `session_id`。

#### `HELLO_REQ.payload`

```json
{
  "agent_id": "agent_claude_ptc",
  "supported_versions": ["0.1"],
  "resume_session_id": null
}
```

#### `HELLO_RES.payload`

```json
{
  "session_id": "sess_01H...",
  "server_version": "0.1",
  "catalog_epoch": 42,
  "retry_budget": 3,
  "seq_start": 1,
  "features": ["CATALOG_SYNC", "CALL", "CALL_BATCH", "APPROVAL"]
}
```

---

### 6.1.2 `CATALOG_SYNC_REQ` / `CATALOG_SYNC_RES`

用于同步能力目录（LLM 看到的“可调用能力清单”）。

#### `CATALOG_SYNC_REQ.payload`

```json
{
  "mode": "FULL",
  "known_epoch": 42
}
```

* `mode`：

  * `FULL`：拉全量目录（首次或失配）
  * `DELTA`：拉增量（未来扩展，v0.1 可返回 full）

#### `CATALOG_SYNC_RES.payload`

```json
{
  "catalog_epoch": 43,
  "alias_table": [
    {
      "idx": 0,
      "cap_id": "cap.search.web.v1",
      "name": "web_search",
      "desc": "Web search over public internet",
      "risk_tier": "LOW",
      "io_class": "READ",
      "arg_template": {
        "query": "string",
        "top_k": "int?"
      },
      "schema_digest": "sha256:abc..."
    },
    {
      "idx": 1,
      "cap_id": "cap.mail.send.v2",
      "name": "send_email",
      "desc": "Send an email to a verified recipient",
      "risk_tier": "HIGH",
      "io_class": "WRITE",
      "arg_template": {
        "to": "email",
        "subject": "string",
        "body": "string"
      },
      "schema_digest": "sha256:def..."
    }
  ],
  "ttl_sec": 600
}
```

> 关键点：LLM 用 `idx` 调用，但必须同时附带 `cap_id`（二次校验），防止 index 漂移事故。

---

### 6.1.3 `CAP_QUERY_REQ` / `CAP_QUERY_RES`（可选但建议）

用于按需获取某个能力的详细参数 schema、例子、限制说明（避免 catalog 太大）。

#### `CAP_QUERY_REQ.payload`

```json
{
  "idx": 1,
  "cap_id": "cap.mail.send.v2",
  "include_examples": true
}
```

#### `CAP_QUERY_RES.payload`

```json
{
  "idx": 1,
  "cap_id": "cap.mail.send.v2",
  "canonical_schema": {
    "type": "object",
    "required": ["to", "subject", "body"],
    "properties": {
      "to": {"type": "string", "format": "email"},
      "subject": {"type": "string", "maxLength": 200},
      "body": {"type": "string", "maxLength": 20000}
    }
  },
  "policy_hints": {
    "requires_approval": true,
    "idempotency_required": true
  },
  "examples": [
    {
      "args": {"to": "a@example.com", "subject": "hello", "body": "test"}
    }
  ]
}
```

---

## 6.2 数据帧（Data Frames）

### 6.2.1 `CALL_REQ`

单次能力调用请求（v0.1 核心帧）。

#### `CALL_REQ.payload`

```json
{
  "call_id": "call_00012",
  "idempotency_key": "idem_9a3f...",
  "idx": 1,
  "cap_id": "cap.mail.send.v2",
  "depends_on": [],
  "attempt": 1,
  "timeout_ms": 15000,
  "approval_token": null,
  "args": {
    "to": "a@example.com",
    "subject": "hello",
    "body": "test"
  }
}
```

#### 字段约束

* `call_id`：会话内唯一，必填
* `idempotency_key`：

  * `WRITE/HIGH` 能力必填
  * `READ/LOW` 可选
* `idx` + `cap_id`：都必填（双校验）
* `depends_on`：依赖的 `call_id` 列表（v0.1 可只支持空或已完成依赖）
* `attempt`：当前尝试次数（初始为 1）
* `approval_token`：

  * 高风险能力可能必填
  * 未获审批时应返回 `NACK`
* `args`：**canonical args**（由 Router 内部映射到具体工具）

---

### 6.2.2 `ACK`

表示 Router 已接收并接受处理（不代表执行成功）。

#### `ACK.payload`

```json
{
  "ack_of_frame_id": "frm_01H...",
  "ack_of_call_id": "call_00012",
  "status": "ACCEPTED",
  "expected_seq_next": 13
}
```

`ACK` 主要用于：

* 顺序确认
* 减少重复发送
* 支持异步执行（未来扩展）

---

### 6.2.3 `NACK`

拒绝处理或调用失败前置检查未通过。

#### `NACK.payload`

```json
{
  "nack_of_frame_id": "frm_01H...",
  "nack_of_call_id": "call_00012",
  "error_class": "ORDER_VIOLATION",
  "error_code": "TRP_1002",
  "message": "seq out of order",
  "retryable": true,
  "retry_hint": {
    "expected_seq": 12,
    "backoff_ms": 200
  }
}
```

---

### 6.2.4 `RESULT`

调用完成，返回结果（高信号格式）。

#### `RESULT.payload`

```json
{
  "call_id": "call_00012",
  "idx": 1,
  "cap_id": "cap.mail.send.v2",
  "status": "SUCCESS",
  "result": {
    "summary": "Email sent successfully",
    "data": {
      "message_id": "msg_abc123"
    }
  },
  "usage": {
    "router_ms": 23,
    "adapter_ms": 2,
    "executor_ms": 19
  }
}
```

#### `result` 设计原则

* `summary`：给 LLM 的高信号摘要（短）
* `data`：结构化关键结果（裁剪后）
* 不要返回大量低层字段（headers/raw/blob）除非显式请求

---

### 6.2.5 `PARTIAL_RESULT`（可选）

长任务可分块返回（v0.1 可选支持）。

* 适合分页查询、批处理部分完成
* 最终仍以 `RESULT` 收口

---

### 6.2.6 `CALL_BATCH_REQ` / `CALL_BATCH_RES`（强烈建议 v0.1 支持）

这是你“超过纯 PTC”的关键之一。LLM 发一批调用，Router 统一执行与策略处理。

#### `CALL_BATCH_REQ.payload`

```json
{
  "batch_id": "batch_001",
  "mode": "PARALLEL",
  "max_concurrency": 4,
  "calls": [
    {
      "call_id": "call_101",
      "idempotency_key": null,
      "idx": 0,
      "cap_id": "cap.search.web.v1",
      "attempt": 1,
      "timeout_ms": 8000,
      "args": {"query": "TRP protocol draft"}
    },
    {
      "call_id": "call_102",
      "idempotency_key": null,
      "idx": 0,
      "cap_id": "cap.search.web.v1",
      "attempt": 1,
      "timeout_ms": 8000,
      "args": {"query": "PTC Anthropic"}
    }
  ]
}
```

#### `CALL_BATCH_RES.payload`

```json
{
  "batch_id": "batch_001",
  "status": "PARTIAL_SUCCESS",
  "results": [
    {
      "call_id": "call_101",
      "status": "SUCCESS",
      "result": {"summary": "...", "data": {...}}
    },
    {
      "call_id": "call_102",
      "status": "FAILED",
      "error_class": "TRANSIENT",
      "error_code": "TRP_3001",
      "retryable": true
    }
  ]
}
```

---

## 7. 目录与别名机制（`idx` / `cap_id` / `catalog_epoch`）

这是你方案的核心，必须严格。

### 7.1 基本规则

1. `cap_id` 是真实身份（稳定）
2. `idx` 是会话短地址（方便 LLM）
3. `idx` 只在 `catalog_epoch` 内有效
4. 调用必须同时携带 `idx` 和 `cap_id`
5. Router 必须校验 `(catalog_epoch, idx) -> cap_id` 是否匹配

### 7.2 失配处理（非常关键）

如果发生以下任一情况：

* `catalog_epoch` 过期
* `idx` 不存在
* `idx` 对应的 `cap_id` 不匹配

Router 返回：

* `NACK`
* `error_class = "CATALOG_MISMATCH"`
* `retryable = true`
* `retry_hint.action = "SYNC_CATALOG"`

这样可以避免最危险的事故：`idx` 漂移导致误调高风险工具。

---

## 8. 顺序控制与重试（802.11 类比的协议化落地）

你提出“顺序不对就重试”，这里必须严谨定义。

## 8.1 顺序规则（`seq`）

* `seq` 是 **会话级单调递增**
* Router 维护 `expected_seq`
* 收到 `seq < expected_seq`：

  * 如果 `call_id` 已执行过：返回幂等结果（或 ACK duplicate）
  * 否则返回 `NACK DUPLICATE_OR_STALE`
* 收到 `seq > expected_seq`：

  * 返回 `NACK ORDER_VIOLATION`
  * 附带 `expected_seq`

## 8.2 重试规则（`attempt` + `retry_budget`）

### 允许重试的错误类（`retryable=true`）

* `TRANSIENT`（下游超时、网络抖动、短暂不可用）
* `ORDER_VIOLATION`（乱序）
* `CATALOG_MISMATCH`（需先同步目录）

### 不允许自动重试的错误类

* `SCHEMA_MISMATCH`
* `POLICY_DENIED`
* `APPROVAL_REQUIRED`
* `NON_IDEMPOTENT_BLOCKED`

## 8.3 退避策略（Router 提示）

Router 在 `NACK.retry_hint` 给出：

* `backoff_ms`
* `jitter_ms`
* `expected_seq`（如适用）
* `max_attempts`

Agent SDK 应默认执行指数退避：

* `base * 2^(attempt-1) + jitter`

---

## 9. 幂等性与副作用控制（必须）

## 9.1 幂等性要求

对以下能力类型，`idempotency_key` 必填：

* `io_class = WRITE`
* `risk_tier = MEDIUM/HIGH/CRITICAL`

## 9.2 Router 行为

如果重复收到同一 `(cap_id, idempotency_key)`：

* 若首次调用已成功：直接返回缓存的 `RESULT`
* 若首次调用仍在进行：返回 `ACK status=IN_PROGRESS`
* 若首次调用失败且可重试：按策略返回 `NACK/RESULT`

## 9.3 幂等窗口

Router 应为 `idempotency_key` 维护 TTL（例如 24h），避免长期无限增长。

---

## 10. 参数语义对齐（Canonical Args）

这是你方案的护城河，不只是“转发”。

## 10.1 Agent 只传 Canonical Args

Agent 不关心每个工具原生字段名，只按 Router 发布的 canonical schema 传 `args`。

例如：

* Agent 传：`{"query":"...", "top_k": 5}`
* Router Adapter 映射：

  * Tool A：`q`, `limit`
  * Tool B：`keyword`, `page_size`

## 10.2 Schema 摘要与校验

Catalog 中每个能力带：

* `schema_digest`

`CALL_REQ` 可选附带 `schema_digest`，Router 可校验是否 Agent 使用的是旧模板。
若过期，返回 `NACK SCHEMA_MISMATCH` + `CAP_QUERY` 提示。

## 10.3 参数校验层级

Router 至少做三层校验：

1. **结构校验**（schema）
2. **语义校验**（业务约束，例如时间范围、最大数量）
3. **安全校验**（敏感字段、越权字段、注入风险）

---

## 11. 安全与策略（Policy）

## 11.1 风险分级（`risk_tier`）

建议标准化为：

* `LOW`：只读、低敏感
* `MEDIUM`：只读但敏感数据 / 批量操作
* `HIGH`：写操作、外部通知、状态变更
* `CRITICAL`：删除、支付、生产环境更改

## 11.2 审批令牌（`approval_token`）

对 `HIGH/CRITICAL` 能力，Router 可要求审批。

流程：

1. `CALL_REQ` 无 `approval_token`
2. Router 返回 `NACK APPROVAL_REQUIRED`
3. 上层系统（人类或策略服务）签发 `approval_token`
4. Agent 重发 `CALL_REQ`

## 11.3 最小权限执行

Router 应按 `auth_context` + `cap_id` 做权限检查，不允许 Agent 仅凭 `idx` 越权。

---

## 12. 错误模型（Error Model）

## 12.1 错误类（`error_class`）

建议 v0.1 固定这些：

* `TRANSIENT`：临时错误，可重试
* `ORDER_VIOLATION`：顺序错误
* `CATALOG_MISMATCH`：目录版本/映射失配
* `SCHEMA_MISMATCH`：参数不符合 schema
* `POLICY_DENIED`：策略拒绝
* `APPROVAL_REQUIRED`：缺少审批
* `NON_IDEMPOTENT_BLOCKED`：无幂等键且高副作用
* `EXECUTOR_ERROR`：下游工具错误（非临时）
* `INTERNAL_ERROR`：Router 内部异常

## 12.2 错误码（`error_code`）

建议统一命名：

* `TRP_1xxx`：协议层（顺序、目录、版本）
* `TRP_2xxx`：参数与 schema
* `TRP_3xxx`：执行与网络
* `TRP_4xxx`：策略与审批
* `TRP_5xxx`：Router 内部

示例：

* `TRP_1002`：ORDER_VIOLATION
* `TRP_1003`：CATALOG_MISMATCH
* `TRP_2001`：SCHEMA_VALIDATION_FAILED
* `TRP_4002`：APPROVAL_REQUIRED

---

## 13. 结果整形（Result Shaping）

为了兑现“比 PTC 更省上下文”的目标，Router 必须做结果治理。

## 13.1 标准结果结构

```json
{
  "summary": "高信号摘要",
  "data": {...},
  "artifacts": [],
  "warnings": []
}
```

## 13.2 设计规则

* `summary` 尽量短（给 LLM 读）
* `data` 保留结构化关键字段
* 大体量原始数据不要直接返回（可返回引用句柄 `artifact_id`）
* 错误信息统一格式，避免下游异构错误污染上下文

---

## 14. 可观测性（Observability）

你这个系统如果没 trace，很难调。v0.1 就应该内置。

## 14.1 最小审计日志字段

每次调用至少记录：

* `trace_id`
* `session_id`
* `catalog_epoch`
* `seq`
* `call_id`
* `idx`
* `cap_id`
* `idempotency_key`（脱敏/哈希）
* `policy_decision`
* `attempt`
* `latency_ms`
* `result_status`
* `error_class/error_code`（如有）

## 14.2 事件流（建议）

Router 内部建议产生事件：

* `catalog.synced`
* `call.accepted`
* `call.policy_denied`
* `call.executed`
* `call.retry_suggested`
* `call.succeeded`
* `call.failed`

后续你要做 dashboard、回放、回归测试都会很有用。

---

## 15. Agent SDK 行为规范（v0.1）

这是给 LLM 所在 Python 执行环境用的，不是给人写的。

## 15.1 SDK 必须做的事

1. 自动维护 `seq`（单调递增）
2. 自动附带 `catalog_epoch`
3. 自动处理 `CATALOG_MISMATCH`（触发 `sync_catalog()` 后重试）
4. 自动处理 `ORDER_VIOLATION`（按 `expected_seq` 重发）
5. 自动对 `TRANSIENT` 做指数退避重试
6. 对高风险调用缺少 `idempotency_key` 时直接抛错（本地预检）

## 15.2 SDK 不应做的事

1. 不应私自修改 `cap_id`
2. 不应吞掉 `POLICY_DENIED` / `APPROVAL_REQUIRED`
3. 不应绕过 Router 直连工具（除非系统显式允许热路径直连模式）

---

## 16. 典型交互流程（v0.1）

## 16.1 会话建立 + 目录同步

1. Agent → `HELLO_REQ`
2. Router → `HELLO_RES(session_id, catalog_epoch)`
3. Agent → `CATALOG_SYNC_REQ`
4. Router → `CATALOG_SYNC_RES(alias_table)`

---

## 16.2 单次调用（正常）

1. Agent 根据 alias table 选 `idx=3, cap_id=...`
2. Agent 发 `CALL_REQ(seq=12, call_id=..., idempotency_key=...)`
3. Router 校验顺序、目录、参数、策略
4. Router 执行下游工具
5. Router 回 `RESULT`

---

## 16.3 顺序错误（乱序）

1. Agent 发 `CALL_REQ(seq=14)`，但 Router 期望 `13`
2. Router 回 `NACK ORDER_VIOLATION + expected_seq=13`
3. SDK 调整后按正确顺序重发

---

## 16.4 目录失配（index 漂移保护）

1. Agent 发 `CALL_REQ(catalog_epoch=43, idx=7, cap_id=A)`
2. Router 当前 `catalog_epoch=44`，且 `idx=7` 现已映射到 `cap_id=B`
3. Router 回 `NACK CATALOG_MISMATCH`
4. SDK 先 `CATALOG_SYNC_REQ`
5. 重新选择 `idx` 后再发 `CALL_REQ`

---

## 16.5 高风险调用审批

1. Agent 发 `CALL_REQ`（`risk_tier=HIGH`，无 `approval_token`）
2. Router 回 `NACK APPROVAL_REQUIRED`
3. 外部审批系统签发 `approval_token`
4. Agent 带 token 重发
5. Router 执行并返回 `RESULT`

---

## 17. 最小实现建议（MVP）

你要快速做 demo，TRP v0.1 可以先实现这些：

### 17.1 Router 必做接口

* `hello()`
* `sync_catalog()`
* `call()`
* `batch()`（强烈建议）

### 17.2 Router 内部模块

* `SessionManager`（`session_id`, `seq`, `catalog_epoch`）
* `CapabilityRegistry`（`cap_id`, `idx`, schema, risk）
* `PolicyEngine`（权限/审批/幂等规则）
* `AdapterLayer`（canonical args → native args）
* `Executor`（真实工具调用）
* `ResultShaper`（摘要/裁剪）
* `AuditLogger`

### 17.3 先接的 5 类能力（正好覆盖场景）

* 只读搜索（LOW）
* 只读详情查询（LOW）
* 结构化查询（MEDIUM）
* 写操作（HIGH，需要幂等）
* 删除/通知（CRITICAL，需要审批）

<!-- ---

## 18. v0.1 到 v0.2 的自然升级路径（你后面肯定会做）

v0.1 足够跑起来，但我建议你下一步往这些方向升级：

1. **`CATALOG_DELTA`**（增量目录同步）
2. **`CALL_DAG_REQ`**（Router 直接执行 DAG，不只是 batch）
3. **`PARTIAL_RESULT`** 流式返回
4. **能力租约（lease）**：`idx` 绑定 TTL，更稳
5. **策略证明（policy proof）**：返回策略决策摘要，便于审计
6. **artifact handle**：大结果不进上下文，返回句柄给后续引用
7. **热路径直连模式**（可选）：TRP 仍做控制平面，部分低风险工具数据平面直连

--- -->