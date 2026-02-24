from __future__ import annotations

import re
import time
import uuid
from typing import Any, Dict, List, Optional

from .router_interfaces import (
    CapabilityMeta,
    Adapter,
    AdapterRegistry,
    Executor,
    ResultShaper,
)


# =========================
# Adapters（参数校验 + canonical -> native）
# =========================

class SearchAdapter(Adapter):
    def validate_canonical_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> None:
        q = args.get("query")
        if not isinstance(q, str) or not q.strip():
            raise ValueError("`query` is required and must be a non-empty string")
        top_k = args.get("top_k", 5)
        if not isinstance(top_k, int) or top_k <= 0 or top_k > 20:
            raise ValueError("`top_k` must be int in [1, 20]")

    def to_native_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"q": args["query"].strip(), "limit": int(args.get("top_k", 5))}


class FetchAdapter(Adapter):
    def validate_canonical_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> None:
        doc_id = args.get("document_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("`document_id` is required and must be a non-empty string")

    def to_native_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"doc_id": args["document_id"].strip()}


class SQLReadAdapter(Adapter):
    _deny_keywords = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke)\b", re.I)

    def validate_canonical_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> None:
        sql = args.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("`sql` is required and must be a non-empty string")
        if self._deny_keywords.search(sql):
            raise ValueError("only read-only SQL is allowed")
        limit = args.get("limit", 50)
        if not isinstance(limit, int) or limit <= 0 or limit > 500:
            raise ValueError("`limit` must be int in [1, 500]")

    def to_native_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> Dict[str, Any]:
        sql = args["sql"].strip().rstrip(";")
        limit = int(args.get("limit", 50))
        return {"sql": f"{sql} LIMIT {limit}" if "limit" not in sql.lower() else sql, "limit": limit}


class TicketCreateAdapter(Adapter):
    def validate_canonical_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> None:
        title = args.get("title")
        body = args.get("body")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("`title` is required")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("`body` is required")
        if len(title) > 200:
            raise ValueError("`title` too long")
        if len(body) > 5000:
            raise ValueError("`body` too long")
        p = args.get("priority", "medium")
        if p not in {"low", "medium", "high"}:
            raise ValueError("`priority` must be one of low/medium/high")

    def to_native_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": args["title"].strip(),
            "description": args["body"].strip(),
            "priority": args.get("priority", "medium"),
        }


class RecordDeleteAdapter(Adapter):
    def validate_canonical_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> None:
        rt = args.get("resource_type")
        rid = args.get("resource_id")
        if not isinstance(rt, str) or not rt.strip():
            raise ValueError("`resource_type` is required")
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError("`resource_id` is required")
        if len(rt) > 50 or len(rid) > 200:
            raise ValueError("resource_type/resource_id too long")

    def to_native_args(self, cap: CapabilityMeta, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "resource_type": args["resource_type"].strip(),
            "resource_id": args["resource_id"].strip(),
            "reason": args.get("reason", ""),
        }


# =========================
# Adapter Registry
# =========================

class BasicAdapterRegistry(AdapterRegistry):
    def __init__(self):
        self._adapters: Dict[str, Adapter] = {
            "search": SearchAdapter(),
            "fetch": FetchAdapter(),
            "sql_read": SQLReadAdapter(),
            "ticket_create": TicketCreateAdapter(),
            "record_delete": RecordDeleteAdapter(),
        }

    def get(self, adapter_key: str) -> Adapter:
        if adapter_key not in self._adapters:
            raise KeyError(f"adapter not found: {adapter_key}")
        return self._adapters[adapter_key]


# =========================
# Demo Executor（假执行器/内存执行器）
# =========================

class BasicExecutor(Executor):
    """
    一个最小可运行执行器：
    - 搜索/查询返回 mock 数据
    - ticket_create / record_delete 维护内存状态，便于测试幂等和审批
    """
    def __init__(self):
        self._tickets: Dict[str, Dict[str, Any]] = {}
        self._deleted: List[Dict[str, Any]] = []

    def execute(self, cap: CapabilityMeta, native_args: Dict[str, Any], timeout_ms: int) -> Dict[str, Any]:
        # 这里可模拟耗时
        if timeout_ms < 5:
            raise TimeoutError("timeout too small")
        time.sleep(0.01)

        if cap.cap_id == "cap.search.web.v1":
            q = native_args["q"]
            limit = native_args["limit"]
            items = [
                {
                    "title": f"Result {i+1} for {q}",
                    "url": f"https://example.com/search/{i+1}",
                    "snippet": f"Snippet about {q} ({i+1})",
                }
                for i in range(limit)
            ]
            return {"items": items, "total": 1000, "query": q}

        if cap.cap_id == "cap.fetch.doc.v1":
            doc_id = native_args["doc_id"]
            return {
                "document_id": doc_id,
                "title": f"Document {doc_id}",
                "content": f"Mock content for document {doc_id}.",
                "updated_at": "2026-02-24T00:00:00Z",
            }

        if cap.cap_id == "cap.query.sql_read.v1":
            sql = native_args["sql"]
            # 简单 mock：返回伪行数据
            rows = [
                {"id": 1, "name": "alpha", "value": 42},
                {"id": 2, "name": "beta", "value": 17},
            ]
            return {"sql": sql, "rows": rows, "row_count": len(rows)}

        if cap.cap_id == "cap.ticket.create.v1":
            ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
            rec = {
                "ticket_id": ticket_id,
                "title": native_args["title"],
                "description": native_args["description"],
                "priority": native_args["priority"],
                "status": "OPEN",
            }
            self._tickets[ticket_id] = rec
            return rec

        if cap.cap_id == "cap.record.delete.v1":
            rec = {
                "resource_type": native_args["resource_type"],
                "resource_id": native_args["resource_id"],
                "reason": native_args.get("reason", ""),
                "deleted": True,
            }
            self._deleted.append(rec)
            return rec

        raise RuntimeError(f"unsupported capability in executor: {cap.cap_id}")


# =========================
# Result Shaper（高信号返回）
# =========================

class BasicResultShaper(ResultShaper):
    def shape_success(self, cap: CapabilityMeta, raw: Dict[str, Any]) -> Dict[str, Any]:
        if cap.cap_id == "cap.search.web.v1":
            items = raw.get("items", [])
            top_titles = [x.get("title", "") for x in items[:3]]
            return {
                "summary": f"Found {len(items)} results for query '{raw.get('query', '')}'",
                "data": {
                    "items": items,
                    "top_titles": top_titles,
                    "total": raw.get("total"),
                },
                "warnings": [],
                "artifacts": [],
            }

        if cap.cap_id == "cap.fetch.doc.v1":
            return {
                "summary": f"Fetched document {raw.get('document_id')}",
                "data": {
                    "document_id": raw.get("document_id"),
                    "title": raw.get("title"),
                    "content": raw.get("content"),
                    "updated_at": raw.get("updated_at"),
                },
                "warnings": [],
                "artifacts": [],
            }

        if cap.cap_id == "cap.query.sql_read.v1":
            rows = raw.get("rows", [])
            return {
                "summary": f"SQL query returned {raw.get('row_count', len(rows))} rows",
                "data": {
                    "sql": raw.get("sql"),
                    "rows": rows,
                    "row_count": raw.get("row_count", len(rows)),
                },
                "warnings": [],
                "artifacts": [],
            }

        if cap.cap_id == "cap.ticket.create.v1":
            return {
                "summary": f"Ticket {raw.get('ticket_id')} created",
                "data": raw,
                "warnings": [],
                "artifacts": [],
            }

        if cap.cap_id == "cap.record.delete.v1":
            return {
                "summary": f"Deleted {raw.get('resource_type')}:{raw.get('resource_id')}",
                "data": raw,
                "warnings": [],
                "artifacts": [],
            }

        # 默认
        return {
            "summary": "Operation completed",
            "data": raw,
            "warnings": [],
            "artifacts": [],
        }

    def shape_error(self, cap: Optional[CapabilityMeta], exc: Exception) -> Dict[str, Any]:
        return {
            "summary": f"{cap.cap_id if cap else 'unknown'} failed",
            "data": {"error": str(exc)},
            "warnings": [],
            "artifacts": [],
        }
