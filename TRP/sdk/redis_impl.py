from __future__ import annotations

import base64
import json
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .router_interfaces import IdempotencyStore, SessionManager, RuntimeStateStore


class RedisConnectionError(Exception):
    pass


class RedisCommandError(Exception):
    pass


class OrderViolationError(Exception):
    def __init__(self, expected_seq: int, got_seq: int):
        super().__init__(f"seq out of order: expected={expected_seq}, got={got_seq}")
        self.expected_seq = expected_seq
        self.got_seq = got_seq


class DuplicateFrameError(Exception):
    def __init__(self, frame_id: str):
        super().__init__(f"duplicate frame: {frame_id}")
        self.frame_id = frame_id


@dataclass(frozen=True)
class RedisURL:
    host: str
    port: int
    db: int
    password: Optional[str]


def parse_redis_url(url: str) -> RedisURL:
    p = urlparse(url)
    if p.scheme != "redis":
        raise ValueError(f"unsupported redis URL scheme: {p.scheme}")
    host = p.hostname or "127.0.0.1"
    port = int(p.port or 6379)
    db = 0
    if p.path and p.path != "/":
        try:
            db = int(p.path.lstrip("/"))
        except ValueError:
            raise ValueError(f"invalid redis db in URL: {p.path}") from None
    password = p.password
    return RedisURL(host=host, port=port, db=db, password=password)


class RedisRESPClient:
    """
    极简 RESP2 客户端（标准库实现），避免引入 redis-py 依赖。
    目标：跨 Windows/Linux 可运行，用于当前原型的持久化验证。
    """

    def __init__(self, url: str, *, timeout_sec: float = 2.0):
        self._cfg = parse_redis_url(url)
        self._timeout_sec = float(timeout_sec)

    def ping(self) -> bool:
        return self.execute("PING") == "PONG"

    def execute(self, *parts: Any) -> Any:
        try:
            with socket.create_connection((self._cfg.host, self._cfg.port), timeout=self._timeout_sec) as sock:
                sock.settimeout(self._timeout_sec)
                f = sock.makefile("rwb")
                if self._cfg.password:
                    self._send(f, "AUTH", self._cfg.password)
                    self._read(f)
                if self._cfg.db:
                    self._send(f, "SELECT", self._cfg.db)
                    self._read(f)
                self._send(f, *parts)
                return self._read(f)
        except (OSError, EOFError, TimeoutError) as e:
            raise RedisConnectionError(str(e)) from e

    def _send(self, f: Any, *parts: Any) -> None:
        f.write(f"*{len(parts)}\r\n".encode("utf-8"))
        for p in parts:
            if isinstance(p, bytes):
                b = p
            elif isinstance(p, (int, float)):
                b = str(p).encode("utf-8")
            else:
                b = str(p).encode("utf-8")
            f.write(f"${len(b)}\r\n".encode("utf-8"))
            f.write(b)
            f.write(b"\r\n")
        f.flush()

    def _read(self, f: Any) -> Any:
        prefix = f.read(1)
        if not prefix:
            raise EOFError("redis connection closed")
        if prefix == b"+":
            return self._readline(f).decode("utf-8", errors="replace")
        if prefix == b"-":
            msg = self._readline(f).decode("utf-8", errors="replace")
            raise RedisCommandError(msg)
        if prefix == b":":
            return int(self._readline(f))
        if prefix == b"$":
            n = int(self._readline(f))
            if n == -1:
                return None
            data = self._readexact(f, n)
            crlf = self._readexact(f, 2)
            if crlf != b"\r\n":
                raise EOFError("invalid bulk string terminator")
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data
        if prefix == b"*":
            n = int(self._readline(f))
            if n == -1:
                return None
            return [self._read(f) for _ in range(n)]
        raise RedisConnectionError(f"unsupported RESP prefix: {prefix!r}")

    @staticmethod
    def _readline(f: Any) -> bytes:
        line = f.readline()
        if not line:
            raise EOFError("redis connection closed")
        if not line.endswith(b"\r\n"):
            raise EOFError("invalid RESP line terminator")
        return line[:-2]

    @staticmethod
    def _readexact(f: Any, n: int) -> bytes:
        data = f.read(n)
        if data is None or len(data) != n:
            raise EOFError("short read from redis")
        return data


def _b64_key(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")


class RedisSessionManager(SessionManager):
    _FEATURES = [
        "CATALOG_SYNC",
        "CAP_QUERY",
        "CALL",
        "CALL_BATCH",
        "RESULT_QUERY",
        "PARTIAL_RESULT",
        "ASYNC_CALL",
        "APPROVAL",
    ]

    _CHECK_SEQ_LUA = """
local meta = KEYS[1]
local seen = KEYS[2]
local frame_id = ARGV[1]
local seq = tonumber(ARGV[2])
local ttl_ms = tonumber(ARGV[3])

if redis.call('EXISTS', meta) == 0 then
  return {'ERR', 'UNKNOWN_SESSION'}
end

if redis.call('SISMEMBER', seen, frame_id) == 1 then
  return {'DUP', frame_id}
end

local expected_raw = redis.call('HGET', meta, 'expected_seq')
if not expected_raw then
  return {'ERR', 'UNKNOWN_SESSION'}
end
local expected = tonumber(expected_raw)

if seq ~= expected then
  return {'ORDER', tostring(expected), tostring(seq)}
end

redis.call('SADD', seen, frame_id)
redis.call('HSET', meta, 'expected_seq', tostring(expected + 1))
redis.call('PEXPIRE', meta, ttl_ms)
redis.call('PEXPIRE', seen, ttl_ms)
return {'OK', tostring(expected + 1)}
"""

    def __init__(
        self,
        *,
        redis: RedisRESPClient,
        catalog_epoch_provider: Callable[[], int],
        default_retry_budget: int = 3,
        session_ttl_sec: int = 86400,
        key_prefix: str = "trp",
    ):
        self._redis = redis
        self._catalog_epoch_provider = catalog_epoch_provider
        self._default_retry_budget = int(default_retry_budget)
        self._session_ttl_ms = max(1000, int(session_ttl_sec * 1000))
        self._prefix = key_prefix.strip() or "trp"

    def hello(self, agent_id: str, resume_session_id: Optional[str]) -> Dict[str, Any]:
        if resume_session_id:
            meta = self._hgetall(self._session_meta_key(resume_session_id))
            if meta:
                self._refresh_session_ttl(resume_session_id)
                return {
                    "session_id": resume_session_id,
                    "catalog_epoch": self._catalog_epoch_provider(),
                    "retry_budget": int(meta.get("retry_budget", self._default_retry_budget)),
                    "seq_start": int(meta.get("expected_seq", 1)),
                    "features": list(self._FEATURES),
                }

        sid = f"sess_{uuid.uuid4().hex[:12]}"
        meta_key = self._session_meta_key(sid)
        seen_key = self._session_seen_key(sid)
        now_ts = str(int(time.time()))
        self._redis.execute(
            "HSET",
            meta_key,
            "agent_id", agent_id,
            "expected_seq", "1",
            "retry_budget", str(self._default_retry_budget),
            "created_at", now_ts,
        )
        # 创建空 set（用占位 member 后立刻删除），确保能设置 TTL
        self._redis.execute("SADD", seen_key, "__bootstrap__")
        self._redis.execute("SREM", seen_key, "__bootstrap__")
        self._redis.execute("PEXPIRE", meta_key, self._session_ttl_ms)
        self._redis.execute("PEXPIRE", seen_key, self._session_ttl_ms)

        return {
            "session_id": sid,
            "catalog_epoch": self._catalog_epoch_provider(),
            "retry_budget": self._default_retry_budget,
            "seq_start": 1,
            "features": list(self._FEATURES),
        }

    def check_and_advance_seq(self, session_id: str, seq: int, frame_id: str) -> Dict[str, Any]:
        try:
            resp = self._redis.execute(
                "EVAL",
                self._CHECK_SEQ_LUA,
                2,
                self._session_meta_key(session_id),
                self._session_seen_key(session_id),
                frame_id,
                int(seq),
                self._session_ttl_ms,
            )
        except RedisCommandError as e:
            raise ValueError(f"unknown session_id: {session_id}") from e

        if not isinstance(resp, list) or not resp:
            raise RuntimeError(f"invalid redis seq script response: {resp!r}")

        tag = str(resp[0])
        if tag == "OK":
            next_seq = int(resp[1]) if len(resp) > 1 else int(seq) + 1
            return {"ok": True, "expected_seq_next": next_seq}
        if tag == "DUP":
            raise DuplicateFrameError(frame_id=str(resp[1]) if len(resp) > 1 else frame_id)
        if tag == "ORDER":
            expected = int(resp[1]) if len(resp) > 1 else -1
            got = int(resp[2]) if len(resp) > 2 else int(seq)
            raise OrderViolationError(expected_seq=expected, got_seq=got)
        if tag == "ERR" and len(resp) > 1 and str(resp[1]) == "UNKNOWN_SESSION":
            raise ValueError(f"unknown session_id: {session_id}")
        raise RuntimeError(f"unexpected redis seq script response: {resp!r}")

    def _session_meta_key(self, sid: str) -> str:
        return f"{self._prefix}:sess:{sid}:meta"

    def _session_seen_key(self, sid: str) -> str:
        return f"{self._prefix}:sess:{sid}:seen"

    def _refresh_session_ttl(self, sid: str) -> None:
        self._redis.execute("PEXPIRE", self._session_meta_key(sid), self._session_ttl_ms)
        self._redis.execute("PEXPIRE", self._session_seen_key(sid), self._session_ttl_ms)

    def _hgetall(self, key: str) -> Dict[str, str]:
        raw = self._redis.execute("HGETALL", key)
        if raw is None:
            return {}
        if not isinstance(raw, list):
            return {}
        out: Dict[str, str] = {}
        for i in range(0, len(raw), 2):
            if i + 1 >= len(raw):
                break
            out[str(raw[i])] = str(raw[i + 1])
        return out


class RedisIdempotencyStore(IdempotencyStore):
    def __init__(
        self,
        *,
        redis: RedisRESPClient,
        key_prefix: str = "trp",
    ):
        self._redis = redis
        self._prefix = key_prefix.strip() or "trp"

    def get(self, cap_id: str, idempotency_key: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.execute("GET", self._key(cap_id, idempotency_key))
        if raw is None:
            return None
        if not isinstance(raw, str):
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def put(self, cap_id: str, idempotency_key: str, result_payload: Dict[str, Any], ttl_sec: int = 86400) -> None:
        ttl_ms = max(1000, int(ttl_sec * 1000))
        body = json.dumps(result_payload, ensure_ascii=False, separators=(",", ":"))
        self._redis.execute("SET", self._key(cap_id, idempotency_key), body, "PX", ttl_ms)

    def _key(self, cap_id: str, idempotency_key: str) -> str:
        return f"{self._prefix}:idem:{_b64_key(cap_id)}:{_b64_key(idempotency_key)}"


class RedisRuntimeStateStore(RuntimeStateStore):
    """
    RouterService 运行时状态持久化（单条 JSON 文档）
    - call_records: depends_on 所需的调用结果状态
    - async_state: 结果查询/partial 事件状态

    说明：
    - 当前为简单 JSON GET/SET，适合单 Router 实例 + 多线程。
    - 若后续扩展到多实例，需要改成 Lua/事务以保证跨进程并发一致性。
    """

    def __init__(self, *, redis: RedisRESPClient, key_prefix: str = "trp"):
        self._redis = redis
        self._prefix = key_prefix.strip() or "trp"

    _MERGE_ASYNC_STATE_LUA = """
local key = KEYS[1]
local patch_json = ARGV[1]
local ttl_ms = tonumber(ARGV[2])

local rec = {}
local raw = redis.call('GET', key)
if raw then
  rec = cjson.decode(raw)
end
local patch = cjson.decode(patch_json)

for k, v in pairs(patch) do
  rec[k] = v
end

if rec['events'] == nil then rec['events'] = {} end
if rec['next_event_id'] == nil then rec['next_event_id'] = 1 end

redis.call('SET', key, cjson.encode(rec), 'PX', ttl_ms)
return cjson.encode(rec)
"""

    _APPEND_ASYNC_EVENT_LUA = """
local key = KEYS[1]
local event_json = ARGV[1]
local event_limit = tonumber(ARGV[2])
local ttl_ms = tonumber(ARGV[3])
local now_ms = tonumber(ARGV[4])
local ttl_window_ms = tonumber(ARGV[5])

local rec = {}
local raw = redis.call('GET', key)
if raw then
  rec = cjson.decode(raw)
end
if rec['events'] == nil then rec['events'] = {} end
if rec['next_event_id'] == nil then rec['next_event_id'] = 1 end

local ev = cjson.decode(event_json)
ev['event_id'] = rec['next_event_id']
if ev['timestamp_ms'] == nil then ev['timestamp_ms'] = now_ms end

table.insert(rec['events'], ev)

local n = #rec['events']
if n > event_limit then
  local overflow = n - event_limit
  for i = 1, overflow do
    table.remove(rec['events'], 1)
  end
  local dropped = tonumber(rec['dropped_event_count'] or 0)
  rec['dropped_event_count'] = dropped + overflow
end

rec['next_event_id'] = tonumber(rec['next_event_id']) + 1
rec['updated_at'] = now_ms
rec['expires_at'] = now_ms + ttl_window_ms

redis.call('SET', key, cjson.encode(rec), 'PX', ttl_ms)
return cjson.encode(rec)
"""

    _CLAIM_ASYNC_EXECUTION_LUA = """
local key = KEYS[1]
local worker_id = ARGV[1]
local lease_ms = tonumber(ARGV[2])
local state_ttl_ms = tonumber(ARGV[3])
local now_ms = tonumber(ARGV[4])

local raw = redis.call('GET', key)
if not raw then
  return cjson.encode({claimed=false, reason='NOT_FOUND'})
end

local rec = cjson.decode(raw)
local status = tostring(rec['status'] or '')

if status == 'SUCCESS' or status == 'FAILED' then
  return cjson.encode({claimed=false, reason='TERMINAL', state=rec})
end

local current_owner = rec['execution_owner']
local current_lease_exp = tonumber(rec['execution_lease_expires_at'] or 0)

if status == 'RUNNING' and current_owner ~= worker_id and current_lease_exp > now_ms then
  return cjson.encode({claimed=false, reason='LEASE_HELD', state=rec})
end

rec['status'] = 'RUNNING'
rec['execution_owner'] = worker_id
rec['execution_lease_expires_at'] = now_ms + lease_ms
rec['updated_at'] = now_ms
if rec['expires_at'] == nil or tonumber(rec['expires_at']) < (now_ms + state_ttl_ms) then
  rec['expires_at'] = now_ms + state_ttl_ms
end
if rec['events'] == nil then rec['events'] = {} end
if rec['next_event_id'] == nil then rec['next_event_id'] = 1 end

redis.call('SET', key, cjson.encode(rec), 'PX', state_ttl_ms)
return cjson.encode({claimed=true, reason='OK', state=rec})
"""

    def get_call_record(self, session_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.execute("GET", self._call_record_key(session_id, call_id))
        return self._loads_obj(raw)

    def put_call_record(self, session_id: str, call_id: str, record: Dict[str, Any], ttl_sec: int) -> None:
        self._set_json(self._call_record_key(session_id, call_id), record, ttl_sec)

    def get_async_call_state(self, session_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.execute("GET", self._async_state_key(session_id, call_id))
        return self._loads_obj(raw)

    def put_async_call_state(self, session_id: str, call_id: str, state: Dict[str, Any], ttl_sec: int) -> None:
        self._set_json(self._async_state_key(session_id, call_id), state, ttl_sec)

    def merge_async_call_state(self, session_id: str, call_id: str, patch: Dict[str, Any], ttl_sec: int) -> Dict[str, Any]:
        ttl_ms = max(1000, int(ttl_sec * 1000))
        patch_json = json.dumps(patch, ensure_ascii=False, separators=(",", ":"))
        raw = self._redis.execute(
            "EVAL",
            self._MERGE_ASYNC_STATE_LUA,
            1,
            self._async_state_key(session_id, call_id),
            patch_json,
            ttl_ms,
        )
        return self._loads_obj(raw) or {}

    def append_async_event(
        self,
        session_id: str,
        call_id: str,
        event: Dict[str, Any],
        *,
        event_limit: int,
        ttl_sec: int,
    ) -> Dict[str, Any]:
        ttl_ms = max(1000, int(ttl_sec * 1000))
        now_ms = int(time.time() * 1000)
        ttl_window_ms = ttl_ms
        event_json = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        raw = self._redis.execute(
            "EVAL",
            self._APPEND_ASYNC_EVENT_LUA,
            1,
            self._async_state_key(session_id, call_id),
            event_json,
            max(1, int(event_limit)),
            ttl_ms,
            now_ms,
            ttl_window_ms,
        )
        return self._loads_obj(raw) or {}

    def claim_async_execution(
        self,
        session_id: str,
        call_id: str,
        *,
        worker_id: str,
        lease_ttl_sec: int,
        state_ttl_sec: int,
    ) -> Dict[str, Any]:
        lease_ms = max(1000, int(lease_ttl_sec * 1000))
        state_ttl_ms = max(1000, int(state_ttl_sec * 1000))
        now_ms = int(time.time() * 1000)
        raw = self._redis.execute(
            "EVAL",
            self._CLAIM_ASYNC_EXECUTION_LUA,
            1,
            self._async_state_key(session_id, call_id),
            worker_id,
            lease_ms,
            state_ttl_ms,
            now_ms,
        )
        return self._loads_obj(raw) or {"claimed": False, "reason": "INVALID_RESPONSE", "state": None}

    def _set_json(self, key: str, obj: Dict[str, Any], ttl_sec: int) -> None:
        ttl_ms = max(1000, int(ttl_sec * 1000))
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        self._redis.execute("SET", key, body, "PX", ttl_ms)

    @staticmethod
    def _loads_obj(raw: Any) -> Optional[Dict[str, Any]]:
        if raw is None or not isinstance(raw, str):
            return None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    def _call_record_key(self, session_id: str, call_id: str) -> str:
        return f"{self._prefix}:rt:callrec:{_b64_key(session_id)}:{_b64_key(call_id)}"

    def _async_state_key(self, session_id: str, call_id: str) -> str:
        return f"{self._prefix}:rt:async:{_b64_key(session_id)}:{_b64_key(call_id)}"
