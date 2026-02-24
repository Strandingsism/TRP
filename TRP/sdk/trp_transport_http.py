from __future__ import annotations

from typing import Any, Dict
import requests

from .trp_client import TRPTransport


class HttpTRPTransport(TRPTransport):
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def send_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        # 统一单端点，保持“LLM 只调用一个工具”的风格
        resp = requests.post(
            f"{self.base_url}/trp/frame",
            json=frame,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()
