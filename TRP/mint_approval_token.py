from __future__ import annotations

import argparse
import json

from sdk.approval_tokens import mint_approval_token


def main() -> int:
    p = argparse.ArgumentParser(description="Mint TRP signed approval token")
    p.add_argument("--secret", required=True, help="HMAC secret")
    p.add_argument("--cap-id", required=True, help="Capability ID to bind")
    p.add_argument("--args-json", default=None, help="Canonical args JSON to bind")
    p.add_argument("--args-file", default=None, help="Path to JSON file containing canonical args")
    p.add_argument("--ttl-sec", type=int, default=300, help="Token TTL seconds")
    p.add_argument("--subject", default=None, help="Optional subject")
    ns = p.parse_args()

    if not ns.args_json and not ns.args_file:
        raise SystemExit("one of --args-json or --args-file is required")
    if ns.args_json and ns.args_file:
        raise SystemExit("use only one of --args-json or --args-file")

    if ns.args_file:
        # PowerShell 5 `Set-Content -Encoding utf8` 会写入 BOM；utf-8-sig 可兼容读取
        with open(ns.args_file, "r", encoding="utf-8-sig") as f:
            args = json.load(f)
    else:
        args = json.loads(ns.args_json)
    if not isinstance(args, dict):
        raise SystemExit("--args-json must decode to a JSON object")

    token = mint_approval_token(
        secret=ns.secret,
        cap_id=ns.cap_id,
        args=args,
        ttl_sec=ns.ttl_sec,
        subject=ns.subject,
    )
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
