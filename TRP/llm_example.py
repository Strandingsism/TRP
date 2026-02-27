from sdk.trp_client import RouterClient
from sdk.trp_transport_http import HttpTRPTransport
from sdk.trp_types import CallSpec

router = RouterClient(HttpTRPTransport("http://localhost:8000"))

# 1) Initialize
router.hello()
catalog = router.sync_catalog()

# 2) Inspect the catalog (LLM can see idx + cap_id + desc)
for c in catalog:
    print(c.idx, c.cap_id, c.name, "-", c.desc)

# 3) Single call (by idx + cap_id)
res = router.call(
    idx=0,
    cap_id="cap.search.web.v1",
    args={"query": "Anthropic advanced tool use"},
    timeout_ms=8000,
)
print(res["result"]["summary"])

# 4) Batch call (the index-based approach shows clear benefits here)
calls = [
    CallSpec(
        call_id="call_a",
        idempotency_key=None,
        idx=0,
        cap_id="cap.search.web.v1",
        attempt=1,
        timeout_ms=8000,
        args={"query": "TRP protocol design"}
    ),
    CallSpec(
        call_id="call_b",
        idempotency_key=None,
        idx=0,
        cap_id="cap.search.web.v1",
        attempt=1,
        timeout_ms=8000,
        args={"query": "PTC programmatic tool calling"}
    ),
]
batch_res = router.batch(calls, mode="PARALLEL", max_concurrency=2)
print(batch_res["status"])
