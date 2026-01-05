"""
step11_observe.py
Version: 1.0
Usage: Prometheus metrics for observability.
"""
from prometheus_client import Counter, Histogram

REQS = Counter("rag_requests_total", "Total RAG requests")
FAILS = Counter("rag_failures_total", "Total RAG failures")
LAT = Histogram("rag_latency_seconds", "RAG latency seconds")

def inc_req(): REQS.inc()
def inc_fail(): FAILS.inc()

