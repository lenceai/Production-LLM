"""
test_regression.py
Version: 1.0
Usage: Regression tests for the pipeline.
"""
from llmops.pipeline import Pipeline

def test_smoke():
    p = Pipeline(canary_ratio=0.0)
    r = p.answer("What is this repository about?")
    assert isinstance(r["answer"], str)
    assert "variant" in r

