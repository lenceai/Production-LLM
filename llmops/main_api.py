"""
main_api.py
Version: 1.0
Usage: Script to start the API server.
"""
import uvicorn
from llmops.config import settings
from llmops.pipeline import Pipeline
from llmops.step10_serve import create_app

def run():
    pipeline = Pipeline(canary_ratio=0.1)  # 10% canary traffic
    app = create_app(pipeline)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)

if __name__ == "__main__":
    run()

