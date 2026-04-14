#!/usr/bin/env python3
"""
Start the Danooc IVR API server.

Usage:
    python scripts/serve.py
    python scripts/serve.py --port 8000 --host 0.0.0.0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from danooc.config import Config
from danooc.api.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Danooc IVR server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--call-flow", default="data/call_flow.json")
    parser.add_argument("--model", default="data/intent_model.json")
    args = parser.parse_args()

    config = Config(
        host=args.host,
        port=args.port,
        base_url=args.base_url,
        call_flow_path=args.call_flow,
        model_path=args.model,
    )

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
