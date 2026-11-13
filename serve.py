# serve.py
# Production runner for the News Verification FastAPI service.

import os
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the Claim Verification API Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port to listen on (default: $PORT or 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")

    args = parser.parse_args()

    print("=" * 60)
    print(f"  Starting News Verification API Service on http://{args.host}:{args.port}")
    print(f"  Web Dashboard: http://localhost:{args.port}/")
    print(f"  Swagger Docs:  http://localhost:{args.port}/docs")
    print(f"  Health Check:  http://localhost:{args.port}/api/v1/health")
    print("=" * 60)

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
