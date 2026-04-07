"""
FastAPI application for the WiFi-HAR Environment.

Exposes the WiFiHAREnvironment over HTTP endpoints compatible with OpenEnv.

Endpoints:
    POST /reset  — Reset the environment, returns initial observation
    POST /step   — Execute an action, returns observation + reward
    GET  /state  — Current environment state
    GET  /schema — Action / observation schemas
    GET  /health — Health check
    GET  /metadata — Environment metadata
    WS   /ws     — WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core is required. Run: pip install openenv-core") from e

try:
    from models import WiFiHARAction, WiFiHARObservation
    from wifi_har.environment import WiFiHAREnvironment
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models import WiFiHARAction, WiFiHARObservation
    from wifi_har.environment import WiFiHAREnvironment


app = create_app(
    WiFiHAREnvironment,
    WiFiHARAction,
    WiFiHARObservation,
    env_name="wifi-har",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for uv run server or direct execution.

    Usage:
        uv run server
        python -m server.app
        docker run -p 7860:7860 wifi-har
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)  # noqa: main()
