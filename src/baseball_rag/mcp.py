"""MLB Stats API MCP client — wraps the mlb-api-mcp subprocess with typed tools."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_MCP_SERVER_PORT = int(os.environ.get("MCP_PORT", 8001))
_MCP_BASE_URL = f"http://localhost:{_MCP_SERVER_PORT}"
_MCP_ENDPOINT = f"{_MCP_BASE_URL}/mcp"
_PROCESS: subprocess.Popen | None = None
_LOCK = threading.Lock()


def _start_server() -> None:
    """Start mlb-api-mcp as a background subprocess on port 8001."""
    global _PROCESS
    with _LOCK:
        if _PROCESS is not None and _PROCESS.poll() is None:
            return  # already running

        repo_root = Path(__file__).parent.parent.parent  # baseball-rag/
        server_script = repo_root / "src" / "mlb_api_mcp" / "main.py"

        _PROCESS = subprocess.Popen(
            ["uv", "run", "python", str(server_script), "--http", "--port", str(_MCP_SERVER_PORT)],
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        # Wait for server to be ready
        _wait_for_server(30)


def _wait_for_server(timeout: int) -> None:
    """Poll /health until the server is up."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{_MCP_BASE_URL}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if json.loads(resp.read())["status"] == "ok":
                    return
        except Exception:
            pass
        time.sleep(0.5)

    raise RuntimeError("MLB Stats MCP server failed to start within timeout")


def _stop_server() -> None:
    """Stop the mlb-api-mcp subprocess."""
    global _PROCESS
    with _LOCK:
        if _PROCESS is None:
            return
        try:
            os.killpg(os.getpgid(_PROCESS.pid), signal.SIGTERM)
        except OSError:
            pass
        _PROCESS = None


def _call_tool(tool_name: str, **kwargs: Any) -> dict:
    """Call an MCP tool over HTTP JSON-RPC."""
    _start_server()

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": kwargs,
        },
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        _MCP_ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise ConnectionError(f"MLB Stats MCP server unreachable: {e}") from e

    if "error" in result:
        raise RuntimeError(f"MCP tool error [{tool_name}]: {result['error']}")

    return result.get("result", {})


# ---------------------------------------------------------------------------
# Typed wrappers
# ---------------------------------------------------------------------------

def get_mlb_player_info(player_id: int) -> dict:
    """Get biographical info for a player by MLB API ID."""
    return _call_tool("get_mlb_player_info", player_id=player_id)


def search_players(fullname: str, sport_id: int = 1) -> dict:
    """Search for players by name. Returns list of matches with IDs."""
    return _call_tool("get_mlb_search_players", fullname=fullname, sport_id=sport_id)


def get_player_stats(player_ids: str, group: str = "hitting", season: int | None = None) -> dict:
    """Get stat lines for one or more players (comma-separated IDs)."""
    return _call_tool(
        "get_multiple_mlb_player_stats",
        player_ids=player_ids,
        group=group,
        type="season" if season else "career",
        season=season,
    )


def get_stat_leaders(stat: str, season: int | None = None) -> dict:
    """Get league leaders in a stat for a given season."""
    # mlb_api_mcp doesn't expose a generic leaderboard tool directly;
    # use get_multiple_mlb_player_stats with a large roster instead.
    # For now, return an error guiding toward the search-then-stats pattern.
    raise NotImplementedError(
        "get_stat_leaders via MLB API requires player ID lookup first. "
        "Use search_players() then get_player_stats()."
    )
