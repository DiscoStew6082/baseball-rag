"""Tests for Docker build — Phase 8."""
import subprocess
from pathlib import Path

import pytest

class TestDocker:
    def test_docker_builds(self):
        """docker build -t baseball-rag . succeeds with exit code 0."""
        result = subprocess.run(
            ["docker", "build", "-t", "baseball-rag", "."],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=300,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:])
            print("STDERR:", result.stderr[-2000:])
        assert result.returncode == 0, f"Docker build failed"

    def test_container_health(self):
        """Container starts and /health endpoint responds on port 8001."""
        import time, requests as _requests
        proc = subprocess.Popen(
            ["docker", "run", "--rm", "-p", "8001:8000", "baseball-rag"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(8)
        try:
            resp = _requests.get("http://localhost:8001/health", timeout=10)
            assert resp.status_code == 200
        finally:
            proc.terminate()
            proc.wait(timeout=10)