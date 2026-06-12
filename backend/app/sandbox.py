"""Execute agent-written solver code in a subprocess with a timeout.

The child process runs the solver and prints the solution as JSON.
The PARENT validates and scores it — agent code is never trusted for evaluation.

Note: this is process isolation with a timeout, not a security sandbox.
You are running LLM-generated Python on your own machine; review the README.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HARNESS = r"""
import json, sys

with open(sys.argv[1], encoding="utf-8") as f:
    payload = json.load(f)

ns = {}
exec(payload["code"], ns)
solve = ns.get("solve")
if solve is None:
    print(json.dumps({"error": "no function named solve(cities) defined"}))
    sys.exit(0)

try:
    result = solve(payload["instance"]["cities"])
    result = [int(i) for i in result]
    print(json.dumps({"solution": result}))
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
"""


def run_solver(code: str, instance: dict, timeout_s: int) -> dict:
    """Returns {solution|None, error|None, exec_time}."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "harness.py").write_text(HARNESS, encoding="utf-8")
        (tmp / "payload.json").write_text(
            json.dumps({"code": code, "instance": instance}), encoding="utf-8")
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp / "harness.py"), str(tmp / "payload.json")],
                capture_output=True, text=True, timeout=timeout_s, cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return {"solution": None, "error": f"timeout after {timeout_s}s",
                    "exec_time": round(time.time() - t0, 3)}
        exec_time = round(time.time() - t0, 3)

        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            return {"solution": None, "error": err[-1] if err else "process crashed",
                    "exec_time": exec_time}
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            return {"solution": None, "error": "solver produced no parseable output",
                    "exec_time": exec_time}
        if "error" in out:
            return {"solution": None, "error": out["error"], "exec_time": exec_time}
        return {"solution": out.get("solution"), "error": None, "exec_time": exec_time}
