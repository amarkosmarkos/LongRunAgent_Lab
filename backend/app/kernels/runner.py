"""Run a kernel submission through the official GPU MODE harness.

This is the kernel-domain replacement for the old TSP `app.sandbox`: same role
(execute agent-written code, never trust it to report its own result), same
position in the design (called only from `Problem.execute`).

Two interchangeable backends, chosen per run by config:

 - "local":  runs the harness in a subprocess on this machine. Correctness is
             real, but on a box without CUDA the timings are CPU timings, so
             this is a development backend for exercising the loop, not a way
             to measure GPU performance. Results are tagged accordingly.
 - "modal":  runs the same harness on a real GPU via Modal. Correctness and
             timings are both real.

Both assemble the identical working directory and parse the identical harness
output, so a submission that scores locally scores the same way remotely.

The harness writes `key: value` lines to the fd named by POPCORN_FD. We point
that at stdout (fd 1) — the alternative, passing a private fd, is POSIX-only
and would not work on Windows.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..config import DATA_DIR
from . import spec

TMP_ROOT = DATA_DIR.parent / "tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# On a CUDA-less machine the harness still calls torch.cuda.synchronize() and
# reference.py allocates on device='cuda'. sitecustomize is imported by `site`
# at interpreter startup (we put the work dir on PYTHONPATH so it is found),
# which lets us neutralise the sync calls without editing the official eval.py.
CPU_SITECUSTOMIZE = '''\
"""Injected by the lab's local backend: make the official harness run on CPU."""
try:
    import torch
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None
except Exception:
    pass
'''

_CUDA_DEVICE_RE = re.compile(r"""device\s*=\s*(['"])cuda\1""")
_RESULT_RE = re.compile(r"^([A-Za-z0-9_.\-]+):\s*(.*)$")


def parse_harness_output(text: str) -> dict:
    """Turn the harness's `key: value` lines into a flat dict.

    Anything else on the stream (torch warnings, triton chatter) is ignored:
    only lines whose key looks like a harness key are kept.
    """
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        m = _RESULT_RE.match(line.strip())
        if m and (m.group(1).split(".")[0] in
                  ("test", "benchmark", "check", "test-count", "benchmark-count",
                   "error", "duration")):
            out[m.group(1)] = m.group(2).strip()
    return out


def _collect(parsed: dict, kind: str) -> list[dict]:
    """Gather per-case results (kind = "test" | "benchmark") in index order."""
    try:
        count = int(parsed.get(f"{kind}-count", 0))
    except ValueError:
        count = 0
    cases = []
    for i in range(count):
        case: dict = {"index": i, "spec": parsed.get(f"{kind}.{i}.spec")}
        for field in ("status", "error", "runs", "mean", "std", "err",
                      "best", "worst"):
            key = f"{kind}.{i}.{field}"
            if key not in parsed:
                continue
            val = parsed[key]
            if field in ("runs",):
                try:
                    val = int(float(val))
                except ValueError:
                    pass
            elif field in ("mean", "std", "err", "best", "worst"):
                try:
                    val = float(val)
                except ValueError:
                    pass
            case[field] = val
        # a benchmark case with timings and no explicit status passed
        if "status" not in case:
            case["status"] = "fail" if "error" in case else (
                "pass" if "mean" in case else "unknown")
        cases.append(case)
    return cases


def _build_workdir(tmp: Path, problem: str, code: str, cases: list[dict],
                   cpu: bool) -> None:
    files = spec.source_files(problem)
    if cpu:
        # the only source rewrite we ever do, and only for the local backend:
        # reference.py hardcodes device='cuda' for input generation
        files["reference.py"] = _CUDA_DEVICE_RE.sub("device='cpu'",
                                                    files["reference.py"])
        (tmp / "sitecustomize.py").write_text(CPU_SITECUSTOMIZE, encoding="utf-8")
    for name, text in files.items():
        (tmp / name).write_text(text, encoding="utf-8")
    (tmp / "submission.py").write_text(code, encoding="utf-8")
    (tmp / "cases.txt").write_text(spec.format_cases(cases), encoding="utf-8")


def run_local(problem: str, code: str, mode: str, cases: list[dict],
              timeout_s: int, seed: int = 42) -> dict:
    """Run the harness in a subprocess on this machine."""
    import os
    cpu = not _cuda_available()
    with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmpdir:
        tmp = Path(tmpdir)
        _build_workdir(tmp, problem, code, cases, cpu)
        env = {**os.environ,
               "POPCORN_FD": "1",          # harness logs to stdout
               "POPCORN_SEED": str(seed),
               "PYTHONPATH": str(tmp),     # so sitecustomize is importable
               "PYTHONWARNINGS": "ignore"}
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp / "eval.py"), mode, str(tmp / "cases.txt")],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout_s, cwd=tmp, env=env)
        except subprocess.TimeoutExpired:
            return {"error": f"harness timed out after {timeout_s}s",
                    "exec_time": round(time.time() - t0, 3),
                    "parsed": {}, "backend": "local-cpu" if cpu else "local-gpu"}
        exec_time = round(time.time() - t0, 3)
        parsed = parse_harness_output(proc.stdout)
        error = None
        if not parsed:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            error = (tail[-1] if tail else
                     f"harness produced no output (exit {proc.returncode})")
        return {"error": error, "exec_time": exec_time, "parsed": parsed,
                "stderr": (proc.stderr or "")[-2000:],
                "backend": "local-cpu" if cpu else "local-gpu"}


def run_modal(problem: str, code: str, mode: str, cases: list[dict],
              timeout_s: int, gpu: str = "T4", seed: int = 42) -> dict:
    """Run the same harness on a real GPU through a deployed Modal function.

    Deploy it once with:  modal deploy backend/app/kernels/modal_app.py
    """
    t0 = time.time()
    try:
        import modal
    except ImportError:
        return {"error": "the 'modal' package is not installed "
                         "(pip install modal) — cannot use the modal backend",
                "exec_time": 0.0, "parsed": {}, "backend": f"modal-{gpu}"}
    try:
        fn = modal.Function.from_name("longrun-kernel-lab", f"run_harness_{gpu.lower()}")
        files = spec.source_files(problem)
        files["submission.py"] = code
        files["cases.txt"] = spec.format_cases(cases)
        stdout = fn.remote(files=files, mode=mode, seed=seed, timeout_s=timeout_s)
    except Exception as e:
        return {"error": f"modal backend failed: {type(e).__name__}: {e}",
                "exec_time": round(time.time() - t0, 3), "parsed": {},
                "backend": f"modal-{gpu}"}
    parsed = parse_harness_output(stdout)
    return {"error": None if parsed else "harness produced no output on Modal",
            "exec_time": round(time.time() - t0, 3), "parsed": parsed,
            "backend": f"modal-{gpu}"}


def run(problem: str, code: str, mode: str, cases: list[dict], timeout_s: int,
        backend: str = "local", gpu: str = "T4", seed: int = 42) -> dict:
    """Execute a submission and return the parsed harness result.

    Returns {error, exec_time, parsed, backend} plus, for convenience,
    `tests` and `benchmarks` as per-case lists.
    """
    if backend == "modal":
        out = run_modal(problem, code, mode, cases, timeout_s, gpu=gpu, seed=seed)
    else:
        out = run_local(problem, code, mode, cases, timeout_s, seed=seed)
    out["tests"] = _collect(out["parsed"], "test")
    out["benchmarks"] = _collect(out["parsed"], "benchmark")
    out["check"] = out["parsed"].get("check")
    return out


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def backend_info(backend: str, gpu: str) -> dict:
    """What the UI/README should say about how these numbers were produced."""
    if backend == "modal":
        return {"backend": "modal", "gpu": gpu, "timings_are_gpu": True,
                "note": f"official harness on a Modal {gpu}"}
    cuda = _cuda_available()
    return {"backend": "local", "gpu": "cuda" if cuda else "cpu",
            "timings_are_gpu": cuda,
            "note": ("official harness on the local CUDA device" if cuda else
                     "official harness on local CPU — correctness is real, "
                     "timings are CPU timings and are NOT a GPU benchmark")}
