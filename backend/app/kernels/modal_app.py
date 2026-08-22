"""Modal app that runs the official GPU MODE harness on a real GPU.

Deploy once, then the lab calls it for every experiment:

    pip install modal && modal setup
    modal deploy backend/app/kernels/modal_app.py

One function per GPU type so a run can pick its target hardware; the lab looks
them up by name (`run_harness_t4`, `run_harness_a100`, …). Each call is
stateless: it receives the whole working directory as text, runs the harness,
and returns its stdout for the caller to parse — exactly what the local
backend does, so the two are interchangeable.

The per-GPU entry points are written out explicitly rather than generated in a
loop: Modal has to serialise each function, and a closure built inside a loop
is both fragile to serialise and invisible to `modal deploy`'s discovery.
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "numpy")
)

app = modal.App("longrun-kernel-lab")

# Wall-clock ceiling for one harness invocation on the remote side. The lab
# passes its own (usually smaller) timeout in; this is the outer backstop that
# stops a hung kernel from holding a GPU indefinitely.
REMOTE_TIMEOUT_S = 900


def _run(files: dict, mode: str, seed: int, timeout_s: int) -> str:
    """Materialise the working directory and run the official harness."""
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name, text in files.items():
            (tmp / name).write_text(text, encoding="utf-8")
        env = {**os.environ, "POPCORN_FD": "1", "POPCORN_SEED": str(seed),
               "PYTHONWARNINGS": "ignore"}
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp / "eval.py"), mode,
                 str(tmp / "cases.txt")],
                capture_output=True, text=True,
                timeout=min(timeout_s, REMOTE_TIMEOUT_S - 30),
                cwd=tmp, env=env)
        except subprocess.TimeoutExpired:
            return f"error: harness timed out after {timeout_s}s"
        return proc.stdout or f"error: {(proc.stderr or '').strip()[-500:]}"


@app.function(image=image, gpu="T4", timeout=REMOTE_TIMEOUT_S)
def run_harness_t4(files: dict, mode: str, seed: int = 42,
                   timeout_s: int = 600) -> str:
    return _run(files, mode, seed, timeout_s)


@app.function(image=image, gpu="L4", timeout=REMOTE_TIMEOUT_S)
def run_harness_l4(files: dict, mode: str, seed: int = 42,
                   timeout_s: int = 600) -> str:
    return _run(files, mode, seed, timeout_s)


@app.function(image=image, gpu="A100", timeout=REMOTE_TIMEOUT_S)
def run_harness_a100(files: dict, mode: str, seed: int = 42,
                     timeout_s: int = 600) -> str:
    return _run(files, mode, seed, timeout_s)


@app.function(image=image, gpu="H100", timeout=REMOTE_TIMEOUT_S)
def run_harness_h100(files: dict, mode: str, seed: int = 42,
                     timeout_s: int = 600) -> str:
    return _run(files, mode, seed, timeout_s)


# GPU types the lab may target; kept in sync with the functions above and
# surfaced by /api/problems so the UI only offers what is deployable.
GPUS = ["T4", "L4", "A100", "H100"]
