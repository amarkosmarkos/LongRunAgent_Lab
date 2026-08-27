"""Modal app that runs the official GPU MODE harness on a real GPU.

Deploy once, then the lab calls it for every experiment:

    pip install modal && modal setup
    modal deploy backend/app/kernels/modal_app.py

One function per GPU type so a run can pick its target hardware; the lab looks
them up by name (`run_harness_l4`, `run_harness_a100`, …). Each call is
stateless: it receives the whole working directory as text, runs the harness,
and returns stdout, stderr and an exit code for the caller to parse — exactly
the shape the local backend produces, so the two are interchangeable.

Two things keep repeated calls cheap. The image is built once and cached by
Modal (its digest only changes when the dependency list does), and
`scaledown_window` keeps a warm container alive between experiments, so a run's
dozens of evaluations pay the CUDA/torch import once rather than every time.

The per-GPU entry points are written out explicitly rather than generated in a
loop: Modal has to serialise each function, and a closure built inside a loop is
both fragile to serialise and invisible to `modal deploy`'s discovery.
"""
from __future__ import annotations

import modal

# Pinned so a redeploy cannot silently change what the numbers were measured on.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "triton==3.2.0", "numpy==2.2.6")
)

app = modal.App("longrun-kernel-lab")

# Outer backstop so a hung kernel cannot hold a GPU indefinitely; the lab passes
# its own, smaller timeout per call.
REMOTE_TIMEOUT_S = 900
# Keep a warm container between experiments — a run makes dozens of calls and
# each cold start re-imports torch and re-initialises CUDA.
IDLE_S = 240

GPUS = ["T4", "L4", "A100", "H100"]


def _run(files: dict, mode: str, seed: int, timeout_s: int) -> dict:
    """Materialise the working directory, run the harness, report everything.

    Returns the same keys for every outcome — success, harness failure, timeout
    — so the caller never has to guess which shape it got.
    """
    import os
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    gpu_name, cuda = "unknown", False
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception as e:                                  # noqa: BLE001
        return {"stdout": "", "stderr": f"torch unavailable: {e}",
                "returncode": -1, "timed_out": False, "exec_time": 0.0,
                "gpu_name": gpu_name, "cuda": False}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name, text in files.items():
            (tmp / name).write_text(text, encoding="utf-8")
        env = {**os.environ, "POPCORN_FD": "1", "POPCORN_SEED": str(seed),
               "PYTHONWARNINGS": "ignore"}
        budget = max(30, min(timeout_s, REMOTE_TIMEOUT_S - 60))
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp / "eval.py"), mode,
                 str(tmp / "cases.txt")],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=budget, cwd=tmp, env=env)
        except subprocess.TimeoutExpired as e:
            return {"stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
                    "stderr": f"harness timed out after {budget}s on {gpu_name}",
                    "returncode": -9, "timed_out": True,
                    "exec_time": round(time.time() - t0, 3),
                    "gpu_name": gpu_name, "cuda": cuda}
        return {"stdout": proc.stdout or "", "stderr": proc.stderr or "",
                "returncode": proc.returncode, "timed_out": False,
                "exec_time": round(time.time() - t0, 3),
                "gpu_name": gpu_name, "cuda": cuda}


@app.function(image=image, gpu="T4", timeout=REMOTE_TIMEOUT_S,
              scaledown_window=IDLE_S)
def run_harness_t4(files: dict, mode: str, seed: int = 42,
                   timeout_s: int = 600) -> dict:
    return _run(files, mode, seed, timeout_s)


@app.function(image=image, gpu="L4", timeout=REMOTE_TIMEOUT_S,
              scaledown_window=IDLE_S)
def run_harness_l4(files: dict, mode: str, seed: int = 42,
                   timeout_s: int = 600) -> dict:
    return _run(files, mode, seed, timeout_s)


@app.function(image=image, gpu="A100", timeout=REMOTE_TIMEOUT_S,
              scaledown_window=IDLE_S)
def run_harness_a100(files: dict, mode: str, seed: int = 42,
                     timeout_s: int = 600) -> dict:
    return _run(files, mode, seed, timeout_s)


@app.function(image=image, gpu="H100", timeout=REMOTE_TIMEOUT_S,
              scaledown_window=IDLE_S)
def run_harness_h100(files: dict, mode: str, seed: int = 42,
                     timeout_s: int = 600) -> dict:
    return _run(files, mode, seed, timeout_s)


@app.local_entrypoint()
def smoke(gpu: str = "L4"):
    """`modal run modal_app.py` — prove the deployment works end to end."""
    fn = {"t4": run_harness_t4, "l4": run_harness_l4,
          "a100": run_harness_a100, "h100": run_harness_h100}[gpu.lower()]
    out = fn.remote(files={"eval.py": "print('POPCORN_FD ok')",
                           "cases.txt": ""},
                    mode="test", seed=42, timeout_s=60)
    print(f"gpu={out['gpu_name']} cuda={out['cuda']} rc={out['returncode']} "
          f"stdout={out['stdout'].strip()!r}")
