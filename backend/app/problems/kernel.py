"""GPU kernel optimization against the GPU MODE reference-kernels benchmark.

The agents write `submission.py` — a `custom_kernel(data)` that must match the
reference implementation — and are scored by the OFFICIAL upstream harness
(problems/pmpp/eval.py), which the lab runs unmodified.

Engine convention is "lower is better", which fits directly: the score is the
geometric mean of the per-shape mean runtimes in nanoseconds. Improvement over
the baseline is therefore the aggregate speedup over the reference kernel.

Correctness is a hard gate, exactly as upstream: a submission that fails any
correctness test scores nothing, no matter how fast it is.
"""
from __future__ import annotations

import math

from ..kernels import runner, spec
from .base import Problem

# Timing runs are the expensive part; give the harness room but stay bounded.
MIN_TIMEOUT_S = 120


def geomean(values: list[float]) -> float:
    vals = [v for v in values if v and v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


class KernelBenchmark(Problem):
    name = "gpu_kernel"
    description = (
        "GPU kernel optimization on the GPU MODE reference-kernels benchmark. "
        "Write submission.py defining custom_kernel(data) that matches the "
        "reference implementation exactly, and make it as fast as possible. "
        "Score = geometric mean of per-shape runtimes (lower is better); a "
        "submission that fails correctness scores nothing."
    )

    # ------------------------------------------------------------ instance
    def generate_instance(self, params: dict) -> dict:
        problem = params.get("kernel_problem") or "matmul_py"
        s = spec.load_spec(problem)
        backend = params.get("backend") or "local"
        gpu = params.get("gpu") or "T4"
        info = runner.backend_info(backend, gpu)

        shapes = list(s["benchmarks"])
        # On a CUDA-less local box the big shapes take minutes each, so the
        # development backend keeps only the N smallest. (A "keep the smallest
        # N" rule travels across problems; an absolute size cap would not,
        # since a shape's cost means something different per kernel.) Timings
        # there are CPU timings anyway — the point is exercising the loop.
        if not info["timings_are_gpu"]:
            keep = int(params.get("local_max_shapes") or 3)
            shapes = sorted(shapes, key=_work)[:keep]

        # dev shapes drive the run; holdout shapes are only used at the end, to
        # expose a kernel that was tuned to the shapes it was developed against
        split = params.get("holdout_count")
        split = int(split) if split is not None else max(1, len(shapes) // 3)
        # always leave at least two dev shapes: with only one, every kernel
        # looks equally "scalable" and the behavior descriptor cannot tell a
        # small-shape winner from a large-shape one
        split = min(split, max(0, len(shapes) - 2))
        dev = shapes[:len(shapes) - split] if split else shapes
        holdout = shapes[len(shapes) - split:] if split else []

        return {
            "kernel_problem": problem,
            "description": s["description"],
            "tests": s["tests"],
            "dev": dev,
            "holdout": holdout,
            "backend": backend,
            "gpu": gpu,
            "backend_info": info,
            "test_timeout": s["test_timeout"],
            "benchmark_timeout": s["benchmark_timeout"],
            # filled in by baseline(): per-shape reference timings, used for
            # per-shape speedups and the behavior descriptor
            "baseline_per_shape": {},
        }

    def _timeout(self, instance: dict, base: int) -> int:
        return max(base, MIN_TIMEOUT_S, int(instance.get("benchmark_timeout") or 0))

    # ------------------------------------------------------------ baseline
    def baseline(self, instance: dict):
        """The upstream reference submission.py, measured the same way."""
        code = spec.reference_submission(instance["kernel_problem"])
        out = self._measure(code, instance, instance["dev"],
                            self._timeout(instance, MIN_TIMEOUT_S))
        if out["error"] or not out["solution"]:
            raise RuntimeError(
                f"could not benchmark the reference kernel: {out['error']}")
        solution = out["solution"]
        instance["baseline_per_shape"] = {
            k: v.get("mean") for k, v in solution["per_shape"].items()}
        score = self.evaluate(instance, solution)
        return solution, score, "reference submission.py (torch eager)"

    # ------------------------------------------------------------- scoring
    def validate(self, instance: dict, solution) -> str | None:
        if not isinstance(solution, dict):
            return "harness produced no structured result"
        if not solution.get("correct"):
            errs = solution.get("errors") or ["correctness check failed"]
            return f"incorrect kernel: {errs[0]}"
        per_shape = solution.get("per_shape") or {}
        missing = [spec.shape_label(c) for c in instance["dev"]
                   if spec.shape_label(c) not in per_shape]
        if missing:
            return f"no timing produced for shape(s): {', '.join(missing)}"
        return None

    def evaluate(self, instance: dict, solution) -> float:
        """Geometric mean of per-shape mean runtimes, in nanoseconds."""
        means = [v.get("mean") for v in (solution.get("per_shape") or {}).values()]
        return round(geomean([m for m in means if m]), 1)

    def instance_stats(self, instance: dict) -> str:
        info = instance["backend_info"]
        shapes = ", ".join(spec.shape_label(c) for c in instance["dev"])
        return (f"{instance['kernel_problem']} on {info['note']}. "
                f"{len(instance['tests'])} correctness tests; "
                f"benchmark shapes: {shapes}"
                + (f"; {len(instance['holdout'])} held-out shapes"
                   if instance["holdout"] else ""))

    def solver_contract(self) -> str:
        return (
            "Write the complete contents of `submission.py`. It must define "
            "exactly one entry point:\n"
            "    def custom_kernel(data: input_t) -> output_t\n"
            "It runs against the official GPU MODE harness, which imports your "
            "module, checks your output against the reference implementation "
            "on every test shape, and only then times it. You may import "
            "torch, triton, and the stdlib, and you may write inline CUDA via "
            "torch.utils.cpp_extension.load_inline. `task.py` provides the "
            "input_t / output_t type aliases. Do NOT modify reference.py, "
            "task.py or eval.py — you only control submission.py. Any output "
            "that does not match the reference within tolerance scores "
            "nothing, so correctness comes first and speed second."
        )

    # ------------------------------------------------------------ execute
    def _measure(self, code: str, instance: dict, shapes: list[dict],
                 timeout_s: int) -> dict:
        """Correctness first, then timing — the upstream order.

        Returns {"solution"|None, "error"|None, "exec_time", "detail"}.
        """
        problem = instance["kernel_problem"]
        backend, gpu = instance["backend"], instance["gpu"]
        elapsed = 0.0

        # 1. correctness on the official test shapes (cheap, fails fast)
        t = runner.run(problem, code, "test", instance["tests"],
                       max(MIN_TIMEOUT_S, instance.get("test_timeout") or 0),
                       backend=backend, gpu=gpu)
        elapsed += t["exec_time"]
        if t["error"]:
            return {"solution": None, "error": t["error"],
                    "exec_time": round(elapsed, 3), "detail": None}
        test_cases = t["tests"]
        failures = [c for c in test_cases if c.get("status") != "pass"]
        if t["check"] != "pass" or failures:
            first = (failures[0].get("error") if failures
                     else "correctness check failed")
            return {
                "solution": {"correct": False,
                             "errors": [str(first)],
                             "per_shape": {}, "tests": test_cases},
                "error": None, "exec_time": round(elapsed, 3),
                "detail": {"tests": test_cases, "benchmarks": []},
            }

        # 2. timing on the benchmark shapes
        b = runner.run(problem, code, "benchmark", shapes, timeout_s,
                       backend=backend, gpu=gpu)
        elapsed += b["exec_time"]
        if b["error"]:
            return {"solution": None, "error": b["error"],
                    "exec_time": round(elapsed, 3),
                    "detail": {"tests": test_cases, "benchmarks": []}}
        bench_cases = b["benchmarks"]
        bad = [c for c in bench_cases if c.get("status") != "pass"]
        if b["check"] != "pass" or bad:
            first = (bad[0].get("error") if bad else "benchmark check failed")
            return {
                "solution": {"correct": False, "errors": [str(first)],
                             "per_shape": {}, "tests": test_cases},
                "error": None, "exec_time": round(elapsed, 3),
                "detail": {"tests": test_cases, "benchmarks": bench_cases},
            }

        per_shape = {}
        base = instance.get("baseline_per_shape") or {}
        for case, result in zip(shapes, bench_cases):
            label = spec.shape_label(case)
            entry = {k: result.get(k)
                     for k in ("mean", "std", "err", "best", "worst", "runs")}
            entry["spec"] = result.get("spec")
            if base.get(label) and result.get("mean"):
                entry["speedup_vs_baseline"] = round(
                    base[label] / result["mean"], 4)
            per_shape[label] = entry

        solution = {"correct": True, "errors": [], "per_shape": per_shape,
                    "tests": test_cases, "backend": b["backend"]}
        return {"solution": solution, "error": None,
                "exec_time": round(elapsed, 3),
                "detail": {"tests": test_cases, "benchmarks": bench_cases,
                           "per_shape": per_shape}}

    def execute(self, code: str, instance: dict, timeout_s: int) -> dict:
        return self._measure(code, instance, instance["dev"],
                             self._timeout(instance, timeout_s))

    # ------------------------------------------------------------ holdout
    def holdout_eval(self, code: str, instance: dict, timeout_s: int) -> dict | None:
        """Re-time the winning kernel on shapes it was never developed against."""
        shapes = instance.get("holdout") or []
        if not shapes:
            return None
        ref = spec.reference_submission(instance["kernel_problem"])
        t = self._timeout(instance, timeout_s)
        base = self._measure(ref, instance, shapes, t)
        win = self._measure(code, instance, shapes, t)
        if base["error"] or not base["solution"]:
            return {"error": f"reference failed on held-out shapes: {base['error']}"}
        if win["error"]:
            return {"error": win["error"]}
        if not (win["solution"] or {}).get("correct"):
            errs = (win["solution"] or {}).get("errors") or ["incorrect"]
            return {"error": f"winning kernel is incorrect on held-out shapes: {errs[0]}"}

        rows, speedups = [], []
        bshapes = base["solution"]["per_shape"]
        wshapes = win["solution"]["per_shape"]
        for case in shapes:
            label = spec.shape_label(case)
            bm = (bshapes.get(label) or {}).get("mean")
            wm = (wshapes.get(label) or {}).get("mean")
            sp = round(bm / wm, 4) if bm and wm else None
            if sp:
                speedups.append(sp)
            rows.append({"name": label, "spec": case,
                         "baseline_ns": bm, "winner_ns": wm, "speedup": sp,
                         "outcome": ("improved" if sp and sp > 1.0 else
                                     "worsened" if sp else "failed")})
        mean_speedup = round(geomean(speedups), 4) if speedups else None
        return {
            "shapes": rows,
            "summary": {
                "generalizes": bool(mean_speedup and mean_speedup > 1.0),
                "mean_speedup": mean_speedup,
                "improved": sum(1 for r in rows if r["outcome"] == "improved"),
                "worsened": sum(1 for r in rows if r["outcome"] == "worsened"),
                "failed": sum(1 for r in rows if r["outcome"] == "failed"),
            },
        }

    # ------------------------------------------------------------ behavior
    def behavior_descriptor(self, instance: dict, solution,
                            exec_time: float, timeout_s: int) -> dict | None:
        """Bin kernels by HOW they win, not by what their code says.

        Two kernels that reach the same score by being fast on small shapes
        versus on large shapes are genuinely different solutions, and the
        population keeps an elite for each.
        """
        per_shape = (solution or {}).get("per_shape") or {}
        base = instance.get("baseline_per_shape") or {}
        pairs = []
        for case in instance["dev"]:
            label = spec.shape_label(case)
            mean = (per_shape.get(label) or {}).get("mean")
            if mean and base.get(label):
                pairs.append((_work(case), base[label] / mean))
        if not pairs:
            return None
        pairs.sort()
        half = max(1, len(pairs) // 2)
        small = geomean([s for _, s in pairs[:half]])
        large = geomean([s for _, s in pairs[-half:]])
        allsp = [s for _, s in pairs]
        mean_sp = geomean(allsp)
        var = (sum((s - mean_sp) ** 2 for s in allsp) / len(allsp)) ** 0.5
        return {
            "small_speedup": round(small, 3),
            "large_speedup": round(large, 3),
            # >1 means it scales up: relatively better on the big shapes
            "scaling": round(large / small, 3) if small else 1.0,
            "consistency_cv": round(var / mean_sp, 3) if mean_sp else 0.0,
        }


def _work(case: dict) -> int:
    """Rough problem size of a shape, used for capping and for ordering."""
    total = 1
    for k, v in case.items():
        if k != "seed" and isinstance(v, int) and v > 0:
            total *= v
    return total


PROBLEMS = {p.name: p for p in [KernelBenchmark()]}
