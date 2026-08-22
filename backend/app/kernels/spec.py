"""Load a GPU MODE reference-kernel problem from its official files.

A problem directory (vendored under backend/data/kernels/<problem>/) holds the
upstream files unchanged: task.yml (shapes + description), task.py (type
schema), reference.py (input generation + correctness check) and submission.py
(the reference implementation, which the lab uses as its baseline). The shared
harness eval.py and utils.py sit one level up, exactly as in the upstream repo.

Nothing here is rewritten: the lab runs the official harness, so a score
produced locally means the same thing it means upstream.
"""
from __future__ import annotations

import yaml

from ..config import DATA_DIR

KERNELS_DIR = DATA_DIR.parent / "kernels"

# Files the harness needs in its working directory, resolved relative to the
# problem directory (mirrors the `files:` section of task.yml).
SHARED_FILES = ("eval.py", "utils.py")
PROBLEM_FILES = ("task.py", "reference.py")


def available_problems() -> list[str]:
    if not KERNELS_DIR.exists():
        return []
    return sorted(p.name for p in KERNELS_DIR.iterdir()
                  if p.is_dir() and (p / "task.yml").exists())


def load_spec(problem: str) -> dict:
    """Parse a problem's task.yml into the shapes and metadata the lab needs."""
    pdir = KERNELS_DIR / problem
    task_yml = pdir / "task.yml"
    if not task_yml.exists():
        raise FileNotFoundError(
            f"unknown kernel problem '{problem}'. Available: "
            f"{available_problems() or '(none vendored under data/kernels/)'}")
    cfg = yaml.safe_load(task_yml.read_text(encoding="utf-8")) or {}
    return {
        "problem": problem,
        "description": (cfg.get("description") or "").strip(),
        "tests": list(cfg.get("tests") or []),
        "benchmarks": list(cfg.get("benchmarks") or []),
        "test_timeout": int(cfg.get("test_timeout") or 180),
        "benchmark_timeout": int(cfg.get("benchmark_timeout") or 180),
    }


def source_files(problem: str) -> dict[str, str]:
    """The official harness + problem sources, as {filename: text}."""
    pdir = KERNELS_DIR / problem
    files = {name: (KERNELS_DIR / name).read_text(encoding="utf-8")
             for name in SHARED_FILES}
    files.update({name: (pdir / name).read_text(encoding="utf-8")
                  for name in PROBLEM_FILES})
    return files


def reference_submission(problem: str) -> str:
    """The upstream submission.py — the lab's baseline to beat."""
    return (KERNELS_DIR / problem / "submission.py").read_text(encoding="utf-8")


def format_cases(cases: list[dict]) -> str:
    """Render shape dicts in the harness's test-file format.

    One case per line, `key: value` parts separated by ';' — see get_test_cases
    in the upstream eval.py.
    """
    return "\n".join(
        "; ".join(f"{k}: {v}" for k, v in case.items()) for case in cases)


def shape_label(case: dict) -> str:
    """Short human label for a shape, used as the per-case result key."""
    return "x".join(str(v) for k, v in case.items() if k != "seed") or "case"
