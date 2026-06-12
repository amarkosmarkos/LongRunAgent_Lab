"""Problem-agnostic interface. Lower score = better (engine convention)."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Problem(ABC):
    name: str = "abstract"
    description: str = ""

    @abstractmethod
    def generate_instance(self, params: dict) -> dict:
        """Deterministic instance from params (must include a seed)."""

    @abstractmethod
    def baseline(self, instance: dict) -> tuple[list, float, str]:
        """(solution, score, algorithm_name) for the reference baseline."""

    @abstractmethod
    def validate(self, instance: dict, solution) -> str | None:
        """None if valid, else error message."""

    @abstractmethod
    def evaluate(self, instance: dict, solution) -> float:
        """Objective score. Lower is better."""

    @abstractmethod
    def instance_stats(self, instance: dict) -> str:
        """Short human/LLM-readable description of the instance."""

    @abstractmethod
    def solver_contract(self) -> str:
        """Prompt text describing the required solver function signature."""

    def execute(self, code: str, instance: dict, timeout_s: int) -> dict:
        """Run agent solver code against the instance.

        Returns {"solution", "error", "exec_time", "detail"}; "detail" is an
        optional problem-specific breakdown attached to the experiment event
        (e.g. per-instance results for benchmark suites). The default runs the
        code once on the whole instance.
        """
        from ..sandbox import run_solver
        out = run_solver(code, instance, timeout_s)
        out["detail"] = None
        return out

    def holdout_eval(self, code: str, instance: dict, timeout_s: int) -> dict | None:
        """Optionally evaluate the winning code on held-out data at the end of
        a run. Return a JSON-serializable report, or None if not supported."""
        return None
