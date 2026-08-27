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

    @abstractmethod
    def execute(self, code: str, instance: dict, timeout_s: int) -> dict:
        """Run agent-written code against the instance and report what happened.

        Returns {"solution", "error", "exec_time", "detail"}. "solution" is
        whatever `validate` and `evaluate` consume; "detail" is an optional
        problem-specific breakdown attached to the experiment event (e.g.
        per-shape benchmark results). Never trust the agent's own claim about
        its result — everything here must come from actually running it.
        """

    def holdout_eval(self, code: str, instance: dict, timeout_s: int) -> dict | None:
        """Optionally evaluate the winning code on held-out data at the end of
        a run. Return a JSON-serializable report, or None if not supported."""
        return None

    def evaluation_domain(self, instance: dict) -> dict:
        """What makes two measurements comparable.

        Scores are only meaningful against other scores produced the same way.
        Anything that changes what a number MEANS — the benchmark, the
        hardware, the execution backend — belongs here, so cross-run memory
        can refuse to mix incomparable results. Keys must be JSON-scalar.
        """
        return {"problem": self.name}

    def behavior_descriptor(self, instance: dict, solution,
                            exec_time: float, timeout_s: int) -> dict | None:
        """Describe what a valid solution BEHAVES like (not what its code says):
        a small dict of floats used to bin programs into MAP-Elites niches.
        Return None if the problem has no meaningful descriptor."""
        return None
