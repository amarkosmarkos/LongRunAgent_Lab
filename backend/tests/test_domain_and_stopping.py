"""Two things that make the lab scientifically valid rather than merely working.

1. A measurement only means something alongside measurements made the same way.
   CPU smoke tests and scripted demos must not steer a real GPU search.
2. A long-running experiment has to actually run. A previous run set itself a
   15% target against a configured 90% and stopped after one round.
"""
import threading

import pytest

from app import knowledge as K
from app.engine.orchestrator import Orchestrator
from app.problems.kernel import KernelBenchmark

CPU = {"problem": "gpu_kernel", "kernel": "conv2d_py",
       "backend": "local", "hardware": "cpu", "timings_are_gpu": False}
L4 = {"problem": "gpu_kernel", "kernel": "conv2d_py",
      "backend": "modal", "hardware": "L4", "timings_are_gpu": True}
A100 = {**L4, "hardware": "A100"}
GRAY_L4 = {**L4, "kernel": "grayscale_py"}
MOCK_L4 = {**L4, "mock": True}


class TestEvaluationDomain:
    def test_hardware_and_benchmark_define_comparability(self):
        keys = {K.domain_key(d) for d in (CPU, L4, A100, GRAY_L4, MOCK_L4)}
        assert len(keys) == 5, keys

    def test_scripted_runs_live_in_their_own_world(self):
        assert K.domain_key(MOCK_L4).startswith("mock:")
        assert K.domain_key(MOCK_L4) != K.domain_key(L4)

    def test_the_problem_reports_its_own_domain(self):
        p = KernelBenchmark()
        cpu = p.evaluation_domain(
            p.generate_instance({"kernel_problem": "conv2d_py", "backend": "local"}))
        gpu = p.evaluation_domain(
            p.generate_instance({"kernel_problem": "conv2d_py",
                                 "backend": "modal", "gpu": "L4"}))
        assert cpu["timings_are_gpu"] is False
        assert gpu["timings_are_gpu"] is True
        assert K.domain_key(cpu) != K.domain_key(gpu)

    @pytest.mark.parametrize("other,comparable", [
        (L4, True), (CPU, False), (A100, False),
        (GRAY_L4, False), (MOCK_L4, False),
    ])
    def test_only_like_for_like_is_comparable(self, other, comparable):
        assert K.same_domain({"domain": other}, L4) is comparable

    def test_entries_without_provenance_are_never_comparable(self):
        assert K.same_domain({"problem": "gpu_kernel"}, L4) is False


class TestArchiveIsolation:
    def archive(self, tmp_path):
        return K.KnowledgeArchive(path=tmp_path / "a.json")

    def ingest(self, arc, run_id, domain, imp, code="def custom_kernel(d):\n    return d\n"):
        return arc.ingest_run(run_id, domain,
                              {"winner_code": code, "improvement_pct": imp,
                               "best_score": 100.0, "winner_branch_name": "w"},
                              [f"insight from {run_id}"])

    def test_a_cpu_result_never_seeds_a_gpu_run(self, tmp_path):
        arc = self.archive(tmp_path)
        self.ingest(arc, "cpu-run", CPU, 90.0)
        assert arc.recall_seed_programs(L4) == []
        assert arc.recall(L4, "conv2d") is None

    def test_a_scripted_result_never_seeds_a_real_run(self, tmp_path):
        arc = self.archive(tmp_path)
        self.ingest(arc, "mock-run", MOCK_L4, 90.6)
        assert arc.recall_seed_programs(L4) == []
        assert arc.recall(L4, "conv2d") is None

    def test_a_matching_result_does_seed(self, tmp_path):
        arc = self.archive(tmp_path)
        self.ingest(arc, "gpu-run", L4, 42.0)
        seeds = arc.recall_seed_programs(L4)
        assert len(seeds) == 1 and seeds[0]["improvement_pct"] == 42.0
        assert arc.recall(L4, "conv2d") is not None

    def test_different_gpus_do_not_share_an_incumbent(self, tmp_path):
        arc = self.archive(tmp_path)
        self.ingest(arc, "l4", L4, 30.0)
        self.ingest(arc, "a100", A100, 80.0)
        assert len(arc.recall_seed_programs(L4)) == 1
        assert arc.recall_seed_programs(L4)[0]["improvement_pct"] == 30.0

    def test_mock_can_be_opted_back_in_for_replay(self, tmp_path):
        arc = self.archive(tmp_path)
        self.ingest(arc, "mock-run", MOCK_L4, 90.6)
        assert arc.recall_seed_programs(MOCK_L4, include_mock=True)

    def test_provenance_is_recorded_on_every_entry(self, tmp_path):
        arc = self.archive(tmp_path)
        out = self.ingest(arc, "gpu-run", L4, 42.0)
        assert out["solver_added"]
        entry = arc.recall(L4, "conv2d")["solvers"][0]
        for field in ("run_id", "domain", "mock"):
            assert field in entry, field
        assert entry["domain"]["hardware"] == "L4"
        assert entry["run_id"] == "gpu-run"

    def test_insights_carry_provenance_too(self, tmp_path):
        arc = self.archive(tmp_path)
        self.ingest(arc, "mock-run", MOCK_L4, 90.6)
        self.ingest(arc, "gpu-run", L4, 10.0)
        recalled = arc.recall(L4, "insight")["insights"]
        assert all("mock-run" not in t for t in recalled)


class TestStoppingPolicy:
    def orch(self, configured=90.0, planned=None, min_rounds=3):
        o = Orchestrator.__new__(Orchestrator)
        o.cfg = {"target_improvement_pct": configured, "min_rounds": min_rounds}
        o.scope = {"success_criteria": {"target_improvement_pct": planned}} \
            if planned is not None else {}
        o._lock = threading.Lock()
        return o

    def test_the_planner_cannot_lower_the_bar(self):
        """The exact regression: configured 90, the Planner set itself 15."""
        assert self.orch(configured=90.0, planned=15)._target_pct() == 90.0

    def test_the_planner_may_raise_the_bar(self):
        assert self.orch(configured=50.0, planned=95)._target_pct() == 95.0

    def test_config_stands_alone_when_the_planner_is_silent(self):
        assert self.orch(configured=90.0)._target_pct() == 90.0

    def test_a_nonsense_planner_target_is_ignored(self):
        assert self.orch(configured=90.0, planned="soon")._target_pct() == 90.0
        assert self.orch(configured=90.0, planned=None)._target_pct() == 90.0

    def test_the_target_cannot_end_the_run_before_min_rounds(self):
        o = self.orch(min_rounds=3)
        assert o._may_stop_on_target(1) is False
        assert o._may_stop_on_target(2) is False
        assert o._may_stop_on_target(3) is True

    def test_min_rounds_of_one_restores_the_old_behaviour(self):
        assert self.orch(min_rounds=1)._may_stop_on_target(1) is True

    def test_defaults_make_a_one_round_run_impossible(self):
        from app.config import DEFAULT_RUN_CONFIG as C
        assert C["min_rounds"] >= 3
        assert C["archive_include_mock"] is False


class TestModalReadiness:
    def test_readiness_is_verified_not_assumed(self):
        """from_name is lazy in modal 1.x and succeeds unauthenticated, so a
        naive check reports a deployment that does not exist."""
        from app.kernels import runner
        r = runner.modal_ready()
        assert isinstance(r["ready"], bool)
        if not r["ready"]:
            assert r.get("hint")           # tell the operator what to run

    def test_the_default_gpu_is_a_sensible_one(self):
        from app.config import DEFAULT_RUN_CONFIG as C
        assert C["problem_params"]["gpu"] in ("L4", "A100")
