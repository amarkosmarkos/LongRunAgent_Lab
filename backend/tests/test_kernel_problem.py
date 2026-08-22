"""Scoring, gating and the dev/holdout split of the kernel benchmark."""
import pytest

from app.kernels import spec
from app.problems.kernel import KernelBenchmark, geomean

P = KernelBenchmark()


def solution(per_shape, correct=True):
    return {"correct": correct, "errors": [] if correct else ["boom"],
            "per_shape": per_shape, "tests": []}


def timings(**kw):
    return {label: {"mean": mean} for label, mean in kw.items()}


class TestGeomean:
    def test_matches_the_definition(self):
        assert geomean([1.0, 100.0]) == pytest.approx(10.0)

    def test_ignores_zero_and_missing(self):
        assert geomean([0, None, 4.0, 9.0]) == pytest.approx(6.0)

    def test_empty_is_zero_not_a_crash(self):
        assert geomean([]) == 0.0


class TestScoring:
    def test_score_is_the_geomean_of_the_shape_means(self):
        inst = {"dev": [{"size": 1, "seed": 0}, {"size": 2, "seed": 0}]}
        s = solution(timings(**{"1": 1000.0, "2": 100000.0}))
        assert P.evaluate(inst, s) == pytest.approx(10000.0)

    def test_lower_is_better_so_improvement_reads_as_speedup(self):
        """The engine computes (baseline - score)/baseline; for runtimes that
        is exactly the aggregate speedup, which the whole UI assumes."""
        inst = {"dev": [{"size": 1, "seed": 0}]}
        baseline = P.evaluate(inst, solution(timings(**{"1": 1000.0})))
        winner = P.evaluate(inst, solution(timings(**{"1": 250.0})))
        improvement = (baseline - winner) / baseline
        assert improvement == pytest.approx(0.75)          # 4x faster
        assert baseline / winner == pytest.approx(4.0)


class TestValidateIsAHardGate:
    inst = {"dev": [{"size": 512, "seed": 1}, {"size": 1024, "seed": 2}]}

    def test_accepts_a_correct_fully_timed_kernel(self):
        s = solution(timings(**{"512": 10.0, "1024": 20.0}))
        assert P.validate(self.inst, s) is None

    def test_rejects_an_incorrect_kernel_however_fast(self):
        s = solution(timings(**{"512": 0.001, "1024": 0.001}), correct=False)
        err = P.validate(self.inst, s)
        assert err and "incorrect" in err

    def test_rejects_a_kernel_missing_a_shape(self):
        err = P.validate(self.inst, solution(timings(**{"512": 10.0})))
        assert err and "1024" in err

    def test_rejects_a_non_dict_result(self):
        assert P.validate(self.inst, None) is not None
        assert P.validate(self.inst, [1, 2, 3]) is not None


class TestInstanceSplit:
    def test_dev_and_holdout_are_disjoint_and_complete(self):
        inst = P.generate_instance({"kernel_problem": "grayscale_py",
                                    "backend": "modal", "gpu": "A100"})
        dev, hold = inst["dev"], inst["holdout"]
        assert dev and hold
        assert not [c for c in dev if c in hold]
        total = len(spec.load_spec("grayscale_py")["benchmarks"])
        assert len(dev) + len(hold) == total

    def test_always_leaves_at_least_two_dev_shapes(self):
        """With one dev shape the behaviour descriptor cannot distinguish a
        small-shape winner from a large-shape one, and every kernel lands in
        the same niche."""
        inst = P.generate_instance({"kernel_problem": "grayscale_py",
                                    "backend": "modal", "gpu": "A100",
                                    "holdout_count": 99})
        assert len(inst["dev"]) >= 2

    def test_local_backend_without_cuda_keeps_only_small_shapes(self):
        inst = P.generate_instance({"kernel_problem": "grayscale_py",
                                    "backend": "local"})
        gpu_inst = P.generate_instance({"kernel_problem": "grayscale_py",
                                        "backend": "modal", "gpu": "A100"})
        if not inst["backend_info"]["timings_are_gpu"]:
            assert len(inst["dev"]) + len(inst["holdout"]) < \
                   len(gpu_inst["dev"]) + len(gpu_inst["holdout"])

    def test_unknown_problem_fails_loudly(self):
        with pytest.raises(FileNotFoundError):
            P.generate_instance({"kernel_problem": "no_such_kernel"})


class TestBehaviorDescriptor:
    inst = {
        "dev": [{"size": 512, "seed": 1}, {"size": 4096, "seed": 2}],
        "baseline_per_shape": {"512": 100.0, "4096": 100.0},
    }

    def test_separates_small_winners_from_large_winners(self):
        small_winner = P.behavior_descriptor(
            self.inst, solution(timings(**{"512": 10.0, "4096": 100.0})), 1.0, 10)
        large_winner = P.behavior_descriptor(
            self.inst, solution(timings(**{"512": 100.0, "4096": 10.0})), 1.0, 10)
        assert small_winner["small_speedup"] > small_winner["large_speedup"]
        assert large_winner["large_speedup"] > large_winner["small_speedup"]
        # scaling is the axis that tells them apart
        assert small_winner["scaling"] < 1 < large_winner["scaling"]

    def test_no_timings_yields_no_descriptor(self):
        assert P.behavior_descriptor(self.inst, solution({}), 1.0, 10) is None


class TestSpecLoading:
    def test_every_vendored_problem_loads_with_shapes(self):
        problems = spec.available_problems()
        assert "grayscale_py" in problems
        for name in problems:
            s = spec.load_spec(name)
            assert s["tests"] and s["benchmarks"], name
            assert s["description"], name

    def test_sources_are_the_files_the_harness_needs(self):
        files = spec.source_files("grayscale_py")
        assert {"eval.py", "utils.py", "task.py", "reference.py"} <= set(files)
        assert "def custom_kernel" in spec.reference_submission("grayscale_py")

    def test_vendored_harness_is_unmodified_upstream(self):
        """The lab's claim to comparable scores rests on not touching this."""
        evalpy = spec.source_files("grayscale_py")["eval.py"]
        assert "POPCORN_FD" in evalpy and "def benchmark" in evalpy
