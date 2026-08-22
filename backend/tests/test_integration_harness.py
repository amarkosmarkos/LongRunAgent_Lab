"""End-to-end against the REAL official harness (needs torch; slow).

Everything else in the suite mocks the boundary. These tests are the only
thing proving the lab and the upstream harness actually agree, so they run the
real subprocess. Skipped automatically when torch is unavailable.

    pytest tests -m "not slow"     # skip these
"""
import pytest

torch = pytest.importorskip("torch", reason="the official harness needs torch")

from app.kernels import runner, spec          # noqa: E402
from app.problems.kernel import KernelBenchmark  # noqa: E402

pytestmark = pytest.mark.slow

PROBLEM = "grayscale_py"

FUSED = """
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    return data[..., 0] * 0.2989 + data[..., 1] * 0.5870 + data[..., 2] * 0.1140
"""

WRONG = """
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    return data[..., 0]          # drops two channels entirely
"""

BROKEN = """
def custom_kernel(data):
    raise RuntimeError("kaboom")
"""


@pytest.fixture(scope="module")
def small_shapes():
    s = spec.load_spec(PROBLEM)
    return sorted(s["benchmarks"], key=lambda c: c["size"])[:1]


class TestRealHarness:
    def test_reference_passes_its_own_correctness_check(self):
        s = spec.load_spec(PROBLEM)
        out = runner.run(PROBLEM, spec.reference_submission(PROBLEM),
                         "test", s["tests"], 600)
        assert out["error"] is None
        assert out["check"] == "pass"
        assert out["tests"] and all(c["status"] == "pass" for c in out["tests"])

    def test_a_wrong_kernel_is_rejected_not_timed(self):
        s = spec.load_spec(PROBLEM)
        out = runner.run(PROBLEM, WRONG, "test", s["tests"], 600)
        assert out["check"] != "pass"
        assert any(c["status"] == "fail" for c in out["tests"])

    def test_a_crashing_kernel_surfaces_its_error(self):
        s = spec.load_spec(PROBLEM)
        out = runner.run(PROBLEM, BROKEN, "test", s["tests"][:1], 600)
        assert out["error"] or out["check"] != "pass"

    def test_benchmark_reports_real_per_shape_timings(self, small_shapes):
        out = runner.run(PROBLEM, FUSED, "benchmark", small_shapes, 900)
        assert out["error"] is None and out["check"] == "pass"
        case = out["benchmarks"][0]
        assert case["mean"] > 0 and case["runs"] >= 1
        assert case["best"] <= case["mean"] <= case["worst"]

    def test_backend_is_labelled_so_cpu_is_never_read_as_gpu(self):
        info = runner.backend_info("local", "T4")
        out = runner.run(PROBLEM, FUSED, "test",
                         spec.load_spec(PROBLEM)["tests"][:1], 600)
        assert out["backend"].startswith("local-")
        assert info["timings_are_gpu"] is (out["backend"] == "local-gpu")


class TestProblemAgainstTheHarness:
    def test_full_measure_scores_a_correct_kernel(self):
        p = KernelBenchmark()
        inst = p.generate_instance({"kernel_problem": PROBLEM,
                                    "backend": "local", "local_max_shapes": 2})
        out = p.execute(FUSED, inst, 600)
        assert out["error"] is None
        assert p.validate(inst, out["solution"]) is None
        score = p.evaluate(inst, out["solution"])
        assert score > 0
        # per-case results are stored, not just the aggregate
        assert len(out["detail"]["per_shape"]) == len(inst["dev"])

    def test_a_wrong_kernel_scores_nothing_however_fast(self):
        p = KernelBenchmark()
        inst = p.generate_instance({"kernel_problem": PROBLEM,
                                    "backend": "local", "local_max_shapes": 2})
        out = p.execute(WRONG, inst, 600)
        err = p.validate(inst, out["solution"])
        assert err and "incorrect" in err

    def test_the_fused_kernel_really_beats_the_reference(self):
        """The migration's core claim: the loop can measure a real speedup."""
        p = KernelBenchmark()
        inst = p.generate_instance({"kernel_problem": PROBLEM,
                                    "backend": "local", "local_max_shapes": 2})
        _, baseline, _ = p.baseline(inst)
        out = p.execute(FUSED, inst, 600)
        assert p.validate(inst, out["solution"]) is None
        assert p.evaluate(inst, out["solution"]) < baseline

    def test_behaviour_descriptor_is_populated_from_real_timings(self):
        p = KernelBenchmark()
        inst = p.generate_instance({"kernel_problem": PROBLEM,
                                    "backend": "local", "local_max_shapes": 2})
        p.baseline(inst)
        out = p.execute(FUSED, inst, 600)
        b = p.behavior_descriptor(inst, out["solution"], out["exec_time"], 600)
        assert b and {"small_speedup", "large_speedup", "scaling"} <= set(b)
        assert b["small_speedup"] > 0
