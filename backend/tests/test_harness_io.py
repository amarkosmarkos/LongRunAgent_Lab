"""The boundary with the official harness: its output format and test files.

Everything the lab believes about a kernel comes through this parser, so a
silent change here would silently corrupt every score.
"""
from app.kernels import runner, spec


class TestParseHarnessOutput:
    def test_parses_key_value_lines(self):
        out = runner.parse_harness_output(
            "test-count: 2\ntest.0.status: pass\ntest.1.status: pass\ncheck: pass\n")
        assert out["test-count"] == "2"
        assert out["test.1.status"] == "pass"
        assert out["check"] == "pass"

    def test_ignores_noise_around_the_results(self):
        """torch and triton chatter shares the stream; it must not be parsed."""
        out = runner.parse_harness_output(
            "UserWarning: Failed to initialize NumPy: whatever\n"
            "some/path/file.py:275: note\n"
            "benchmark-count: 1\n"
            "benchmark.0.mean: 1234.5\n"
            "Downloading: 100%\n")
        assert out == {"benchmark-count": "1", "benchmark.0.mean": "1234.5"}

    def test_empty_and_garbage_are_safe(self):
        assert runner.parse_harness_output("") == {}
        assert runner.parse_harness_output(None) == {}
        assert runner.parse_harness_output("no colons here at all") == {}


class TestCollect:
    def test_benchmark_cases_are_typed_and_ordered(self):
        parsed = {
            "benchmark-count": "2",
            "benchmark.0.spec": "size: 512", "benchmark.0.mean": "1000.5",
            "benchmark.0.best": "900", "benchmark.0.runs": "100",
            "benchmark.1.spec": "size: 1024", "benchmark.1.mean": "2000",
            "benchmark.1.runs": "50",
        }
        cases = runner._collect(parsed, "benchmark")
        assert [c["index"] for c in cases] == [0, 1]
        assert cases[0]["mean"] == 1000.5 and isinstance(cases[0]["mean"], float)
        assert cases[0]["runs"] == 100 and isinstance(cases[0]["runs"], int)
        # a timed case with no explicit status counts as a pass
        assert cases[0]["status"] == "pass"

    def test_case_with_an_error_is_a_failure(self):
        cases = runner._collect(
            {"benchmark-count": "1", "benchmark.0.error": "mismatch found"},
            "benchmark")
        assert cases[0]["status"] == "fail"

    def test_missing_count_yields_nothing(self):
        assert runner._collect({}, "benchmark") == []


class TestCaseFileFormat:
    def test_matches_the_upstream_grammar(self):
        """Upstream get_test_cases splits on ';' and matches `key: value`."""
        text = spec.format_cases([{"m": 64, "n": 64, "k": 64, "seed": 53124}])
        assert text == "m: 64; n: 64; k: 64; seed: 53124"

    def test_one_line_per_case(self):
        text = spec.format_cases([{"size": 512, "seed": 1}, {"size": 1024, "seed": 2}])
        assert text.splitlines() == ["size: 512; seed: 1", "size: 1024; seed: 2"]

    def test_round_trips_through_the_upstream_regex(self):
        import re
        grammar = r"\s*([a-zA-Z]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*"
        for line in spec.format_cases(
                [{"size": 8192, "seed": 6252}, {"m": 4096, "n": 5120, "k": 4096,
                                                "seed": 1}]).splitlines():
            for part in line.split(";"):
                assert re.fullmatch(grammar, part), part


class TestShapeLabels:
    def test_seed_is_not_part_of_the_label(self):
        assert spec.shape_label({"size": 512, "seed": 999}) == "512"

    def test_duplicate_shapes_get_distinct_labels(self):
        """conv2d_py really does benchmark one shape under two seeds; a
        collision would silently drop a measurement from the score."""
        cases = [{"size": 256, "kernelsize": 16, "channels": 128, "batch": 2,
                  "seed": 6256},
                 {"size": 256, "kernelsize": 16, "channels": 128, "batch": 2,
                  "seed": 8841}]
        labels = spec.shape_labels(cases)
        assert len(set(labels)) == 2, labels

    def test_unique_shapes_keep_their_plain_label(self):
        labels = spec.shape_labels([{"size": 512, "seed": 1},
                                    {"size": 1024, "seed": 2}])
        assert labels == ["512", "1024"]

    def test_every_vendored_problem_labels_uniquely(self):
        for name in spec.available_problems():
            s = spec.load_spec(name)
            for key in ("benchmarks", "tests"):
                labels = spec.shape_labels(s[key])
                assert len(set(labels)) == len(labels), (name, key, labels)
