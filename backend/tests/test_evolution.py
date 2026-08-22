"""The evolution substrate: population, niches, parent selection, bandit, git."""
import random

from app.bandit import OPERATORS, OperatorBandit
from app.gitrepo import RunRepo
from app.novelty import fingerprint, similarity
from app.population import Population, Program, niche_key


def prog(pid, score, behavior=None, **kw):
    return Program(id=pid, code=f"# {pid}", branch_id="b1", round=1,
                   score=score, valid=True, behavior=behavior, **kw)


class TestNiches:
    def test_kernel_descriptors_drive_the_niche(self):
        small = prog("a", 1.0, {"small_speedup": 8.0, "large_speedup": 1.0,
                                "scaling": 0.125})
        large = prog("b", 1.0, {"small_speedup": 1.0, "large_speedup": 8.0,
                                "scaling": 8.0})
        assert niche_key(small) != niche_key(large)

    def test_similar_behaviour_shares_a_niche(self):
        a = prog("a", 1.0, {"small_speedup": 4.0, "large_speedup": 4.0, "scaling": 1.0})
        b = prog("b", 2.0, {"small_speedup": 4.02, "large_speedup": 4.01, "scaling": 1.0})
        assert niche_key(a) == niche_key(b)

    def test_falls_back_to_technique_tags(self):
        p = prog("a", 1.0, behavior=None)
        p.tags = ["triton", "tiling"]
        assert niche_key(p) == "triton+tiling"

    def test_unclassifiable_still_gets_a_key(self):
        assert niche_key(prog("a", 1.0)) == "unclassified"


class TestPopulation:
    def make(self):
        return Population(baseline_score=100.0, rng=random.Random(0))

    def test_elite_per_niche_keeps_the_best(self):
        pop = self.make()
        b = {"small_speedup": 2.0, "large_speedup": 2.0, "scaling": 1.0}
        assert pop.add(prog("slow", 50.0, b))["became_elite"] is True
        assert pop.add(prog("fast", 10.0, b))["became_elite"] is True
        assert pop.add(prog("mid", 30.0, b))["became_elite"] is False
        elites = pop.elites_public()
        assert len(elites) == 1 and elites[0]["id"] == "fast"

    def test_a_new_behaviour_opens_a_new_niche(self):
        pop = self.make()
        pop.add(prog("a", 50.0, {"small_speedup": 1.0, "large_speedup": 1.0,
                                 "scaling": 1.0}))
        out = pop.add(prog("b", 90.0, {"small_speedup": 9.0, "large_speedup": 1.0,
                                       "scaling": 0.1}))
        # worse-scoring but behaviourally new: it survives anyway
        assert out["new_niche"] is True and out["became_elite"] is True
        assert pop.stats()["niches"] == 2

    def test_invalid_programs_never_become_parents(self):
        pop = self.make()
        pop.add(Program(id="broken", code="x", branch_id="b", round=1,
                        score=None, valid=False))
        assert pop.select_parent() is None

    def test_dgm_rule_favours_the_unexplored(self):
        """Weight is quality / (1 + children), so an equally good parent that
        has already been mutated a lot should be picked less often."""
        pop = self.make()
        pop.add(prog("explored", 50.0))
        pop.add(prog("fresh", 50.0))
        pop.get("explored").children = 30
        picks = [pop.select_parent().id for _ in range(300)]
        assert picks.count("fresh") > picks.count("explored") * 3

    def test_inspirations_come_from_distinct_niches(self):
        """k-1 elites, each from a different niche, plus one deliberately odd
        non-elite — so only the elite part is required to be niche-distinct."""
        pop = self.make()
        for i in range(4):
            pop.add(prog(f"p{i}", 50.0 - i,
                         {"small_speedup": 2.0 ** i,
                          "large_speedup": 1.0, "scaling": 1.0}))
        picked = pop.select_inspirations(3)
        assert len(picked) == 3
        elite_ids = {e["id"] for e in pop.elites_public()}
        elites = [p for p in picked if p.id in elite_ids]
        assert len({niche_key(p) for p in elites}) == len(elites)

    def test_speedup_niches_do_not_saturate(self):
        """A linear bin collapsed everything past 3x into one bucket, which is
        exactly the range where kernels get interesting."""
        keys = {niche_key(prog(f"p{v}", 1.0,
                               {"small_speedup": v, "large_speedup": 1.0,
                                "scaling": 1.0}))
                for v in (1.0, 2.0, 4.0, 8.0, 16.0)}
        assert len(keys) == 5

    def test_excluded_parent_is_not_offered_back(self):
        pop = self.make()
        pop.add(prog("a", 10.0))
        pop.add(prog("b", 20.0))
        assert all(p.id != "a" for p in pop.select_inspirations(3, {"a"}))


class TestBandit:
    def test_explores_every_arm_before_exploiting(self):
        b = OperatorBandit(list(OPERATORS), ["m"], random.Random(0))
        seen = set()
        for _ in range(len(OPERATORS)):
            arm = b.pick()           # an arm counts as explored once rewarded,
            b.reward(arm, True, True, False, 0.0)   # which is how the loop uses it
            seen.add(arm[0])
        assert seen == set(OPERATORS)

    def test_converges_on_the_paying_operator(self):
        b = OperatorBandit(["good", "bad"], ["m"], random.Random(0))
        for _ in range(60):
            op, model = b.pick()
            b.reward((op, model), improved=(op == "good"), valid=True,
                     new_niche=False, cost_usd=0.0)
        stats = {a["operator"]: a["pulls"] for a in b.stats()}
        assert stats["good"] > stats["bad"]

    def test_opening_a_niche_is_rewarded_even_without_improvement(self):
        b = OperatorBandit(["x"], ["m"], random.Random(0))
        r = b.reward(("x", "m"), improved=False, valid=True, new_niche=True,
                     cost_usd=0.0)
        plain = OperatorBandit(["x"], ["m"], random.Random(0)).reward(
            ("x", "m"), improved=False, valid=True, new_niche=False, cost_usd=0.0)
        assert r > plain

    def test_reward_is_discounted_by_spend(self):
        b = OperatorBandit(["x"], ["m"], random.Random(0))
        cheap = b.reward(("x", "m"), True, True, False, cost_usd=0.0)
        dear = b.reward(("x", "m"), True, True, False, cost_usd=1.0)
        assert dear < cheap


class TestNoveltyFingerprint:
    def test_renaming_variables_does_not_look_novel(self):
        a = "def custom_kernel(data):\n    total = data * 2\n    return total\n"
        b = "def custom_kernel(data):\n    acc = data * 2\n    return acc\n"
        assert similarity(fingerprint(a), fingerprint(b)) > 0.9

    def test_a_different_algorithm_looks_different(self):
        a = "def custom_kernel(d):\n    return d.sum(-1)\n"
        b = ("import torch\n"
             "def custom_kernel(d):\n"
             "    w = torch.tensor([1.0, 2.0, 3.0])\n"
             "    out = torch.empty(d.shape[0])\n"
             "    for i in range(d.shape[0]):\n"
             "        out[i] = (d[i] * w).sum()\n"
             "    return out\n")
        assert similarity(fingerprint(a), fingerprint(b)) < 0.6

    def test_unparseable_code_still_fingerprints(self):
        assert fingerprint("def broken(:\n  ???") != frozenset()


class TestGitRepo:
    def test_records_a_real_dag_with_merges(self, tmp_path):
        repo = RunRepo(tmp_path, "run1")
        if not repo.available:                      # git absent: degrade, don't fail
            assert repo.commit_root("x", {}) is None
            return
        root = repo.commit_root("# baseline\n", {"baseline": True})
        assert root
        repo.create_branch("a", root)
        repo.create_branch("b", root)
        a1 = repo.commit_attempt("a", "# a1", {"score": 10, "summary": "a1"})
        b1 = repo.commit_attempt("b", "# b1", {"score": 20, "summary": "b1"})
        merge = repo.commit_attempt("a", "# merged", {"summary": "merge"},
                                    extra_parents=[b1])
        assert a1 and b1 and merge
        log = repo.log_graph()
        assert "merge" in log and "baseline" in log

    def test_absent_git_never_breaks_a_run(self, tmp_path, monkeypatch):
        repo = RunRepo(tmp_path, "run2")
        monkeypatch.setattr(repo, "available", False)
        assert repo.commit_attempt("a", "code", {}) is None
        assert repo.head("a") is None
        assert repo.log_graph() is None
