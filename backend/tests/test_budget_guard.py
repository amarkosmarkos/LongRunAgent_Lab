"""Regression tests for the incident where one call drained a whole run.

A researcher turn reached 1,506,736 input tokens and $4.66 on a $2 budget:
server-side tool loops resend the growing conversation on every pause_turn, so
cost grows quadratically, and the budget was only checked BETWEEN calls.
"""
import pytest

from app import llm as llmmod
from app.config import MAX_CALL_INPUT_TOKENS, MAX_TOOL_CONTINUATIONS

SONNET = "claude-sonnet-4-6"


class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _Block:
    type = "text"
    text = "partial findings"


class _Msg:
    def __init__(self, in_tok, stop="pause_turn"):
        self.usage = _Usage(in_tok, 1500)
        self.content = [_Block()]
        self.stop_reason = stop


class _Messages:
    """Reproduces the incident's growth: each continuation bills more."""

    def __init__(self, growth=250_000, stop_after=None):
        self.calls = 0
        self.growth = growth
        self.stop_after = stop_after

    def create(self, **kw):
        self.calls += 1
        stop = "end_turn" if (self.stop_after and self.calls >= self.stop_after) \
            else "pause_turn"
        return _Msg(self.growth * self.calls, stop)


def client(**kw):
    c = llmmod.LLMClient.__new__(llmmod.LLMClient)
    c.mock = False
    c._client = type("C", (), {"messages": _Messages(**kw)})()
    return c


class TestPerCallCeiling:
    def test_a_single_call_cannot_drain_the_run(self):
        c = client()
        res = c.call("researcher", "s", "p", budget_left_usd=2.00)
        assert res.cost_usd <= 2.00, f"still overspent: ${res.cost_usd:.2f}"
        assert res.over_budget is True

    def test_it_is_far_cheaper_than_the_incident(self):
        c = client()
        res = c.call("researcher", "s", "p", budget_left_usd=2.00)
        assert res.input_tokens < 1_506_736 / 4

    def test_partial_text_is_still_returned(self):
        """Cutting the loop must degrade the answer, not lose it."""
        c = client()
        res = c.call("researcher", "s", "p", budget_left_usd=2.00)
        assert res.text.strip()

    def test_a_tighter_budget_stops_sooner(self):
        cheap = client()
        cheap.call("researcher", "s", "p", budget_left_usd=0.50)
        rich = client()
        rich.call("researcher", "s", "p", budget_left_usd=50.0)
        assert cheap._client.messages.calls <= rich._client.messages.calls


class TestAbsoluteCeilings:
    def test_token_ceiling_bounds_even_an_unlimited_budget(self):
        c = client()
        res = c.call("researcher", "s", "p", budget_left_usd=10_000.0)
        assert res.input_tokens < 1_506_736
        assert res.over_budget is True

    def test_continuations_are_capped(self):
        # tiny growth so neither the budget nor the token ceiling ever trips
        c = client(growth=10)
        c.call("researcher", "s", "p", budget_left_usd=10_000.0)
        assert c._client.messages.calls <= MAX_TOOL_CONTINUATIONS

    def test_ceilings_are_configured_sanely(self):
        assert 0 < MAX_TOOL_CONTINUATIONS <= 4
        assert MAX_CALL_INPUT_TOKENS < 1_000_000


class TestNormalCallsAreUnaffected:
    def test_a_one_shot_answer_costs_one_round_trip(self):
        c = client(growth=1000, stop_after=1)
        res = c.call("planner", "s", "p", budget_left_usd=2.00)
        assert c._client.messages.calls == 1
        assert res.over_budget is False

    def test_no_ceiling_is_applied_when_no_budget_is_passed(self):
        c = client(growth=10, stop_after=2)
        res = c.call("planner", "s", "p")
        assert res.over_budget is False

    def test_web_search_only_goes_to_the_researcher(self):
        seen = {}

        class Spy(_Messages):
            def create(self, **kw):
                seen.update(kw)
                return _Msg(100, stop="end_turn")

        for role, expect_tools in (("researcher", True), ("planner", False)):
            c = llmmod.LLMClient.__new__(llmmod.LLMClient)
            c.mock = False
            c._client = type("C", (), {"messages": Spy()})()
            seen.clear()
            c.call(role, "s", "p", budget_left_usd=1.0)
            assert ("tools" in seen) is expect_tools, role
            if expect_tools:
                assert seen["tools"][0]["max_uses"] > 0


class TestPricing:
    def test_price_matches_the_incident_invoice(self):
        assert llmmod._price(SONNET, 1_506_736, 9_597) == pytest.approx(4.664, abs=0.01)

    def test_unknown_models_do_not_crash(self):
        assert llmmod._price("some-future-model", 1000, 100) > 0


class TestRoleBudgetShare:
    """A preliminary phase must not be able to starve the actual work.

    The incident spent a whole $2 run on research before one hypothesis
    existed. Bounding the call alone still left research taking 63% of it.
    """

    def orch(self, budget=2.0, spent_by=None):
        from app.engine.orchestrator import Orchestrator
        o = Orchestrator.__new__(Orchestrator)
        import threading
        o.cfg = {"budget_usd": budget}
        o.total_cost = sum((spent_by or {}).values())
        o.cost_by_agent = dict(spent_by or {})
        o._lock = threading.Lock()
        return o

    def test_researcher_is_capped_to_its_share(self):
        from app.config import ROLE_BUDGET_SHARE
        o = self.orch(budget=2.0)
        allowed = o._budget_for("researcher")
        assert allowed == pytest.approx(ROLE_BUDGET_SHARE["researcher"] * 2.0)
        assert allowed < 2.0

    def test_roles_without_a_share_get_the_whole_remaining_budget(self):
        o = self.orch(budget=2.0, spent_by={"planner": 0.5})
        assert o._budget_for("experimenter") == pytest.approx(1.5)

    def test_the_share_is_cumulative_across_calls(self):
        o = self.orch(budget=2.0, spent_by={"researcher": 0.25})
        # 15% of $2 is $0.30, of which $0.25 is already spent
        assert o._budget_for("researcher") == pytest.approx(0.05)

    def test_an_exhausted_share_yields_zero_not_a_negative(self):
        o = self.orch(budget=2.0, spent_by={"researcher": 1.0})
        assert o._budget_for("researcher") == 0.0

    def test_the_share_never_exceeds_what_the_run_has_left(self):
        o = self.orch(budget=2.0, spent_by={"experimenter": 1.95})
        assert o._budget_for("researcher") == pytest.approx(0.05)

    def test_research_leaves_the_budget_for_the_work(self):
        """End to end, worst case: a runaway search under the real share.

        $4.66 of a $2 run before the share; comfortably inside it after.
        """
        o = self.orch(budget=2.0)
        c = client()
        res = c.call("researcher", "s", "p",
                     budget_left_usd=o._budget_for("researcher"))
        assert res.cost_usd < 1.0, f"research took ${res.cost_usd:.2f} of $2"
        assert 2.0 - res.cost_usd > 1.0, "not enough left to do any work"

    def test_the_first_turn_is_the_irreducible_exposure(self):
        """What the guard CANNOT do, stated so nobody assumes otherwise.

        A call's cost is only observable once it returns, so the opening turn
        is never preemptable — it is bounded by WEB_SEARCH_MAX_USES and
        MAX_OUTPUT_TOKENS, not by the budget. Everything after it is.
        """
        c = client()
        res = c.call("researcher", "s", "p", budget_left_usd=0.000001)
        assert c._client.messages.calls == 1     # stopped as early as possible
        assert res.cost_usd > 0                  # but that one turn was paid for
        assert res.over_budget is True
