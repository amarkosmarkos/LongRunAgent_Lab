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
