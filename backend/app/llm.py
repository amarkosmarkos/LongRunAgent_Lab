"""LLM client: real Anthropic API or deterministic mock. Tracks cost per call."""
from __future__ import annotations

from .config import (AGENT_MODELS, ANTHROPIC_API_KEY, LLM_MOCK,
                     MAX_CALL_INPUT_TOKENS, MAX_OUTPUT_TOKENS,
                     MAX_TOOL_CONTINUATIONS, MODEL_PRICING,
                     WEB_SEARCH_MAX_USES)


def _price(model: str, in_tok: int, out_tok: int) -> float:
    in_price, out_price = MODEL_PRICING.get(model, MODEL_PRICING["mock"])
    return (in_tok * in_price + out_tok * out_price) / 1_000_000


class LLMResult:
    def __init__(self, text: str, model: str, input_tokens: int, output_tokens: int,
                 truncated: bool = False):
        self.text = text
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.truncated = truncated  # response hit the output token cap (cut off)
        self.over_budget = False    # tool loop cut short to protect the budget
        in_price, out_price = MODEL_PRICING.get(model, MODEL_PRICING["mock"])
        self.cost_usd = (input_tokens * in_price + output_tokens * out_price) / 1_000_000


class LLMClient:
    """call(role, system, prompt, context) -> LLMResult.

    `context` is only used by the mock implementation, which scripts the demo arc
    deterministically while real execution/evaluation still happens.
    """

    def __init__(self):
        self.mock = LLM_MOCK
        self._client = None
        if not self.mock:
            import anthropic
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def call(self, role: str, system: str, prompt: str, context: dict | None = None,
             model: str | None = None,
             budget_left_usd: float | None = None) -> LLMResult:
        """One agent turn.

        `budget_left_usd` is a HARD ceiling for this single call. Server-side
        tool loops (web search) resend the whole growing conversation on every
        `pause_turn`, so their cost grows quadratically — without this, one
        call can burn a whole run's budget before anyone checks it again.
        """
        if self.mock:
            from .engine.mock_responses import mock_call
            text = mock_call(role, context or {})
            # simulate plausible token counts so cost panels work in mock mode
            in_tok = max(200, len(prompt) // 4)
            out_tok = max(80, len(text) // 4)
            return LLMResult(text, "mock", in_tok, out_tok)

        model = model or AGENT_MODELS.get(role, AGENT_MODELS["experimenter"])
        # The researcher gets Anthropic's server-side web search; the API runs
        # the search loop itself and may return stop_reason="pause_turn" to be
        # resumed. max_uses bounds how many searches it runs per turn.
        tools = ([{"type": "web_search_20260209", "name": "web_search",
                   "max_uses": WEB_SEARCH_MAX_USES}]
                 if role == "researcher" else None)
        messages = [{"role": "user", "content": prompt}]
        in_tok = out_tok = 0
        texts: list[str] = []
        truncated = False
        over_budget = False
        for _ in range(MAX_TOOL_CONTINUATIONS):
            kwargs = dict(model=model, max_tokens=MAX_OUTPUT_TOKENS,
                          system=system, messages=messages)
            if tools:
                kwargs["tools"] = tools
            msg = self._client.messages.create(**kwargs)
            in_tok += msg.usage.input_tokens
            out_tok += msg.usage.output_tokens
            texts.append("".join(b.text for b in msg.content if b.type == "text"))
            if msg.stop_reason == "max_tokens":
                truncated = True
            if msg.stop_reason != "pause_turn":
                break
            # Stop resuming the tool loop before the run can no longer afford
            # it. A continuation resends the entire conversation so far, so the
            # next round bills AT LEAST the tokens already accumulated — that
            # lower bound is what we project with. Checking only what has been
            # spent would discover each overrun one expensive round too late.
            # A thin answer is recoverable; a blown budget ends the run before
            # it has done any work.
            # A hard ceiling on how much context one turn may accumulate. No
            # legitimate agent turn approaches this; a runaway search loop
            # blows past it within a couple of rounds.
            if in_tok >= MAX_CALL_INPUT_TOKENS:
                over_budget = True
                break
            if budget_left_usd is not None:
                spent = _price(model, in_tok, out_tok)
                # the next round bills at least everything accumulated so far,
                # and in practice rather more once fresh search results land,
                # so project it at 2x before deciding we can afford another
                if spent * 3 >= budget_left_usd:
                    over_budget = True
                    break
            messages = messages + [{"role": "assistant", "content": msg.content}]
        text = "\n".join(t for t in texts if t)
        res = LLMResult(text, model, in_tok, out_tok, truncated=truncated)
        res.over_budget = over_budget
        return res
