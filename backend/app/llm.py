"""LLM client: real Anthropic API or deterministic mock. Tracks cost per call."""
from __future__ import annotations

from .config import (AGENT_MODELS, ANTHROPIC_API_KEY, LLM_MOCK, MAX_OUTPUT_TOKENS,
                     MODEL_PRICING)


class LLMResult:
    def __init__(self, text: str, model: str, input_tokens: int, output_tokens: int,
                 truncated: bool = False):
        self.text = text
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.truncated = truncated  # response hit the output token cap (cut off)
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

    def call(self, role: str, system: str, prompt: str, context: dict | None = None) -> LLMResult:
        if self.mock:
            from .engine.mock_responses import mock_call
            text = mock_call(role, context or {})
            # simulate plausible token counts so cost panels work in mock mode
            in_tok = max(200, len(prompt) // 4)
            out_tok = max(80, len(text) // 4)
            return LLMResult(text, "mock", in_tok, out_tok)

        model = AGENT_MODELS.get(role, AGENT_MODELS["experimenter"])
        # the researcher gets Anthropic's server-side web search; the API runs the
        # search loop and may return stop_reason="pause_turn" to be resumed
        tools = ([{"type": "web_search_20260209", "name": "web_search"}]
                 if role == "researcher" else None)
        messages = [{"role": "user", "content": prompt}]
        in_tok = out_tok = 0
        texts: list[str] = []
        truncated = False
        for _ in range(6):  # allow server-tool continuation (pause_turn)
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
            if msg.stop_reason == "pause_turn":
                messages = messages + [{"role": "assistant", "content": msg.content}]
                continue
            break
        text = "\n".join(t for t in texts if t)
        return LLMResult(text, model, in_tok, out_tok, truncated=truncated)

    def judge_originality(self, code: str) -> tuple[dict, "LLMResult"]:
        """Score how original a solver is. Returns (verdict, LLMResult) so the
        caller can attribute cost. Mock mode returns a deterministic, API-free
        verdict so the demo still shows the originality panel."""
        from . import originality
        if self.mock:
            verdict = originality.mock_verdict(code)
            in_tok = max(200, len(code) // 4)
            out_tok = 120
            return verdict, LLMResult("", "mock", in_tok, out_tok)
        verdict, in_tok, out_tok = originality.judge(self._client, code)
        return verdict, LLMResult("", originality.JUDGE_MODEL, in_tok, out_tok)
