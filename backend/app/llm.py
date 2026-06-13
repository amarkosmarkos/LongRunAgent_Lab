"""LLM client: real Anthropic API or deterministic mock. Tracks cost per call."""
from __future__ import annotations

from .config import (AGENT_MAX_TOKENS, AGENT_MODELS, ANTHROPIC_API_KEY, LLM_MOCK,
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
        msg = self._client.messages.create(
            model=model,
            max_tokens=AGENT_MAX_TOKENS.get(role, 4000),
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResult(text, model, msg.usage.input_tokens, msg.usage.output_tokens,
                         truncated=(msg.stop_reason == "max_tokens"))
