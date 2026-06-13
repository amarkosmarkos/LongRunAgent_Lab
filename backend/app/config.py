"""Settings and model pricing."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or ""
LLM_MOCK = os.getenv("LLM_MOCK", "") == "1" or not ANTHROPIC_API_KEY

# USD per million tokens (input, output). Edit to match current pricing.
MODEL_PRICING = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-opus-4-8": (15.00, 75.00),
    "mock": (3.00, 15.00),  # mock mode simulates sonnet-level pricing
}

AGENT_MODELS = {
    "planner": os.getenv("MODEL_PLANNER", "claude-sonnet-4-6"),
    "strategist": os.getenv("MODEL_STRATEGIST", "claude-sonnet-4-6"),
    "experimenter": os.getenv("MODEL_EXPERIMENTER", "claude-sonnet-4-6"),
    "critic": os.getenv("MODEL_CRITIC", "claude-haiku-4-5-20251001"),
    "supervisor": os.getenv("MODEL_SUPERVISOR", "claude-sonnet-4-6"),
}

DEFAULT_RUN_CONFIG = {
    "problem": "tsp",
    "problem_params": {"n_cities": 60, "seed": 42},
    "num_hypotheses": 4,
    "max_rounds": 5,
    "budget_usd": 2.0,
    "experiment_timeout_s": 10,
    # immediate retries when an experiment reply has no code / errors / is invalid,
    # before the round counts as a failure
    "experiment_max_attempts": 3,
    # high enough that no single basic strategy reaches it -> forces real exploration
    "target_improvement_pct": 18.0,
    "stagnation_rounds": 2,
}
