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

# optional ensemble for the experimenter: comma-separated model ids. With more
# than one, a per-run UCB1 bandit learns which model mutates best (ShinkaEvolve
# style); with one, the bandit only learns over operators.
EXPERIMENTER_MODELS = [m.strip() for m in
                       os.getenv("MODELS_EXPERIMENTER", "").split(",")
                       if m.strip()]

AGENT_MODELS = {
    "planner": os.getenv("MODEL_PLANNER", "claude-sonnet-4-6"),
    "strategist": os.getenv("MODEL_STRATEGIST", "claude-sonnet-4-6"),
    "experimenter": os.getenv("MODEL_EXPERIMENTER", "claude-sonnet-4-6"),
    "critic": os.getenv("MODEL_CRITIC", "claude-haiku-4-5-20251001"),
    "supervisor": os.getenv("MODEL_SUPERVISOR", "claude-sonnet-4-6"),
    "researcher": os.getenv("MODEL_RESEARCHER", "claude-sonnet-4-6"),
}

# Output token ceiling for every agent. The API requires max_tokens, so it can't
# be removed — set it as high as a non-streaming request safely allows (above
# ~16K the SDK refuses non-streaming calls / risks HTTP timeouts). This is well
# above anything an agent actually emits, so it never truncates real work. To go
# higher (sonnet 64K / opus 128K) we'd have to switch the client to streaming.
MAX_OUTPUT_TOKENS = 16000

DEFAULT_RUN_CONFIG = {
    "problem": "tsp",
    "problem_params": {"n_cities": 60, "seed": 42},
    # fallback initial hypothesis count if the Planner doesn't specify one
    "num_hypotheses": 4,
    # hard cap on concurrent branches (the Planner decides the actual number,
    # and may grow it across rounds, but never beyond this)
    "max_branches": 12,
    "max_rounds": 5,
    "budget_usd": 2.0,
    "experiment_timeout_s": 10,
    # immediate retries when an experiment reply has no code / errors / is invalid,
    # before the round counts as a failure
    "experiment_max_attempts": 3,
    # run a web-research agent (Anthropic web_search) before planning
    "enable_web_research": True,
    # after a winner is found, judge how original its algorithm is (Anthropic
    # web_search): does the idea already exist online, or did the lab create it?
    "enable_originality_judge": True,
    # long-term memory: recall past runs' elite solvers + insights before
    # planning, and archive this run's outcome at the end (app.knowledge)
    "enable_knowledge_archive": True,
    # high enough that no single basic strategy reaches it -> forces real exploration
    "target_improvement_pct": 18.0,
    "stagnation_rounds": 2,
    # ---- evolution substrate (git DAG + population + novelty pressure) ----
    # every attempt becomes a real git commit in data/runs/<id>/repo
    "enable_git_repo": True,
    # reject near-copies of already-evaluated programs BEFORE spending on them
    "enable_novelty_gate": True,
    # containment similarity above which a candidate counts as a near-copy of
    # ANOTHER program (own-parent refinements are compared at 0.995)
    "novelty_threshold": 0.92,
    # exploratory operators (rewrite/explore) are held to a stricter bar
    "novelty_threshold_explore": 0.85,
    # probability that an attempt mutates a parent sampled from the WHOLE
    # population (DGM-style) instead of the branch's own best code
    "population_parent_prob": 0.35,
    # elite programs from distinct niches shown as inspiration in prompts
    "inspiration_count": 3,
    # UCB1 bandit over mutation operators (refine/rewrite/recombine/explore)
    "enable_operator_bandit": True,
}
