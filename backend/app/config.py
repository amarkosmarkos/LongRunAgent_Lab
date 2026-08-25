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
    "problem": "gpu_kernel",
    "problem_params": {
        # Which vendored GPU MODE reference-kernel to optimize.
        # grayscale_py is the default because its correctness tolerance
        # (rtol/atol 1e-4 on fp32) leaves room for a genuinely different
        # implementation, and its reference materialises a full-size temporary,
        # so there is real headroom to win. conv2d_py (rtol 1e-3) is the other
        # good target on a real GPU. NOTE: matmul_py is vendored but its
        # tolerance (rtol 1e-5 on fp16) is ~100x tighter than one fp16 ULP, so
        # only a bit-identical accumulation order passes — it cannot reward
        # optimisation. See the README before selecting it.
        "kernel_problem": "grayscale_py",
        # "local"  -> official harness in a subprocess on this machine (real
        #             correctness; GPU timings only if this box has CUDA)
        # "modal"  -> the same harness on a real GPU (see kernels/modal_app.py)
        "backend": "local",
        "gpu": "T4",
        # benchmark shapes kept aside to expose shape-overfitting at the end
        "holdout_count": 3,
    },
    # fallback initial hypothesis count if the Planner doesn't specify one
    "num_hypotheses": 4,
    # hard cap on concurrent branches (the Planner decides the actual number,
    # and may grow it across rounds, but never beyond this)
    "max_branches": 12,
    "max_rounds": 5,
    "budget_usd": 2.0,
    # floor for one harness invocation; the problem raises it to the shapes'
    # upstream timeout (kernel benchmarking is far slower than a TSP solve)
    "experiment_timeout_s": 180,
    # immediate retries when an experiment reply has no code / errors / is invalid,
    # before the round counts as a failure
    "experiment_max_attempts": 3,
    # run a web-research agent (Anthropic web_search) before planning
    "enable_web_research": True,
    # long-term memory: recall past runs' elite solvers + insights before
    # planning, and archive this run's outcome at the end (app.knowledge)
    "enable_knowledge_archive": True,
    # Aggregate speedup over the reference kernel the run aims for, as a
    # percentage (90% == a 10x kernel). Deliberately set above what any single
    # obvious rewrite achieves, so reaching it forces the lab to actually
    # explore and combine — the same intent the TSP target had.
    "target_improvement_pct": 90.0,
    "stagnation_rounds": 2,
    # ---- evolution substrate (git DAG + population) ----
    # every attempt becomes a real git commit in data/runs/<id>/repo
    "enable_git_repo": True,
    # Off by design for kernels: rediscovering a standard optimisation
    # (tiling, coalescing, tensor cores) is a perfectly good outcome here, so
    # the lab does not push candidates away from known techniques. The gate
    # still exists (app.novelty) and can be switched on purely as a
    # don't-pay-twice guard against re-evaluating an identical kernel.
    "enable_novelty_gate": False,
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

# --- guards against a single agent turn eating the whole run budget ---------
# Server-side tool loops (web search) resend the entire growing conversation on
# every pause_turn, so their input tokens grow quadratically. A researcher call
# once reached 1.5M input tokens and $4.66 on a $2 run — these bound it, and
# LLMClient.call additionally hard-stops on the remaining budget.
MAX_TOOL_CONTINUATIONS = int(os.getenv("MAX_TOOL_CONTINUATIONS", "3"))
WEB_SEARCH_MAX_USES = int(os.getenv("WEB_SEARCH_MAX_USES", "5"))
# Hard ceiling on the context one agent turn may accumulate across tool
# continuations. No legitimate turn comes close; the runaway hit 1.5M.
MAX_CALL_INPUT_TOKENS = int(os.getenv("MAX_CALL_INPUT_TOKENS", "300000"))

# Fraction of the RUN's total budget a role may ever spend. Preliminary,
# nice-to-have phases must not be able to starve the work the run exists to do:
# a researcher turn once spent a $2 budget before a single hypothesis existed,
# and even bounded it would still have taken 63% of it. Roles absent from this
# map are limited only by the run budget itself.
ROLE_BUDGET_SHARE = {
    "researcher": float(os.getenv("SHARE_RESEARCHER", "0.15")),
}
