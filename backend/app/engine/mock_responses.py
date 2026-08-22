"""Deterministic mock LLM: scripts the canonical demo arc.

The mock replaces only the *reasoning* (LLM text). Every submission below is
real, is really executed by the official GPU MODE harness, and is really
checked for correctness and timed — so mock-mode results are objectively
verified, not faked. Nothing here can invent a speedup the harness did not
measure.

Arc (on grayscale_py): the reference materialises a full-size temporary, so
there is real headroom. A fused elementwise branch wins on small images, a
BLAS mat-vec branch wins on large ones, the low-precision branch is rejected
outright by the correctness gate, and the two survivors merge into a
shape-specialised kernel that takes the best of both.
"""
from __future__ import annotations

import json

# --------------------------------------------------------------- submissions

CODE_FUSED_1 = '''
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # The reference builds `data * weights` — a full H x W x 3 temporary — and
    # only then reduces. Folding the weights into one fused multiply-add over
    # the three channel planes never materialises it.
    return data[..., 0] * 0.2989 + data[..., 1] * 0.5870 + data[..., 2] * 0.1140
'''

CODE_FUSED_2 = '''
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # Same fusion, but unbind the channel dimension so each plane is a clean
    # strided view and the adds chain without re-indexing.
    r, g, b = data.unbind(-1)
    return torch.add(torch.add(r * 0.2989, g * 0.5870), b * 0.1140)
'''

CODE_MATVEC_1 = '''
import torch
from task import input_t, output_t

_W = {}


def custom_kernel(data: input_t) -> output_t:
    # An (H*W, 3) x (3,) mat-vec is exactly this reduction, and it hands the
    # work to the tuned GEMV path instead of a generic elementwise reduce.
    key = (data.device, data.dtype)
    w = _W.get(key)
    if w is None:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device,
                         dtype=data.dtype)
        _W[key] = w
    h, width = data.shape[0], data.shape[1]
    return (data.reshape(-1, 3) @ w).view(h, width)
'''

CODE_MATVEC_2 = '''
import torch
from task import input_t, output_t

_W = {}


def custom_kernel(data: input_t) -> output_t:
    # Same mat-vec, but keep the weight vector and skip the reshape when the
    # input is already contiguous, so no copy can sneak into the timed region.
    key = (data.device, data.dtype)
    w = _W.get(key)
    if w is None:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device,
                         dtype=data.dtype)
        _W[key] = w
    h, width = data.shape[0], data.shape[1]
    flat = data.view(-1, 3) if data.is_contiguous() else data.reshape(-1, 3)
    return torch.mv(flat, w).view(h, width)
'''

CODE_LOWPREC_1 = '''
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # Hypothesis: this kernel is bandwidth-bound, so halving the bytes read
    # should nearly halve the runtime.
    h = data.half()
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device,
                     dtype=torch.float16)
    return torch.sum(h * w, dim=-1).float()
'''

CODE_LOWPREC_2 = '''
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # Less aggressive: read in bf16 but accumulate the weighted sum in fp32.
    b = data.bfloat16()
    return (b[..., 0].float() * 0.2989 + b[..., 1].float() * 0.5870
            + b[..., 2].float() * 0.1140)
'''

CODE_EINSUM_1 = '''
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # Let the einsum planner pick the contraction order and backend.
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device,
                     dtype=data.dtype)
    return torch.einsum("hwc,c->hw", data, w)
'''

CODE_EINSUM_2 = '''
import torch
from task import input_t, output_t

_W = {}


def custom_kernel(data: input_t) -> output_t:
    # Adopt the lab insight that rebuilding the weight tensor shows up in the
    # measured window at small sizes, and cache it.
    key = (data.device, data.dtype)
    w = _W.get(key)
    if w is None:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device,
                         dtype=data.dtype)
        _W[key] = w
    return torch.einsum("hwc,c->hw", data, w)
'''

CODE_MERGED_1 = '''
import torch
from task import input_t, output_t

_W = {}
# Below this many pixels the fused elementwise form wins; above it the GEMV
# path amortises its setup and pulls ahead. Measured, not guessed.
_CROSSOVER = 1_000_000


def custom_kernel(data: input_t) -> output_t:
    h, width = data.shape[0], data.shape[1]
    if h * width <= _CROSSOVER:
        return data[..., 0] * 0.2989 + data[..., 1] * 0.5870 + data[..., 2] * 0.1140
    key = (data.device, data.dtype)
    w = _W.get(key)
    if w is None:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device,
                         dtype=data.dtype)
        _W[key] = w
    flat = data.view(-1, 3) if data.is_contiguous() else data.reshape(-1, 3)
    return torch.mv(flat, w).view(h, width)
'''

CODE_MERGED_2 = '''
import torch
from task import input_t, output_t

_W = {}
_CROSSOVER = 4_000_000


def custom_kernel(data: input_t) -> output_t:
    # Push the crossover up: the fused form keeps winning further than the
    # first estimate suggested, so give it more of the range.
    h, width = data.shape[0], data.shape[1]
    if h * width <= _CROSSOVER:
        r, g, b = data.unbind(-1)
        return torch.add(torch.add(r * 0.2989, g * 0.5870), b * 0.1140)
    key = (data.device, data.dtype)
    w = _W.get(key)
    if w is None:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device,
                         dtype=data.dtype)
        _W[key] = w
    flat = data.view(-1, 3) if data.is_contiguous() else data.reshape(-1, 3)
    return torch.mv(flat, w).view(h, width)
'''

# ------------------------------------------------------------------ script

_HYPOTHESES = [
    {"name": "Fused elementwise", "strategy": "fused-elementwise",
     "hypothesis": "The reference materialises a full H x W x 3 temporary before reducing it; folding the weights into a single multiply-add over the three channel planes removes that whole array from the traffic and should be substantially faster.",
     "risk": "Three strided plane reads may coalesce worse than one contiguous pass over the packed tensor."},
    {"name": "BLAS mat-vec", "strategy": "blas-matvec",
     "hypothesis": "Reducing over the channel axis is exactly an (H*W, 3) x (3,) mat-vec, so reshaping and calling GEMV hands the work to a tuned kernel instead of a generic elementwise reduction.",
     "risk": "The reshape may force a copy, and GEMV setup could dominate at small image sizes."},
    {"name": "Low precision", "strategy": "low-precision",
     "hypothesis": "This kernel is bandwidth-bound, so reading the image in half precision should nearly halve the time.",
     "risk": "The correctness tolerance is tight enough that fp16 rounding may be rejected outright."},
    {"name": "Einsum contraction", "strategy": "einsum",
     "hypothesis": "Expressing the reduction as an einsum lets the contraction planner choose the backend and ordering rather than fixing one by hand.",
     "risk": "einsum may simply dispatch to the same generic reduction, adding parsing overhead for nothing."},
]

_EXPERIMENTS = {
    ("fused-elementwise", 1): ("Fold the three weights into one fused multiply-add over the channel planes, never building the temporary.",
                               CODE_FUSED_1),
    ("fused-elementwise", 2): ("Unbind the channel dimension so each plane is a clean strided view and the adds chain directly.",
                               CODE_FUSED_2),
    ("blas-matvec", 1): ("Reshape to (H*W, 3) and multiply by the cached weight vector, dispatching to GEMV.",
                         CODE_MATVEC_1),
    ("blas-matvec", 2): ("Use torch.mv on a view instead of a reshape so no copy enters the timed region.",
                         CODE_MATVEC_2),
    ("low-precision", 1): ("Downcast the image to fp16 and reduce there to halve the bytes read.",
                           CODE_LOWPREC_1),
    ("low-precision", 2): ("Read in bf16 but accumulate the weighted sum in fp32 to recover accuracy.",
                           CODE_LOWPREC_2),
    ("einsum", 1): ("Express the channel reduction as an einsum contraction.",
                    CODE_EINSUM_1),
    ("einsum", 2): ("Adopt the lab insight and cache the weight tensor across calls.",
                    CODE_EINSUM_2),
    ("merged", 1): ("Specialise by size: fused elementwise below the measured crossover, GEMV above it.",
                    CODE_MERGED_1),
    ("merged", 2): ("Raise the crossover — the fused form keeps winning further up the range than first estimated.",
                    CODE_MERGED_2),
}

_INSIGHTS = {
    ("fused-elementwise", 1): "The reference's cost is dominated by materialising the H x W x 3 product before reducing it; any formulation that never writes that temporary wins immediately, and the margin is largest at small sizes where allocation is a big share of the time.",
    ("fused-elementwise", 2): None,
    ("blas-matvec", 1): "Rebuilding the weight tensor on every call is measurable at small sizes — hoist any per-call allocation to module scope and key it by device and dtype.",
    ("blas-matvec", 2): "The GEMV path has a fixed setup cost but far better large-size scaling than the elementwise form, so which formulation is fastest depends on the shape rather than being a single global winner.",
    ("low-precision", 1): "Reducing precision below the reference's is not a speed/accuracy trade-off here, it is a hard failure: the correctness gate rejects the kernel outright, so the experiment scores nothing at all.",
    ("low-precision", 2): None,
    ("einsum", 1): "einsum dispatches to the same generic reduction as the reference and adds its own parsing overhead; it is a rewrite, not an optimisation.",
    ("einsum", 2): None,
    ("merged", 1): "Because the two survivors win at opposite ends of the shape range, dispatching on size beats either of them alone — the crossover point is worth measuring rather than guessing.",
    ("merged", 2): None,
}


def mock_call(role: str, ctx: dict) -> str:
    if role == "researcher":
        return ("State of the art for an RGB-to-grayscale reduction:\n"
                "- The operation is purely bandwidth-bound; the roofline is one "
                "read of the image plus one write of the output, so the target is "
                "to touch each byte exactly once.\n"
                "- The naive form (weights broadcast, then reduce) materialises a "
                "full-size intermediate and therefore moves ~2x the necessary "
                "traffic — fusing the weighted sum is the single biggest win.\n"
                "- Vectorised loads (float4 / packed 128-bit accesses) and reading "
                "the packed RGB layout contiguously matter more than any "
                "arithmetic change.\n"
                "- A channel reduction of width 3 can also be expressed as a GEMV, "
                "which reaches tuned library bandwidth at large sizes.\n"
                "- Pitfall: the correctness tolerance does not absorb a precision "
                "downgrade, so reading in fp16 fails the check rather than "
                "trading accuracy for speed.")

    if role == "planner" and ctx.get("review"):
        # planner reviews the round; the mock keeps the demo arc stable by not
        # spawning extra branches (real mode generates new hypotheses here)
        rnd = ctx.get("round", 1)
        return json.dumps({
            "new_hypotheses": [],
            "continue": True,
            "reasoning": f"Round {rnd}: branches are differentiating; continue and let the supervisor prune/merge."})

    if role == "planner":
        cfg = ctx.get("config", {})
        target = cfg.get("target_improvement_pct", 18.0)
        return json.dumps({
            "initial_hypotheses": cfg.get("num_hypotheses", 4),
            "objective": "Produce a correct custom_kernel at least {:.0f}% faster than the reference submission across the benchmark shapes.".format(target),
            "success_criteria": {
                "target_improvement_pct": target,
                "rationale": "The reference is a naive two-pass formulation that materialises a full-size temporary, so a fused implementation should clear {:.0f}% comfortably. The real question is whether one formulation wins across the whole shape range or the lab has to specialise.".format(target)},
            "constraints": [
                "Every submission must pass the official correctness check on all test shapes before it is timed",
                f"One harness invocation must finish within {cfg.get('experiment_timeout_s', 180)}s, including any JIT compilation",
                "Only submission.py may change; reference.py, task.py and eval.py are fixed",
                "Timings come from the official harness, never from the agent"],
            "stop_conditions": [
                f"Budget of ${cfg.get('budget_usd', 2.0)} USD exhausted",
                f"{cfg.get('max_rounds', 5)} rounds completed",
                "Target speedup reached and confirmed on held-out shapes",
                "All branches collapsed"],
            "reasoning": "This kernel is bandwidth-bound, so the wins come from moving fewer bytes rather than from cleverer arithmetic. Exploring fusion, a library GEMV path, precision and a contraction planner in parallel covers the plausible strategies; whichever survive can then be combined.",
        })

    if role == "strategist":
        k = ctx.get("k", 4)
        return json.dumps({"hypotheses": _HYPOTHESES[:k]})

    if role == "experimenter":
        key = (ctx.get("strategy"), min(ctx.get("attempt", 1), 2))
        approach, code = _EXPERIMENTS.get(key, _EXPERIMENTS[("fused-elementwise", 1)])
        return json.dumps({"approach": approach,
                           "expectation": "Lower measured runtime than this branch's previous best, with correctness preserved."}) \
            + "\n```python\n" + code + "\n```"

    if role == "critic":
        strategy = ctx.get("strategy")
        attempt = min(ctx.get("attempt", 1), 2)
        error = ctx.get("error")
        improved = ctx.get("improved", False)
        beats_baseline = ctx.get("beats_baseline", False)
        if error and "incorrect" in str(error).lower():
            verdict, analysis = "failed", (
                f"Rejected by the correctness gate: {error}. Speed is irrelevant "
                "until the output matches the reference — this direction has to "
                "recover accuracy before it can be timed at all.")
        elif error:
            verdict, analysis = "failed", f"The experiment failed: {error}. The implementation, not the hypothesis, is at fault; fix and retry."
        elif improved and beats_baseline:
            verdict, analysis = "improved", "The kernel is measurably faster than both the branch's previous best and the reference, while still passing every correctness shape."
        elif improved:
            verdict, analysis = "improved", "Faster than this branch's previous attempt but still slower than the reference; the direction works yet is not competitive."
        else:
            verdict, analysis = "no_improvement", "No measurable gain over the branch's best. The change did not move the bottleneck the measurement is dominated by."
        insight = _INSIGHTS.get((strategy, attempt))
        return json.dumps({
            "verdict": verdict, "analysis": analysis, "insight": insight,
            "suggestion": "Apply the shared lab insights and target whatever dominates the measured time, not what looks slow in the source."})

    if role == "supervisor":
        rnd = ctx.get("round", 1)
        branches = ctx.get("branches", [])
        by_strategy = {b["strategy"]: b for b in branches if b["status"] == "active"}
        decision = {"collapse": [], "merge": None,
                    "reasoning": "Branches are still differentiating; everyone continues."}
        if rnd >= 2 and "low-precision" in by_strategy:
            b = by_strategy["low-precision"]
            decision["collapse"].append({
                "branch_id": b["id"],
                "reason": "Two rounds rejected by the correctness gate: the reference's accuracy is a hard floor, so trading mantissa bits for bandwidth can never score here. Evidence: no valid timing produced while every other branch was measured."})
            decision["reasoning"] = "Low precision is structurally incompatible with the correctness gate and is collapsed to stop spending budget on it."
        if rnd >= 2 and "einsum" in by_strategy:
            b = by_strategy["einsum"]
            decision["collapse"].append({
                "branch_id": b["id"],
                "reason": "Dominated by both surviving branches at every shape: einsum dispatches to the same generic reduction as the reference, so it is a rewrite rather than an optimisation."})
        if rnd >= 3 and "fused-elementwise" in by_strategy and "blas-matvec" in by_strategy:
            a, b = by_strategy["fused-elementwise"], by_strategy["blas-matvec"]
            decision["merge"] = {
                "source_ids": [a["id"], b["id"]],
                "name": "Shape-specialised grayscale",
                "hypothesis": "The fused form wins at small sizes and the GEMV path wins at large ones, so a kernel that dispatches on pixel count beats either branch across the whole benchmark range.",
                "strategy": "merged",
                "reason": "These two are not competing implementations of the same idea — they win at opposite ends of the shape range. Neither dominates, so the highest-value move is to combine them behind a size check rather than pick one."}
            decision["reasoning"] = "The two survivors win on disjoint parts of the shape range; merging them captures both wins instead of discarding one."
        return json.dumps(decision)

    return "{}"
