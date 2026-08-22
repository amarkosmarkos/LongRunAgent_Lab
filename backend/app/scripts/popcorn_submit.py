"""Validate a run's winning kernel against the OFFICIAL GPU MODE leaderboard.

This is deliberately outside the run loop. popcorn-cli submits to the shared
GPU MODE cluster, which means queue latency and rate limits — fine for
checking a finished result, not for the dozens of evaluations an experiment
loop makes. The loop uses app.kernels.runner (local / Modal); this script is
the final "does it hold up on the real thing" step.

Prerequisites (one-off):
    curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
    popcorn register discord

Usage:
    python -m app.scripts.popcorn_submit <run_id> [--gpu A100] [--mode benchmark]
    python -m app.scripts.popcorn_submit --file submission.py --leaderboard grayscale_v2

Modes: test (correctness only), benchmark (timings), leaderboard (ranked entry).
Nothing is submitted without --yes, because a leaderboard submission is public.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ..config import DATA_DIR


def winner_from_run(run_id: str) -> tuple[str, dict]:
    """Pull the winning submission.py and its recorded result out of a run."""
    events = DATA_DIR / run_id / "events.jsonl"
    if not events.exists():
        raise SystemExit(f"no such run: {run_id} ({events} not found)")
    results = None
    with open(events, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("type") == "run.completed":
                results = (ev.get("payload") or {}).get("results")
    if not results:
        raise SystemExit(f"run {run_id} never completed")
    code = results.get("winner_code")
    if not code:
        raise SystemExit(f"run {run_id} produced no winning kernel")
    return code, results


def submit(code: str, leaderboard: str, gpu: str, mode: str,
           dry_run: bool) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "submission.py"
        path.write_text(code, encoding="utf-8")
        cmd = ["popcorn", "submit", "--leaderboard", leaderboard,
               "--gpu", gpu, "--mode", mode, "--no-tui", str(path)]
        print("$", " ".join(cmd))
        if dry_run:
            print("\n[dry run] not submitting. Re-run with --yes to send this to "
                  "the public GPU MODE leaderboard.")
            print("\n--- submission.py ---")
            print(code)
            return 0
        try:
            proc = subprocess.run(cmd, text=True)
        except FileNotFoundError:
            raise SystemExit(
                "popcorn-cli not found on PATH. Install it with:\n"
                "  curl -fsSL https://raw.githubusercontent.com/gpu-mode/"
                "popcorn-cli/main/install.sh | bash\n"
                "then authenticate with:  popcorn register discord")
        return proc.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("run_id", nargs="?", help="run whose winner to submit")
    src.add_argument("--file", help="submit this submission.py instead")
    ap.add_argument("--leaderboard", help="official leaderboard name "
                                          "(default: derived from the run)")
    ap.add_argument("--gpu", default="A100", help="A100 | H100 | B200 | T4 | L4")
    ap.add_argument("--mode", default="benchmark",
                    choices=["test", "benchmark", "leaderboard", "profile"])
    ap.add_argument("--yes", action="store_true",
                    help="actually submit (without this it is a dry run)")
    args = ap.parse_args(argv)

    if args.file:
        code = Path(args.file).read_text(encoding="utf-8")
        leaderboard = args.leaderboard
        if not leaderboard:
            raise SystemExit("--leaderboard is required with --file")
    else:
        code, results = winner_from_run(args.run_id)
        leaderboard = args.leaderboard or results.get("leaderboard")
        print(f"run {args.run_id}: winner \"{results.get('winner_branch_name')}\" "
              f"at {results.get('best_score')} ns "
              f"({results.get('improvement_pct')}% vs the local reference)")
        if not leaderboard:
            raise SystemExit(
                "could not derive the leaderboard name from the run — pass "
                "--leaderboard explicitly (e.g. grayscale_v2). The vendored "
                "problem directory name is not always the leaderboard name.")

    return submit(code, leaderboard, args.gpu, args.mode, dry_run=not args.yes)


if __name__ == "__main__":
    sys.exit(main())
