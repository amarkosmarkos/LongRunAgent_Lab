"""Git substrate for a run: the version DAG is a real git repository.

Every run owns a bare-ish repo at data/runs/<id>/repo. Each solver attempt is a
real commit (solver.py + meta.json) on the git branch of its hypothesis; a
supervisor merge is a true two-parent merge commit. The branch graph the UI
draws is therefore a faithful view of `git log --graph`, and every program's
full lineage (who mutated whom, with which scores) is inspectable with plain
git tooling.

Implementation notes:
 - Only plumbing commands are used (hash-object / mktree / commit-tree /
   update-ref), so there is no index and no working tree to fight over —
   parallel branches can commit concurrently without checkouts or locks on
   files. A single process lock serializes the (fast) git calls.
 - If git is not installed the substrate degrades to a no-op: the run works
   exactly as before, just without the on-disk DAG.
"""
from __future__ import annotations

import json
import subprocess
import threading


class RunRepo:
    def __init__(self, run_dir, run_id: str):
        self.path = run_dir / "repo"
        self.run_id = run_id
        self.available = False
        self._lock = threading.Lock()
        self._root_sha: str | None = None
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            self._git("init", "-q", "-b", "master")
            self._git("config", "user.email", "lab@longrun.local")
            self._git("config", "user.name", "Long Run Agent Lab")
            self.available = True
        except (OSError, subprocess.SubprocessError):
            self.available = False

    # ------------------------------------------------------------- plumbing
    def _git(self, *args: str, input_text: str | None = None) -> str:
        out = subprocess.run(
            ["git", *args], cwd=str(self.path), capture_output=True,
            text=True, encoding="utf-8",
            input=input_text, timeout=30)
        if out.returncode != 0:
            raise subprocess.SubprocessError(
                f"git {' '.join(args)}: {out.stderr.strip()}")
        return out.stdout.strip()

    def _blob(self, content: str) -> str:
        return self._git("hash-object", "-w", "--stdin", input_text=content)

    def _tree(self, files: dict[str, str]) -> str:
        entries = "".join(
            f"100644 blob {self._blob(content)}\t{name}\n"
            for name, content in sorted(files.items()))
        return self._git("mktree", input_text=entries)

    def _commit(self, tree: str, parents: list[str], message: str) -> str:
        args = ["commit-tree", tree]
        for p in parents:
            args += ["-p", p]
        return self._git(*args, input_text=message)

    # -------------------------------------------------------------- public
    def head(self, branch: str) -> str | None:
        if not self.available:
            return None
        try:
            with self._lock:
                return self._git("rev-parse", f"refs/heads/{branch}")
        except subprocess.SubprocessError:
            return None

    def commit_root(self, baseline_code: str, meta: dict) -> str | None:
        """The root of the DAG: the baseline solver every lineage descends from."""
        if not self.available:
            return None
        try:
            with self._lock:
                tree = self._tree({
                    "solver.py": baseline_code,
                    "meta.json": json.dumps(meta, indent=2),
                })
                sha = self._commit(tree, [], "baseline\n\n" + json.dumps(meta))
                self._git("update-ref", "refs/heads/master", sha)
                self._root_sha = sha
                return sha
        except subprocess.SubprocessError:
            return None

    def create_branch(self, branch: str, from_sha: str | None = None) -> str | None:
        """Point a new hypothesis branch at its starting commit."""
        if not self.available:
            return None
        try:
            with self._lock:
                base = from_sha or self._root_sha
                if base is None:
                    return None
                self._git("update-ref", f"refs/heads/{branch}", base)
                return base
        except subprocess.SubprocessError:
            return None

    def commit_attempt(self, branch: str, code: str, meta: dict,
                       extra_parents: list[str] | None = None) -> str | None:
        """One solver attempt = one commit on the hypothesis branch.

        meta (score, valid, round, operator, behavior, ...) is stored both as
        meta.json in the tree and in the commit message, so `git log` alone
        tells the whole story. extra_parents turns the commit into a true
        merge commit (crossover)."""
        if not self.available:
            return None
        try:
            with self._lock:
                parents = []
                try:
                    parents.append(self._git("rev-parse", f"refs/heads/{branch}"))
                except subprocess.SubprocessError:
                    if self._root_sha:
                        parents.append(self._root_sha)
                for p in extra_parents or []:
                    if p and p not in parents:
                        parents.append(p)
                tree = self._tree({
                    "solver.py": code,
                    "meta.json": json.dumps(meta, indent=2),
                })
                summary = meta.get("summary") or (
                    f"r{meta.get('round', '?')} "
                    f"score={meta.get('score')} valid={meta.get('valid')}")
                sha = self._commit(tree, parents, summary + "\n\n" + json.dumps(meta))
                self._git("update-ref", f"refs/heads/{branch}", sha)
                return sha
        except subprocess.SubprocessError:
            return None

    def log_graph(self, max_count: int = 400) -> str | None:
        """Human-readable DAG (for the conclusion event / debugging)."""
        if not self.available:
            return None
        try:
            with self._lock:
                return self._git("log", "--all", "--graph", "--oneline",
                                 f"--max-count={max_count}")
        except subprocess.SubprocessError:
            return None
