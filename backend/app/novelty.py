"""Novelty pressure: refuse to pay for programs the lab has already tried.

ShinkaEvolve-style novelty rejection, API-free: every parsed solver is
fingerprinted (canonicalized AST token n-grams, so renaming variables or
reshuffling comments does not fool it) and compared against everything the lab
has evaluated — this run AND the cross-run archive. A candidate that is a
near-copy of an existing program is bounced back to the Experimenter with an
explicit "this already exists, do something different" before any sandbox time
or evaluation budget is spent.

The gate is SOFT: after the retry allowance is exhausted the candidate runs
anyway (flagged as non-novel) — a stuck mock script or a stubborn model can
never wedge a run.
"""
from __future__ import annotations

import ast
import re
import threading

NGRAM = 4
_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


class _Canonicalizer(ast.NodeVisitor):
    """Emit a token stream where identifiers/constants are normalized so only
    the SHAPE of the computation remains."""

    def __init__(self):
        self.tokens: list[str] = []

    def generic_visit(self, node):
        self.tokens.append(type(node).__name__)
        if isinstance(node, ast.Name):
            self.tokens.append("v")
        elif isinstance(node, ast.Attribute):
            self.tokens.append(f".{node.attr}")
        elif isinstance(node, ast.Constant):
            self.tokens.append("num" if isinstance(node.value, (int, float)) else "lit")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.tokens.append("def")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # stdlib calls keep their name: min/sorted/range are structural
            self.tokens.append(f"call:{node.func.id}")
        super().generic_visit(node)


def fingerprint(code: str) -> frozenset:
    """Canonical n-gram fingerprint of a program. AST-based when the code
    parses; lexical fallback otherwise (a broken candidate still deserves
    dedup — resubmitting the same broken code is the worst waste)."""
    tokens: list[str]
    try:
        canon = _Canonicalizer()
        canon.visit(ast.parse(code))
        tokens = canon.tokens
    except SyntaxError:
        tokens = _FALLBACK_TOKEN_RE.findall(code.lower())
    if len(tokens) < NGRAM:
        return frozenset([" ".join(tokens)]) if tokens else frozenset()
    return frozenset(
        " ".join(tokens[i:i + NGRAM]) for i in range(len(tokens) - NGRAM + 1))


def similarity(fp_a: frozenset, fp_b: frozenset) -> float:
    """Containment similarity: how much of the smaller program is inside the
    larger one. Catches both near-copies and copy-plus-padding."""
    if not fp_a or not fp_b:
        return 0.0
    inter = len(fp_a & fp_b)
    return inter / min(len(fp_a), len(fp_b))


class NoveltyIndex:
    """Everything the lab has already tried, fingerprinted. Thread-safe."""

    def __init__(self):
        self._entries: list[tuple[str, frozenset]] = []  # (label, fingerprint)
        self._by_label: dict[str, frozenset] = {}
        self._lock = threading.Lock()

    def add(self, label: str, code: str) -> None:
        fp = fingerprint(code)
        with self._lock:
            self._entries.append((label, fp))
            self._by_label[label] = fp

    def nearest(self, code: str, exclude: set[str] | None = None
                ) -> tuple[float, str | None]:
        """(max similarity, label of the nearest existing program)."""
        fp = fingerprint(code)
        best_sim, best_label = 0.0, None
        with self._lock:
            entries = list(self._entries)
        for label, other in entries:
            if exclude and label in exclude:
                continue
            sim = similarity(fp, other)
            if sim > best_sim:
                best_sim, best_label = sim, label
        return best_sim, best_label

    def similarity_to(self, code: str, label: str) -> float:
        with self._lock:
            other = self._by_label.get(label)
        return similarity(fingerprint(code), other) if other else 0.0

    def size(self) -> int:
        with self._lock:
            return len(self._entries)
