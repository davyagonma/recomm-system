"""
Classic Top-N strategies for handling business constraints.

Each method starts from the same scores R_hat and candidate sets Cu as the CSP.
They differ in how (and whether) constraints are enforced after ranking.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Any, Callable

import numpy as np

from streamlit_app.selection import (
    ConstraintConfig,
    _violations,
    select_heuristic,
    select_hybrid_greedy,
    select_topn_classic,
)

MethodFn = Callable[..., list[int]]


def _score_sum(u: int, items: list[int], R_hat: np.ndarray) -> float:
    """Return the sum of predicted scores for one user's slate."""
    return float(sum(float(R_hat[u, i]) for i in items))


def _provider_counts(
    recommendations: dict[int, list[int]],
    item_providers: np.ndarray | None,
) -> dict[int, int]:
    """Aggregate provider exposure counts across all recommended slates."""
    counts: dict[int, int] = {}
    if item_providers is None:
        return counts
    for items in recommendations.values():
        for item in items:
            provider = int(item_providers[item])
            counts[provider] = counts.get(provider, 0) + 1
    return counts


def count_user_violations(
    slate: list[int],
    constraints: ConstraintConfig,
    global_provider_counts: dict[int, int] | None = None,
) -> int:
    """Count how many constraint violations a slate triggers."""
    return len(_violations(slate, constraints, global_provider_counts))


def is_user_feasible(
    slate: list[int],
    constraints: ConstraintConfig,
    global_provider_counts: dict[int, int] | None = None,
) -> bool:
    """Return True when the slate satisfies all checked constraints."""
    return count_user_violations(slate, constraints, global_provider_counts) == 0


def has_explanation_support(
    u: int,
    item: int,
    Hu: np.ndarray,
    W: np.ndarray,
    explanation_min: int,
    support_threshold: float,
) -> bool:
    """
    Check whether item i has enough history support for user u.

    A history item j supports i when abs(W[j, i]) exceeds support_threshold.
    """
    if explanation_min <= 0:
        return True
    consumed = np.where(Hu[u] == 1)[0]
    supports = sum(
        1 for j in consumed if abs(float(W[int(j), item])) > support_threshold
    )
    return supports >= explanation_min


# ---------------------------------------------------------------------------
# S1 — Top-N only (constraints ignored)
# ---------------------------------------------------------------------------

def method_s1_topn(
    u: int,
    Cu: list[np.ndarray],
    R_hat: np.ndarray,
    slate_size: int,
    **_kwargs: Any,
) -> list[int]:
    """S1: pure score ranking, no constraint handling."""
    return select_topn_classic(u, Cu, R_hat, slate_size)


# ---------------------------------------------------------------------------
# S2 — Top-N + local heuristic repair
# ---------------------------------------------------------------------------

def method_s2_heuristic_repair(
    u: int,
    Cu: list[np.ndarray],
    R_hat: np.ndarray,
    slate_size: int,
    constraints: ConstraintConfig,
    pool_size: int | None = None,
    **_kwargs: Any,
) -> list[int]:
    """S2: start from top pool, swap lowest-scoring item until local constraints hold."""
    return select_heuristic(
        u, Cu, R_hat, slate_size, constraints, pool_size=pool_size
    )


# ---------------------------------------------------------------------------
# S3 — Hybrid greedy (soft diversity, not hard constraints)
# ---------------------------------------------------------------------------

def method_s3_hybrid_greedy(
    u: int,
    Cu: list[np.ndarray],
    R_hat: np.ndarray,
    slate_size: int,
    constraints: ConstraintConfig,
    diversity_weight: float = 0.5,
    genre_matrix: np.ndarray | None = None,
    **_kwargs: Any,
) -> list[int]:
    """S3: relevance + diversity score; does not guarantee hard constraint satisfaction."""
    return select_hybrid_greedy(
        u,
        Cu,
        R_hat,
        slate_size,
        diversity_weight,
        constraints.item_categories,
        genre_matrix,
    )


# ---------------------------------------------------------------------------
# C4 — Constraint-aware greedy construction (local hard constraints)
# ---------------------------------------------------------------------------

def method_c4_constraint_greedy(
    u: int,
    Cu: list[np.ndarray],
    R_hat: np.ndarray,
    slate_size: int,
    constraints: ConstraintConfig,
    global_provider_counts: dict[int, int] | None = None,
    **_kwargs: Any,
) -> list[int]:
    """
    C4: repeatedly pick the highest-scoring item that keeps the partial slate feasible.

    Only checks constraints that can be evaluated on the current partial slate.
    """
    pool = list(Cu[u])
    slate: list[int] = []

    while len(slate) < slate_size and pool:
        best_item = None
        best_score = -np.inf
        for item in pool:
            if item in slate:
                continue
            trial = slate + [item]
            if is_user_feasible(trial, constraints, global_provider_counts):
                score = float(R_hat[u, item])
                if score > best_score:
                    best_score = score
                    best_item = item
        if best_item is None:
            break
        slate.append(int(best_item))

    return slate[:slate_size]


# ---------------------------------------------------------------------------
# C5 — Filter-then-rank (category pre-filter + Top-N)
# ---------------------------------------------------------------------------

def method_c5_filter_then_rank(
    u: int,
    Cu: list[np.ndarray],
    R_hat: np.ndarray,
    slate_size: int,
    constraints: ConstraintConfig,
    **_kwargs: Any,
) -> list[int]:
    """
    C5: keep the best item per category, rank them, take Top-N, then repair with S2.

    This is a cheap pre-filter for category diversity before ranking.
    """
    pool = list(Cu[u])
    if constraints.item_categories is None:
        return select_topn_classic(u, Cu, R_hat, slate_size)

    best_per_category: dict[int, int] = {}
    for item in pool:
        category = int(constraints.item_categories[item])
        if (
            category not in best_per_category
            or float(R_hat[u, item]) > float(R_hat[u, best_per_category[category]])
        ):
            best_per_category[category] = int(item)

    filtered = sorted(
        best_per_category.values(),
        key=lambda item: float(R_hat[u, item]),
        reverse=True,
    )
    if len(filtered) < slate_size:
        return method_s2_heuristic_repair(
            u, Cu, R_hat, slate_size, constraints, pool_size=len(pool)
        )

    fake_Cu = list(Cu)
    fake_Cu[u] = np.array(filtered, dtype=int)
    return method_s2_heuristic_repair(
        u, fake_Cu, R_hat, slate_size, constraints, pool_size=len(filtered)
    )


# ---------------------------------------------------------------------------
# C6 — Exhaustive search on a bounded pool (locally optimal in pool)
# ---------------------------------------------------------------------------

def method_c6_exhaustive_pool(
    u: int,
    Cu: list[np.ndarray],
    R_hat: np.ndarray,
    slate_size: int,
    constraints: ConstraintConfig,
    pool_size: int = 12,
    global_provider_counts: dict[int, int] | None = None,
    **_kwargs: Any,
) -> list[int]:
    """
    C6: enumerate all slate_size-combinations from the top pool_size candidates.

    Exponential cost O(C(pool, slate_size)) but optimal within the pool for local constraints.
    Falls back to S2 when the pool is too small or no feasible slate exists.
    """
    pool = list(Cu[u][:pool_size])
    if len(pool) < slate_size:
        return method_s2_heuristic_repair(
            u, Cu, R_hat, slate_size, constraints, pool_size=pool_size
        )

    best_slate: list[int] | None = None
    best_score = -np.inf

    for combo in combinations(pool, slate_size):
        slate = list(combo)
        if not is_user_feasible(slate, constraints, global_provider_counts):
            continue
        score = _score_sum(u, slate, R_hat)
        if score > best_score:
            best_score = score
            best_slate = slate

    if best_slate is not None:
        return best_slate

    return method_s2_heuristic_repair(
        u, Cu, R_hat, slate_size, constraints, pool_size=pool_size
    )


# ---------------------------------------------------------------------------
# C7 — Global sequential assignment (local + provider exposure)
# ---------------------------------------------------------------------------

def method_c7_global_sequential(
    user_indices: list[int],
    Cu: list[np.ndarray],
    R_hat: np.ndarray,
    slate_size: int,
    constraints: ConstraintConfig,
    item_providers: np.ndarray | None = None,
    **_kwargs: Any,
) -> dict[int, list[int]]:
    """
    C7: process users one by one while updating global provider counts.

    This is the classic answer to cross-user provider bounds, but order-dependent.
    """
    recommendations: dict[int, list[int]] = {}
    global_counts: dict[int, int] = {}

    for u in user_indices:
        slate = method_c4_constraint_greedy(
            u,
            Cu,
            R_hat,
            slate_size,
            constraints,
            global_provider_counts=global_counts,
        )
        recommendations[u] = slate
        if item_providers is not None:
            for item in slate:
                provider = int(item_providers[item])
                global_counts[provider] = global_counts.get(provider, 0) + 1

    return recommendations


# ---------------------------------------------------------------------------
# C8 — Top-N + explanation post-filter and refill
# ---------------------------------------------------------------------------

def method_c8_explanation_filter(
    u: int,
    Cu: list[np.ndarray],
    R_hat: np.ndarray,
    slate_size: int,
    constraints: ConstraintConfig,
    Hu: np.ndarray,
    W: np.ndarray,
    pool_size: int | None = None,
    **_kwargs: Any,
) -> list[int]:
    """
    C8: build a Top-N slate, drop items without explanatory support, refill greedily.

    Mimics how one might bolt explainability onto a classic pipeline.
    """
    pool_size = pool_size or max(slate_size * 3, slate_size)
    pool = list(Cu[u][:pool_size])
    slate: list[int] = []

    for item in pool:
        if len(slate) >= slate_size:
            break
        if not has_explanation_support(
            u,
            item,
            Hu,
            W,
            constraints.explanation_min,
            constraints.support_threshold,
        ):
            continue
        trial = slate + [item]
        if is_user_feasible(trial, constraints):
            slate.append(int(item))

    if len(slate) < slate_size:
        slate = method_s2_heuristic_repair(
            u, Cu, R_hat, slate_size, constraints, pool_size=pool_size
        )
    return slate[:slate_size]


CLASSIC_METHODS: dict[str, dict[str, Any]] = {
    "S1": {
        "label": "Top-N only (constraints ignored)",
        "scope": "per-user",
        "handles_global_providers": False,
        "handles_explanation": False,
        "guarantees_feasibility": False,
        "complexity": "O(n log n)",
        "fn": method_s1_topn,
    },
    "S2": {
        "label": "Top-N + local heuristic repair",
        "scope": "per-user",
        "handles_global_providers": False,
        "handles_explanation": False,
        "guarantees_feasibility": False,
        "complexity": "O(pool * iterations)",
        "fn": method_s2_heuristic_repair,
    },
    "S3": {
        "label": "Hybrid greedy (relevance + soft diversity)",
        "scope": "per-user",
        "handles_global_providers": False,
        "handles_explanation": False,
        "guarantees_feasibility": False,
        "complexity": "O(slate * candidates)",
        "fn": method_s3_hybrid_greedy,
    },
    "C4": {
        "label": "Constraint-aware greedy construction",
        "scope": "per-user",
        "handles_global_providers": False,
        "handles_explanation": False,
        "guarantees_feasibility": False,
        "complexity": "O(slate * candidates * checks)",
        "fn": method_c4_constraint_greedy,
    },
    "C5": {
        "label": "Filter-then-rank (category pre-filter + repair)",
        "scope": "per-user",
        "handles_global_providers": False,
        "handles_explanation": False,
        "guarantees_feasibility": False,
        "complexity": "O(categories + S2)",
        "fn": method_c5_filter_then_rank,
    },
    "C6": {
        "label": "Exhaustive search on bounded pool",
        "scope": "per-user",
        "handles_global_providers": False,
        "handles_explanation": False,
        "guarantees_feasibility": False,
        "complexity": "O(C(pool, slate))",
        "fn": method_c6_exhaustive_pool,
    },
    "C7": {
        "label": "Global sequential greedy (provider exposure)",
        "scope": "global",
        "handles_global_providers": True,
        "handles_explanation": False,
        "guarantees_feasibility": False,
        "complexity": "O(users * slate * candidates)",
        "fn": method_c7_global_sequential,
    },
    "C8": {
        "label": "Top-N + explanation post-filter",
        "scope": "per-user",
        "handles_global_providers": False,
        "handles_explanation": True,
        "guarantees_feasibility": False,
        "complexity": "O(pool + S2)",
        "fn": method_c8_explanation_filter,
    },
}


def run_classic_method(
    method_id: str,
    R_hat: np.ndarray,
    Cu: list[np.ndarray],
    slate_size: int,
    constraints: ConstraintConfig,
    Hu: np.ndarray | None = None,
    W: np.ndarray | None = None,
    item_providers: np.ndarray | None = None,
    pool_size: int | None = None,
    diversity_weight: float = 0.5,
) -> dict[str, Any]:
    """
    Run one classic selection method for all users and collect metrics.

    Returns timing, objective value, feasibility counts, and per-user details.
    """
    meta = CLASSIC_METHODS[method_id]
    fn = meta["fn"]
    n_users = R_hat.shape[0]
    user_indices = list(range(n_users))
    t0 = time.perf_counter()

    if method_id == "C7":
        recommendations = fn(
            user_indices=user_indices,
            Cu=Cu,
            R_hat=R_hat,
            slate_size=slate_size,
            constraints=constraints,
            item_providers=item_providers,
        )
    else:
        recommendations = {}
        for u in user_indices:
            kwargs = {
                "u": u,
                "Cu": Cu,
                "R_hat": R_hat,
                "slate_size": slate_size,
                "constraints": constraints,
                "pool_size": pool_size,
                "diversity_weight": diversity_weight,
            }
            if method_id == "C8":
                kwargs["Hu"] = Hu
                kwargs["W"] = W
            recommendations[u] = fn(**kwargs)

    elapsed = time.perf_counter() - t0
    global_counts = _provider_counts(recommendations, item_providers)

    per_user = []
    total_violations = 0
    feasible_users = 0
    objective = 0.0

    for u in user_indices:
        slate = recommendations.get(u, [])
        violations = _violations(slate, constraints, global_counts)
        n_viol = len(violations)
        total_violations += n_viol
        if n_viol == 0 and len(slate) == slate_size:
            feasible_users += 1
        obj_u = _score_sum(u, slate, R_hat) if slate else 0.0
        objective += obj_u
        per_user.append({
            "user": u,
            "slate": slate,
            "n_violations": n_viol,
            "violations": violations,
            "objective": obj_u,
        })

    return {
        "method_id": method_id,
        "label": meta["label"],
        "scope": meta["scope"],
        "elapsed_s": round(elapsed, 6),
        "recommendations": recommendations,
        "objective": objective,
        "feasible_users": feasible_users,
        "all_users_feasible": feasible_users == n_users,
        "total_violations": total_violations,
        "per_user": per_user,
    }
