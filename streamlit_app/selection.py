"""Top-N selection strategies S1 to S4 (thesis chapter 3)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from algos1 import recommend_complete, solve_cp_recommendations


@dataclass
class ConstraintConfig:
    """Constraint parameters shared by S2, S3, and S4."""

    item_categories: np.ndarray | None = None
    category_min: int | None = None
    category_max: int | None = None
    forbidden_pairs: list[tuple[int, int]] = field(default_factory=list)
    item_providers: np.ndarray | None = None
    provider_min: int | None = None
    provider_max: int | None = None
    explanation_min: int = 0
    support_threshold: float = 1e-12

    def category_bounds(self):
        """Return (min, max) per-category bounds, or None if unset."""
        if self.category_min is None and self.category_max is None:
            return None
        return (self.category_min, self.category_max)

    def provider_bounds(self):
        """Return (min, max) global provider exposure bounds, or None if unset."""
        if self.provider_min is None and self.provider_max is None:
            return None
        return (self.provider_min, self.provider_max)


def _violations(
    selected: list[int],
    constraints: ConstraintConfig,
    global_provider_counts: dict[int, int] | None = None,
) -> list[str]:
    """List constraint violations triggered by a slate."""
    issues: list[str] = []

    if constraints.item_categories is not None:
        lb, ub = constraints.category_bounds() or (None, None)
        if lb is not None or ub is not None:
            counts: dict[int, int] = {}
            for item in selected:
                cat = int(constraints.item_categories[item])
                counts[cat] = counts.get(cat, 0) + 1
            for cat, count in counts.items():
                if lb is not None and count < lb:
                    issues.append(f"category_min:{cat}")
                if ub is not None and count > ub:
                    issues.append(f"category_max:{cat}")

    for i, j in constraints.forbidden_pairs:
        if i in selected and j in selected:
            issues.append(f"pair:{i},{j}")

    if (
        constraints.item_providers is not None
        and global_provider_counts is not None
        and (constraints.provider_min is not None or constraints.provider_max is not None)
    ):
        lb, ub = constraints.provider_bounds() or (None, None)
        for provider, count in global_provider_counts.items():
            if lb is not None and count < lb:
                issues.append(f"provider_min:{provider}")
            if ub is not None and count > ub:
                issues.append(f"provider_max:{provider}")

    return issues


def _score_sum(selected: list[int], scores: np.ndarray) -> float:
    """Return the sum of predicted scores for items in a slate."""
    return float(sum(float(scores[i]) for i in selected))


def _dissimilarity(
    item: int,
    slate: list[int],
    item_categories: np.ndarray | None,
    genre_matrix: np.ndarray | None,
) -> float:
    """Mean dissimilarity between item and items already in the partial slate."""
    if not slate:
        return 1.0
    values = []
    for other in slate:
        if genre_matrix is not None:
            overlap = np.dot(genre_matrix[item], genre_matrix[other])
            values.append(0.0 if overlap > 0 else 1.0)
        elif item_categories is not None:
            values.append(1.0 if item_categories[item] != item_categories[other] else 0.0)
        else:
            values.append(1.0 if item != other else 0.0)
    return float(np.mean(values))


def select_topn_classic(
    u: int,
    Cu: list[np.ndarray],
    R_hat: np.ndarray,
    slate_size: int,
) -> list[int]:
    """S1 — classic Top-N by descending score."""
    ordered = list(Cu[u][:slate_size])
    if len(ordered) < slate_size:
        candidates = np.argsort(R_hat[u])[::-1]
        seen = set(ordered)
        for item in candidates:
            item = int(item)
            if item not in seen:
                ordered.append(item)
                seen.add(item)
            if len(ordered) >= slate_size:
                break
    return ordered[:slate_size]


def select_heuristic(
    u: int,
    Cu: list[np.ndarray],
    R_hat: np.ndarray,
    slate_size: int,
    constraints: ConstraintConfig,
    pool_size: int | None = None,
) -> list[int]:
    """S2 — heuristic post-processing (local replacement)."""
    pool_size = pool_size or max(slate_size * 3, slate_size)
    pool = list(Cu[u][:pool_size])
    if len(pool) < slate_size:
        return select_topn_classic(u, Cu, R_hat, slate_size)

    slate = pool[:slate_size]
    remaining = pool[slate_size:]

    for _ in range(len(pool) * 2):
        issues = _violations(slate, constraints)
        if not issues:
            break

        worst_item = min(slate, key=lambda i: float(R_hat[u, i]))
        replacement = None
        for candidate in remaining:
            if candidate in slate:
                continue
            trial = [i for i in slate if i != worst_item] + [candidate]
            if not _violations(trial, constraints):
                replacement = candidate
                break

        if replacement is None:
            for candidate in remaining:
                if candidate in slate:
                    continue
                trial = [i for i in slate if i != worst_item] + [candidate]
                if len(_violations(trial, constraints)) < len(_violations(slate, constraints)):
                    replacement = candidate
                    break

        if replacement is None:
            break

        slate = [i for i in slate if i != worst_item] + [replacement]

    return slate[:slate_size]


def select_hybrid_greedy(
    u: int,
    Cu: list[np.ndarray],
    R_hat: np.ndarray,
    slate_size: int,
    diversity_weight: float,
    item_categories: np.ndarray | None = None,
    genre_matrix: np.ndarray | None = None,
) -> list[int]:
    """S3 — greedy selection combining relevance and diversity."""
    candidates = list(Cu[u])
    slate: list[int] = []

    while len(slate) < slate_size and candidates:
        best_item = None
        best_score = -np.inf
        for item in candidates:
            if item in slate:
                continue
            relevance = float(R_hat[u, item])
            diversity = _dissimilarity(item, slate, item_categories, genre_matrix)
            combined = relevance + diversity_weight * diversity
            if combined > best_score:
                best_score = combined
                best_item = item
        if best_item is None:
            break
        slate.append(int(best_item))

    return slate


def run_selection(
    approach: str,
    R: np.ndarray,
    R_hat: np.ndarray,
    Cu: list[np.ndarray],
    W: np.ndarray,
    Hu: np.ndarray,
    user_indices: list[int],
    slate_size: int,
    constraints: ConstraintConfig,
    diversity_weight: float = 0.5,
    solver_name: str = "ortools",
    heuristic_pool_size: int | None = None,
    genre_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Run S1, S2, S3, or S4 for a list of users (matrix row indices).

    Returns recommendations, timing, objectives, and feasibility status.
    """
    approach = approach.upper()
    t0 = time.perf_counter()
    recommendations: dict[int, list[int]] = {}
    explanations: dict[int, dict[int, list[int]]] = {}
    objectives: dict[int, float] = {}
    feasible = True

    if approach == "S4":
        sub_Cu = [Cu[u] for u in user_indices]
        sub_R_hat = R_hat[user_indices, :]
        sub_Hu = Hu[user_indices, :]
        index_map = {local: global_u for local, global_u in enumerate(user_indices)}

        cp_kwargs = {
            "solver_name": solver_name,
            "explanation_min": constraints.explanation_min,
            "support_threshold": constraints.support_threshold,
            "forbidden_pairs": constraints.forbidden_pairs or None,
        }
        if constraints.item_categories is not None and constraints.category_bounds() is not None:
            cp_kwargs["item_categories"] = constraints.item_categories
            cp_kwargs["category_bounds"] = constraints.category_bounds()
        if constraints.item_providers is not None and constraints.provider_bounds() is not None:
            cp_kwargs["item_providers"] = constraints.item_providers
            cp_kwargs["provider_bounds"] = constraints.provider_bounds()

        result = solve_cp_recommendations(
            R_hat=sub_R_hat,
            Cu=sub_Cu,
            slate_size=slate_size,
            Hu=sub_Hu,
            W=W,
            **cp_kwargs,
        )

        feasible = bool(result["status"])
        for local_u, global_u in index_map.items():
            recommendations[global_u] = result["recommendations"].get(local_u, [])
            explanations[global_u] = result.get("explanations", {}).get(local_u, {})
            if recommendations[global_u]:
                objectives[global_u] = _score_sum(recommendations[global_u], R_hat[global_u])

        return {
            "status": feasible,
            "approach": approach,
            "recommendations": recommendations,
            "explanations": explanations,
            "objectives": objectives,
            "objective": result.get("objective"),
            "elapsed_s": time.perf_counter() - t0,
            "R_hat": R_hat,
            "Cu": Cu,
            "W": W,
            "cp_result": result,
        }

    for u in user_indices:
        if approach == "S1":
            slate = select_topn_classic(u, Cu, R_hat, slate_size)
        elif approach == "S2":
            slate = select_heuristic(
                u, Cu, R_hat, slate_size, constraints, pool_size=heuristic_pool_size
            )
        elif approach == "S3":
            slate = select_hybrid_greedy(
                u,
                Cu,
                R_hat,
                slate_size,
                diversity_weight,
                constraints.item_categories,
                genre_matrix,
            )
        else:
            raise ValueError(f"Unknown approach: {approach}")

        recommendations[u] = slate
        objectives[u] = _score_sum(slate, R_hat[u])
        explanations[u] = {}

    return {
        "status": True,
        "approach": approach,
        "recommendations": recommendations,
        "explanations": explanations,
        "objectives": objectives,
        "objective": sum(objectives.values()) if objectives else None,
        "elapsed_s": time.perf_counter() - t0,
        "R_hat": R_hat,
        "Cu": Cu,
        "W": W,
    }


def run_full_pipeline(
    R: np.ndarray,
    method: str = "svd",
    k: int = 20,
    n_candidates: int = 50,
    exclude_consumed: bool = True,
    min_score: float | None = None,
    approach: str = "S4",
    user_indices: list[int] | None = None,
    slate_size: int = 5,
    constraints: ConstraintConfig | None = None,
    diversity_weight: float = 0.5,
    solver_name: str = "ortools",
    genre_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full pipeline: R → W → R_hat → Cu → selection."""
    from algos1 import build_candidate_sets, build_history, build_W
    from uniform import compute_R_hat

    constraints = constraints or ConstraintConfig()
    R = np.asarray(R, dtype=float)
    Hu = build_history(R)
    W = build_W(R, method=method, k=k)
    R_hat = compute_R_hat(R, W)
    Cu = build_candidate_sets(
        R_hat=R_hat,
        Hu=Hu,
        n_candidates=n_candidates,
        exclude_consumed=exclude_consumed,
        min_score=min_score,
    )

    if user_indices is None:
        user_indices = list(range(R.shape[0]))

    selection = run_selection(
        approach=approach,
        R=R,
        R_hat=R_hat,
        Cu=Cu,
        W=W,
        Hu=Hu,
        user_indices=user_indices,
        slate_size=slate_size,
        constraints=constraints,
        diversity_weight=diversity_weight,
        solver_name=solver_name,
        genre_matrix=genre_matrix,
    )

    selection["method"] = method
    selection["k"] = k
    selection["Hu"] = Hu
    selection["slate_size"] = slate_size
    selection["n_candidates"] = n_candidates
    return selection
