"""Shared utilities for CSP vs classic validation."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos1 import (  # noqa: E402
    build_candidate_sets,
    build_history,
    build_W,
    recommend_complete as recommend_csp_with_constraints,
    solve_cp_recommendations,
)
from algos1_without_constraints import recommend_complete as recommend_csp_no_constraints  # noqa: E402
from streamlit_app.selection import (  # noqa: E402
    ConstraintConfig,
    select_heuristic,
    select_topn_classic,
)
from uniform import compute_R_hat  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def ensure_results_dir() -> Path:
    """Create the validation/results directory if needed and return its path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def load_movielens_u_data(path: str | Path | None = None) -> np.ndarray:
    """
    Build a user-item matrix from MovieLens u.data.

    Each row is a tab-separated rating: user_id, item_id, rating, timestamp.
    Missing ratings are stored as 0 in the pivot matrix.
    """
    path = Path(path or ROOT / "u.data")
    if not path.exists():
        raise FileNotFoundError(f"MovieLens file not found: {path}")

    ratings: list[tuple[int, int, float]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            user_id, item_id, rating = int(parts[0]), int(parts[1]), float(parts[2])
            ratings.append((user_id, item_id, rating))

    users = sorted({u for u, _, _ in ratings})
    items = sorted({i for _, i, _ in ratings})
    user_index = {u: idx for idx, u in enumerate(users)}
    item_index = {i: idx for idx, i in enumerate(items)}

    R = np.zeros((len(users), len(items)), dtype=float)
    for user_id, item_id, rating in ratings:
        R[user_index[user_id], item_index[item_id]] = rating

    return R


def submatrix_from_size(
    R: np.ndarray,
    target_size: int,
    min_unrated: int = 3,
) -> np.ndarray:
    """
    Extract a sparse submatrix of roughly sqrt(target_size) users.

    We keep about 3x as many items as users so that each user still has
    unrated candidates, which is required for meaningful Top-N benchmarks.
    """
    target_size = max(4, int(target_size))
    n = max(2, int(round(target_size**0.5)))
    m = min(R.shape[1], max(n + 1, 3 * n))

    user_activity = np.count_nonzero(R, axis=1)
    item_activity = np.count_nonzero(R, axis=0)

    top_users = np.argsort(user_activity)[::-1]
    top_items = np.argsort(item_activity)[::-1][:m]

    selected_users = []
    for user_idx in top_users:
        sub_row = R[user_idx, top_items]
        if np.count_nonzero(sub_row) > 0 and (len(top_items) - np.count_nonzero(sub_row)) >= min_unrated:
            selected_users.append(user_idx)
        if len(selected_users) >= n:
            break

    if len(selected_users) < 2:
        selected_users = top_users[:n].tolist()
    else:
        selected_users = selected_users[:n]

    return R[np.ix_(selected_users, top_items)]


def build_pipeline(
    R: np.ndarray,
    method: str = "fc",
    k: int = 2,
    n_candidates: int | None = None,
    exclude_consumed: bool = True,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Build W, R_hat, Hu, and Cu for matrix R."""
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
    return {"R": R, "Hu": Hu, "W": W, "R_hat": R_hat, "Cu": Cu}


def recommend_classic_topn(
    R_hat: np.ndarray,
    Cu: list[np.ndarray],
    slate_size: int,
) -> dict[int, list[int]]:
    """Classic Top-N (S1): sort candidates by descending score in R_hat."""
    n_users = R_hat.shape[0]
    return {
        u: select_topn_classic(u, Cu, R_hat, slate_size)
        for u in range(n_users)
    }


def recommend_classic_with_constraints(
    R_hat: np.ndarray,
    Cu: list[np.ndarray],
    slate_size: int,
    constraints: ConstraintConfig,
    pool_size: int | None = None,
) -> dict[int, list[int]]:
    """
    Classic Top-N plus heuristic post-processing (S2).

    This is the standard way to add constraints to a classic recommender:
    start from score-based ranking, then locally repair the slate when a
    business rule is violated.
    """
    n_users = R_hat.shape[0]
    return {
        u: select_heuristic(
            u, Cu, R_hat, slate_size, constraints, pool_size=pool_size
        )
        for u in range(n_users)
    }


def objective_value(
    recommendations: dict[int, list[int]],
    R_hat: np.ndarray,
) -> float:
    """Sum predicted scores over all users and all recommended items."""
    return float(
        sum(float(R_hat[u, i]) for u, items in recommendations.items() for i in items)
    )


def compare_recommendation_dicts(
    rec_a: dict[int, list[int]],
    rec_b: dict[int, list[int]],
    R_hat: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
) -> dict[str, Any]:
    """Compare two recommendation dictionaries user by user."""
    users = sorted(set(rec_a) | set(rec_b))
    per_user = []

    for u in users:
        set_a = set(rec_a.get(u, []))
        set_b = set(rec_b.get(u, []))
        per_user.append({
            "user": u,
            label_a: sorted(rec_a.get(u, [])),
            label_b: sorted(rec_b.get(u, [])),
            "identical": set_a == set_b,
            "jaccard": len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 1.0,
            f"objective_{label_a}": sum(float(R_hat[u, i]) for i in set_a),
            f"objective_{label_b}": sum(float(R_hat[u, i]) for i in set_b),
        })

    all_identical = all(row["identical"] for row in per_user)
    return {
        "all_identical": all_identical,
        "n_users": len(users),
        "n_identical_users": sum(1 for row in per_user if row["identical"]),
        "mean_jaccard": float(np.mean([row["jaccard"] for row in per_user])) if per_user else 1.0,
        "per_user": per_user,
    }


def timed_call(fn, *args, **kwargs) -> tuple[Any, float]:
    """Execute fn and return (result, elapsed_seconds) using perf_counter."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def save_json(data: dict, filename: str) -> Path:
    """Write a dictionary as indented JSON under validation/results/."""
    ensure_results_dir()
    path = RESULTS_DIR / filename
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=_json_default)
    return path


def save_csv_table(rows: list[dict], filename: str) -> Path:
    """Write a list of row dictionaries as CSV under validation/results/."""
    ensure_results_dir()
    path = RESULTS_DIR / filename
    if not rows:
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _json_default(obj):
    """JSON serializer hook for numpy scalars and arrays."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def matrix_5x5() -> np.ndarray:
    """Toy 5-user x 5-item matrix for constrained comparison tests."""
    return np.array([
        [5, 3, 0, 1, 0],
        [4, 2, 1, 0, 0],
        [1, 0, 5, 4, 0],
        [0, 4, 4, 5, 0],
        [0, 0, 2, 3, 0],  # item 4 remains unrated -> diverse category candidates
    ], dtype=float)


def constraints_5x5() -> dict[str, Any]:
    """Documented constraint set for the 5x5 toy matrix."""
    return {
        "item_categories": np.array([0, 0, 1, 1, 2]),
        "category_bounds": (0, 1),
        "item_providers": np.array([0, 1, 0, 1, 0]),
        "provider_bounds": (1, 8),
        "forbidden_pairs": [(0, 1), (2, 3)],
        "explanation_min": 0,
        "support_threshold": 1e-12,
    }
