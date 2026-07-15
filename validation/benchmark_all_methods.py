#!/usr/bin/env python3
"""
Benchmark all classic constraint-handling methods against CSP (S4).

Methods compared:
  S1  Top-N only
  S2  Top-N + local heuristic repair
  S3  Hybrid greedy (soft diversity)
  C4  Constraint-aware greedy
  C5  Filter-then-rank
  C6  Exhaustive search on bounded pool
  C7  Global sequential greedy (provider exposure)
  C8  Top-N + explanation post-filter
  S4  CSP / CPMpy (reference)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VALIDATION = Path(__file__).resolve().parent
for path in (str(ROOT), str(VALIDATION)):
    if path not in sys.path:
        sys.path.insert(0, path)

from algos1 import solve_cp_recommendations  # noqa: E402
from classic_methods import CLASSIC_METHODS, run_classic_method
from common import (
    build_pipeline,
    constraints_5x5,
    load_movielens_u_data,
    matrix_5x5,
    save_csv_table,
    save_json,
    submatrix_from_size,
    timed_call,
)
from streamlit_app.selection import ConstraintConfig


def _constraint_config_from_dict(params: dict) -> ConstraintConfig:
    """Build a ConstraintConfig object from a plain parameter dictionary."""
    bounds = params.get("category_bounds")
    provider_bounds = params.get("provider_bounds")
    return ConstraintConfig(
        item_categories=params.get("item_categories"),
        category_min=bounds[0] if bounds else None,
        category_max=bounds[1] if bounds else None,
        forbidden_pairs=list(params.get("forbidden_pairs", [])),
        item_providers=params.get("item_providers"),
        provider_min=provider_bounds[0] if provider_bounds else None,
        provider_max=provider_bounds[1] if provider_bounds else None,
        explanation_min=params.get("explanation_min", 0),
        support_threshold=params.get("support_threshold", 1e-12),
    )


def _build_movielens_constraints(n_items: int) -> dict:
    """Synthetic metadata aligned with MovieLens item columns."""
    item_categories = np.arange(n_items) % 5
    item_providers = np.arange(n_items) % 4
    return {
        "item_categories": item_categories,
        "category_bounds": (0, 1),
        "item_providers": item_providers,
        "provider_bounds": (2, n_items),
        "forbidden_pairs": [],
        "explanation_min": 0,
        "support_threshold": 1e-12,
    }


def run_csp(
    pipe: dict,
    slate_size: int,
    constraints: ConstraintConfig,
) -> dict:
    """Run the CSP reference model (S4)."""
    csp_kwargs = {
        "R_hat": pipe["R_hat"],
        "Cu": pipe["Cu"],
        "slate_size": slate_size,
        "Hu": pipe["Hu"],
        "W": pipe["W"],
        "solver_name": "ortools",
        "explanation_min": constraints.explanation_min,
        "support_threshold": constraints.support_threshold,
        "forbidden_pairs": constraints.forbidden_pairs or None,
    }
    if constraints.item_categories is not None and constraints.category_bounds() is not None:
        csp_kwargs["item_categories"] = constraints.item_categories
        csp_kwargs["category_bounds"] = constraints.category_bounds()
    if constraints.item_providers is not None and constraints.provider_bounds() is not None:
        csp_kwargs["item_providers"] = constraints.item_providers
        csp_kwargs["provider_bounds"] = constraints.provider_bounds()

    result, elapsed = timed_call(solve_cp_recommendations, **csp_kwargs)
    objective = result.get("objective")
    if objective is None:
        objective = 0.0

    n_users = pipe["R_hat"].shape[0]
    feasible_users = sum(
        1 for u in range(n_users) if len(result.get("recommendations", {}).get(u, [])) == slate_size
    )

    return {
        "method_id": "S4",
        "label": "CSP / CPMpy (reference)",
        "scope": "global",
        "elapsed_s": round(elapsed, 6),
        "recommendations": result.get("recommendations", {}),
        "objective": objective,
        "csp_status": result.get("status", False),
        "feasible_users": feasible_users if result.get("status") else 0,
        "all_users_feasible": bool(result.get("status")),
        "total_violations": 0 if result.get("status") else n_users,
    }


def benchmark_matrix(
    R: np.ndarray,
    constraints_params: dict,
    method: str = "fc",
    k: int = 2,
    slate_size: int = 2,
    n_candidates: int | None = None,
    pool_size: int | None = None,
    classic_ids: list[str] | None = None,
) -> dict:
    """
    Benchmark all requested classic methods and CSP (S4) on one rating matrix.

    Builds W and R_hat once, runs each selector, and compares timing, objective,
    feasibility, and violations against the CSP reference.
    """
    pipe, t_build = timed_call(
        build_pipeline,
        R,
        method=method,
        k=k,
        n_candidates=n_candidates,
    )
    constraints = _constraint_config_from_dict(constraints_params)
    classic_ids = [m for m in (classic_ids or list(CLASSIC_METHODS)) if m != "S4"]

    rows = []
    for method_id in classic_ids:
        result = run_classic_method(
            method_id=method_id,
            R_hat=pipe["R_hat"],
            Cu=pipe["Cu"],
            slate_size=slate_size,
            constraints=constraints,
            Hu=pipe["Hu"],
            W=pipe["W"],
            item_providers=constraints_params.get("item_providers"),
            pool_size=pool_size,
        )
        rows.append({
            "method_id": method_id,
            "label": result["label"],
            "scope": result["scope"],
            "time_s": result["elapsed_s"],
            "objective": round(result["objective"], 6),
            "feasible_users": result["feasible_users"],
            "all_feasible": result["all_users_feasible"],
            "total_violations": result["total_violations"],
            "ratio_vs_fastest": None,
        })

    csp = run_csp(pipe, slate_size, constraints)
    rows.append({
        "method_id": csp["method_id"],
        "label": csp["label"],
        "scope": csp["scope"],
        "time_s": csp["elapsed_s"],
        "objective": round(float(csp["objective"]), 6),
        "feasible_users": csp["feasible_users"],
        "all_feasible": csp["all_users_feasible"],
        "total_violations": csp["total_violations"],
        "ratio_vs_fastest": None,
    })

    fastest = min(row["time_s"] for row in rows if row["time_s"] > 0)
    for row in rows:
        row["ratio_vs_fastest"] = round(row["time_s"] / fastest, 4) if fastest > 0 else None

    csp_objective = float(csp["objective"])
    for row in rows:
        row["objective_gap_vs_csp"] = round(csp_objective - row["objective"], 6)
        row["slower_than_csp"] = row["time_s"] > csp["elapsed_s"]

    return {
        "matrix_shape": list(R.shape),
        "method_w": method,
        "k": k,
        "slate_size": slate_size,
        "time_build_w_s": round(t_build, 6),
        "constraints": {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in constraints_params.items()
        },
        "results": rows,
        "csp_reference": csp,
    }


def benchmark_movielens_sizes(
    sizes: list[int],
    method: str = "fc",
    k: int = 10,
    slate_size: int = 3,
    n_candidates: int = 30,
    pool_size: int = 12,
    u_data_path: Path | None = None,
    classic_ids: list[str] | None = None,
) -> dict:
    """
    Run benchmark_matrix on several MovieLens submatrix sizes.

    Each target_size extracts roughly sqrt(size) active users and 3x items.
    """
    R_full = load_movielens_u_data(u_data_path)
    all_rows = []

    for target_size in sizes:
        R = submatrix_from_size(R_full, target_size)
        params = _build_movielens_constraints(R.shape[1])
        result = benchmark_matrix(
            R,
            params,
            method=method,
            k=k,
            slate_size=slate_size,
            n_candidates=n_candidates,
            pool_size=pool_size,
            classic_ids=classic_ids,
        )
        for row in result["results"]:
            all_rows.append({
                "target_size": target_size,
                "n_users": R.shape[0],
                "n_items": R.shape[1],
                **row,
            })
        print(f"size={target_size:5d} | {R.shape[0]}x{R.shape[1]} done")

    return {
        "dataset": "MovieLens 100K",
        "method_w": method,
        "k": k,
        "slate_size": slate_size,
        "n_candidates": n_candidates,
        "pool_size": pool_size,
        "sizes": sizes,
        "results": all_rows,
    }


def main() -> None:
    """CLI entry point for 5x5 and MovieLens multi-method benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=["5x5", "movielens", "both"],
        default="both",
    )
    parser.add_argument("--method", choices=["fc", "svd", "both"], default="fc")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--slate-size", type=int, default=2)
    parser.add_argument("--n-candidates", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=12)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 400, 1600, 6400],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(CLASSIC_METHODS),
        help="Classic method ids to include",
    )
    args = parser.parse_args()

    w_methods = ["fc", "svd"] if args.method == "both" else [args.method]
    outputs = {}

    if args.scenario in {"5x5", "both"}:
        for w_method in w_methods:
            print(f"\n=== 5x5 constrained benchmark — {w_method.upper()} ===")
            result = benchmark_matrix(
                matrix_5x5(),
                constraints_5x5(),
                method=w_method,
                k=args.k,
                slate_size=args.slate_size,
                pool_size=args.pool_size,
                classic_ids=args.methods,
            )
            outputs[f"5x5_{w_method}"] = result
            save_json(result, f"benchmark_all_methods_5x5_{w_method}.json")
            save_csv_table(result["results"], f"benchmark_all_methods_5x5_{w_method}.csv")
            for row in result["results"]:
                print(
                    f"  {row['method_id']:3s} | {row['time_s']:.4f}s | "
                    f"obj={row['objective']:.4f} | feasible={row['all_feasible']} | "
                    f"violations={row['total_violations']}"
                )

    if args.scenario in {"movielens", "both"}:
        for w_method in w_methods:
            print(f"\n=== MovieLens constrained benchmark — {w_method.upper()} ===")
            k_val = args.k if w_method == "svd" else 10
            result = benchmark_movielens_sizes(
                sizes=args.sizes,
                method=w_method,
                k=k_val,
                slate_size=max(3, args.slate_size),
                n_candidates=args.n_candidates,
                pool_size=args.pool_size,
                classic_ids=args.methods,
            )
            outputs[f"movielens_{w_method}"] = result
            save_json(result, f"benchmark_all_methods_movielens_{w_method}.json")
            save_csv_table(result["results"], f"benchmark_all_methods_movielens_{w_method}.csv")

    save_json(outputs, "benchmark_all_methods_all.json")
    print("\nSaved: validation/results/benchmark_all_methods_all.json")


if __name__ == "__main__":
    main()
