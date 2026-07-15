#!/usr/bin/env python3
"""
Large-scale benchmark: when do classic methods exceed CSP runtime?

Tests increasing MovieLens submatrix sizes with constrained selection.
Focus on methods most likely to become slower than CSP (C6, S2, S3, C4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATION = Path(__file__).resolve().parent
for path in (str(ROOT), str(VALIDATION)):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_all_methods import benchmark_matrix  # noqa: E402
from common import load_movielens_u_data, save_csv_table, save_json, submatrix_from_size  # noqa: E402


DEFAULT_SIZES = [1600, 6400, 16000, 36000, 64000]
DEFAULT_METHODS = ["S2", "S3", "C4", "C6", "C5", "S4"]


def _movielens_constraints(n_items: int) -> dict:
    """Build synthetic category and provider metadata for MovieLens columns."""
    import numpy as np

    return {
        "item_categories": np.arange(n_items) % 5,
        "category_bounds": (0, 1),
        "item_providers": np.arange(n_items) % 4,
        "provider_bounds": (2, n_items),
        "forbidden_pairs": [],
        "explanation_min": 0,
        "support_threshold": 1e-12,
    }


def run_large_benchmark(
    sizes: list[int],
    method_w: str = "fc",
    k: int = 10,
    slate_size: int = 3,
    n_candidates: int = 50,
    pool_sizes: list[int] | None = None,
    classic_ids: list[str] | None = None,
) -> dict:
    """
    Benchmark classic methods at increasing matrix sizes and pool sizes.

    For each submatrix, CSP timing is measured once, then each classic method
    is compared against that reference across multiple pool sizes.
    """
    pool_sizes = pool_sizes or [12, 25, 30]
    classic_ids = [m for m in (classic_ids or DEFAULT_METHODS) if m != "S4"]
    R_full = load_movielens_u_data()
    all_rows = []

    for target_size in sizes:
        R = submatrix_from_size(R_full, target_size)
        params = _movielens_constraints(R.shape[1])
        n_users, n_items = R.shape

        # CSP reference once per matrix size
        csp_only = benchmark_matrix(
            R,
            params,
            method=method_w,
            k=k,
            slate_size=slate_size,
            n_candidates=n_candidates,
            pool_size=pool_sizes[0],
            classic_ids=[],
        )
        csp_row = next(r for r in csp_only["results"] if r["method_id"] == "S4")
        csp_time = csp_row["time_s"]
        csp_objective = csp_row["objective"]

        for pool_size in pool_sizes:
            result = benchmark_matrix(
                R,
                params,
                method=method_w,
                k=k,
                slate_size=slate_size,
                n_candidates=n_candidates,
                pool_size=pool_size,
                classic_ids=classic_ids,
            )
            for row in result["results"]:
                if row["method_id"] == "S4":
                    continue
                all_rows.append({
                    "target_size": target_size,
                    "n_users": n_users,
                    "n_items": n_items,
                    "n_ratings": int((R > 0).sum()),
                    "pool_size": pool_size,
                    "method_id": row["method_id"],
                    "label": row["label"],
                    "time_s": row["time_s"],
                    "csp_time_s": csp_time,
                    "time_ratio_vs_csp": round(row["time_s"] / csp_time, 4) if csp_time > 0 else None,
                    "slower_than_csp": row["time_s"] > csp_time,
                    "objective": row["objective"],
                    "csp_objective": csp_objective,
                    "all_feasible": row["all_feasible"],
                    "total_violations": row["total_violations"],
                })

        # Append CSP row once per size
        all_rows.append({
            "target_size": target_size,
            "n_users": n_users,
            "n_items": n_items,
            "n_ratings": int((R > 0).sum()),
            "pool_size": None,
            "method_id": "S4",
            "label": "CSP / CPMpy (reference)",
            "time_s": csp_time,
            "csp_time_s": csp_time,
            "time_ratio_vs_csp": 1.0,
            "slower_than_csp": False,
            "objective": csp_objective,
            "csp_objective": csp_objective,
            "all_feasible": csp_row["all_feasible"],
            "total_violations": csp_row["total_violations"],
        })

        slower = [r for r in all_rows if r["target_size"] == target_size and r.get("slower_than_csp")]
        names = ", ".join(f"{r['method_id']}(pool={r['pool_size']})" for r in slower) or "none"
        print(f"size={target_size:6d} | {n_users}x{n_items} | CSP={csp_time:.3f}s | slower: {names}")

    return {
        "dataset": "MovieLens 100K",
        "method_w": method_w,
        "slate_size": slate_size,
        "n_candidates": n_candidates,
        "pool_sizes": pool_sizes,
        "sizes": sizes,
        "results": all_rows,
    }


def main() -> None:
    """CLI entry point for large-scale classic vs CSP crossover benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["fc", "svd"], default="fc")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--slate-size", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[12, 25, 30])
    args = parser.parse_args()

    result = run_large_benchmark(
        sizes=args.sizes,
        method_w=args.method,
        k=args.k,
        slate_size=args.slate_size,
        n_candidates=args.n_candidates,
        pool_sizes=args.pool_sizes,
    )
    save_json(result, "benchmark_large_scale.json")
    save_csv_table(result["results"], "benchmark_large_scale.csv")
    print("\nSaved validation/results/benchmark_large_scale.csv")


if __name__ == "__main__":
    main()
