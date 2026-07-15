#!/usr/bin/env python3
"""
Modular benchmark: CSP vs classic runtime on MovieLens submatrices.

Target sizes approximate n_users x n_items ~= size, with n ~= sqrt(size).
Default sizes: 40, 100, 200, 400, 800, 1600, 3200, 6400.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    build_pipeline,
    load_movielens_u_data,
    recommend_classic_topn,
    save_csv_table,
    save_json,
    submatrix_from_size,
    timed_call,
)
from algos1_without_constraints import recommend_complete as recommend_csp_no_constraints


DEFAULT_SIZES = [40, 100, 200, 400, 800, 1600, 3200, 6400]


def benchmark_one(
    R,
    method: str,
    k: int,
    slate_size: int,
    n_candidates: int,
) -> dict:
    """
    Time classic Top-N (S1) vs CSP without business constraints on one matrix.

    W and R_hat are built once and reused for both selectors.
    """
    n_users, n_items = R.shape

    pipe, t_build = timed_call(
        build_pipeline,
        R,
        method=method,
        k=k,
        n_candidates=n_candidates,
    )

    _, t_classic = timed_call(
        recommend_classic_topn,
        pipe["R_hat"],
        pipe["Cu"],
        slate_size,
    )

    csp_result, t_csp = timed_call(
        recommend_csp_no_constraints,
        R,
        method=method,
        k=k,
        slate_size=slate_size,
        n_candidates=n_candidates,
        W=pipe["W"],
        Hu=pipe["Hu"],
    )

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_cells": n_users * n_items,
        "n_ratings": int((R > 0).sum()),
        "time_build_w_s": round(t_build, 6),
        "time_classic_s": round(t_classic, 6),
        "time_csp_s": round(t_csp, 6),
        "speedup_classic_over_csp": round(t_csp / t_classic, 4) if t_classic > 0 else None,
        "csp_status": csp_result["status"],
        "csp_objective": csp_result.get("objective"),
    }


def run_benchmark(
    sizes: list[int],
    method: str = "fc",
    k: int = 10,
    slate_size: int = 3,
    n_candidates: int = 50,
    u_data_path: Path | None = None,
) -> dict:
    """Benchmark S1 vs cardinality-only CSP across multiple submatrix sizes."""
    R_full = load_movielens_u_data(u_data_path)
    rows = []

    for target_size in sizes:
        R = submatrix_from_size(R_full, target_size)
        row = benchmark_one(R, method, k, slate_size, n_candidates)
        row["target_size"] = target_size
        row["method"] = method
        rows.append(row)
        print(
            f"size={target_size:5d} | {row['n_users']}x{row['n_items']} | "
            f"classic={row['time_classic_s']:.4f}s | csp={row['time_csp_s']:.4f}s | "
            f"ratio={row['speedup_classic_over_csp']}"
        )

    return {
        "dataset": "MovieLens 100K (u.data)",
        "method": method,
        "k": k,
        "slate_size": slate_size,
        "n_candidates": n_candidates,
        "sizes": sizes,
        "results": rows,
    }


def main() -> None:
    """CLI entry point for unconstrained CSP vs classic efficiency benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["fc", "svd", "both"],
        default="both",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--slate-size", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Target submatrix sizes (approximate number of cells)",
    )
    parser.add_argument(
        "--u-data",
        type=Path,
        default=None,
        help="Path to MovieLens u.data",
    )
    args = parser.parse_args()

    methods = ["fc", "svd"] if args.method == "both" else [args.method]
    all_results = {}

    for method in methods:
        print(f"\n=== Efficiency benchmark — {method.upper()} ===")
        result = run_benchmark(
            sizes=args.sizes,
            method=method,
            k=args.k,
            slate_size=args.slate_size,
            n_candidates=args.n_candidates,
            u_data_path=args.u_data,
        )
        all_results[method] = result
        save_json(result, f"benchmark_efficiency_{method}.json")
        save_csv_table(result["results"], f"benchmark_efficiency_{method}.csv")

    save_json(all_results, "benchmark_efficiency_all.json")
    print("\nResults: validation/results/benchmark_efficiency_all.json")


if __name__ == "__main__":
    main()
