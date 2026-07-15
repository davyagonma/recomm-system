#!/usr/bin/env python3
"""
Compare CSP+SVD without business constraints to classic SVD.

Classic baseline: Top-N on R_hat = R @ W_SVD.
CSP variant: cardinality constraint only (slate_size).
"""

from __future__ import annotations

import argparse

from common import (
    build_pipeline,
    compare_recommendation_dicts,
    matrix_5x5,
    recommend_classic_topn,
    save_csv_table,
    save_json,
    timed_call,
)


def run_comparison(
    R,
    slate_size: int = 2,
    k: int = 2,
    n_candidates: int | None = None,
    min_score: float | None = None,
) -> dict:
    """
    Compare CSP+SVD against classic Top-N on the same R_hat and candidate sets.

    Both approaches use cardinality only; no business constraints are applied.
    """
    pipe, t_pipe = timed_call(
        build_pipeline,
        R,
        method="svd",
        k=k,
        n_candidates=n_candidates,
        min_score=min_score,
    )

    csp_result, t_csp = timed_call(
        __import__("algos1_without_constraints", fromlist=["recommend_complete"]).recommend_complete,
        R,
        method="svd",
        k=k,
        slate_size=slate_size,
        n_candidates=n_candidates,
        min_score=min_score,
        W=pipe["W"],
        Hu=pipe["Hu"],
    )

    classic_rec, t_classic = timed_call(
        recommend_classic_topn,
        pipe["R_hat"],
        pipe["Cu"],
        slate_size,
    )

    comparison = compare_recommendation_dicts(
        csp_result["recommendations"],
        classic_rec,
        pipe["R_hat"],
        label_a="csp_svd",
        label_b="classic_svd",
    )

    return {
        "method": "svd",
        "k": k,
        "constraints": "none (slate_size only)",
        "matrix_shape": list(R.shape),
        "slate_size": slate_size,
        "n_candidates": n_candidates,
        "min_score": min_score,
        "timing_s": {
            "pipeline": round(t_pipe, 6),
            "csp": round(t_csp, 6),
            "classic": round(t_classic, 6),
        },
        "csp_status": csp_result["status"],
        "csp_objective": csp_result.get("objective"),
        "classic_objective": sum(
            float(pipe["R_hat"][u, i])
            for u, items in classic_rec.items()
            for i in items
        ),
        "comparison": comparison,
        "csp_recommendations": csp_result["recommendations"],
        "classic_recommendations": classic_rec,
    }


def main() -> None:
    """CLI entry point for SVD comparison without business constraints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slate-size", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n-candidates", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--use-memoire-matrix",
        action="store_true",
        help="Use the 4x4 toy matrix from the project README",
    )
    args = parser.parse_args()

    R = matrix_5x5()[:4, :4] if args.use_memoire_matrix else matrix_5x5()

    print(f"Matrix R: {R.shape[0]} users x {R.shape[1]} items")
    print(f"Parameters: method=svd, k={args.k}, slate_size={args.slate_size}")

    result = run_comparison(
        R,
        slate_size=args.slate_size,
        k=args.k,
        n_candidates=args.n_candidates,
        min_score=args.min_score,
    )

    json_path = save_json(result, "compare_svd_sans_contraintes.json")
    csv_path = save_csv_table(result["comparison"]["per_user"], "compare_svd_sans_contraintes.csv")

    print(f"\nAll users identical: {result['comparison']['all_identical']}")
    print(
        f"Identical users: {result['comparison']['n_identical_users']}/"
        f"{result['comparison']['n_users']}"
    )
    print(f"Mean Jaccard: {result['comparison']['mean_jaccard']:.4f}")
    print(f"CSP objective: {result['csp_objective']}")
    print(f"Classic objective: {result['classic_objective']}")
    print(
        f"Time CSP: {result['timing_s']['csp']:.4f}s | "
        f"classic: {result['timing_s']['classic']:.4f}s"
    )
    print(f"\nResults: {json_path}")
    print(f"Table:   {csv_path}")


if __name__ == "__main__":
    main()
