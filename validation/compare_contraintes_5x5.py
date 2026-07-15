#!/usr/bin/env python3
"""
Compare CSP with constraints to classic selection with constraints on a 5x5 matrix.

Shows how constraints are added to the classic pipeline:
  1. Compute R_hat = R @ W (FC or SVD)
  2. Build candidate sets Cu
  3. Top-N by score (S1)
  4. Heuristic repair pass (S2) to reduce constraint violations
"""

from __future__ import annotations

import argparse

from common import (
    build_pipeline,
    compare_recommendation_dicts,
    constraints_5x5,
    matrix_5x5,
    recommend_classic_with_constraints,
    save_csv_table,
    save_json,
    timed_call,
)
from streamlit_app.selection import ConstraintConfig, _violations


def _constraint_config(params: dict) -> ConstraintConfig:
    """Convert the 5x5 constraint dictionary into a ConstraintConfig object."""
    return ConstraintConfig(
        item_categories=params["item_categories"],
        category_min=params["category_bounds"][0],
        category_max=params["category_bounds"][1],
        forbidden_pairs=list(params["forbidden_pairs"]),
        item_providers=params["item_providers"],
        provider_min=params["provider_bounds"][0],
        provider_max=params["provider_bounds"][1],
        explanation_min=params["explanation_min"],
        support_threshold=params["support_threshold"],
    )


def run_comparison(
    method: str = "fc",
    k: int = 2,
    slate_size: int = 2,
) -> dict:
    """
    Compare CSP with full constraints against classic S2 on the toy 5x5 matrix.

    Returns timing, objectives, feasibility flags, and per-user recommendation diffs.
    """
    R = matrix_5x5()
    params = constraints_5x5()
    cfg = _constraint_config(params)

    pipe, t_pipe = timed_call(build_pipeline, R, method=method, k=k)

    from algos1 import recommend_complete as recommend_csp

    csp_result, t_csp = timed_call(
        recommend_csp,
        R,
        method=method,
        k=k,
        slate_size=slate_size,
        W=pipe["W"],
        Hu=pipe["Hu"],
        item_categories=params["item_categories"],
        category_bounds=params["category_bounds"],
        item_providers=params["item_providers"],
        provider_bounds=params["provider_bounds"],
        forbidden_pairs=params["forbidden_pairs"],
        explanation_min=params["explanation_min"],
        support_threshold=params["support_threshold"],
    )

    classic_rec, t_classic = timed_call(
        recommend_classic_with_constraints,
        pipe["R_hat"],
        pipe["Cu"],
        slate_size,
        cfg,
        pool_size=slate_size * 3,
    )

    comparison = compare_recommendation_dicts(
        csp_result["recommendations"] if csp_result["status"] else {},
        classic_rec,
        pipe["R_hat"],
        label_a=f"csp_{method}",
        label_b=f"classic_{method}",
    )

    classic_violations = {
        u: _violations(items, cfg)
        for u, items in classic_rec.items()
    }

    return {
        "method": method,
        "matrix_shape": [5, 5],
        "matrix_R": R.tolist(),
        "slate_size": slate_size,
        "constraints": {
            "item_categories": params["item_categories"].tolist(),
            "category_bounds": params["category_bounds"],
            "item_providers": params["item_providers"].tolist(),
            "provider_bounds": params["provider_bounds"],
            "forbidden_pairs": params["forbidden_pairs"],
            "explanation_min": params["explanation_min"],
        },
        "classic_constraint_strategy": (
            "Top-N by score (S1), then heuristic repair (S2) that replaces "
            "the lowest-scoring item involved in a violation"
        ),
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
        "csp_recommendations": csp_result.get("recommendations", {}),
        "classic_recommendations": classic_rec,
        "csp_explanations": csp_result.get("explanations", {}),
        "classic_violations": classic_violations,
        "classic_feasible": all(len(v) == 0 for v in classic_violations.values()),
    }


def main() -> None:
    """CLI entry point for constrained 5x5 CSP vs classic comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["fc", "svd", "both"],
        default="both",
        help="Method used to build W",
    )
    parser.add_argument("--slate-size", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()

    methods = ["fc", "svd"] if args.method == "both" else [args.method]
    all_results = {}

    for method in methods:
        print(f"\n=== Constrained comparison — {method.upper()} — 5x5 matrix ===")
        result = run_comparison(method=method, k=args.k, slate_size=args.slate_size)
        all_results[method] = result

        suffix = f"compare_contraintes_5x5_{method}"
        save_json(result, f"{suffix}.json")
        save_csv_table(result["comparison"]["per_user"], f"{suffix}.csv")

        print(f"CSP feasible: {result['csp_status']}")
        print(f"Identical: {result['comparison']['all_identical']}")
        print(
            f"CSP objective: {result['csp_objective']} | "
            f"classic: {result['classic_objective']}"
        )
        print(
            f"Time CSP: {result['timing_s']['csp']:.4f}s | "
            f"classic: {result['timing_s']['classic']:.4f}s"
        )

        for row in result["comparison"]["per_user"]:
            u = row["user"]
            print(
                f"  User {u}: CSP={row[f'csp_{method}']} | "
                f"Classic={row[f'classic_{method}']} | identical={row['identical']}"
            )

    save_json(all_results, "compare_contraintes_5x5_all.json")
    print("\nSummary: validation/results/compare_contraintes_5x5_all.json")


if __name__ == "__main__":
    main()
