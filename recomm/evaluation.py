"""
Evaluation protocol aligned with ``test/``: classical greedy Top-N/FC vs CSP (CPMpy).

Three scenarios (same as the thesis experiments):
  1. Simple — score-only Top-N (no parity constraints).
  2. With parity — forbidden pairs (same rule on classical and CSP).
  3. Two constraints — forbidden pairs + category cap (completeness + time).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from recomm.constraints import ConstraintConfig, compliance_summary, constraints_are_active
from recomm.operators import compute_R_hat
from recomm.pipeline import (
    build_W,
    build_candidate_sets,
    build_history,
    solve_cp_recommendations,
)

MethodId = Literal["classical", "csp"]


@dataclass(frozen=True)
class EvalScenario:
    id: str
    label: str
    detail: str


EVAL_SCENARIOS: tuple[EvalScenario, ...] = (
    EvalScenario(
        "simple",
        "Simple (no parity)",
        "Classical Top-N by R̂ vs CSP with cardinality only — same scoring, no business rules.",
    ),
    EvalScenario(
        "parity",
        "With parity",
        "Forbidden pairs applied to both methods (as in ``test/simple_csp_with_parity``).",
    ),
    EvalScenario(
        "two_constraints",
        "Parity + category cap",
        "Forbidden pairs plus max items per genre/category (second constraint).",
    ),
)


def classical_topn(u: int, Cu: list[np.ndarray], slate_size: int) -> list[int]:
    """Pure Top-N on the candidate list (unconstrained)."""
    return [int(i) for i in Cu[u][:slate_size]]


def classical_greedy_constrained(
    u: int,
    Cu: list[np.ndarray],
    slate_size: int,
    constraints: ConstraintConfig,
) -> list[int]:
    """
    Greedy FC-style selection on sorted candidates (``test/simple_csp_with_parity.recommend_fc``).
    """
    slate: list[int] = []
    forbidden = constraints.forbidden_pairs or []

    def blocked(item: int, current: list[int]) -> bool:
        for other in current:
            if (item, other) in forbidden or (other, item) in forbidden:
                return True
        if constraints.item_categories is not None and constraints.category_max is not None:
            cat = int(constraints.item_categories[item])
            same = sum(
                1 for i in current if int(constraints.item_categories[i]) == cat
            )
            if same >= constraints.category_max:
                return True
        return False

    for item in Cu[u]:
        item = int(item)
        if blocked(item, slate):
            continue
        slate.append(item)
        if len(slate) >= slate_size:
            break
    return slate


def scenario_constraints(
    scenario_id: str,
    metadata: dict,
    *,
    forbidden_pairs: list[tuple[int, int]],
    category_max: int,
) -> ConstraintConfig:
    base = ConstraintConfig(
        item_categories=metadata.get("item_categories"),
        item_providers=metadata.get("item_providers"),
    )
    if scenario_id == "simple":
        return base
    if scenario_id == "parity":
        return ConstraintConfig(
            item_categories=base.item_categories,
            item_providers=base.item_providers,
            forbidden_pairs=list(forbidden_pairs),
        )
    if scenario_id == "two_constraints":
        return ConstraintConfig(
            item_categories=base.item_categories,
            item_providers=base.item_providers,
            forbidden_pairs=list(forbidden_pairs),
            category_min=0,
            category_max=category_max,
        )
    raise ValueError(f"Unknown scenario: {scenario_id}")


def _score_sum(u: int, slate: list[int], R_hat: np.ndarray) -> float:
    return float(sum(float(R_hat[u, i]) for i in slate))


def run_one(
    method: MethodId,
    *,
    u: int,
    R_hat: np.ndarray,
    Cu: list[np.ndarray],
    Hu: np.ndarray,
    W: np.ndarray,
    slate_size: int,
    constraints: ConstraintConfig,
    solver_name: str = "ortools",
) -> dict[str, Any]:
    t0 = time.perf_counter()

    if method == "classical":
        if constraints_are_active(constraints):
            slate = classical_greedy_constrained(u, Cu, slate_size, constraints)
        else:
            slate = classical_topn(u, Cu, slate_size)
        elapsed = time.perf_counter() - t0
        comp = compliance_summary(slate, constraints, slate_size=slate_size)
        return {
            "method": "classical",
            "method_label": "Classical (greedy Top-N / FC)",
            "slate": slate,
            "status": True,
            "cp_feasible": None,
            "elapsed_s": elapsed,
            "score_sum": _score_sum(u, slate, R_hat),
            **comp,
        }

    cp_kwargs: dict[str, Any] = {
        "solver_name": solver_name,
        "forbidden_pairs": constraints.forbidden_pairs or None,
        "explanation_min": constraints.explanation_min,
        "support_threshold": constraints.support_threshold,
    }
    if constraints.item_categories is not None and constraints.category_bounds() is not None:
        cp_kwargs["item_categories"] = constraints.item_categories
        cp_kwargs["category_bounds"] = constraints.category_bounds()

    sub_R = R_hat[u : u + 1, :]
    sub_Cu = [Cu[u]]
    sub_Hu = Hu[u : u + 1, :]

    cp_result = solve_cp_recommendations(
        R_hat=sub_R,
        Cu=sub_Cu,
        slate_size=slate_size,
        Hu=sub_Hu,
        W=W,
        **cp_kwargs,
    )
    elapsed = time.perf_counter() - t0
    slate = cp_result["recommendations"].get(0, [])
    comp = compliance_summary(slate, constraints, slate_size=slate_size)

    return {
        "method": "csp",
        "method_label": "CSP (CPMpy)",
        "slate": slate,
        "status": bool(cp_result["status"]),
        "cp_feasible": bool(cp_result["status"]),
        "elapsed_s": elapsed,
        "score_sum": _score_sum(u, slate, R_hat) if slate else 0.0,
        "objective": cp_result.get("objective"),
        **comp,
    }


def run_evaluation_protocol(
    R: np.ndarray,
    user_index: int,
    *,
    metadata: dict,
    method: str = "svd",
    k: int = 20,
    n_candidates: int = 50,
    slate_size: int = 5,
    exclude_consumed: bool = True,
    min_score: float | None = None,
    forbidden_pairs: list[tuple[int, int]] | None = None,
    category_max: int = 2,
    solver_name: str = "ortools",
) -> dict[str, Any]:
    """Run all scenarios × {classical, CSP} for one user."""
    forbidden_pairs = forbidden_pairs or []
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

    rows: list[dict[str, Any]] = {}
    by_scenario: dict[str, dict[str, dict]] = {}

    for scenario in EVAL_SCENARIOS:
        cfg = scenario_constraints(
            scenario.id,
            metadata,
            forbidden_pairs=forbidden_pairs,
            category_max=category_max,
        )
        by_scenario[scenario.id] = {}
        for method_id in ("classical", "csp"):
            out = run_one(
                method_id,
                u=user_index,
                R_hat=R_hat,
                Cu=Cu,
                Hu=Hu,
                W=W,
                slate_size=slate_size,
                constraints=cfg,
                solver_name=solver_name,
            )
            by_scenario[scenario.id][method_id] = out
            rows[f"{scenario.id}:{method_id}"] = {
                "scenario_id": scenario.id,
                "scenario": scenario.label,
                **out,
            }

    return {
        "user_index": user_index,
        "R_hat": R_hat,
        "Cu": Cu,
        "W": W,
        "Hu": Hu,
        "slate_size": slate_size,
        "rows": rows,
        "by_scenario": by_scenario,
        "scenarios": EVAL_SCENARIOS,
    }
