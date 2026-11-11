"""Constraint configuration and feasibility checks (shared by tests and UI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ConstraintConfig:
    """Business constraints for parity / category / provider / explanation."""

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
        if self.category_min is None and self.category_max is None:
            return None
        return (self.category_min, self.category_max)

    def provider_bounds(self):
        if self.provider_min is None and self.provider_max is None:
            return None
        return (self.provider_min, self.provider_max)


def constraint_violations(
    selected: list[int],
    constraints: ConstraintConfig,
    *,
    global_provider_counts: dict[int, int] | None = None,
) -> list[str]:
    issues: list[str] = []

    if constraints.item_categories is not None:
        lower, upper = constraints.category_bounds() or (None, None)
        if lower is not None or upper is not None:
            counts: dict[int, int] = {}
            for item in selected:
                cat = int(constraints.item_categories[item])
                counts[cat] = counts.get(cat, 0) + 1
            for cat, count in counts.items():
                if lower is not None and count < lower:
                    issues.append(f"category_min:{cat}")
                if upper is not None and count > upper:
                    issues.append(f"category_max:{cat}")

    for i, j in constraints.forbidden_pairs:
        if i in selected and j in selected:
            issues.append(f"pair:{i},{j}")

    if (
        constraints.item_providers is not None
        and global_provider_counts is not None
        and (constraints.provider_min is not None or constraints.provider_max is not None)
    ):
        lower, upper = constraints.provider_bounds() or (None, None)
        for provider, count in global_provider_counts.items():
            if lower is not None and count < lower:
                issues.append(f"provider_min:{provider}")
            if upper is not None and count > upper:
                issues.append(f"provider_max:{provider}")

    return issues


def constraints_are_active(constraints: ConstraintConfig) -> bool:
    if constraints.forbidden_pairs:
        return True
    if constraints.explanation_min > 0:
        return True
    if constraints.category_min is not None or constraints.category_max is not None:
        return True
    if constraints.provider_min is not None or constraints.provider_max is not None:
        return True
    return False


def compliance_summary(
    selected: list[int],
    constraints: ConstraintConfig,
    *,
    slate_size: int,
    global_provider_counts: dict[int, int] | None = None,
) -> dict[str, Any]:
    violations = constraint_violations(
        selected, constraints, global_provider_counts=global_provider_counts
    )
    return {
        "n_violations": len(violations),
        "violations": violations,
        "complete": len(selected) >= slate_size,
        "constraint_satisfied": len(violations) == 0 and len(selected) >= slate_size,
    }
