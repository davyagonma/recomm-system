"""Métriques de qualité des listes Top-N."""

from __future__ import annotations

import numpy as np


def intra_list_diversity(
    selected: list[int],
    item_categories: np.ndarray | None = None,
    genre_matrix: np.ndarray | None = None,
) -> float:
    """Diversité intra-liste : proportion de paires d'items dissimilaires."""
    if len(selected) < 2:
        return 0.0

    pairs = 0
    diverse = 0
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            pairs += 1
            a, b = selected[i], selected[j]
            if genre_matrix is not None:
                ga = genre_matrix[a]
                gb = genre_matrix[b]
                overlap = np.dot(ga, gb)
                if overlap == 0:
                    diverse += 1
            elif item_categories is not None:
                if item_categories[a] != item_categories[b]:
                    diverse += 1
            else:
                diverse += 1 if a != b else 0

    return diverse / pairs if pairs else 0.0


def slate_metrics(
    selected: list[int],
    scores: np.ndarray,
    item_categories: np.ndarray | None = None,
    genre_matrix: np.ndarray | None = None,
) -> dict:
    """Résumé numérique d'une slate pour un utilisateur."""
    if not selected:
        return {
            "count": 0,
            "score_sum": 0.0,
            "score_mean": 0.0,
            "score_min": 0.0,
            "ild": 0.0,
            "n_categories": 0,
        }

    slate_scores = [float(scores[i]) for i in selected]
    categories = (
        {int(item_categories[i]) for i in selected}
        if item_categories is not None
        else set()
    )

    return {
        "count": len(selected),
        "score_sum": sum(slate_scores),
        "score_mean": float(np.mean(slate_scores)),
        "score_min": min(slate_scores),
        "ild": intra_list_diversity(selected, item_categories, genre_matrix),
        "n_categories": len(categories),
    }
