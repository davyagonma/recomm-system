import argparse
import csv
import sys
import time
from pathlib import Path

import cpmpy as cp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app.metadata import build_item_metadata


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator != 0 else 0.0


def item_similarity_matrix(R: np.ndarray) -> np.ndarray:
    """Build the item-item cosine similarity matrix used by the FC script."""
    item_norms = np.linalg.norm(R, axis=0)
    denominator = np.outer(item_norms, item_norms)
    similarity = np.divide(
        R.T @ R,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    np.fill_diagonal(similarity, 0.0)
    return similarity


def predict_rating(
    R: np.ndarray,
    similarity: np.ndarray,
    user_id: int,
    item_id: int,
    k: int = 3,
) -> float:
    """Predict one score using the same FC formula."""
    user_ratings = R[user_id]
    rated_items = np.where(user_ratings > 0)[0]

    if len(rated_items) == 0:
        return 0.0

    rated_items_array = np.fromiter(rated_items, dtype=int)
    similarities = similarity[item_id, rated_items_array]
    neighbor_count = min(k, len(rated_items_array))
    neighbor_positions = np.argsort(
        -np.abs(similarities),
        kind="stable",
    )[:neighbor_count]
    neighbor_similarities = similarities[neighbor_positions]
    neighbor_ratings = user_ratings[rated_items_array[neighbor_positions]]

    numerator = np.dot(neighbor_similarities, neighbor_ratings)
    denominator = np.abs(neighbor_similarities).sum()
    return float(numerator / denominator) if denominator != 0 else 0.0


def recommend_fc(
    scores: np.ndarray,
    rated_items: set[int],
    top_n: int,
    forbidden_pairs: list[tuple[int, int]],
    item_genres: dict[int, str],
    max_items_per_genre: int,
    max_candidates: int | None = None,
) -> list[tuple[int, float]]:
    """Select recommendations greedily, as in the FC algorithm."""
    recommendations: list[tuple[int, float]] = []
    ordered_items = sorted(
        (
            (item, float(score))
            for item, score in enumerate(scores)
            if item not in rated_items
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if max_candidates is not None:
        ordered_items = ordered_items[:max_candidates]

    for item, score in ordered_items:
        conflicts_with_pair = any(
            (item, selected) in forbidden_pairs
            or (selected, item) in forbidden_pairs
            for selected, _ in recommendations
        )
        if conflicts_with_pair:
            continue

        genre = item_genres.get(item)
        if genre is not None:
            genre_count = sum(
                item_genres.get(selected) == genre
                for selected, _ in recommendations
            )
            if genre_count >= max_items_per_genre:
                continue

        recommendations.append((item, float(score)))
        if len(recommendations) == top_n:
            break

    return recommendations


def solve_simple_csp(
    scores: np.ndarray,
    rated_items: set[int],
    top_n: int,
    forbidden_pairs: list[tuple[int, int]],
    item_genres: dict[int, str],
    max_items_per_genre: int = 1,
    max_candidates: int | None = None,
) -> list[tuple[int, float]]:
    """Select the highest-scoring valid slate with parity and diversity."""
    candidates = [item for item in range(len(scores)) if item not in rated_items]
    if max_candidates is not None:
        candidates = sorted(
            candidates,
            key=lambda item: scores[item],
            reverse=True,
        )[:max_candidates]
    selected = {item: cp.boolvar(name=f"x_{item}") for item in candidates}

    constraints = [cp.sum(selected.values()) <= top_n]

    for first, second in forbidden_pairs:
        if first in selected and second in selected:
            constraints.append(selected[first] + selected[second] <= 1)

    genres = set(item_genres.values())
    for genre in genres:
        genre_items = [
            selected[item]
            for item in candidates
            if item_genres.get(item) == genre
        ]
        if genre_items:
            constraints.append(cp.sum(genre_items) <= max_items_per_genre)

    objective = cp.sum(scores[item] * variable for item, variable in selected.items())
    model = cp.Model(constraints, maximize=objective)

    if not model.solve():
        raise RuntimeError("No feasible recommendation slate was found.")

    recommendations = [
        (item, float(scores[item]))
        for item, variable in selected.items()
        if variable.value() == 1
    ]
    return sorted(recommendations, key=lambda pair: pair[1], reverse=True)


def load_pivot_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a pivot CSV whose first column is the user identifier."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError(
                "Le CSV doit contenir une colonne user_id et au moins un item."
            )

    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(header):
        raise ValueError("Le nombre de colonnes du CSV est incohérent.")

    try:
        item_labels = np.array([int(item) for item in header[1:]])
    except ValueError as error:
        raise ValueError(
            "Les en-têtes d'items du CSV doivent être des identifiants numériques."
        ) from error

    return data[:, 1:], data[:, 0].astype(int), item_labels


def parse_pair(value: str) -> tuple[int, int]:
    """Parse a forbidden pair written as ITEM1:ITEM2."""
    try:
        first, second = value.split(":", maxsplit=1)
        return int(first), int(second)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Une paire interdite doit être écrite ITEM1:ITEM2."
        ) from error


def resolve_pairs(
    pairs: list[tuple[int, int]],
    item_labels: np.ndarray,
) -> list[tuple[int, int]]:
    """Convert item identifiers from the CLI into matrix column indices."""
    positions = {int(label): index for index, label in enumerate(item_labels)}
    resolved = []
    for first, second in pairs:
        if first not in positions or second not in positions:
            raise ValueError(
                f"Paire inconnue: {first}:{second}. "
                "Les identifiants doivent exister dans l'en-tête du CSV."
            )
        resolved.append((positions[first], positions[second]))
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare les recommandations FC et CSP avec parité."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV pivoté (première colonne: user_id). Sans cette option, utilise l'exemple intégré.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Identifiant utilisateur présent dans le CSV (par défaut: premier utilisateur).",
    )
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Nombre maximal de candidats non notés; utile pour accélérer un grand CSV.",
    )
    parser.add_argument(
        "--forbidden-pair",
        type=parse_pair,
        action="append",
        default=[],
        metavar="ITEM1:ITEM2",
        help="Paire d'indices d'items interdite; peut être répétée.",
    )
    parser.add_argument("--max-items-per-genre", type=int, default=1)
    parser.add_argument("--u-item", type=Path, default=Path("u.item"))
    parser.add_argument(
        "--category-max",
        type=int,
        default=None,
        help="Maximum d'items par catégorie pour le CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv is None:
        R = np.array(
            [
                [5, 4, 0, 1, 0, 0, 0, 0],
                [4, 3, 2, 0, 5, 4, 1, 0],
                [1, 0, 5, 4, 4, 0, 5, 3],
                [0, 4, 4, 5, 0, 5, 4, 5],
            ],
            dtype=float,
        )
        user_id = 0
        user_labels = np.arange(R.shape[0])
        item_labels = np.arange(R.shape[1])
        forbidden_pairs = args.forbidden_pair or [(4, 6)]
        item_genres = {
            0: "action",
            1: "comedy",
            2: "drama",
            3: "thriller",
            4: "action",
            5: "action",
            6: "drama",
            7: "comedy",
        }
    else:
        R, user_labels, item_labels = load_pivot_csv(args.csv)
        if args.user_id is None:
            user_id = 0
        else:
            matching_users = np.where(user_labels == args.user_id)[0]
            if len(matching_users) == 0:
                raise ValueError(
                    f"Utilisateur introuvable dans le CSV: {args.user_id}"
                )
            user_id = int(matching_users[0])
        forbidden_pairs = args.forbidden_pair
        metadata = build_item_metadata(item_labels, args.u_item)
        item_genres = {
            item: str(category)
            for item, category in enumerate(metadata["item_categories"])
        }

    forbidden_pairs = resolve_pairs(forbidden_pairs, item_labels)
    max_candidates = args.max_candidates
    if args.top_n < 1 or args.k < 1 or args.max_items_per_genre < 1:
        raise ValueError("top-n, k et max-items-per-genre doivent être positifs.")
    if max_candidates is not None and max_candidates < args.top_n:
        raise ValueError("max-candidates doit être au moins égal à top-n.")
    if args.category_max is not None and args.category_max < 0:
        raise ValueError("category-max doit être positif ou nul.")
    if args.csv is not None and args.category_max is not None:
        args.max_items_per_genre = args.category_max

    start_time = time.perf_counter()
    similarity = item_similarity_matrix(R)
    scores = np.array(
        [
            predict_rating(R, similarity, user_id, item_id)
            for item_id in range(R.shape[1])
        ]
    )
    rated_items = set(np.where(R[user_id] > 0)[0])
    recommendations_fc = recommend_fc(
        scores=scores,
        rated_items=rated_items,
        top_n=args.top_n,
        forbidden_pairs=forbidden_pairs,
        item_genres=item_genres,
        max_items_per_genre=args.max_items_per_genre,
        max_candidates=max_candidates,
    )
    recommendations_csp = solve_simple_csp(
        scores=scores,
        rated_items=rated_items,
        top_n=args.top_n,
        forbidden_pairs=forbidden_pairs,
        item_genres=item_genres,
        max_items_per_genre=args.max_items_per_genre,
        max_candidates=max_candidates,
    )
    elapsed_time = time.perf_counter() - start_time

    print(f"Utilisateur: {user_labels[user_id]}")
    print(f"Source: {args.csv or 'exemple intégré'}")
    print("\nScores FC pour les items :")
    print(scores)
    print(f"\nPaires interdites : {forbidden_pairs}")
    print(f"Genres des items : {item_genres}")
    print(f"Maximum d'items par genre : {args.max_items_per_genre}")
    print(
        "Recommandations FC :",
        [(int(item_labels[item]), score) for item, score in recommendations_fc],
    )
    print(
        "Recommandations CSP :",
        [(int(item_labels[item]), score) for item, score in recommendations_csp],
    )
    print(f"Temps d'exécution : {elapsed_time:.6f} secondes")


if __name__ == "__main__":
    main()
