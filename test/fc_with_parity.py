import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app.metadata import build_item_metadata

def cosine_similarity(a, b):
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return num / den if den != 0 else 0.0

def item_similarity_matrix(R):
    """
    R : matrice users x items
    Les zéros représentent les notes manquantes.
    """
    norms = np.linalg.norm(R, axis=0)
    denominator = np.outer(norms, norms)
    S = np.divide(
        R.T @ R,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    np.fill_diagonal(S, 0.0)
    return S

def predict_rating(R, S, user_id, item_id, k=3):
    """
    Prédit la note de user_id pour item_id.
    """
    user_ratings = R[user_id]

    # Items déjà notés par l'utilisateur
    rated_items = np.where(user_ratings > 0)[0]

    if len(rated_items) == 0:
        return 0.0

    # Similarités entre l'item cible et les items notés
    sims = []
    for j in rated_items:
        sims.append((j, S[item_id, j], user_ratings[j]))

    # Garder les k plus grandes similarités absolues
    sims = sorted(sims, key=lambda x: abs(x[1]), reverse=True)[:k]

    num = 0.0
    den = 0.0
    for j, sim, rating in sims:
        num += sim * rating
        den += abs(sim)

    return num / den if den != 0 else 0.0

def recommend(
    R,
    S,
    user_id,
    top_n=3,
    k=3,
    forbidden_pairs=None,
    item_genres=None,
    max_items_per_genre=1,
):
    """
    Retourne les top_n items non notés avec leurs scores.

    forbidden_pairs contient les paires d'items qui ne doivent pas
    apparaître ensemble dans la recommandation.
    """
    n_items = R.shape[1]
    rated = set(np.where(R[user_id] > 0)[0])
    forbidden_pairs = forbidden_pairs or []
    item_genres = item_genres or {}

    def conflicts_with_slate(item_id, slate):
        violates_pair = any(
            (item_id, selected) in forbidden_pairs
            or (selected, item_id) in forbidden_pairs
            for selected, _ in slate
        )
        if violates_pair:
            return True

        genre = item_genres.get(item_id)
        if genre is None:
            return False

        same_genre_count = sum(
            item_genres.get(selected) == genre
            for selected, _ in slate
        )
        return same_genre_count >= max_items_per_genre

    preds = []
    for item_id in range(n_items):
        if item_id not in rated:
            score = predict_rating(R, S, user_id, item_id, k=k)
            preds.append((item_id, score))

    preds.sort(key=lambda x: x[1], reverse=True)

    selected = []
    for candidate in preds:
        if not conflicts_with_slate(candidate[0], selected):
            selected.append(candidate)
        if len(selected) == top_n:
            break

    return selected


def load_pivot_csv(path):
    """Load a pivot CSV with user_id as its first column."""
    with path.open(newline="", encoding="utf-8") as file:
        header = next(csv.reader(file), None)
    if header is None or len(header) < 2:
        raise ValueError("Le CSV doit contenir user_id et au moins un item.")

    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(header):
        raise ValueError("Le nombre de colonnes du CSV est incohérent.")

    try:
        item_labels = np.array([int(label) for label in header[1:]])
    except ValueError as error:
        raise ValueError(
            "Les identifiants d'items du CSV doivent être numériques."
        ) from error
    return data[:, 1:], data[:, 0].astype(int), item_labels


def parse_pair(value):
    """Parse a forbidden pair written as ITEM1:ITEM2."""
    try:
        first, second = value.split(":", maxsplit=1)
        return int(first), int(second)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Une paire interdite doit être écrite ITEM1:ITEM2."
        ) from error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tester FC avec parité sur une matrice ou un CSV pivoté."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV pivoté dont la première colonne est user_id.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        help="Identifiant utilisateur du CSV; par défaut, le premier.",
    )
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--forbidden-pair",
        type=parse_pair,
        action="append",
        default=[],
        metavar="ITEM1:ITEM2",
    )
    parser.add_argument("--u-item", type=Path, default=Path("u.item"))
    parser.add_argument(
        "--category-max",
        type=int,
        default=None,
        help="Nombre maximal d'items d'une même catégorie.",
    )
    return parser.parse_args()


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


def main():
    args = parse_args()
    if args.top_n < 1 or args.k < 1:
        raise ValueError("top-n et k doivent être positifs.")
    if args.category_max is not None and args.category_max < 0:
        raise ValueError("category-max doit être positif ou nul.")
    category_limit = (
        args.category_max
        if args.category_max is not None
        else (1 if args.csv is None else 10**9)
    )

    if args.csv is None:
        ratings = np.array([
            [5, 4, 0, 1, 0, 0, 0, 0],
            [4, 3, 2, 0, 5, 4, 1, 0],
            [1, 0, 5, 4, 4, 0, 5, 3],
            [0, 4, 4, 5, 0, 5, 4, 5],
        ], dtype=float)
        row_index = 0
        user_labels = np.arange(ratings.shape[0])
        item_labels = np.arange(ratings.shape[1])
        configured_pairs = args.forbidden_pair or [(4, 6)]
        configured_genres = {
            0: "action", 1: "comedy", 2: "drama", 3: "thriller",
            4: "action", 5: "action", 6: "drama", 7: "comedy",
        }
    else:
        ratings, user_labels, item_labels = load_pivot_csv(args.csv)
        if args.user_id is None:
            row_index = 0
        else:
            matches = np.where(user_labels == args.user_id)[0]
            if len(matches) == 0:
                raise ValueError(f"Utilisateur introuvable: {args.user_id}")
            row_index = int(matches[0])
        configured_pairs = args.forbidden_pair
        metadata = build_item_metadata(item_labels, args.u_item)
        configured_genres = {
            item: str(category)
            for item, category in enumerate(metadata["item_categories"])
        }
    configured_pairs = resolve_pairs(configured_pairs, item_labels)

    start_time = time.perf_counter()
    similarity = item_similarity_matrix(ratings)
    recommendations = recommend(
        ratings, similarity, row_index, top_n=args.top_n, k=args.k,
        forbidden_pairs=configured_pairs,
        item_genres=configured_genres,
        max_items_per_genre=category_limit,
    )
    elapsed_time = time.perf_counter() - start_time

    print(f"Utilisateur : {user_labels[row_index]}")
    print(f"Source : {args.csv or 'exemple intégré'}")
    print(f"Paires interdites : {configured_pairs}")
    print(f"Limite par catégorie : {args.category_max}")
    print("Recommandations FC :")
    print([(int(item_labels[item]), score) for item, score in recommendations])
    print(f"Temps d'exécution : {elapsed_time:.6f} secondes")


if __name__ == "__main__":
    main()