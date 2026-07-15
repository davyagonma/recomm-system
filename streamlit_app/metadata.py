"""Item metadata loading (MovieLens genres, categories, providers)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def load_u_item(path: str | Path) -> tuple[dict[int, str], np.ndarray, np.ndarray]:
    """
    Load MovieLens 100K u.item metadata.

    Returns (titles, primary categories, binary multi-genre vectors).
    The primary category is the index of the first active genre flag.
    """
    path = Path(path)
    titles: dict[int, str] = {}
    primary: dict[int, int] = {}
    genre_vectors: dict[int, np.ndarray] = {}

    with path.open(encoding="latin-1", errors="replace") as file:
        for line in file:
            parts = line.strip().split("|")
            if len(parts) < 6:
                continue
            movie_id = int(parts[0])
            titles[movie_id] = parts[1]
            flags = np.array([int(x) for x in parts[5:5 + len(GENRES)]], dtype=int)
            active = np.where(flags == 1)[0]
            primary[movie_id] = int(active[0]) if len(active) else 0
            genre_vectors[movie_id] = flags

    return titles, primary, genre_vectors


def build_item_metadata(
    item_ids: np.ndarray,
    u_item_path: str | Path | None = None,
    num_synthetic_categories: int = 19,
) -> dict:
    """
    Build titles, categories, and providers aligned with pivot item_ids.

    If u.item is missing, categories are derived from movie_id mod N
    (documented synthetic approximation for demo purposes).
    """
    item_ids = np.asarray(item_ids)
    n_items = len(item_ids)
    titles: dict[int, str] = {}
    item_categories = np.zeros(n_items, dtype=int)
    genre_matrix = np.zeros((n_items, len(GENRES)), dtype=int)
    item_providers = np.zeros(n_items, dtype=int)

    ml_titles: dict[int, str] = {}
    ml_primary: dict[int, int] = {}
    ml_genres: dict[int, np.ndarray] = {}

    if u_item_path and Path(u_item_path).exists():
        ml_titles, ml_primary, ml_genres = load_u_item(u_item_path)

    for index, raw_id in enumerate(item_ids):
        movie_id = int(raw_id)
        if movie_id in ml_titles:
            titles[movie_id] = ml_titles[movie_id]
            item_categories[index] = ml_primary[movie_id]
            genre_matrix[index] = ml_genres[movie_id]
        else:
            titles[movie_id] = f"Item {movie_id}"
            item_categories[index] = movie_id % num_synthetic_categories
            genre_matrix[index, item_categories[index]] = 1

        item_providers[index] = movie_id % 5

    return {
        "titles": titles,
        "item_categories": item_categories,
        "genre_matrix": genre_matrix,
        "item_providers": item_providers,
        "genre_names": GENRES,
        "metadata_source": "u.item" if ml_titles else "synthetic",
    }
