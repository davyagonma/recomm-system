import numpy as np
import time

def cosine_similarity(a, b):
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return num / den if den != 0 else 0.0

def item_similarity_matrix(R):
    """
    R : matrice users x items
    Les zéros représentent les notes manquantes.
    """
    n_items = R.shape[1]
    S = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(n_items):
            if i != j:
                S[i, j] = cosine_similarity(R[:, i], R[:, j])

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
            score = predict_rating(R, S, user_id, item_id)
            preds.append((item_id, score))

    preds.sort(key=lambda x: x[1], reverse=True)

    selected = []
    for candidate in preds:
        if not conflicts_with_slate(candidate[0], selected):
            selected.append(candidate)
        if len(selected) == top_n:
            break

    return selected


# ---------------- Exemple ----------------
R = np.array([
    [5, 4, 0, 1, 0, 0, 0, 0],
    [4, 3, 2, 0, 5, 4, 1, 0],
    [1, 0, 5, 4, 4, 0, 5, 3],
    [0, 4, 4, 5, 0, 5, 4, 5],
], dtype=float)

start_time = time.perf_counter()
S = item_similarity_matrix(R)
prediction = predict_rating(R, S, 0, 4)
forbidden_pairs = [(4, 6)]
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
recommendations = recommend(
    R,
    S,
    0,
    top_n=3,
    forbidden_pairs=forbidden_pairs,
    item_genres=item_genres,
    max_items_per_genre=1,
)
elapsed_time = time.perf_counter() - start_time

print("Matrice de similarité :")
print(S)

print("Prédiction user 0 -> item 4 :", prediction)
print(
    "Recommandations pour user 0 :",
    recommendations,
)
print("Genres des items :", item_genres)
print("Maximum d'items par genre :", 1)
print(f"Temps d'exécution des recommandations : {elapsed_time:.6f} secondes")