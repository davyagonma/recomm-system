import numpy as np

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

def recommend(R, S, user_id, top_n=3):
    """
    Retourne les top_n items non notés avec leurs scores.
    """
    n_items = R.shape[1]
    rated = set(np.where(R[user_id] > 0)[0])

    preds = []
    for item_id in range(n_items):
        if item_id not in rated:
            score = predict_rating(R, S, user_id, item_id)
            preds.append((item_id, score))

    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:top_n]


# ---------------- Exemple ----------------
R = np.array([
    [5, 3, 0, 1],
    [4, 2, 1, 0],
    [1, 0, 5, 4],
    [0, 4, 4, 5]
], dtype=float)

S = item_similarity_matrix(R)

print("Matrice de similarité :")
print(S)

print("Prédiction user 0 -> item 2 :", predict_rating(R, S, 0, 2))
print("Recommandations pour user 0 :", recommend(R, S, 0, top_n=2))
