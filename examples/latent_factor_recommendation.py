import numpy as np
# Importation de NumPy pour manipuler les matrices et effectuer les calculs mathématiques


def build_user_item_matrix(ratings):
    """
    Construit une matrice utilisateurs x items à partir d'une liste de notes.

    ratings : liste de tuples (user_id, item_id, rating)

    Exemple :
    (0, 1, 5) signifie :
    utilisateur 0 a donné la note 5 à l’item 1
    """

    # Extraction de tous les identifiants utilisateurs
    users = [u for u, _, _ in ratings]

    # Extraction de tous les identifiants des items
    items = [i for _, i, _ in ratings]

    # Nombre total d'utilisateurs
    # max(users) + 1 car les IDs commencent à 0
    n_users = max(users) + 1

    # Nombre total d’items
    n_items = max(items) + 1

    # Création d’une matrice remplie de 0
    # dimensions : utilisateurs x items
    R = np.zeros((n_users, n_items), dtype=float)

    # Remplissage de la matrice avec les notes
    for u, i, r in ratings:
        R[u, i] = r

    # Retour de la matrice finale
    return R


def train_svd(R, k=2):
    """
    Entraîne un modèle de recommandation basé sur SVD.

    R : matrice users x items
        Les valeurs manquantes sont représentées par 0.

    k : nombre de facteurs latents.
        Plus k est grand, plus le modèle peut capturer
        des relations complexes.

    Retour :
    dictionnaire contenant :
    - moyennes utilisateurs
    - moyenne globale
    - facteurs utilisateurs
    - facteurs items
    - matrice originale
    """

    # Dimensions de la matrice
    n_users, n_items = R.shape

    # Tableau des moyennes de chaque utilisateur
    user_means = np.zeros(n_users)

    # Moyenne globale de toutes les notes non nulles
    # Sert de fallback si un utilisateur n’a aucune note
    global_mean = R[R > 0].mean() if np.any(R > 0) else 0.0

    # Matrice centrée
    # On retire la moyenne de chaque utilisateur
    centered = np.zeros_like(R)

    # Parcours de chaque utilisateur
    for u in range(n_users):

        # Détecte les items notés par l’utilisateur
        rated = R[u] > 0

        # Si l'utilisateur possède au moins une note
        if np.any(rated):

            # Moyenne des notes de l'utilisateur
            user_means[u] = R[u, rated].mean()

            # Centrage des notes :
            # note - moyenne utilisateur
            centered[u, rated] = R[u, rated] - user_means[u]

        else:
            # Si aucune note :
            # on utilise la moyenne globale
            user_means[u] = global_mean

    # Décomposition SVD
    # centered ≈ U * S * Vt
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)

    # On limite k à la taille disponible
    k = min(k, len(s))

    # Sélection des k premières composantes
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]

    # Construction des facteurs utilisateurs
    # sqrt(s) répartit les poids entre users/items
    user_factors = U_k * np.sqrt(s_k)

    # Construction des facteurs items
    item_factors = (np.sqrt(s_k)[:, None] * Vt_k).T

    # Retour du modèle
    return {
        "user_means": user_means,
        "global_mean": global_mean,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "R": R,
    }


def predict_rating(model, user_id, item_id):
    """
    Prédit la note qu’un utilisateur donnerait à un item.
    """

    # Récupération des données du modèle
    R = model["R"]
    user_means = model["user_means"]
    global_mean = model["global_mean"]
    user_factors = model["user_factors"]
    item_factors = model["item_factors"]

    # Si utilisateur ou item inconnu :
    # retourne la moyenne globale
    if user_id >= R.shape[0] or item_id >= R.shape[1]:
        return global_mean

    # Base de prédiction :
    # moyenne personnelle de l’utilisateur
    base = user_means[user_id] if user_id < len(user_means) else global_mean

    # Produit scalaire entre :
    # vecteur latent utilisateur
    # vecteur latent item
    #
    # Cela mesure la compatibilité entre eux
    score = base + np.dot(user_factors[user_id], item_factors[item_id])

    return float(score)


def recommend(model, user_id, top_n=3):
    """
    Génère les meilleures recommandations pour un utilisateur.

    top_n : nombre d'items à recommander
    """

    R = model["R"]

    # Liste des items déjà notés
    rated = set(np.where(R[user_id] > 0)[0])

    preds = []

    # Parcours de tous les items
    for item_id in range(R.shape[1]):

        # Ignore les items déjà notés
        if item_id not in rated:

            # Calcul de la note prédite
            preds.append(
                (item_id, predict_rating(model, user_id, item_id))
            )

    # Tri décroissant selon les scores prédits
    preds.sort(key=lambda x: x[1], reverse=True)

    # Retour des meilleurs items
    return preds[:top_n]


# ---------------- Exemple d'utilisation ----------------
if __name__ == "__main__":

    # Données :
    # (utilisateur, item, note)
    ratings = [
        (0, 0, 5), (0, 1, 3), (0, 3, 1),
        (1, 0, 4), (1, 1, 2), (1, 2, 1),
        (2, 0, 1), (2, 2, 5), (2, 3, 4),
        (3, 1, 4), (3, 2, 4), (3, 3, 5),
    ]

    # Construction de la matrice user-item
    R = build_user_item_matrix(ratings)

    # Entraînement du modèle SVD
    model = train_svd(R, k=2)

    # Prédiction :
    # quelle note user 0 donnerait à item 2 ?
    print("Prédiction user 0 -> item 2 :",
          predict_rating(model, 0, 2))

    # Recommandation des 2 meilleurs items
    print("Recommandations pour user 0 :",
          recommend(model, 0, top_n=2))