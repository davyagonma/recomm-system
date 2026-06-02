import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fonction : R̂ = R @ W
# ─────────────────────────────────────────────────────────────────────────────

def compute_R_hat(R: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice reconstruite R̂ = R @ W.

    Cette fonction est l'interface commune aux deux algorithmes :
    il suffit de lui passer R et le W retourné par
    ``W_collaborative_filtering`` ou ``W_svd``.

    Paramètres
    ----------
    R : np.ndarray, shape (n_users, n_items)
        Matrice utilisateurs × items d'origine (creux, 0 = note manquante).
    W : np.ndarray, shape (n_items, n_items)
        Matrice de pondération produite par l'un des deux algorithmes.

    Retour
    ------
    R_hat : np.ndarray, shape (n_users, n_items)
        Matrice reconstruite avec les scores de recommandation.
    """
    if W.shape[0] != R.shape[1] or W.shape[1] != R.shape[1]:
        raise ValueError(
            f"W doit être de shape ({R.shape[1]}, {R.shape[1]}), "
            f"reçu {W.shape}"
        )
    return R @ W


# ─────────────────────────────────────────────────────────────────────────────
# Algorithme 1 : Filtrage Collaboratif Basé sur les Items
# ─────────────────────────────────────────────────────────────────────────────

def W_collaborative_filtering(R: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de pondération W pour le filtrage collaboratif item-item.

    Formule :
        C  = RᵀR                    (matrice de co-occurrence items × items)
        D  = diag(C)                (vecteur des éléments diagonaux de C)
        W  = D⁻¹ C D⁻¹             (normalisation symétrique)

    Avec D⁻¹[i,i] = 1/D[i] (0 si D[i] = 0 pour éviter la division par zéro).

    Paramètres
    ----------
    R : np.ndarray, shape (n_users, n_items)
        Matrice utilisateurs × items avec les creux représentés par 0.

    Retour
    ------
    W : np.ndarray, shape (n_items, n_items)
        Matrice de pondération normalisée.
    """
    # Co-occurrence items × items
    C = R.T @ R                              # shape : (n_items, n_items)

    # Diagonale de C (auto-produits de chaque item)
    d = np.diag(C)                           # shape : (n_items,)

    # Inverse de D (robuste aux zéros)
    d_inv = np.where(d != 0, 1.0 / d, 0.0)  # shape : (n_items,)

    # W = D⁻¹ C D⁻¹  (broadcast sur lignes puis colonnes)
    W = (d_inv[:, None] * C) * d_inv[None, :]   # shape : (n_items, n_items)

    return W


def apply_collaborative_filtering(R: np.ndarray) -> np.ndarray:
    """
    Calcule R̂ = R @ W_CF.

    Paramètres
    ----------
    R : np.ndarray, shape (n_users, n_items)

    Retour
    ------
    R_hat : np.ndarray, shape (n_users, n_items)
        Matrice reconstruite avec les scores de recommandation.
    """
    W = W_collaborative_filtering(R)
    return compute_R_hat(R, W)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithme 2 : Facteurs Latents (SVD tronquée)
# ─────────────────────────────────────────────────────────────────────────────

def W_svd(R: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Calcule la matrice de pondération W pour la décomposition SVD tronquée.

    Formule :
        Décomposition SVD :  R_centré ≈ U Σ Vᵀ
        Q  = Vᵀ_k.T          (k facteurs latents items, shape n_items × k)
        W  = Q Qᵀ            (projecteur dans l'espace latent, shape n_items × n_items)

    Paramètres
    ----------
    R : np.ndarray, shape (n_users, n_items)
        Matrice utilisateurs × items avec les creux représentés par 0.
    k : int
        Nombre de facteurs latents à retenir (rang de la troncature).

    Retour
    ------
    W : np.ndarray, shape (n_items, n_items)
        Matrice de projection dans l'espace latent des items.
    """
    n_users, n_items = R.shape

    # ── Centrage par utilisateur (on ignore les 0) ──────────────────────────
    centered = np.zeros_like(R, dtype=float)
    for u in range(n_users):
        rated_mask = R[u] > 0
        if np.any(rated_mask):
            user_mean = R[u, rated_mask].mean()
            centered[u, rated_mask] = R[u, rated_mask] - user_mean

    # ── SVD tronquée ─────────────────────────────────────────────────────────
    # centered ≈ U Σ Vᵀ   (full_matrices=False → Vt shape : min(m,n) × n_items)
    _U, _s, Vt = np.linalg.svd(centered, full_matrices=False)

    # On limite k à la dimension disponible
    k = min(k, Vt.shape[0])

    # Q : facteurs latents des items  (n_items × k)
    Q = Vt[:k, :].T                  # shape : (n_items, k)

    # W = Q Qᵀ  (projecteur items × items)
    W = Q @ Q.T                      # shape : (n_items, n_items)

    return W


def apply_svd(R: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Calcule R̂ = R @ W_SVD.

    Paramètres
    ----------
    R : np.ndarray, shape (n_users, n_items)
    k : int
        Nombre de facteurs latents.

    Retour
    ------
    R_hat : np.ndarray, shape (n_users, n_items)
        Matrice reconstruite avec les scores de recommandation.
    """
    W = W_svd(R, k=k)
    return compute_R_hat(R, W)


# ─────────────────────────────────────────────────────────────────────────────
# Exemple d'utilisation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Matrice R avec creux (0 = note manquante)
    R = np.array([
        [5, 3, 0, 1],
        [4, 2, 1, 0],
        [1, 0, 5, 4],
        [0, 4, 4, 5],
    ], dtype=float)

    print("Matrice R originale :")
    print(R)
    print()

    # ── Filtrage collaboratif ────────────────────────────────────────────────
    W_cf = W_collaborative_filtering(R)        # W retourné par l'algo CF
    R_hat_cf = compute_R_hat(R, W_cf)          # R̂ via la fonction commune

    print("W  (filtrage collaboratif) :")
    print(np.round(W_cf, 4))
    print()
    print("R̂ = R @ W_CF :")
    print(np.round(R_hat_cf, 4))
    print()

    # ── SVD facteurs latents ─────────────────────────────────────────────────
    W_s = W_svd(R, k=2)                        # W retourné par l'algo SVD
    R_hat_svd = compute_R_hat(R, W_s)          # R̂ via la même fonction commune

    print("W  (SVD, k=2) :")
    print(np.round(W_s, 4))
    print()
    print("R̂ = R @ W_SVD :")
    print(np.round(R_hat_svd, 4))
    print()

    # ── Utilisation directe de compute_R_hat avec n'importe quel W ───────────
    print("=" * 50)
    print("Usage générique : compute_R_hat(R, W)")
    print("  W_cf  → R̂[0] :", np.round(compute_R_hat(R, W_cf)[0], 4))
    print("  W_svd → R̂[0] :", np.round(compute_R_hat(R, W_s)[0], 4))
