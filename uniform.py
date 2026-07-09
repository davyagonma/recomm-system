import numpy as np

# Function: R̂ = R @ W
def compute_R_hat(R: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute the reconstructed matrix R̂ = R @ W.

    This function is the common interface for both algorithms:
    simply pass R and the W returned by
    ``W_collaborative_filtering`` or ``W_svd``.

    Parameters
    ----------
    R : np.ndarray, shape (n_users, n_items)
        Original user × item matrix (sparse, 0 = missing rating).
    W : np.ndarray, shape (n_items, n_items)
        Weight matrix produced by one of the two algorithms.

    Returns
    -------
    R_hat : np.ndarray, shape (n_users, n_items)
        Reconstructed matrix with recommendation scores.
    """
    if W.shape[0] != R.shape[1] or W.shape[1] != R.shape[1]:
        raise ValueError(
            f"W must have shape ({R.shape[1]}, {R.shape[1]}), "
            f"got {W.shape}"
        )
    return R @ W

# Algorithm 1: Item-Based Collaborative Filtering

def W_collaborative_filtering(R: np.ndarray) -> np.ndarray:
    """
    Compute the weight matrix W for item-item collaborative filtering.

    Formula:
        C  = RᵀR                    (item × item co-occurrence matrix)
        D  = diag(C)                (vector of diagonal elements of C)
        W  = D⁻¹ C D⁻¹             (symmetric normalization)

    With D⁻¹[i,i] = 1/D[i] (0 if D[i] = 0 to avoid division by zero).

    Parameters
    ----------
    R : np.ndarray, shape (n_users, n_items)
        User × item matrix with missing values represented by 0.

    Returns
    -------
    W : np.ndarray, shape (n_items, n_items)
        Normalized weight matrix.
    """
    # Item × item co-occurrence
    C = R.T @ R                              # shape: (n_items, n_items)

    # Diagonal of C (self-products of each item)
    d = np.diag(C)                           # shape: (n_items,)

    # Inverse of D (robust to zeros)
    d_inv = np.where(d != 0, 1.0 / d, 0.0)  # shape: (n_items,)

    # W = D⁻¹ C D⁻¹  (broadcast over rows then columns)
    W = (d_inv[:, None] * C) * d_inv[None, :]   # shape: (n_items, n_items)

    return W


def apply_collaborative_filtering(R: np.ndarray) -> np.ndarray:
    """
    Compute R̂ = R @ W_CF.

    Parameters
    ----------
    R : np.ndarray, shape (n_users, n_items)

    Returns
    -------
    R_hat : np.ndarray, shape (n_users, n_items)
        Reconstructed matrix with recommendation scores.
    """
    W = W_collaborative_filtering(R)
    return compute_R_hat(R, W)


# Algorithm 2: Latent Factors (Truncated SVD)

def W_svd(R: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Compute the weight matrix W for truncated SVD decomposition.

    Formula:
        SVD decomposition:  R_centered ≈ U Σ Vᵀ
        Q  = Vᵀ_k.T          (k item latent factors, shape n_items × k)
        W  = Q Qᵀ            (projector in latent space, shape n_items × n_items)

    Parameters
    ----------
    R : np.ndarray, shape (n_users, n_items)
        User × item matrix with missing values represented by 0.
    k : int
        Number of latent factors to retain (truncation rank).

    Returns
    -------
    W : np.ndarray, shape (n_items, n_items)
        Projection matrix in the item latent space.
    """
    n_users, n_items = R.shape

    # Per-user centering (zeros are ignored)
    centered = np.zeros_like(R, dtype=float)
    for u in range(n_users):
        rated_mask = R[u] > 0
        if np.any(rated_mask):
            user_mean = R[u, rated_mask].mean()
            centered[u, rated_mask] = R[u, rated_mask] - user_mean

    # Truncated SVD
    # centered ≈ U Σ Vᵀ   (full_matrices=False → Vt shape: min(m,n) × n_items)
    _U, _s, Vt = np.linalg.svd(centered, full_matrices=False)

    # Clamp k to the available dimension
    k = min(k, Vt.shape[0])

    # Q: item latent factors  (n_items × k)
    Q = Vt[:k, :].T                  # shape: (n_items, k)

    # W = Q Qᵀ  (item × item projector)
    W = Q @ Q.T                      # shape: (n_items, n_items)

    return W


def apply_svd(R: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Compute R̂ = R @ W_SVD.

    Parameters
    ----------
    R : np.ndarray, shape (n_users, n_items)
    k : int
        Number of latent factors.

    Returns
    -------
    R_hat : np.ndarray, shape (n_users, n_items)
        Reconstructed matrix with recommendation scores.
    """
    W = W_svd(R, k=k)
    return compute_R_hat(R, W)

# Usage example
if __name__ == "__main__":

    # Matrix R with missing values (0 = missing rating)
    R = np.array([
        [5, 3, 0, 1],
        [4, 2, 1, 0],
        [1, 0, 5, 4],
        [0, 4, 4, 5],
    ], dtype=float)

    print("Original matrix R:")
    print(R)
    print()

    # Collaborative filtering
    W_cf = W_collaborative_filtering(R)        # W returned by the CF algorithm
    R_hat_cf = compute_R_hat(R, W_cf)          # R̂ via the common function

    print("W  (collaborative filtering):")
    print(np.round(W_cf, 4))
    print()
    print("R̂ = R @ W_CF:")
    print(np.round(R_hat_cf, 4))
    print()

    # SVD latent factors
    W_s = W_svd(R, k=2)                        # W returned by the SVD algorithm
    R_hat_svd = compute_R_hat(R, W_s)          # R̂ via the same common function

    print("W  (SVD, k=2):")
    print(np.round(W_s, 4))
    print()
    print("R̂ = R @ W_SVD:")
    print(np.round(R_hat_svd, 4))
    print()

    # Direct use of compute_R_hat with any W
    print("=" * 50)
    print("Generic usage: compute_R_hat(R, W)")
    print("  W_cf  → R̂[0]:", np.round(compute_R_hat(R, W_cf)[0], 4))
    print("  W_svd → R̂[0]:", np.round(compute_R_hat(R, W_s)[0], 4))
