import numpy as np
import cpmpy as cp

from uniform import compute_R_hat, W_collaborative_filtering, W_svd


def build_history(R: np.ndarray) -> np.ndarray:
    """
    Construit Hu, la matrice binaire d'historique utilisateur-item.

    Hu[u, i] = 1 si l'utilisateur u a deja consomme/note l'item i.
    Ici, une note strictement positive dans R signifie que l'item est connu.
    """
    return (np.asarray(R) > 0).astype(int)


def build_W(R: np.ndarray, method: str = "fc", k: int = 2) -> np.ndarray:
    """
    Construit la matrice W selon l'algorithme choisi.

    method = "fc" : filtrage collaboratif item-item
    method = "fl" ou "svd" : facteurs latents par SVD tronquee
    """
    method = method.lower()
    if method == "fc":
        return W_collaborative_filtering(R)
    if method in {"fl", "svd", "facteurs_latents"}:
        return W_svd(R, k=k)
    raise ValueError("method doit etre 'fc', 'fl' ou 'svd'")


def build_candidate_sets(
    R_hat: np.ndarray,
    Hu: np.ndarray | None = None,
    n_candidates: int | None = None,
    exclude_consumed: bool = True,
) -> list[np.ndarray]:
    """
    Construit Cu pour chaque utilisateur avec les meilleurs scores de R_hat.

    Si exclude_consumed=True, les items deja presents dans Hu sont retires.
    """
    R_hat = np.asarray(R_hat, dtype=float)
    n_users, n_items = R_hat.shape
    if Hu is None:
        Hu = np.zeros((n_users, n_items), dtype=int)

    Cu = []
    for u in range(n_users):
        candidates = np.arange(n_items)
        if exclude_consumed:
            candidates = candidates[Hu[u] == 0]

        ordered = candidates[np.argsort(R_hat[u, candidates])[::-1]]
        if n_candidates is not None:
            ordered = ordered[:n_candidates]
        Cu.append(ordered.astype(int))
    return Cu


def _bounds_for(value, key, default_min=None, default_max=None):
    """Retourne (min, max) depuis None, tuple global ou dict par cle."""
    if value is None:
        return default_min, default_max
    if isinstance(value, dict):
        return value.get(key, (default_min, default_max))
    return value


def solve_cp_recommendations(
    R_hat: np.ndarray,
    Cu: list[np.ndarray],
    slate_size: int,
    Hu: np.ndarray | None = None,
    W: np.ndarray | None = None,
    item_categories: np.ndarray | None = None,
    category_bounds=None,
    item_providers: np.ndarray | None = None,
    provider_bounds=None,
    forbidden_pairs: list[tuple[int, int]] | None = None,
    explanation_min: int = 0,
    support_threshold: float = 0.0,
    solver_name: str = "ortools",
):
    """
    Resout le modele CP et retourne les slates recommandes.

    Contraintes supportees :
    - cardinalite : exactement slate_size items par utilisateur ;
    - categorie : category_bounds=(min,max) ou {categorie: (min,max)} ;
    - fournisseur : provider_bounds=(min,max) ou {provider: (min,max)} sur tous les users ;
    - diversite : forbidden_pairs=[(i,j), ...] interdit deux items ensemble ;
    - explication : au moins explanation_min items historiques j avec |W[j,i]| >= seuil.
    """
    R_hat = np.asarray(R_hat, dtype=float)
    n_users, n_items = R_hat.shape
    if Hu is None:
        Hu = np.zeros((n_users, n_items), dtype=int)
    if forbidden_pairs is None:
        forbidden_pairs = []

    x_ui = {
        (u, int(i)): cp.boolvar(name=f"x_{u}_{int(i)}")
        for u in range(n_users)
        for i in Cu[u]
    }

    constraints = []
    for u in range(n_users):
        user_vars = [x_ui[(u, int(i))] for i in Cu[u]]
        constraints.append(cp.sum(user_vars) == slate_size)

    if item_categories is not None and category_bounds is not None:
        item_categories = np.asarray(item_categories)
        for u in range(n_users):
            for c in np.unique(item_categories):
                lb, ub = _bounds_for(category_bounds, int(c))
                cat_vars = [
                    x_ui[(u, int(i))]
                    for i in Cu[u]
                    if item_categories[int(i)] == c
                ]
                if lb is not None:
                    constraints.append(cp.sum(cat_vars) >= lb)
                if ub is not None:
                    constraints.append(cp.sum(cat_vars) <= ub)

    for u in range(n_users):
        for i, j in forbidden_pairs:
            if (u, i) in x_ui and (u, j) in x_ui:
                constraints.append(x_ui[(u, i)] + x_ui[(u, j)] <= 1)

    if item_providers is not None and provider_bounds is not None:
        item_providers = np.asarray(item_providers)
        for p in np.unique(item_providers):
            lb, ub = _bounds_for(provider_bounds, int(p))
            provider_vars = [
                var
                for (u, i), var in x_ui.items()
                if item_providers[i] == p
            ]
            if lb is not None:
                constraints.append(cp.sum(provider_vars) >= lb)
            if ub is not None:
                constraints.append(cp.sum(provider_vars) <= ub)

    y_uij = {}
    if explanation_min > 0:
        if W is None:
            raise ValueError("W est requis quand explanation_min > 0")
        for u in range(n_users):
            consumed = np.where(Hu[u] == 1)[0]
            for i in Cu[u]:
                i = int(i)
                supports = [
                    int(j)
                    for j in consumed
                    if abs(float(W[int(j), i])) >= support_threshold
                ]
                support_vars = []
                for j in supports:
                    y_uij[(u, i, j)] = cp.boolvar(name=f"y_{u}_{i}_{j}")
                    support_vars.append(y_uij[(u, i, j)])
                    constraints.append(y_uij[(u, i, j)] <= x_ui[(u, i)])
                constraints.append(cp.sum(support_vars) >= explanation_min * x_ui[(u, i)])

    objective = cp.sum(
        [float(R_hat[u, i]) * var for (u, i), var in x_ui.items()]
    )
    model = cp.Model(constraints, maximize=objective)
    solver = cp.SolverLookup.get(solver_name, model)
    status = solver.solve()

    if not status:
        return {
            "status": False,
            "objective": None,
            "recommendations": {},
            "R_hat": R_hat,
            "Cu": Cu,
            "model": model,
            "x_ui": x_ui,
            "y_uij": y_uij,
        }

    recommendations = {}
    explanations = {}
    for u in range(n_users):
        selected = [
            int(i)
            for i in Cu[u]
            if (u, int(i)) in x_ui and x_ui[(u, int(i))].value() == 1
        ]
        recommendations[u] = selected
        explanations[u] = {
            i: [j for (uu, ii, j), y in y_uij.items() if uu == u and ii == i and y.value() == 1]
            for i in selected
        }

    return {
        "status": True,
        "objective": solver.objective_value(),
        "recommendations": recommendations,
        "explanations": explanations,
        "R_hat": R_hat,
        "Cu": Cu,
        "model": model,
        "x_ui": x_ui,
        "y_uij": y_uij,
    }


def recommend_complete(
    R: np.ndarray,
    method: str = "fc",
    W: np.ndarray | None = None,
    k: int = 2,
    slate_size: int = 3,
    n_candidates: int | None = None,
    Hu: np.ndarray | None = None,
    exclude_consumed: bool = True,
    **cp_constraints,
):
    """
    Pipeline complet : R -> W -> R_hat -> Cu -> CP -> recommandations.

    W peut etre fourni directement. Sinon il est calcule avec method="fc"
    ou method="fl". Les contraintes CP sont transmises a solve_cp_recommendations.
    """
    R = np.asarray(R, dtype=float)
    if Hu is None:
        Hu = build_history(R)
    if W is None:
        W = build_W(R, method=method, k=k)

    R_hat = compute_R_hat(R, W)
    Cu = build_candidate_sets(
        R_hat,
        Hu=Hu,
        n_candidates=n_candidates,
        exclude_consumed=exclude_consumed,
    )

    return solve_cp_recommendations(
        R_hat=R_hat,
        Cu=Cu,
        slate_size=slate_size,
        Hu=Hu,
        W=W,
        **cp_constraints,
    )


if __name__ == "__main__":
    R = np.array([
        [5, 3, 0, 1, 0, 0],
        [4, 2, 1, 0, 0, 0],
        [1, 0, 5, 4, 0, 0],
        [0, 4, 4, 5, 0, 0],
    ], dtype=float)

    item_categories = np.array([0, 0, 1, 1, 2, 2])
    item_providers = np.array([0, 1, 0, 1, 0, 1])

    result = recommend_complete(
        R,
        method="fl",
        slate_size=2,
        n_candidates=4,
        item_categories=item_categories,
        category_bounds=(0, 1),
        item_providers=item_providers,
        provider_bounds=(1, 6),
        forbidden_pairs=[(0, 1), (2, 3)],
        explanation_min=1,
        support_threshold=0.0,
    )

    if result["status"]:
        print("Objectif :", result["objective"])
        print("Recommandations :", result["recommendations"])
        print("Explications :", result["explanations"])
    else:
        print("Aucune solution faisable avec ces contraintes.")
