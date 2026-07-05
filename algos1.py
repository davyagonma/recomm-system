import argparse
import csv
from pathlib import Path

import numpy as np
import cpmpy as cp

from uniform import compute_R_hat, W_collaborative_filtering, W_svd


def load_pivot_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge un CSV pivot utilisateur-item.

    La premiere colonne doit contenir les identifiants utilisateurs, et les
    colonnes suivantes les identifiants items. Les valeurs sont les notes, avec
    0 pour une note absente.
    """
    path = Path(path)
    with path.open(newline="") as file:
        reader = csv.reader(file)
        header = next(reader)

    if len(header) < 2:
        raise ValueError("Le CSV doit contenir une colonne utilisateur et au moins un item")

    item_ids = np.array(header[1:])
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    user_ids = data[:, 0].astype(int)
    R = data[:, 1:].astype(float)

    return user_ids, item_ids, R


def save_recommendations_csv(
    path: str | Path,
    result: dict,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> None:
    """Ecrit les recommandations sous forme user_id, rang, item_id, score."""
    path = Path(path)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "rank", "item_id", "score", "support_item_ids"])

        R_hat = result["R_hat"]
        explanations = result.get("explanations", {})

        for u, selected_items in result["recommendations"].items():
            ordered_items = sorted(
                selected_items,
                key=lambda item_index: float(R_hat[u, item_index]),
                reverse=True,
            )

            for rank, item_index in enumerate(ordered_items, start=1):
                support_indices = explanations.get(u, {}).get(item_index, [])
                support_ids = [str(item_ids[support_index]) for support_index in support_indices]
                writer.writerow([
                    int(user_ids[u]),
                    rank,
                    item_ids[item_index],
                    float(R_hat[u, item_index]),
                    "|".join(support_ids),
                ])


def build_history(R: np.ndarray) -> np.ndarray:
    """
    Construit Hu, la matrice binaire d'historique utilisateur-item.

    Hu[u, i] = 1 si l'utilisateur u a deja consomme/note l'item i.
    Dans cette version, une note strictement positive dans R signifie que
    l'item appartient a l'historique de l'utilisateur.
    """
    return (np.asarray(R) > 0).astype(int)


def build_W(R: np.ndarray, method: str = "fc", k: int = 2) -> np.ndarray:
    """
    Construit W avec l'algorithme choisi.

    method="fc" utilise le filtrage collaboratif item-item.
    method="fl" ou method="svd" utilise les facteurs latents par SVD.
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
    min_score: float | None = None,
) -> list[np.ndarray]:
    """
    Construit Cu, l'ensemble des items candidats pour chaque utilisateur.

    Les candidats sont tries par score decroissant dans R_hat.
    - exclude_consumed=True retire les items deja consommes.
    - min_score retire les items dont le score est trop faible.
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

        if min_score is not None:
            candidates = candidates[R_hat[u, candidates] >= min_score]

        ordered = candidates[np.argsort(R_hat[u, candidates])[::-1]]

        if n_candidates is not None:
            ordered = ordered[:n_candidates]

        Cu.append(ordered.astype(int))

    return Cu


def _bounds_for(bounds, key, default_min=None, default_max=None):
    """
    Lit une borne globale ou une borne specifique.

    Exemple global : (0, 2)
    Exemple par cle : {0: (1, 2), 1: (0, 1)}
    """
    if bounds is None:
        return default_min, default_max

    if isinstance(bounds, dict):
        return bounds.get(key, (default_min, default_max))

    return bounds


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
    support_threshold: float = 1e-12,
    solver_name: str = "ortools",
):
    """
    Resout le modele CP et retourne les recommandations.

    Contraintes disponibles :
    - cardinalite : exactement slate_size items par utilisateur ;
    - categorie : category_bounds=(min,max) ou {categorie: (min,max)} ;
    - fournisseur : provider_bounds=(min,max) ou {provider: (min,max)} ;
    - diversite : forbidden_pairs=[(i,j), ...] interdit deux items ensemble ;
    - explication : au moins explanation_min items historiques j tels que
      abs(W[j, i]) > support_threshold.
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
                for (_u, i), var in x_ui.items()
                if item_providers[i] == p
            ]
            if lb is not None:
                constraints.append(cp.sum(provider_vars) >= lb)
            if ub is not None:
                constraints.append(cp.sum(provider_vars) <= ub)

    y_uij = {}
    if explanation_min > 0:
        if W is None:
            raise ValueError("W est obligatoire quand explanation_min > 0")

        for u in range(n_users):
            consumed_items = np.where(Hu[u] == 1)[0]

            for i in Cu[u]:
                i = int(i)
                supports = [
                    int(j)
                    for j in consumed_items
                    if abs(float(W[int(j), i])) > support_threshold
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
            "explanations": {},
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
            i: [
                j
                for (uu, ii, j), y in y_uij.items()
                if uu == u and ii == i and y.value() == 1
            ]
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
    min_score: float | None = None,
    **cp_constraints,
):
    """
    Pipeline complet : R -> W -> R_hat -> Cu -> modele CP.

    Si W est fourni, il est utilise directement. Sinon W est construit avec
    method="fc" ou method="fl"/"svd".
    """
    R = np.asarray(R, dtype=float)

    if Hu is None:
        Hu = build_history(R)

    if W is None:
        W = build_W(R, method=method, k=k)

    R_hat = compute_R_hat(R, W)
    Cu = build_candidate_sets(
        R_hat=R_hat,
        Hu=Hu,
        n_candidates=n_candidates,
        exclude_consumed=exclude_consumed,
        min_score=min_score,
    )

    return solve_cp_recommendations(
        R_hat=R_hat,
        Cu=Cu,
        slate_size=slate_size,
        Hu=Hu,
        W=W,
        **cp_constraints,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Applique le pipeline de recommandation sur un CSV pivot utilisateur-item."
    )
    parser.add_argument("--input", default="u_data_pivot.csv", help="CSV pivot a charger")
    parser.add_argument(
        "--output",
        default="recommendations_algos1.csv",
        help="CSV de recommandations a produire",
    )
    parser.add_argument(
        "--method",
        default="svd",
        choices=["fc", "svd", "fl", "facteurs_latents"],
        help="Methode de construction de W",
    )
    parser.add_argument("--k", type=int, default=20, help="Nombre de facteurs pour SVD")
    parser.add_argument(
        "--slate-size",
        type=int,
        default=5,
        help="Nombre d'items a recommander par utilisateur",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=50,
        help="Nombre maximum de candidats gardes par utilisateur avant CP",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Score minimum dans R_hat pour garder un candidat",
    )
    parser.add_argument(
        "--include-consumed",
        action="store_true",
        help="Autorise la recommandation d'items deja notes/consommes",
    )
    parser.add_argument(
        "--explanation-min",
        type=int,
        default=0,
        help="Nombre minimal d'items historiques supportant chaque recommandation",
    )
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=1e-12,
        help="Seuil de support utilise pour les explications",
    )
    parser.add_argument("--solver", default="ortools", help="Solveur CPMpy a utiliser")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_ids, item_ids, R = load_pivot_csv(args.input)

    print(f"Matrice chargee : {R.shape[0]} utilisateurs x {R.shape[1]} items")
    print(
        "Parametres : "
        f"method={args.method}, slate_size={args.slate_size}, "
        f"n_candidates={args.n_candidates}"
    )

    result = recommend_complete(
        R=R,
        method=args.method,
        k=args.k,
        slate_size=args.slate_size,
        n_candidates=args.n_candidates,
        exclude_consumed=not args.include_consumed,
        min_score=args.min_score,
        explanation_min=args.explanation_min,
        support_threshold=args.support_threshold,
        solver_name=args.solver,
    )

    if not result["status"]:
        print("Aucune solution faisable avec ces contraintes.")
        return

    save_recommendations_csv(args.output, result, user_ids, item_ids)
    total_recommendations = sum(len(items) for items in result["recommendations"].values())

    # print("Objectif :", result["objective"])
    print(f"Recommandations produites : {total_recommendations}")
    print(f"Fichier ecrit : {args.output}")

    first_user = 0
    first_items = result["recommendations"].get(first_user, [])
    preview = [str(item_ids[item_index]) for item_index in first_items]
    print(f"Apercu utilisateur {int(user_ids[first_user])} : {preview}")


if __name__ == "__main__":
    main()
