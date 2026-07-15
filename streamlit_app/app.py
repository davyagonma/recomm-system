"""
Streamlit UI — constrained Top-N recommendation (CPMpy).

Run locally:
    cd codes
    streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos1 import load_pivot_csv, save_recommendations_csv  # noqa: E402
from streamlit_app.metadata import build_item_metadata, GENRES  # noqa: E402
from streamlit_app.metrics import slate_metrics  # noqa: E402
from streamlit_app.selection import (  # noqa: E402
    ConstraintConfig,
    run_full_pipeline,
)

DEFAULT_PIVOT = ROOT / "u_data_pivot.csv"
DEFAULT_U_ITEM = ROOT / "u.item"

APPROACHES = {
    "S1": "Top-N classique (baseline)",
    "S2": "Post-traitement heuristique",
    "S3": "Hybride glouton (pertinence + diversité)",
    "S4": "CPMpy — approche proposée",
}


def init_session_state() -> None:
    """Initialize Streamlit session state keys with default values."""
    defaults = {
        "last_result": None,
        "comparison_results": None,
        "user_ids": None,
        "item_ids": None,
        "R": None,
        "metadata": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_dataset(uploaded_file, pivot_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load user-item pivot matrix from an upload or a local CSV path."""
    if uploaded_file is not None:
        content = uploaded_file.read()
        tmp = ROOT / ".streamlit_upload.csv"
        tmp.write_bytes(content)
        return load_pivot_csv(tmp)
    if not pivot_path.exists():
        raise FileNotFoundError(f"Pivot file not found: {pivot_path}")
    return load_pivot_csv(pivot_path)


def parse_forbidden_pairs(text: str, item_id_to_index: dict[int, int]) -> list[tuple[int, int]]:
    """Parse semicolon-separated item_id pairs into matrix index pairs."""
    pairs: list[tuple[int, int]] = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk or "," not in chunk:
            continue
        left, right = chunk.split(",", 1)
        try:
            a = item_id_to_index[int(left.strip())]
            b = item_id_to_index[int(right.strip())]
            if a != b:
                pairs.append((min(a, b), max(a, b)))
        except (ValueError, KeyError):
            continue
    return list(dict.fromkeys(pairs))


def build_constraints_from_sidebar(metadata: dict) -> ConstraintConfig:
    """Build a ConstraintConfig from sidebar widget values in session state."""
    cfg = ConstraintConfig()
    cfg.item_categories = metadata["item_categories"]
    cfg.item_providers = metadata["item_providers"]

    if st.session_state.get("use_category_constraint"):
        cfg.category_min = st.session_state.get("category_min")
        cfg.category_max = st.session_state.get("category_max")

    if st.session_state.get("use_provider_constraint"):
        cfg.provider_min = st.session_state.get("provider_min")
        cfg.provider_max = st.session_state.get("provider_max")

    if st.session_state.get("use_explanation"):
        cfg.explanation_min = int(st.session_state.get("explanation_min", 1))
        cfg.support_threshold = float(st.session_state.get("support_threshold", 1e-12))

    text = st.session_state.get("forbidden_pairs_text", "")
    item_ids = st.session_state.get("item_ids")
    if text and item_ids is not None:
        id_map = {int(v): i for i, v in enumerate(item_ids)}
        cfg.forbidden_pairs = parse_forbidden_pairs(text, id_map)

    return cfg


def recommendations_to_dataframe(
    result: dict,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    titles: dict[int, str],
    metadata: dict,
    selected_user_index: int,
) -> pd.DataFrame:
    """Format one user's recommendations as a display DataFrame."""
    u = selected_user_index
    R_hat = result["R_hat"]
    rows = []
    explanations = result.get("explanations", {}).get(u, {})

    for rank, item_index in enumerate(result["recommendations"].get(u, []), start=1):
        movie_id = int(item_ids[item_index])
        cat = int(metadata["item_categories"][item_index])
        support = explanations.get(item_index, explanations.get(int(item_index), []))
        support_ids = [int(item_ids[j]) for j in support]

        rows.append(
            {
                "Rang": rank,
                "Item ID": movie_id,
                "Titre": titles.get(movie_id, f"Item {movie_id}"),
                "Score R̂": round(float(R_hat[u, item_index]), 4),
                "Catégorie": GENRES[cat] if cat < len(GENRES) else str(cat),
                "Support (IDs)": ", ".join(map(str, support_ids)) if support_ids else "—",
            }
        )

    return pd.DataFrame(rows)


def export_csv_bytes(result: dict, user_ids: np.ndarray, item_ids: np.ndarray) -> bytes:
    """Serialize recommendations to CSV bytes for download."""
    buffer = io.StringIO()
    path = ROOT / ".streamlit_export.csv"
    save_recommendations_csv(path, result, user_ids, item_ids)
    return path.read_bytes()


def render_sidebar() -> dict:
    """Render sidebar widgets and return the current run configuration."""
    st.sidebar.header("Configuration")

    uploaded = st.sidebar.file_uploader("CSV pivot (optionnel)", type=["csv"])
    pivot_path = Path(st.sidebar.text_input("Chemin pivot local", str(DEFAULT_PIVOT)))

    st.sidebar.subheader("Modèle de scores R̂ = RW")
    method_label = st.sidebar.radio("Méthode W", ["SVD (facteurs latents)", "FC (item-item)"])
    method = "svd" if "SVD" in method_label else "fc"
    k = st.sidebar.slider("k (SVD)", 2, 50, 20, disabled=(method == "fc"))

    st.sidebar.subheader("Top-N")
    slate_size = st.sidebar.slider("slate_size (N)", 1, 15, 5)
    n_candidates = st.sidebar.slider("n_candidates", 10, 200, 50)
    min_score_enabled = st.sidebar.checkbox("Filtrer par score minimum", value=False)
    min_score_val = st.sidebar.number_input("min_score", value=0.0, step=0.1, disabled=not min_score_enabled)
    min_score = min_score_val if min_score_enabled else None
    exclude_consumed = st.sidebar.checkbox("Exclure items déjà notés", value=True)

    st.sidebar.subheader("Approche de sélection")
    approach = st.sidebar.selectbox(
        "Scénario",
        options=list(APPROACHES.keys()),
        format_func=lambda k: f"{k} — {APPROACHES[k]}",
    )
    diversity_weight = st.sidebar.slider("λ diversité (S3)", 0.0, 2.0, 0.5, step=0.1)
    solver = st.sidebar.selectbox("Solveur CPMpy (S4)", ["ortools"], index=0)

    st.sidebar.subheader("Périmètre utilisateurs")
    scope = st.sidebar.radio("Exécuter pour", ["Un utilisateur", "N premiers", "Tous"])
    if scope == "Un utilisateur":
        user_pick = st.sidebar.number_input("user_id MovieLens", min_value=1, value=1, step=1)
        user_scope = ("single", int(user_pick))
    elif scope == "N premiers":
        n_users = st.sidebar.slider("Nombre d'utilisateurs", 1, 50, 5)
        user_scope = ("first_n", n_users)
    else:
        st.sidebar.warning("Tous les utilisateurs peut être lent en S4.")
        user_scope = ("all", None)

    st.sidebar.subheader("Contraintes (S2 / S4)")
    st.session_state["use_category_constraint"] = st.sidebar.checkbox("Quota catégories", value=False)
    if st.session_state["use_category_constraint"]:
        st.session_state["category_min"] = st.sidebar.number_input("Min / catégorie / user", 0, 10, 0)
        st.session_state["category_max"] = st.sidebar.number_input("Max / catégorie / user", 0, 10, 2)

    st.session_state["use_provider_constraint"] = st.sidebar.checkbox("Exposition fournisseurs (global)", value=False)
    if st.session_state["use_provider_constraint"]:
        st.session_state["provider_min"] = st.sidebar.number_input("Min / fournisseur (global)", 0, 100, 0)
        st.session_state["provider_max"] = st.sidebar.number_input("Max / fournisseur (global)", 1, 500, 200)

    st.session_state["use_explanation"] = st.sidebar.checkbox("Support explicatif (S4)", value=False)
    if st.session_state["use_explanation"]:
        st.session_state["explanation_min"] = st.sidebar.number_input("explanation_min", 0, 5, 1)
        st.session_state["support_threshold"] = st.sidebar.number_input(
            "support_threshold", min_value=0.0, value=1e-12, format="%.2e"
        )

    st.session_state["forbidden_pairs_text"] = st.sidebar.text_area(
        "Paires interdites (item_id,item_id ; …)",
        placeholder="50,51; 120,121",
        height=80,
    )

    u_item_path = st.sidebar.text_input("u.item (genres MovieLens)", str(DEFAULT_U_ITEM))

    return {
        "uploaded": uploaded,
        "pivot_path": pivot_path,
        "method": method,
        "k": k,
        "slate_size": slate_size,
        "n_candidates": n_candidates,
        "min_score": min_score,
        "exclude_consumed": exclude_consumed,
        "approach": approach,
        "diversity_weight": diversity_weight,
        "solver": solver,
        "user_scope": user_scope,
        "u_item_path": Path(u_item_path),
    }


def resolve_user_indices(user_ids: np.ndarray, scope: tuple) -> list[int]:
    """Map sidebar user scope selection to matrix row indices."""
    kind, value = scope
    id_to_index = {int(v): i for i, v in enumerate(user_ids)}
    if kind == "single":
        if value not in id_to_index:
            raise ValueError(f"user_id {value} not found in the dataset.")
        return [id_to_index[value]]
    if kind == "first_n":
        return list(range(min(int(value), len(user_ids))))
    return list(range(len(user_ids)))


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="Recommandation Top-N — CPMpy",
        page_icon="U",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    st.title("Recommandation Top-N sous contraintes")
    st.caption(
        "Pipeline R → W → R̂ → C_u → sélection (S1–S4) · MovieLens 100K · CPMpy / OR-Tools"
    )

    cfg = render_sidebar()

    col_run, col_cmp = st.columns(2)
    run_clicked = col_run.button("Lancer la recommandation", type="primary", use_container_width=True)
    cmp_clicked = col_cmp.button("Comparer S1–S4 (même périmètre)", use_container_width=True)

    if run_clicked or cmp_clicked:
        try:
            with st.spinner("Chargement des données…"):
                user_ids, item_ids, R = load_dataset(cfg["uploaded"], cfg["pivot_path"])
                metadata = build_item_metadata(item_ids, cfg["u_item_path"])
                st.session_state.update(
                    user_ids=user_ids,
                    item_ids=item_ids,
                    R=R,
                    metadata=metadata,
                )

            constraints = build_constraints_from_sidebar(metadata)
            user_indices = resolve_user_indices(user_ids, cfg["user_scope"])

            if cmp_clicked:
                results = {}
                progress = st.progress(0.0, text="Comparaison des scénarios…")
                for i, approach in enumerate(["S1", "S2", "S3", "S4"]):
                    progress.progress((i + 1) / 4, text=f"Exécution {approach}…")
                    results[approach] = run_full_pipeline(
                        R=R,
                        method=cfg["method"],
                        k=cfg["k"],
                        n_candidates=cfg["n_candidates"],
                        exclude_consumed=cfg["exclude_consumed"],
                        min_score=cfg["min_score"],
                        approach=approach,
                        user_indices=user_indices,
                        slate_size=cfg["slate_size"],
                        constraints=constraints,
                        diversity_weight=cfg["diversity_weight"],
                        solver_name=cfg["solver"],
                        genre_matrix=metadata["genre_matrix"],
                    )
                progress.empty()
                st.session_state["comparison_results"] = results
                st.session_state["last_result"] = results[cfg["approach"]]
                st.success("Comparaison S1–S4 terminée.")
            else:
                with st.spinner(f"Exécution {cfg['approach']}…"):
                    result = run_full_pipeline(
                        R=R,
                        method=cfg["method"],
                        k=cfg["k"],
                        n_candidates=cfg["n_candidates"],
                        exclude_consumed=cfg["exclude_consumed"],
                        min_score=cfg["min_score"],
                        approach=cfg["approach"],
                        user_indices=user_indices,
                        slate_size=cfg["slate_size"],
                        constraints=constraints,
                        diversity_weight=cfg["diversity_weight"],
                        solver_name=cfg["solver"],
                        genre_matrix=metadata["genre_matrix"],
                    )
                st.session_state["last_result"] = result
                st.session_state["comparison_results"] = None
                if result["status"]:
                    st.success(
                        f"{cfg['approach']} terminé en {result['elapsed_s']:.2f}s · "
                        f"objectif = {result.get('objective')}"
                    )
                else:
                    st.error("Aucune solution faisable avec ces contraintes (S4).")

        except Exception as exc:
            st.error(f"Erreur : {exc}")

    tab_rec, tab_cmp, tab_data, tab_export, tab_about = st.tabs(
        ["Recommandations", "Comparaison", "Données", "Export", "À propos"]
    )

    result = st.session_state.get("last_result")
    user_ids = st.session_state.get("user_ids")
    item_ids = st.session_state.get("item_ids")
    metadata = st.session_state.get("metadata")

    with tab_rec:
        if result is None or user_ids is None:
            st.info("Configurez les paramètres dans la barre latérale, puis lancez une exécution.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Approche", result.get("approach", "—"))
            c2.metric("Temps (s)", f"{result.get('elapsed_s', 0):.2f}")
            c3.metric("Faisable", "Oui" if result.get("status") else "Non")
            c4.metric("Objectif", f"{result.get('objective', 0):.3f}" if result.get("objective") else "—")

            rec_users = sorted(result["recommendations"].keys())
            if not rec_users:
                st.warning("Aucune recommandation produite.")
            else:
                id_to_index = {int(v): i for i, v in enumerate(user_ids)}
                index_to_id = {i: int(v) for i, v in enumerate(user_ids)}
                default_u = rec_users[0]
                selected_id = st.selectbox(
                    "Utilisateur",
                    options=[index_to_id[u] for u in rec_users],
                    index=0,
                )
                u = id_to_index[selected_id]

                df = recommendations_to_dataframe(
                    result, user_ids, item_ids, metadata["titles"], metadata, u
                )
                st.subheader(f"Liste Top-{result.get('slate_size', '?')} — user {selected_id}")
                st.dataframe(df, use_container_width=True, hide_index=True)

                if not df.empty:
                    st.bar_chart(df.set_index("Titre")["Score R̂"])

                m = slate_metrics(
                    result["recommendations"].get(u, []),
                    result["R_hat"][u],
                    metadata["item_categories"],
                    metadata["genre_matrix"],
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Score moyen", f"{m['score_mean']:.3f}")
                m2.metric("ILD", f"{m['ild']:.2f}")
                m3.metric("Catégories distinctes", m["n_categories"])
                m4.metric("Items", m["count"])

                with st.expander("Historique de l'utilisateur (notes > 0)"):
                    R = st.session_state["R"]
                    history = [
                        {
                            "Item ID": int(item_ids[i]),
                            "Titre": metadata["titles"].get(int(item_ids[i]), ""),
                            "Note": float(R[u, i]),
                        }
                        for i in np.where(R[u] > 0)[0]
                    ]
                    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)

    with tab_cmp:
        comparison = st.session_state.get("comparison_results")
        if not comparison:
            st.info("Cliquez sur « Comparer S1–S4 » pour afficher un tableau comparatif.")
        else:
            rows = []
            u = next(iter(comparison["S1"]["recommendations"]))
            for approach, res in comparison.items():
                slate = res["recommendations"].get(u, [])
                m = slate_metrics(
                    slate,
                    res["R_hat"][u],
                    metadata["item_categories"],
                    metadata["genre_matrix"],
                )
                rows.append(
                    {
                        "Scénario": approach,
                        "Libellé": APPROACHES[approach],
                        "Faisable": res["status"],
                        "Temps (s)": round(res["elapsed_s"], 3),
                        "Score moyen": round(m["score_mean"], 4),
                        "ILD": round(m["ild"], 3),
                        "Catégories": m["n_categories"],
                        "Items": m["count"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.subheader("Listes recommandées (premier utilisateur du périmètre)")
            for approach, res in comparison.items():
                slate = res["recommendations"].get(u, [])
                ids = [int(item_ids[i]) for i in slate]
                st.markdown(f"**{approach}** — {', '.join(map(str, ids)) or '∅'}")

    with tab_data:
        if user_ids is None:
            st.info("Chargez d'abord les données.")
        else:
            R = st.session_state["R"]
            d1, d2, d3 = st.columns(3)
            d1.metric("Utilisateurs", len(user_ids))
            d2.metric("Items", len(item_ids))
            d3.metric("Métadonnées", metadata["metadata_source"])
            density = np.count_nonzero(R) / R.size
            st.write(f"Densité de la matrice R : **{density:.4%}**")
            st.caption(
                "Source des genres : fichier u.item MovieLens si disponible, "
                "sinon catégories synthétiques (movie_id mod 19)."
            )

    with tab_export:
        if result is None or user_ids is None:
            st.info("Aucun résultat à exporter.")
        else:
            csv_bytes = export_csv_bytes(result, user_ids, item_ids)
            st.download_button(
                "Télécharger recommendations.csv",
                data=csv_bytes,
                file_name=f"recommendations_{result.get('approach', 'run')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_about:
        st.markdown(
            """
            ### Goal
            This interface tests the **R → W → R_hat → C_u → Top-N**
            pipeline from the thesis without editing Python code directly.

            ### Scenarios
            - **S1** : score-based ranking (baseline)
            - **S2** : local heuristic constraint repair
            - **S3** : greedy relevance + diversity (λ)
            - **S4** : CPMpy optimization under declarative constraints

            ### Local run
            ```bash
            cd codes
            pip install -r requirements.txt
            streamlit run streamlit_app/app.py
            ```
            """
        )


if __name__ == "__main__":
    main()
