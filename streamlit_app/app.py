"""
Streamlit — classical (greedy FC / Top-N) vs CSP, same protocol as ``test/``.

Run:
    streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recomm.evaluation import EVAL_SCENARIOS, run_evaluation_protocol
from recomm.pipeline import load_pivot_csv, save_recommendations_csv
from streamlit_app.metadata import GENRES, build_item_metadata

DEFAULT_PIVOT = ROOT / "u_data_pivot.csv"
DEFAULT_U_ITEM = ROOT / "u.item"


def load_dataset(uploaded_file, pivot_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if uploaded_file is not None:
        tmp = ROOT / ".streamlit_upload.csv"
        tmp.write_bytes(uploaded_file.read())
        return load_pivot_csv(tmp)
    if not pivot_path.exists():
        raise FileNotFoundError(f"Pivot file not found: {pivot_path}")
    return load_pivot_csv(pivot_path)


def parse_forbidden_pairs(text: str, item_id_to_index: dict[int, int]) -> list[tuple[int, int]]:
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


def slate_to_df(
    slate: list[int],
    u: int,
    R_hat: np.ndarray,
    item_ids: np.ndarray,
    titles: dict[int, str],
    categories: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for rank, idx in enumerate(slate, start=1):
        mid = int(item_ids[idx])
        cat = int(categories[idx])
        rows.append(
            {
                "Rank": rank,
                "Item ID": mid,
                "Title": titles.get(mid, str(mid)),
                "Score R̂": round(float(R_hat[u, idx]), 4),
                "Genre bucket": GENRES[cat] if cat < len(GENRES) else str(cat),
            }
        )
    return pd.DataFrame(rows)


def results_to_table(protocol: dict) -> pd.DataFrame:
    records = []
    for scenario in EVAL_SCENARIOS:
        block = protocol["by_scenario"][scenario.id]
        for method_id in ("classical", "csp"):
            r = block[method_id]
            records.append(
                {
                    "Scenario": scenario.label,
                    "Method": r["method_label"],
                    "Time (s)": round(r["elapsed_s"], 4),
                    "Complete (|slate|=N)": "Yes" if r["complete"] else "No",
                    "Constraints OK": "Yes" if r["constraint_satisfied"] else "No",
                    "Violations": r["n_violations"],
                    "CSP feasible": "—"
                    if method_id == "classical"
                    else ("Yes" if r["cp_feasible"] else "No"),
                    "Score sum": round(r["score_sum"], 4),
                }
            )
    return pd.DataFrame(records)


def render_sidebar() -> dict:
    st.sidebar.header("Dataset")
    uploaded = st.sidebar.file_uploader("Pivot CSV", type=["csv"])
    pivot_path = Path(st.sidebar.text_input("Pivot path", str(DEFAULT_PIVOT)))
    u_item_path = Path(st.sidebar.text_input("u.item path", str(DEFAULT_U_ITEM)))

    st.sidebar.header("Score model (shared)")
    method = "svd" if st.sidebar.radio("W", ["SVD", "FC"]) == "SVD" else "fc"
    k = st.sidebar.slider("k (SVD)", 2, 50, 20, disabled=(method == "fc"))
    n_candidates = st.sidebar.slider("Candidates / user", 10, 200, 50)
    slate_size = st.sidebar.slider("Slate size N", 1, 15, 5)
    user_pick = st.sidebar.number_input("user_id", min_value=1, value=196)

    st.sidebar.header("Parity constraints")
    st.sidebar.caption("Used in scenarios 2 and 3 (same as ``test/``).")
    forbidden_text = st.sidebar.text_input("Forbidden pairs (id,id; …)", "50,51")
    category_max = st.sidebar.number_input("Max items per genre (scenario 3)", 1, 10, 2)

    return {
        "uploaded": uploaded,
        "pivot_path": pivot_path,
        "u_item_path": u_item_path,
        "method": method,
        "k": k,
        "n_candidates": n_candidates,
        "slate_size": slate_size,
        "user_pick": int(user_pick),
        "forbidden_text": forbidden_text,
        "category_max": int(category_max),
    }


def main() -> None:
    st.set_page_config(page_title="Classical vs CSP", layout="wide")
    st.title("Classical vs CSP — evaluation protocol")
    st.markdown(
        """
        Same progression as the **`test/`** scripts:

        1. **Simple** — unconstrained Top-N (classical) vs CSP with cardinality only.
        2. **With parity** — forbidden pairs on both sides (`simple_csp_with_parity` / `fc_with_parity`).
        3. **Two constraints** — parity **plus** genre/category cap (`csp_with_parity` on MovieLens).

        Metrics: **time**, **completeness** (full slate of size *N*), **constraint satisfaction**.
        Only two selectors: **classical greedy FC** and **CSP (CPMpy)** — no extra heuristics.
        """
    )

    cfg = render_sidebar()
    run = st.button("Run full protocol (3 scenarios × 2 methods)", type="primary")

    if run:
        try:
            user_ids, item_ids, R = load_dataset(cfg["uploaded"], cfg["pivot_path"])
            metadata = build_item_metadata(item_ids, cfg["u_item_path"])
            id_map = {int(v): i for i, v in enumerate(item_ids)}
            pairs = parse_forbidden_pairs(cfg["forbidden_text"], id_map)
            if cfg["user_pick"] not in id_map:
                raise ValueError(f"user_id {cfg['user_pick']} not in pivot.")
            u = id_map[cfg["user_pick"]]

            with st.spinner("Building R̂ = R W and running 6 experiments…"):
                protocol = run_evaluation_protocol(
                    R,
                    u,
                    metadata=metadata,
                    method=cfg["method"],
                    k=cfg["k"],
                    n_candidates=cfg["n_candidates"],
                    slate_size=cfg["slate_size"],
                    forbidden_pairs=pairs,
                    category_max=cfg["category_max"],
                )
            st.session_state["protocol"] = protocol
            st.session_state["user_ids"] = user_ids
            st.session_state["item_ids"] = item_ids
            st.session_state["metadata"] = metadata
            st.success("Protocol finished.")
        except Exception as exc:
            st.error(str(exc))

    protocol = st.session_state.get("protocol")
    metadata = st.session_state.get("metadata")
    item_ids = st.session_state.get("item_ids")

    tab_table, tab_detail, tab_about = st.tabs(["Summary", "Slates", "About"])

    with tab_table:
        if not protocol:
            st.info("Run the protocol to populate the table.")
        else:
            df = results_to_table(protocol)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("How to read this")
            st.markdown(
                """
                - **Complete** — classical greedy may stop before *N* items when constraints block candidates;
                  CSP either returns *N* items or reports infeasibility.
                - **Constraints OK** — parity / genre rules satisfied on the final slate.
                - When scenario ≥ 2, compare **violations** and **score sum**: CSP trades score for feasibility.
                """
            )

            # Highlight rows where CSP wins on feasibility
            for scenario in EVAL_SCENARIOS:
                block = protocol["by_scenario"][scenario.id]
                if scenario.id == "simple":
                    continue
                c_ok = block["csp"]["constraint_satisfied"]
                g_ok = block["classical"]["constraint_satisfied"]
                if c_ok and not g_ok:
                    st.success(
                        f"**{scenario.label}:** CSP satisfies constraints; classical does not."
                    )
                elif not block["csp"]["cp_feasible"]:
                    st.warning(f"**{scenario.label}:** CSP infeasible — relax constraints or widen candidates.")

    with tab_detail:
        if not protocol or metadata is None:
            st.info("Run the protocol first.")
        else:
            scenario_pick = st.selectbox(
                "Scenario",
                EVAL_SCENARIOS,
                format_func=lambda s: s.label,
            )
            u = protocol["user_index"]
            R_hat = protocol["R_hat"]
            block = protocol["by_scenario"][scenario_pick.id]
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Classical (greedy)**")
                st.dataframe(
                    slate_to_df(
                        block["classical"]["slate"],
                        u,
                        R_hat,
                        item_ids,
                        metadata["titles"],
                        metadata["item_categories"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                if block["classical"]["violations"]:
                    st.code("\n".join(block["classical"]["violations"]))
            with col_b:
                st.markdown("**CSP (CPMpy)**")
                st.dataframe(
                    slate_to_df(
                        block["csp"]["slate"],
                        u,
                        R_hat,
                        item_ids,
                        metadata["titles"],
                        metadata["item_categories"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                if not block["csp"]["cp_feasible"]:
                    st.error("Solver: no feasible solution.")
                elif block["csp"]["violations"]:
                    st.code("\n".join(block["csp"]["violations"]))

            if scenario_pick.id == "two_constraints" and item_ids is not None:
                csp_slate = block["csp"]["slate"]
                if csp_slate and block["csp"]["cp_feasible"]:
                    fake_result = {
                        "recommendations": {u: csp_slate},
                        "explanations": {},
                        "R_hat": R_hat,
                    }
                    user_ids = st.session_state.get("user_ids")
                    if user_ids is not None and st.button("Prepare CSP CSV export"):
                        path = ROOT / ".streamlit_export.csv"
                        save_recommendations_csv(path, fake_result, user_ids, item_ids)
                        st.download_button(
                            "Download CSV",
                            path.read_bytes(),
                            file_name="recommendations_csp.csv",
                            mime="text/csv",
                        )

    with tab_about:
        st.markdown(
            """
            ### Mapping to `test/`

            | Script | Role |
            | --- | --- |
            | `simple_csp_with_parity.py` | Toy or CSV: **FC greedy vs simple CSP** with pairs + genre |
            | `fc_with_parity.py` | **Classical only** with parity on 5×5 or MovieLens |
            | `csp_with_parity.py` | Full **R̂ = R W** CSP with **category bounds** on MovieLens |

            Implementation shared with CLI: `recomm/evaluation.py`.

            ```bash
            python test/simple_csp_with_parity.py
            python test/fc_with_parity.py --csv u_data_pivot.csv --user-id 196
            python test/csp_with_parity.py --input u_data_pivot.csv --category-max 2
            ```
            """
        )


if __name__ == "__main__":
    main()
