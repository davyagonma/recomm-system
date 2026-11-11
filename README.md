# Constraint-based Recommendation System

Operator-based scores **R̂ = R W** (collaborative filtering or latent factors), then **Top-N selection** under business rules. The research focus is the contrast between:

- **Classical greedy Top-N / FC** — scan candidates by score and skip items that violate constraints locally;
- **CSP (CPMpy + OR-Tools)** — optimize total score subject to **hard** constraints (feasible solution or explicit infeasibility).

---

## Quick start

**Requirements:** Python 3.10+, virtual environment recommended.

```bash
cd codes
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Use `setuptools>=65,<81` (pinned in `requirements.txt`) so CPMpy can import `pkg_resources`.

```bash
# Syntax check
.venv/bin/python -m py_compile recomm/pipeline.py recomm/operators.py streamlit_app/app.py

# Batch CSP on MovieLens pivot
.venv/bin/python -m recomm.pipeline --input u_data_pivot.csv --method svd --slate-size 5

# Classical vs CSP (toy matrix, instant)
.venv/bin/python test/simple_csp_with_parity.py

# Interactive comparison (same protocol as tests)
.venv/bin/streamlit run streamlit_app/app.py
```

---

## Repository layout

| Path | Role |
| --- | --- |
| `recomm/operators.py` | Build `W` (FC / SVD), compute `R_hat = R @ W`. |
| `recomm/pipeline.py` | Candidates `Cu`, CP model, CLI (`python -m recomm.pipeline`). |
| `recomm/evaluation.py` | **Evaluation protocol** — classical vs CSP, three scenarios (Streamlit + tests). |
| `recomm/constraints.py` | Constraint config and violation checks. |
| `recomm/legacy_pipeline.py`, `recomm/scoring_baseline.py` | Older / baseline variants. |
| `streamlit_app/` | Demo: 3 scenarios × 2 methods on one user. |
| `test/` | Reproducible benchmarks — see [test/README.md](test/README.md). |
| `examples/` | Standalone CF / latent-factor examples. |
| `scripts/build_pivot_from_udata.py` | Regenerate `u_data_pivot.csv` from `u.data`. |
| `model/` | Archived CSV/plots only — not source code. |
| `docs/catalogue_contraintes_globales.md` | Reference table of global CP constraints (French). |

**Data (MovieLens 100k, included):** `u.data`, `u_data_pivot.csv` (**943 × 1682**), `u.item` (genres).

```bash
.venv/bin/python scripts/build_pivot_from_udata.py   # rebuild pivot if needed
```

Optional: `curl -o u.item https://files.grouplens.org/datasets/movielens/ml-100k/u.item`

---

## Evaluation methodology

Only **two selectors** are compared (no multi-heuristic S1–S4 grid).

| Method | Behaviour |
| --- | --- |
| **Classical** | Greedy FC-style pass over sorted candidates (`test/simple_csp_with_parity.recommend_fc` logic in `recomm/evaluation.py`). |
| **CSP** | CPMpy model on the same scores / same `R̂` and candidate pool. |

**Three scenarios** (progression used in the thesis experiments):

| # | Scenario | Classical | CSP |
| --- | --- | --- | --- |
| 1 | Simple | Top-N by score | Cardinality only |
| 2 | With parity | + forbidden pairs | + forbidden pairs |
| 3 | Two constraints | + pairs + genre/category cap | + pairs + genre cap |

**Metrics:** execution **time**, **completeness** (full slate of size *N*), **constraint satisfaction**, **score sum**, CSP **feasibility**.

**Scripts:**

| Script | Purpose |
| --- | --- |
| `test/simple_csp_with_parity.py` | Side-by-side **FC greedy vs simple CSP** (toy 4×8 or `--csv`). |
| `test/fc_with_parity.py` | Classical only, parity rules. |
| `test/csp_with_parity.py` | Full **R → W → R̂ → Cu → CP** on MovieLens with category bounds. |

**Streamlit:** loads the full pivot (**943 × 1682**), but each run builds `W` and `R̂` on the whole matrix, then compares **one** `user_id` with at most **`n_candidates`** (default 50) and **`slate_size`** (default 5). CSP solves a **single-user** model. First SVD run can take tens of seconds; later interactions reuse the same session until you click **Run full protocol** again.

---

## Test cases — limits of the classical algorithm

Use these to show when greedy selection **stops short**, **violates** rules, or **misses** feasible high-score slates that CSP can find.

| ID | Intent | Command (from repo root) | What to look for |
| --- | --- | --- | --- |
| **T1** | Explainable toy | `.venv/bin/python test/simple_csp_with_parity.py` | User 0, 4×8 matrix, pair (4,6), `max_items_per_genre=1`, `top_n=3`. Classical **incomplete** or **violations**; CSP feasible slate. |
| **T2** | Parity only (ML) | `.venv/bin/python test/simple_csp_with_parity.py --csv u_data_pivot.csv --user-id 196 --top-n 5 --k 3 --max-candidates 30 --forbidden-pair 50:51` | Top scores blocked by pair; classical may not reach 5 items. |
| **T3** | Hard genre cap | Add `--max-items-per-genre 1` to **T2** (or `--category-max 1` via CSV path) | Early picks exhaust one genre; classical **Complete = No**; CSP trades score for feasibility. |
| **T4** | Small candidate pool | Same as T3 with `--max-candidates 6` and `--top-n 5` | Greedy exhausts pool; CSP may report **infeasible** explicitly. |
| **T5** | Contradictory rules | Toy only: several `--forbidden-pair` among top items + `--max-items-per-genre 1` + high `--top-n` | Classical short slate; CSP **no solution** if constraints cannot be met. |
| **T6** | Full pipeline CSP | `.venv/bin/python test/csp_with_parity.py --input u_data_pivot.csv --slate-size 5 --n-candidates 20 --category-max 1 --method svd --k 20` | Compare with Streamlit or `recomm/evaluation` for **classical** on same `category_max` and pairs. |

**Reading results:** high **score sum** with **Constraints OK = No** means a good ranking that is **not** a valid recommendation list — the main argument for CSP.

**Suggested demo for a supervisor:** T1 → T3 on user 196 → optional T4.

---

## Streamlit demo

```bash
.venv/bin/streamlit run streamlit_app/app.py
```

Click **Run full protocol (3 scenarios × 2 methods)**. Defaults: `user_id=196`, forbidden pair `50,51`, genre cap `2` in scenario 3. Adjust sidebar sliders to replay **T2–T4** without the CLI.

---

## CSP pipeline (CLI)

```bash
.venv/bin/python -m recomm.pipeline --help
```

| Argument | Default | Description |
| --- | --- | --- |
| `--input` | `u_data_pivot.csv` | Pivot CSV (`user_id` + item columns) |
| `--output` | `recommendations.csv` | Output CSV |
| `--method` | `svd` | `fc`, `svd`, or `fl` |
| `--k` | `20` | Latent dimension (SVD) |
| `--slate-size` | `5` | Items per user |
| `--n-candidates` | `50` | Max candidates before CP |
| `--min-score` | — | Drop low `R_hat` candidates |
| `--explanation-min` | `0` | Explanatory support count |
| `--support-threshold` | `1e-12` | Support strength in `W` |
| `--solver` | `ortools` | CPMpy backend |

Examples:

```bash
.venv/bin/python -m recomm.pipeline --method fc --slate-size 3
.venv/bin/python -m recomm.pipeline --method svd --k 10 --explanation-min 1 --support-threshold 0.05
```

---

## Pipeline overview

```text
R  →  W  →  R_hat  →  Cu  →  selection (classical or CP)
```

- **R[u,i]** — rating; 0 if missing. **Hu = (R > 0)**.
- **W** — item-item operator: **FC** (`W = D⁻¹ Rᵀ R D⁻¹`) or **SVD** (`W ≈ Q Qᵀ` on centred `R`).
- **R_hat = R @ W** — scores per user-item (SVD scores can be negative).
- **Cu[u]** — top `n_candidates` unscored items for user `u` (optionally exclude consumed / apply `min_score`).
- **CP** — maximize Σ R_hat[u,i]·x_ui with cardinality, categories, forbidden pairs, providers, explanatory support (see `recomm/pipeline.solve_cp_recommendations`).

### Constraints (summary)

| Constraint | Idea |
| --- | --- |
| Cardinality | Exactly `slate_size` items per user |
| Category bounds | Min/max items per category in a slate |
| Forbidden pairs | `x_i + x_j ≤ 1` for listed pairs |
| Provider exposure | Global min/max per provider across users |
| Explanatory support | Each pick supported by history via `W[j,i]` |

### CLI result object

| Key | Meaning |
| --- | --- |
| `status` | CSP found a feasible solution |
| `objective` | Sum of selected `R_hat` |
| `recommendations` | Item indices per user |
| `explanations` | Support items when enabled |

**Infeasibility:** tighten constraints incrementally; typical causes are `slate_size` too large vs `Cu`, harsh `category_bounds`, too many `forbidden_pairs`, or aggressive `min_score` / `support_threshold`.

### Compact formulation

```text
max Σ_u Σ_i R̂[u,i] · x_ui
s.t. Σ_i x_ui = N
     (+ category, pairs, provider, support constraints as configured)
```

---

## Remote

GitHub: [github.com/davyagonma/recomm-system](https://github.com/davyagonma/recomm-system)
