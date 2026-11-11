# Constraint-based Recommendation System

This project implements a recommendation pipeline combining:

- a user-item matrix `R`;
- a weight matrix `W` built either by collaborative filtering or by latent factors;
- a reconstructed score matrix `R_hat = R @ W`;
- a constraint programming model with `CPMpy` to produce feasible recommendations under constraints.

The goal is not simply to recommend the highest-scoring items, but to produce recommendation lists that satisfy constraints such as list size, diversity, category balance, provider exposure, and explanatory support.

## Project Structure

| Path | Role |
| --- | --- |
| `recomm/operators.py` | Build `W` (FC / SVD) and compute `R_hat = R @ W`. |
| `recomm/pipeline.py` | **Main pipeline**: candidates `Cu`, CP model (CPMpy), CSV I/O, CLI (`python -m recomm.pipeline`). |
| `recomm/legacy_pipeline.py` | First pipeline iteration (reference only). |
| `recomm/scoring_baseline.py` | Scoring + CP without extended constraints (benchmarks). |
| `streamlit_app/` | **Classical vs CSP** demo (3 scenarios × 2 methods, same as `test/`). |
| `recomm/evaluation.py` | Shared evaluation protocol (used by Streamlit). |
| `examples/` | Standalone pedagogical scripts (cosine CF, latent factors). |
| `test/` | Parity and timing on toy matrices and MovieLens. See [test/README.md](test/README.md). |
| `exercices/` | CPMpy / OR-Tools / MiniZinc learning demos. |
| `model/` | Archived experiment outputs (CSVs, figures). See [model/README.md](model/README.md). |
| `scripts/build_pivot_from_udata.py` | Rebuild `u_data_pivot.csv` from `u.data`. |
| `u.data`, `u_data_pivot.csv`, `u.item` | MovieLens 100k data. |

## General Pipeline

The main pipeline is implemented in `recomm/pipeline.py`:

```text
R -> W -> R_hat -> Cu -> CP model -> recommendations
```

### 1. User-item matrix `R`

`R` is a matrix of size `n_users x n_items`.

```python
R[u, i] = rating  # rating given by user u for item i; 0 if missing
```

Example:

```python
R = np.array([
    [5, 3, 0, 1, 0, 0],
    [4, 2, 1, 0, 0, 0],
    [1, 0, 5, 4, 0, 0],
    [0, 4, 4, 5, 0, 0],
], dtype=float)
```

### 2. User history `Hu`

`Hu` is a binary matrix built from `R`.

```python
Hu[u, i] = 1  # if user u has already consumed or rated item i
Hu[u, i] = 0  # otherwise
```

In the pipeline module, it is built by:

```python
Hu = (R > 0).astype(int)
```

### 3. Building `W`

The matrix `W` (`n_items x n_items`) represents relationships between items. Two methods are available.

#### Method `fc`: collaborative filtering

This method uses item-item co-occurrence:

```text
C = R.T @ R
W = D^-1 C D^-1
```

```python
method="fc"
```

#### Method `fl` / `svd`: latent factors

This method uses a truncated SVD to project items into a latent space:

```text
R_centered ≈ U Σ Vᵀ
Q = Vᵀ_k.T
W = Q @ Q.T
```

```python
method="fl"   # or method="svd"
```

The parameter `k` controls the number of latent factors (ignored with `method="fc"`).

- small `k`: compact representation, more general recommendations;
- large `k`: detailed representation, but potentially more sensitive to noise;
- `k` does not change the number of recommendations produced, only how `W` is built in SVD.

### 4. Reconstructed scores `R_hat`

Once `W` is obtained, recommendation scores are computed with:

```python
R_hat = R @ W
```

`R_hat[u, i]` is the predicted score of item `i` for user `u`.

Note: with the SVD method, scores can be negative because the matrix is centered before decomposition. This is not necessarily an error, but it can affect the CP objective.

### 5. Candidate sets `Cu`

For each user `u`, a candidate set `Cu[u]` is built from the best scores in `R_hat`.

Selection steps:

1. Compute all scores `R_hat[u, i]`;
2. Remove already consumed items if `exclude_consumed=True`;
3. Remove items below `min_score` if this parameter is set;
4. Sort remaining items by descending score;
5. Keep the top `n_candidates`.

| Parameter | Role |
| --- | --- |
| `n_candidates` | Maximum number of candidates kept per user. |
| `exclude_consumed` | If `True`, already consumed items are excluded. |
| `min_score` | Minimum score to accept a candidate item. |

`n_candidates` is not the final number of recommendations. It only limits how many items the CP model may consider per user. The final count is given by `slate_size`.

Example:

```python
Cu = build_candidate_sets(
    R_hat=R_hat,
    Hu=Hu,
    n_candidates=5,
    exclude_consumed=True,
    min_score=0,
)
```

## CP Model

The CP model selects binary variables:

```text
x_ui = 1  if item i is recommended to user u
x_ui = 0  otherwise
```

The objective is to maximize the sum of scores of selected items:

```text
max sum(R_hat[u, i] * x_ui)
```

The model is solved with `CPMpy` and the `ortools` solver by default.

## Available Constraints

### Cardinality

Each user receives exactly `slate_size` recommendations.

```python
slate_size=3
```

### Category balance

If items have categories, a minimum and maximum number of items per category can be imposed.

```python
item_categories = np.array([0, 0, 1, 1, 2, 2])
category_bounds = (0, 1)
```

Here, each category can appear between 0 and 1 times in a user's list.

Per-category bounds can also be specified:

```python
category_bounds = {
    0: (1, 2),
    1: (0, 1),
    2: (0, 1),
}
```

### Diversity — forbidden pairs

Certain pairs of items can be forbidden from appearing together.

```python
forbidden_pairs = [(0, 1), (2, 3)]
# imposes: x_u0 + x_u1 <= 1  for each user
```

### Provider exposure

If items belong to providers, global exposure of each provider across all users can be bounded.

```python
item_providers = np.array([0, 1, 0, 1, 0, 1])
provider_bounds = (1, 6)
```

Here, each provider must appear at least 1 time and at most 6 times across all recommendations.

Per-provider bounds can also be specified:

```python
provider_bounds = {
    0: (2, 5),
    1: (1, 4),
}
```

### Explanatory support

Explanatory support forces each recommendation to be justified by already consumed items.

```text
y_uij = 1  if history item j supports the recommendation of item i for user u
```

The link between `j` and `i` is accepted if:

```python
abs(W[j, i]) > support_threshold
```

| Parameter | Role |
| --- | --- |
| `explanation_min` | Minimum number of history items that must support each recommendation. |
| `support_threshold` | Minimum threshold to consider the link `W[j, i]` strong enough. |

Example:

```python
explanation_min=1
support_threshold=0.05
```

This means: each recommended item must have at least one history item linked with a strength above `0.05`.

If `support_threshold` is too low, explanations will be numerous but unselective. If too high, the model may become infeasible.

## Running the pipeline (CLI)

```bash
python -m recomm.pipeline --help
```

### Basic command

```bash
python -m recomm.pipeline --input u_data_pivot.csv --output recommendations.csv
```

### All arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--input` | `u_data_pivot.csv` | Pivot CSV file to load |
| `--output` | `recommendations.csv` | Output recommendations CSV file |
| `--method` | `svd` | Method used to build `W`: `fc`, `svd`, or `fl` |
| `--k` | `20` | Number of latent factors (SVD only) |
| `--slate-size` | `5` | Number of items to recommend per user |
| `--n-candidates` | `50` | Maximum number of candidates per user before CP |
| `--min-score` | `None` | Minimum score in `R_hat` to keep a candidate |
| `--include-consumed` | *(flag)* | Allow recommending already rated/consumed items |
| `--explanation-min` | `0` | Minimum number of history items supporting each recommendation |
| `--support-threshold` | `1e-12` | Support threshold used for explanations |
| `--solver` | `ortools` | CPMpy solver to use |

### Examples

```bash
# Collaborative filtering, 3 recommendations per user
python -m recomm.pipeline --method fc --slate-size 3

# SVD with k=10, up to 50 candidates per user
python -m recomm.pipeline --method svd --k 10 --slate-size 5 --n-candidates 50

# With minimum score filter and explanatory support
python -m recomm.pipeline --method svd --min-score 0 --explanation-min 1 --support-threshold 0.05

# Allow already consumed items
python -m recomm.pipeline --include-consumed --slate-size 5
```

## Testing `fc` and `svd`

```python
result_fc  = recommend_complete(R, method="fc",  slate_size=2)
result_svd = recommend_complete(R, method="svd", k=2, slate_size=2)
```

If the SVD objective is negative, it may come from the negative scores produced by the latent projection. To avoid recommending items with negative scores:

```python
result_svd = recommend_complete(
    R,
    method="svd",
    k=2,
    slate_size=2,
    min_score=0,
)
```

Note: if `min_score=0` removes too many candidates, the CP model may become infeasible.

## Streamlit demo (CSP vs classical)

From the repository root:

```bash
.venv/bin/streamlit run streamlit_app/app.py
```

The UI mirrors the **`test/` evaluation protocol** on one user and one shared `R̂ = R W`:

| Scenario | Classical (greedy FC) | CSP (CPMpy) |
| --- | --- | --- |
| 1. Simple | Top-N by score | CP with cardinality only |
| 2. With parity | Greedy + forbidden pairs | CP + forbidden pairs |
| 3. Two constraints | Greedy + pairs + genre cap | CP + pairs + genre cap |

Reported for each run: **time**, **completeness** (slate of size *N*), **constraint satisfaction**, **score sum**. See [test/README.md](test/README.md).

```bash
curl -o u.item https://files.grouplens.org/datasets/movielens/ml-100k/u.item
```

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install numpy cpmpy ortools
# or:
.venv/bin/pip install -r requirements.txt
```

Syntax check:

```bash
.venv/bin/python -m py_compile recomm/pipeline.py recomm/operators.py streamlit_app/app.py
```

## Interpreting the Output

A typical output looks like:

```python
{
    "status": True,
    "objective": 0.46,
    "recommendations": {0: [2, 4], 1: [3, 4]},
    "explanations": {0: {2: [0, 1], 4: [1]}},
    "R_hat": ...,
    "Cu": ...,
}
```

| Key | Meaning |
| --- | --- |
| `status` | `True` if the solver found a feasible solution |
| `objective` | Optimized sum of selected `R_hat[u, i]` scores |
| `recommendations` | Recommended items per user |
| `explanations` | History items supporting each recommendation |
| `R_hat` | Full predicted score matrix |
| `Cu` | Candidates considered per user |

## Common Infeasibility Causes

The model may return `status=False` if constraints are too tight. Common causes:

| Cause |
| --- |
| `slate_size` exceeds the number of available candidates |
| `min_score` filters out too many items |
| `category_bounds` requires categories absent from candidates |
| `provider_bounds` imposes an impossible exposure |
| `support_threshold` is too high |
| `explanation_min` requires too many history support items |
| `forbidden_pairs` forbids too many combinations |

To debug: start with no constraints, then add them incrementally.

## Mathematical Formulation

```text
U = {1, ..., m}    # users
I = {1, ..., n}    # items

R  ∈ ℝ^(m×n)      # observed matrix
R̂  = R W          # reconstructed scores

x_ui ∈ {0, 1}     # decision variable

max  Σ_u Σ_i R̂[u,i] · x_ui
s.t. Σ_i x_ui = N              (cardinality)
     Σ_j y_uij ≥ K · x_ui      (explanatory support)
     y_uij ≤ x_ui
     abs(W[j,i]) > support_threshold  → j is a valid support for i
```


