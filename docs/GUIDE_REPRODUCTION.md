# Reproduction guide

Companion to the [README](../README.md). Steps to install, run the CSP pipeline, and open the comparison demo.

## Requirements

- Python 3.10+
- Virtual environment recommended (`.venv` is git-ignored)

```bash
cd codes
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m py_compile recomm/pipeline.py recomm/operators.py streamlit_app/app.py
```

## MovieLens 100k files

| File | Purpose |
| --- | --- |
| `u.data` | Raw ratings |
| `u_data_pivot.csv` | User-item pivot for the pipeline |
| `u.item` | Genres for Streamlit metadata |

Regenerate the pivot:

```bash
.venv/bin/python scripts/build_pivot_from_udata.py
```

## Batch CSP pipeline

```bash
.venv/bin/python -m recomm.pipeline \
  --input u_data_pivot.csv \
  --output recommendations.csv \
  --method svd --k 20 --slate-size 5 --n-candidates 50
```

## Streamlit: classical vs CSP (test protocol)

```bash
.venv/bin/streamlit run streamlit_app/app.py
```

**Run full protocol** — three scenarios × two methods (greedy classical vs CSP), same logic as `recomm/evaluation.py` and `test/README.md`.

## Tests

See [test/README.md](../test/README.md).

```bash
.venv/bin/python test/simple_csp_with_parity.py
```

## `model/` folder

Archived CSV/plot outputs only — implementation code lives under `recomm/`.
