#!/usr/bin/env python3
"""Build u_data_pivot.csv from MovieLens u.data (tab-separated ratings)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_pivot(
    udata_path: Path,
    output_path: Path,
    *,
    n_users: int = 943,
    n_items: int = 1682,
) -> tuple[int, int]:
    ratings = pd.read_csv(
        udata_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    pivot = ratings.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        fill_value=0.0,
    )

    full_index = np.arange(1, n_users + 1)
    full_columns = np.arange(1, n_items + 1)
    pivot = pivot.reindex(index=full_index, columns=full_columns, fill_value=0.0)

    out = pivot.copy()
    out.insert(0, "user_id", out.index)
    out.to_csv(output_path, index=False)

    return pivot.shape[0], pivot.shape[1]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "u.data",
        help="MovieLens u.data file (default: repo u.data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "u_data_pivot.csv",
        help="Output pivot CSV (default: repo u_data_pivot.csv)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    n_users, n_items = build_pivot(args.input, args.output)
    print(f"Wrote {args.output} ({n_users} users x {n_items} items)")


if __name__ == "__main__":
    main()
