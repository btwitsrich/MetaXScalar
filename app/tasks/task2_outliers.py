"""
app/tasks/task2_outliers.py — Task 2: Outlier Detection & Unit Errors

Generates a synthetic material stress-testing dataset, then injects:
  - ~5 % numeric values multiplied by 10× (outliers)
  - ~20 % of rows use wrong unit for length_mm (should be mm, stored as cm × 10)

Helper: rescale_column used by SciCleanEnv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate(seed: int | None = None) -> dict:
    """
    Returns:
        {
          "dirty": pd.DataFrame,
          "clean": pd.DataFrame,
          "outlier_ids": list[int],     # row indices of injected outliers
        }
    """
    rng = np.random.default_rng(seed if seed is not None else 1)

    n = 150

    # Ground-truth clean dataset (material stress test measurements)
    clean = pd.DataFrame(
        {
            "specimen_id": np.arange(1, n + 1, dtype=int),
            "length_mm": rng.normal(50.0, 5.0, n).round(2),       # all in mm
            "tensile_strength_mpa": rng.normal(200.0, 20.0, n).round(2),
            "yield_point_mpa": rng.normal(150.0, 15.0, n).round(2),
            "elongation_pct": rng.uniform(5.0, 30.0, n).round(2),
            "batch": rng.choice(["batch_1", "batch_2", "batch_3"], size=n),
        }
    )

    dirty = clean.copy()

    # ── Inject outliers (~5 % of numeric rows) ─────────────────────────────
    numeric_cols = ["tensile_strength_mpa", "yield_point_mpa", "elongation_pct"]
    n_outliers = max(1, int(n * 0.05))
    outlier_row_ids: list[int] = sorted(
        rng.choice(n, size=n_outliers, replace=False).tolist()
    )
    for row_id in outlier_row_ids:
        col = rng.choice(numeric_cols)
        dirty.loc[row_id, col] = dirty.loc[row_id, col] * 10.0

    # ── Unit error: ~20 % of rows store length in cm×10 instead of mm ──────
    unit_error_rows = rng.choice(n, size=int(n * 0.20), replace=False)
    dirty.loc[unit_error_rows, "length_mm"] = dirty.loc[unit_error_rows, "length_mm"] * 10.0

    return {
        "dirty": dirty,
        "clean": clean,
        "outlier_ids": outlier_row_ids,
    }


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def rescale_column(df: pd.DataFrame, column: str, factor: float) -> pd.DataFrame:
    """Multiply all values in *column* by *factor*."""
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    df[column] = pd.to_numeric(df[column], errors="coerce") * factor
    return df
