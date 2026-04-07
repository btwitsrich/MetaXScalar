"""
app/tasks/task1_hygiene.py — Task 1: Basic Data Hygiene

Generates a synthetic biology/lab measurement dataset, then injects:
  - Duplicate rows (5–10 %)
  - Missing values (10–15 % of cells)
  - Columns stored with the wrong dtype (numeric → string)

Helper functions used by SciCleanEnv to apply agent actions.
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
          "dirty": pd.DataFrame,   # what the agent receives
          "clean": pd.DataFrame,   # ground truth
        }
    """
    rng = np.random.default_rng(seed if seed is not None else 0)

    n = 120  # base rows

    # Ground-truth clean dataset (simulates lab measurements)
    clean = pd.DataFrame(
        {
            "sample_id": np.arange(1, n + 1, dtype=int),
            "temperature_c": rng.normal(37.0, 1.5, n).round(2),
            "ph_level": rng.uniform(6.5, 7.5, n).round(3),
            "cell_count": rng.integers(1_000, 50_000, n).astype(int),
            "incubation_hours": rng.choice([24, 48, 72], size=n),
            "treatment_group": rng.choice(["control", "treated_A", "treated_B"], size=n),
        }
    )

    # ── Inject errors ──────────────────────────────────────────────────────

    dirty = clean.copy()

    # 1. Duplicate rows (~7 %)
    dup_indices = rng.choice(n, size=int(n * 0.07), replace=False)
    duplicates = dirty.iloc[dup_indices].copy()
    dirty = pd.concat([dirty, duplicates], ignore_index=True).sample(
        frac=1, random_state=int(rng.integers(0, 999_999))
    ).reset_index(drop=True)

    # 2. Missing values (~12 % of numeric cells)
    numeric_cols = ["temperature_c", "ph_level", "cell_count", "incubation_hours"]
    for col in numeric_cols:
        mask = rng.random(len(dirty)) < 0.12
        dirty.loc[mask, col] = np.nan

    # 3. Wrong dtype: sample_id and cell_count stored as strings
    dirty["sample_id"] = dirty["sample_id"].astype(str)
    dirty["cell_count"] = dirty["cell_count"].apply(
        lambda x: str(int(x)) if not pd.isna(x) else np.nan
    )

    return {"dirty": dirty, "clean": clean}


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def fill_null(df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
    """Fill nulls in *column* using *strategy*."""
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    if strategy == "mean":
        df[column] = df[column].fillna(pd.to_numeric(df[column], errors="coerce").mean())
    elif strategy == "median":
        df[column] = df[column].fillna(pd.to_numeric(df[column], errors="coerce").median())
    elif strategy == "mode":
        mode_val = df[column].mode()
        if not mode_val.empty:
            df[column] = df[column].fillna(mode_val.iloc[0])
    elif strategy == "drop":
        df = df.dropna(subset=[column]).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown fill strategy: '{strategy}'. Use mean|median|mode|drop.")
    return df


def cast_column(df: pd.DataFrame, column: str, dtype: str) -> pd.DataFrame:
    """Attempt to cast *column* to *dtype*."""
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    type_map = {"float": float, "int": "Int64", "str": str}
    if dtype not in type_map:
        raise ValueError(f"Unknown dtype '{dtype}'. Use float|int|str.")

    target = type_map[dtype]
    if dtype in ("float", "int"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if dtype == "int":
        df[column] = df[column].astype("Int64")
    elif dtype == "float":
        df[column] = df[column].astype(float)
    else:
        df[column] = df[column].astype(str)
    return df
