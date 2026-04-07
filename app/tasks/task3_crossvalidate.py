"""
app/tasks/task3_crossvalidate.py — Task 3: Cross-Experiment Consistency

Generates two related clinical trial datasets (arm A and arm B).
Arm B has:
  - Column name mismatches (e.g. 'subject_id' vs 'patient_id')
  - A contradictory measurement column (some values inverted/wrong)
  - Extra junk / phantom rows

The agent must rename columns, drop junk rows, flag contradictions,
and merge the two arms into one consistent dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

COLUMN_MAP_B: dict[str, str] = {
    # B name → correct aligned name
    "patient_id": "subject_id",
    "temp_celsius": "body_temp_c",
    "bp_systolic": "systolic_bp_mmhg",
}


def generate(seed: int | None = None) -> dict:
    """
    Returns:
        {
          "dirty":        pd.DataFrame,   # dataset A (mostly clean, minor issues)
          "dirty_B":      pd.DataFrame,   # dataset B (mismatched cols, junk rows, contradictions)
          "clean":        pd.DataFrame,   # ground-truth dataset A
          "clean_merged": pd.DataFrame,   # ground-truth merged A+B
        }
    """
    rng = np.random.default_rng(seed if seed is not None else 2)

    n = 80  # subjects per arm

    # --- Ground truth arm A ---------------------------------------------
    clean_A = pd.DataFrame(
        {
            "subject_id": np.arange(1, n + 1, dtype=int),
            "body_temp_c": rng.normal(37.0, 0.5, n).round(2),
            "systolic_bp_mmhg": rng.normal(120.0, 10.0, n).round(1),
            "diastolic_bp_mmhg": rng.normal(80.0, 8.0, n).round(1),
            "heart_rate_bpm": rng.normal(70.0, 10.0, n).round(0).astype(int),
            "arm": "A",
        }
    )

    # --- Ground truth arm B (same schema, different subjects) -----------
    clean_B_aligned = pd.DataFrame(
        {
            "subject_id": np.arange(n + 1, 2 * n + 1, dtype=int),
            "body_temp_c": rng.normal(37.1, 0.5, n).round(2),
            "systolic_bp_mmhg": rng.normal(121.0, 10.0, n).round(1),
            "diastolic_bp_mmhg": rng.normal(81.0, 8.0, n).round(1),
            "heart_rate_bpm": rng.normal(72.0, 10.0, n).round(0).astype(int),
            "arm": "B",
        }
    )

    # Ground-truth merged
    clean_merged = pd.concat([clean_A, clean_B_aligned], ignore_index=True)

    # --- Dirty arm B (mismatched names, junk, contradictions) -----------
    dirty_B = clean_B_aligned.copy()

    # 1. Rename columns to mismatched names
    dirty_B = dirty_B.rename(
        columns={
            "subject_id": "patient_id",
            "body_temp_c": "temp_celsius",
            "systolic_bp_mmhg": "bp_systolic",
        }
    )

    # 2. Contradictory values: flip sign on 10 % of systolic readings
    contra_rows = rng.choice(n, size=int(n * 0.10), replace=False)
    dirty_B.loc[contra_rows, "bp_systolic"] = (
        -dirty_B.loc[contra_rows, "bp_systolic"]
    )

    # 3. Junk rows: 5 phantom rows with nonsense data
    junk = pd.DataFrame(
        {
            "patient_id": [-1] * 5,
            "temp_celsius": [999.0] * 5,
            "bp_systolic": [0.0] * 5,
            "diastolic_bp_mmhg": [0.0] * 5,
            "heart_rate_bpm": [0] * 5,
            "arm": ["JUNK"] * 5,
        }
    )
    dirty_B = pd.concat([dirty_B, junk], ignore_index=True)

    # Dataset A is provided mostly clean (minor presentation)
    dirty_A = clean_A.copy()

    return {
        "dirty": dirty_A,
        "dirty_B": dirty_B,
        "clean": clean_A,
        "clean_merged": clean_merged,
    }


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def merge_datasets(df_A: pd.DataFrame, df_B: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate A and B on common columns (inner intersection).
    Drops rows where subject_id < 1 (junk sentinel).
    """
    common_cols = list(set(df_A.columns) & set(df_B.columns))
    if not common_cols:
        raise ValueError(
            "No common columns between datasets A and B. "
            "Rename columns in B to match A before merging."
        )

    merged = pd.concat(
        [df_A[common_cols], df_B[common_cols]], ignore_index=True
    )

    # Remove obvious junk rows (subject_id ≤ 0)
    if "subject_id" in merged.columns:
        merged = merged[merged["subject_id"] > 0].reset_index(drop=True)

    return merged
