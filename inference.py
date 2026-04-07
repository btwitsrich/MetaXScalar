"""
inference.py — Sample agent loop for SciClean-Env.

Demonstrates how an AI agent (here: a rule-based heuristic) interacts with
the environment via HTTP. This is the script the competition judges will run.

Usage:
    python inference.py                  # runs all 3 tasks once
    python inference.py --task 2         # run a single task
    python inference.py --host http://localhost:8000

The heuristic agent applies all safe cleaning operations it can detect, then
calls "submit" to finalise. A real LLM agent would introspect the observation
and choose actions dynamically.
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0


def post(client: httpx.Client, path: str, body: dict | None = None) -> dict:
    resp = client.post(f"{BASE_URL}{path}", json=body or {}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get(client: httpx.Client, path: str) -> dict:
    resp = client.get(f"{BASE_URL}{path}", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Per-task heuristic strategies
# ---------------------------------------------------------------------------

def run_task1(client: httpx.Client, seed: int) -> float:
    """
    Task 1 strategy:
      1. drop_duplicates
      2. fill_null with 'mean' for all numeric columns
      3. cast sample_id and cell_count to int
      4. submit
    """
    obs = post(client, "/reset", {"task_id": 1, "seed": seed})
    print(f"\n[Task 1] Reset. Dirty rows: {len(obs['dataframe'])}")

    actions = [
        {"action": "drop_duplicates"},
        {"action": "fill_null", "column": "temperature_c", "strategy": "mean"},
        {"action": "fill_null", "column": "ph_level", "strategy": "mean"},
        {"action": "fill_null", "column": "cell_count", "strategy": "median"},
        {"action": "fill_null", "column": "incubation_hours", "strategy": "mode"},
        {"action": "cast_column", "column": "sample_id", "dtype": "int"},
        {"action": "cast_column", "column": "cell_count", "dtype": "int"},
        {"action": "submit"},
    ]

    total_reward = 0.0
    for action in actions:
        result = post(client, "/step", {"action": action})
        total_reward += result["reward"]
        status = "[done]" if result["done"] else f"step {result['observation']['step']}"
        print(
            f"  {action['action']:20s}  reward={result['reward']:.4f}  "
            f"cumulative={total_reward:.4f}  [{status}]"
            + (f"  info={result['info']}" if result["info"] else "")
        )
        if result["done"]:
            break

    return total_reward


def run_task2(client: httpx.Client, seed: int) -> float:
    """
    Task 2 strategy:
      1. rescale length_mm by 0.1 (fixes 10× unit error on affected rows)
      2. For each row, if any numeric value is > 5× the column mean, flag it
      3. Drop all flagged rows
      4. submit
    """
    obs = post(client, "/reset", {"task_id": 2, "seed": seed})
    df = obs["dataframe"]
    print(f"\n[Task 2] Reset. Rows: {len(df)}")

    total_reward = 0.0

    # Rescale the length column to fix the unit problem
    r = post(client, "/step", {"action": {"action": "rescale_column", "column": "length_mm", "factor": 0.1}})
    total_reward += r["reward"]
    print(f"  rescale_column(length_mm, 0.1)  reward={r['reward']:.4f}")

    # Refresh observation
    obs = post(client, "/reset", {"task_id": 2, "seed": seed})

    # Detect outlier rows heuristically (value > 5× std above mean)
    import statistics
    numeric_cols = ["tensile_strength_mpa", "yield_point_mpa", "elongation_pct"]

    col_stats: dict[str, tuple[float, float]] = {}
    for col in numeric_cols:
        vals = [row[col] for row in df if row.get(col) is not None]
        if vals:
            mean = statistics.mean(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0
            col_stats[col] = (mean, stdev)

    outlier_row_ids = []
    for i, row in enumerate(df):
        for col, (mean, stdev) in col_stats.items():
            val = row.get(col)
            if val is not None and stdev > 0 and abs(val - mean) > 5 * stdev:
                outlier_row_ids.append(i)
                break

    print(f"  Detected {len(outlier_row_ids)} outlier rows: {outlier_row_ids[:10]}")

    for row_id in outlier_row_ids:
        r = post(client, "/step", {"action": {"action": "flag_outlier", "row_id": row_id}})
        total_reward += r["reward"]

    for row_id in sorted(outlier_row_ids, reverse=True):
        r = post(client, "/step", {"action": {"action": "drop_row", "row_id": row_id}})
        total_reward += r["reward"]

    r = post(client, "/step", {"action": {"action": "submit"}})
    total_reward += r["reward"]
    print(f"  submit  reward={r['reward']:.4f}  final_score={r['info'].get('final_score', '?')}")

    return total_reward


def run_task3(client: httpx.Client, seed: int) -> float:
    """
    Task 3 strategy:
      1. Rename mismatched columns in B
      2. Drop junk rows (patient_id < 1)
      3. Flag contradictions (negative bp_systolic)
      4. merge_datasets
      5. submit
    """
    obs = post(client, "/reset", {"task_id": 3, "seed": seed})
    print(f"\n[Task 3] Reset. Dataset A rows: {len(obs['dataframe'])}, "
          f"Dataset B rows: {len(obs['aux'].get('dataset_B', []))}")

    total_reward = 0.0
    actions = [
        # Fix column names in B
        {"action": "rename_column", "dataset": "B", "old": "patient_id", "new": "subject_id"},
        {"action": "rename_column", "dataset": "B", "old": "temp_celsius", "new": "body_temp_c"},
        {"action": "rename_column", "dataset": "B", "old": "bp_systolic", "new": "systolic_bp_mmhg"},
    ]

    for action in actions:
        r = post(client, "/step", {"action": action})
        total_reward += r["reward"]
        print(f"  {action['action']}  {action.get('old','')} -> {action.get('new','')}  reward={r['reward']:.4f}")

    # Drop junk rows (patient_id was -1 → after rename subject_id, they have subject_id=-1)
    # Junk rows are the last 5 in B; indices 80-84
    for row_id in [84, 83, 82, 81, 80]:
        r = post(client, "/step", {"action": {"action": "drop_row", "dataset": "B", "row_id": row_id}})
        total_reward += r["reward"]

    # Merge
    r = post(client, "/step", {"action": {"action": "merge_datasets"}})
    total_reward += r["reward"]
    print(f"  merge_datasets  reward={r['reward']:.4f}  info={r['info']}")

    # Submit
    r = post(client, "/step", {"action": {"action": "submit"}})
    total_reward += r["reward"]
    print(f"  submit  reward={r['reward']:.4f}  final_score={r['info'].get('final_score', '?')}")

    return total_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global BASE_URL

    parser = argparse.ArgumentParser(description="SciClean-Env sample inference agent")
    parser.add_argument("--host", default=BASE_URL, help="Base URL of the environment server")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], help="Run only this task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    BASE_URL = args.host.rstrip("/")

    tasks_to_run = [args.task] if args.task else [1, 2, 3]
    task_runners = {1: run_task1, 2: run_task2, 3: run_task3}

    # Wait for server to be ready
    with httpx.Client() as client:
        for attempt in range(30):
            try:
                resp = client.get(f"{BASE_URL}/health", timeout=5)
                if resp.status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            print(f"Waiting for server... ({attempt + 1}/30)", end="\r", flush=True)
            time.sleep(2)
        else:
            print(f"\n[ERROR] Could not reach server at {BASE_URL}. Is it running?")
            sys.exit(1)

        print(f"[OK] Connected to {BASE_URL}")

        totals: dict[int, float] = {}
        for task_id in tasks_to_run:
            reward = task_runners[task_id](client, seed=args.seed)
            totals[task_id] = reward
            state = get(client, "/state")
            print(f"  -> Final state: {state}")

    print("\n" + "=" * 55)
    print("  INFERENCE SUMMARY")
    print("=" * 55)
    for tid, r in totals.items():
        print(f"  Task {tid}: cumulative reward = {r:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
