"""
Experiment Runner
=================
Runs all four algorithms (MMR-orig, MMR-mod, ACO-orig, ACO-mod) on all 150
benchmark instances and saves results to CSV.

Usage:
    python experiment_runner.py [--categories 5x5,10x10] [--instances 1-10]
"""

import os
import json
import time
import argparse
import csv
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from dataset_generator  import CATEGORIES, load_instance, generate_all_instances
from wta_utils          import expand_instance
from mmr_original       import mmr_original
from mmr_modified       import mmr_modified
from aco_original       import aco_original
from aco_modified       import aco_modified


DATASET_DIR   = "datasets"
RESULTS_DIR   = "results"
RESULTS_CSV   = os.path.join(RESULTS_DIR, "experiment_results.csv")


# Time budget for ACO: scale with problem size, capped at 30 s
def aco_budget(n_weapons: int, n_targets: int) -> float:
    size = n_weapons * n_targets
    return min(2.0 + size / 500.0, 30.0)


def run_instance(instance_path: str) -> List[Dict[str, Any]]:
    """Run all four algorithms on one instance and return list of result rows.

    Instances are expanded from the typed model (weapon_quantities) into the
    flat individual-weapon representation before being passed to algorithms.
    Type i with quantity w_i becomes w_i identical rows in kill_prob, so the
    algorithms naturally model x_ij (integer assignments) via repeated tokens.
    """
    raw  = load_instance(instance_path)
    inst = expand_instance(raw)          # typed -> flat individual-weapon format
    n_w  = inst["n_weapons"]            # total weapon tokens after expansion
    n_t  = inst["n_targets"]
    budget = aco_budget(n_w, n_t)

    rows = []

    n_types    = inst.get("n_weapon_types", n_w)
    total_wpns = n_w   # sum of weapon_quantities after expansion

    # --- MMR Original ---
    r = mmr_original(inst)
    rows.append({
        "algorithm":      "MMR_Original",
        "n_weapon_types": n_types,
        "n_weapons":      total_wpns,
        "n_targets":      n_t,
        "value":          r["value"],
        "time_sec":       r["time_sec"],
        "iterations":     r["iterations"],
        "extra":          "",
    })

    # --- MMR Modified ---
    r = mmr_modified(inst)
    rows.append({
        "algorithm":      "MMR_Modified",
        "n_weapon_types": n_types,
        "n_weapons":      total_wpns,
        "n_targets":      n_t,
        "value":          r["value"],
        "time_sec":       r["time_sec"],
        "iterations":     r["ls_passes"],
        "extra":          f"improvement_pct={r['improvement_pct']:.4f}",
    })

    # --- ACO Original ---
    r = aco_original(inst, time_budget_sec=budget)
    rows.append({
        "algorithm":      "ACO_Original",
        "n_weapon_types": n_types,
        "n_weapons":      total_wpns,
        "n_targets":      n_t,
        "value":          r["value"],
        "time_sec":       r["time_sec"],
        "iterations":     r["iterations"],
        "extra":          "",
    })

    # --- ACO Modified ---
    r = aco_modified(inst, time_budget_sec=budget)
    rows.append({
        "algorithm":      "ACO_Modified",
        "n_weapon_types": n_types,
        "n_weapons":      total_wpns,
        "n_targets":      n_t,
        "value":          r["value"],
        "time_sec":       r["time_sec"],
        "iterations":     r["iterations"],
        "extra":          f"improvement_pct={r['improvement_pct']:.4f}",
    })

    return rows


def run_all_experiments(cat_filter: Optional[List[str]] = None, inst_range: Optional[Tuple[int, int]] = None, append: bool = False):
    """
    Run experiments across all (filtered) categories and instances.
    Saves incremental results to CSV.
    """
    # Generate dataset if not present
    if not os.path.isdir(DATASET_DIR):
        print("Dataset not found - generating ...")
        generate_all_instances(DATASET_DIR)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    fieldnames = ["algorithm", "n_weapon_types", "n_weapons", "n_targets",
                  "value", "time_sec", "iterations", "extra",
                  "category", "instance_id", "instance_file"]

    # Load existing rows if appending
    existing_rows = []
    if append and os.path.isfile(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        print(f"Loaded {len(existing_rows)} existing rows (append mode)")


    all_rows = []
    total_start = time.perf_counter()

    for cat in CATEGORIES:
        if cat_filter and cat["name"] not in cat_filter:
            continue

        cat_dir  = os.path.join(DATASET_DIR, cat["name"])
        inst_start, inst_end = (1, cat["count"]) if inst_range is None else inst_range

        for idx in range(inst_start, inst_end + 1):
            fname = f"instance_{idx:03d}.json"
            fpath = os.path.join(cat_dir, fname)
            if not os.path.isfile(fpath):
                continue

            print(f"  [{cat['name']}] instance {idx:3d}/{inst_end} ...", end="", flush=True)
            t0 = time.perf_counter()
            rows = run_instance(fpath)
            elapsed = time.perf_counter() - t0

            for row in rows:
                row["category"]      = cat["name"]
                row["instance_id"]   = idx
                row["instance_file"] = fname
                all_rows.append(row)

            vals = {r["algorithm"]: f"{r['value']:.2f}" for r in rows}
            print(f"done in {elapsed:.1f}s  "
                  f"MMR-O={vals['MMR_Original']}  MMR-M={vals['MMR_Modified']}  "
                  f"ACO-O={vals['ACO_Original']}  ACO-M={vals['ACO_Modified']}")

    # Write CSV (merge existing + new)
    combined = existing_rows + all_rows
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)

    total_elapsed = time.perf_counter() - total_start
    print(f"\nResults saved -> {RESULTS_CSV}")
    print(f"Total rows: {len(all_rows)}  |  Total time: {total_elapsed:.1f} s")
    return all_rows


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WTA Experiment Runner")
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated category names to run (e.g. 5x5,10x10). Default: all."
    )
    parser.add_argument(
        "--instances", type=str, default=None,
        help="Instance range as START-END (e.g. 1-10). Default: all 30."
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing results CSV instead of overwriting."
    )
    args = parser.parse_args()

    cat_filter  = args.categories.split(",") if args.categories else None
    inst_range  = None
    if args.instances:
        parts = args.instances.split("-")
        inst_range = (int(parts[0]), int(parts[1]))

    run_all_experiments(cat_filter=cat_filter, inst_range=inst_range, append=args.append)
