"""
Master script — generates dataset, runs experiments, produces all plots.
Usage:  python run_all.py
"""

import os
import sys

print("=" * 60)
print("WTA Checkpoint 2 — Full Pipeline")
print("=" * 60)

# Step 1: Generate dataset
print("\n[1/3] Generating benchmark dataset …")
from dataset_generator import generate_all_instances
generate_all_instances("datasets")

# Step 2: Run experiments
print("\n[2/3] Running experiments …")
from experiment_runner import run_all_experiments
run_all_experiments()

# Step 3: Analysis & plots
print("\n[3/3] Generating plots and statistics …")
from analysis import main as analysis_main
analysis_main()

print("\n" + "=" * 60)
print("Pipeline complete.")
print("  Results : results/experiment_results.csv")
print("  Figures : results/figures/")
print("=" * 60)
