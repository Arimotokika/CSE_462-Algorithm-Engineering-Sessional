"""
WTA Benchmark Dataset Generator
=================================
Generates synthetic Weapon-Target Assignment (WTA) problem instances.

Justification for synthetic data:
    No standardized, publicly available WTA benchmark datasets exist in the
    literature. Prior studies (Ahuja et al. 2007, Lee et al. 2003, Manne 1958)
    each used proprietary or problem-specific instances that were never released.
    Following the methodology of Hasan & Barua and common practice in
    combinatorial-optimization research, we generate random instances whose
    statistical properties (kill-probability range, target-value range) match
    those reported in the surveyed literature.

Why W != T instances are included:
    The WTA problem is defined for any number of weapons W and targets T.
    n_weapons = n_targets is NOT a requirement — it was only a simplification
    in earlier work. Three structurally distinct scenarios exist:

        W < T  (weapon-scarce):  militarily the most realistic and hardest case.
                                  Not every target can be directly engaged.
        W = T  (balanced):       each weapon can be matched to one target.
        W > T  (weapon-rich):    concentrated fire is possible; easier to solve.

    The ACO pseudocode (Hasan & Barua, line 5-9) already handles this via
        noOfAnts = max(noOfWeapons, noOfTargets)
    and the kill_prob matrix is always shaped (W x T), so all algorithms
    work without modification for any W and T.

Instance categories:
    Square   (W = T): 5x5, 10x10, 20x20, 50x50, 100x100  — 30 each
    W < T  (scarce) : 5w10t, 10w20t, 20w50t               — 20 each
    W > T  (rich)   : 10w5t, 20w10t, 50w20t               — 20 each

Total: 150 + 60 + 60 = 270 instances (150 square + 120 non-square)
"""

import numpy as np
import json
import os

# Reproducibility
RNG_SEED = 42

# Square instances (W = T) — original design
SQUARE_CATEGORIES = [
    {"name": "5x5",     "weapons": 5,   "targets": 5,   "count": 30},
    {"name": "10x10",   "weapons": 10,  "targets": 10,  "count": 30},
    {"name": "20x20",   "weapons": 20,  "targets": 20,  "count": 30},
    {"name": "50x50",   "weapons": 50,  "targets": 50,  "count": 30},
    {"name": "100x100", "weapons": 100, "targets": 100, "count": 30},
]

# Non-square instances (W != T)
NONSQUARE_CATEGORIES = [
    # Weapon-scarce (W < T): harder, realistic
    {"name": "5w10t",  "weapons": 5,  "targets": 10, "count": 20, "type": "scarce"},
    {"name": "10w20t", "weapons": 10, "targets": 20, "count": 20, "type": "scarce"},
    {"name": "20w50t", "weapons": 20, "targets": 50, "count": 20, "type": "scarce"},
    # Weapon-rich (W > T): easier, concentrated fire
    {"name": "10w5t",  "weapons": 10, "targets": 5,  "count": 20, "type": "rich"},
    {"name": "20w10t", "weapons": 20, "targets": 10, "count": 20, "type": "rich"},
    {"name": "50w20t", "weapons": 50, "targets": 20, "count": 20, "type": "rich"},
]

# All categories combined
CATEGORIES = SQUARE_CATEGORIES + NONSQUARE_CATEGORIES

TARGET_VALUE_RANGE = (1, 100)     # integer threat/value scores
KILL_PROB_RANGE   = (0.01, 0.99)  # kill probabilities p_ij in (0,1)
WEAPON_QTY_RANGE  = (1, 3)        # w_i: each weapon TYPE has 1-3 weapons


def generate_instance(n_weapon_types: int, n_targets: int, rng: np.random.Generator) -> dict:
    """
    Generate one typed WTA instance.

    Parameters match the mathematical formulation (slide 7, Checkpoint 1):
        n_weapon_types : m  — number of distinct weapon types
        n_targets      : n  — number of targets
        weapon_quantities[i] : w_i — how many weapons of type i exist
        kill_prob[i][j]      : p_ij — P(one weapon of type i kills target j)

    kill_prob shape is (n_weapon_types x n_targets) — one row per TYPE,
    not per individual weapon. Use expand_instance() from wta_utils to
    unroll into individual weapon tokens for the algorithms.

    Returns a dict with:
        n_weapon_types   : int
        n_weapons        : int  (= n_weapon_types, kept for compatibility)
        n_targets        : int
        weapon_quantities: list[int]        w_i per type  (sum = total weapons)
        target_values    : list[float]      V_j per target
        kill_prob        : list[list[float]] p_ij shape (n_weapon_types x n_targets)
    """
    target_values = rng.integers(
        TARGET_VALUE_RANGE[0], TARGET_VALUE_RANGE[1] + 1,
        size=n_targets
    ).tolist()

    kill_prob = rng.uniform(
        KILL_PROB_RANGE[0], KILL_PROB_RANGE[1],
        size=(n_weapon_types, n_targets)
    ).tolist()

    # w_i: each weapon type has 1 to 3 individual weapons
    weapon_quantities = rng.integers(
        WEAPON_QTY_RANGE[0], WEAPON_QTY_RANGE[1] + 1,
        size=n_weapon_types
    ).tolist()

    return {
        "n_weapon_types":   n_weapon_types,
        "n_weapons":        n_weapon_types,   # kept for backward compat
        "n_targets":        n_targets,
        "weapon_quantities": weapon_quantities,
        "target_values":    target_values,
        "kill_prob":        kill_prob,
    }


def generate_all_instances(output_dir: str = "datasets") -> dict:
    """
    Generate all benchmark instances (square + non-square) and save them.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    metadata = {
        "seed": RNG_SEED,
        "description": (
            "Synthetic WTA benchmark instances. "
            "Kill probabilities drawn uniformly from [0.01, 0.99]. "
            "Target values drawn uniformly from integers in [1, 100]. "
            "Includes square (W=T), weapon-scarce (W<T), and weapon-rich (W>T) instances."
        ),
        "categories": [],
        "total_instances": 0,
    }

    for cat in CATEGORIES:
        cat_dir = os.path.join(output_dir, cat["name"])
        os.makedirs(cat_dir, exist_ok=True)
        instances = []

        for idx in range(cat["count"]):
            inst = generate_instance(cat["weapons"], cat["targets"], rng)
            fname = f"instance_{idx+1:03d}.json"
            fpath = os.path.join(cat_dir, fname)
            with open(fpath, "w") as f:
                json.dump(inst, f, separators=(",", ":"))
            instances.append(fname)

        ratio = cat["weapons"] / cat["targets"]
        scenario = (
            "square"  if abs(ratio - 1.0) < 1e-9 else
            "scarce"  if ratio < 1.0             else
            "rich"
        )

        metadata["categories"].append({
            "name":      cat["name"],
            "weapons":   cat["weapons"],
            "targets":   cat["targets"],
            "count":     cat["count"],
            "scenario":  scenario,
            "directory": cat_dir,
            "files":     instances,
        })
        metadata["total_instances"] += cat["count"]
        print(f"  [{scenario:6s}] {cat['name']:8s} W={cat['weapons']:3d} T={cat['targets']:3d}"
              f"  -> {cat['count']} instances")

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved -> {meta_path}")
    print(f"Total instances: {metadata['total_instances']}")
    return metadata


def load_instance(path: str) -> dict:
    """Load a single WTA instance from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    data["kill_prob"] = [list(row) for row in data["kill_prob"]]
    return data


if __name__ == "__main__":
    print("Generating WTA benchmark dataset ...")
    generate_all_instances()
