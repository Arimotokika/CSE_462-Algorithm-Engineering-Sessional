"""
Original MMR (Maximum Marginal Return) Algorithm
=================================================
Faithfully implements the pseudocode from:
    Hasan & Barua, "Weapon-Target Assignment Problem", wta.pdf, pp. 1-26.

Algorithm logic:
    At each step, pick the (unallocated weapon, target) pair that gives the
    greatest *instantaneous* marginal decrease in expected threat value,
    defined as:
        decrease(weapon k, target i) = currentTargetValue[i] * p_ij[i][k]

    After assigning weapon k to target i:
        currentTargetValue[i] -= maxDecrease   (greedy update)

Complexity: O(W * W * T) ≈ O(W² T)  — outer loop W times, inner loop W×T.
(The pseudocode iterates over unallocatedWeapons in the inner scan; the
first iteration scans all W weapons × T targets.)
"""

import time
import numpy as np
from typing import List, Dict, Any

from wta_utils import compute_solution_value


def mmr_original(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the original MMR algorithm on a WTA instance.

    Parameters
    ----------
    instance : dict with keys
        n_weapons, n_targets, target_values, kill_prob

    Returns
    -------
    dict with keys
        allocation   : list[int]  weapon i → target allocation[i]
        value        : float      final expected survival value
        time_sec     : float      wall-clock time
        iterations   : int        number of weapon allocations made
    """
    n_weapons    = instance["n_weapons"]
    n_targets    = instance["n_targets"]
    target_values = list(instance["target_values"])          # mutable copy
    kill_prob    = instance["kill_prob"]                     # p[weapon][target]

    # --- initialise ---
    allocation          = [-1] * n_weapons                   # solution.Allocations
    current_values      = list(map(float, target_values))    # working target values
    unallocated_weapons = list(range(n_weapons))             # 0-indexed weapon list
    allocated_count     = 0

    t0 = time.perf_counter()

    # line 4: while allocatedWeaponCount <= noOfWeapons
    while allocated_count < n_weapons:
        max_decrease    = float("-inf")                      # line 5
        best_weapon     = -1
        best_target     = -1

        # line 7: while k < unallocatedWeapons.Count
        for k in unallocated_weapons:                        # weapon index
            # line 8-17: scan all targets
            for i in range(n_targets):
                # line 10: decrease = targetValues[i] * killProbabilities[i][k]
                decrease = current_values[i] * kill_prob[k][i]
                if decrease > max_decrease:                  # line 11
                    max_decrease = decrease                  # line 12
                    best_target  = i                         # line 13 (allocatedTarget)
                    best_weapon  = k                         # line 14 (allocatedWeapon — typo fix)

        # line 20-23
        unallocated_weapons.remove(best_weapon)
        allocation[best_weapon]         = best_target        # line 21
        current_values[best_target]    -= max_decrease       # line 22
        allocated_count                += 1                  # line 23

    elapsed = time.perf_counter() - t0

    # line 25: recalculate exact objective
    value = compute_solution_value(allocation, target_values, kill_prob, n_targets)

    return {
        "allocation": allocation,
        "value":      value,
        "time_sec":   elapsed,
        "iterations": n_weapons,
    }


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 3-weapon / 3-target example from the naval-defense scenario in Checkpoint 1
    inst = {
        "n_weapons":     3,
        "n_targets":     3,
        "target_values": [10, 20, 30],
        "kill_prob": [
            [0.3, 0.2, 0.1],
            [0.1, 0.4, 0.2],
            [0.2, 0.3, 0.5],
        ],
    }
    result = mmr_original(inst)
    print("MMR Original — smoke test")
    print(f"  Allocation : {result['allocation']}")
    print(f"  Value      : {result['value']:.4f}")
    print(f"  Time       : {result['time_sec']*1000:.3f} ms")
