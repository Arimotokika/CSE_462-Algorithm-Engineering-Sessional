"""
Modified MMR Algorithm — MMR with Iterative Re-evaluation (MMR-IR)
===================================================================
Proposed improvement over the original MMR (Hasan & Barua, wta.pdf).

Motivation / Rationale
----------------------
The original MMR is a pure greedy algorithm: once a weapon is assigned and
the target value is updated, no weapon assignment is ever reconsidered.  This
leads to suboptimal solutions when an early high-marginal-return assignment
prevents a better global configuration.

Proposed Modification: Iterative Re-evaluation Pass
----------------------------------------------------
After the standard greedy allocation pass (identical to original MMR), we
add a *local-search re-evaluation* phase:

    For each weapon w (in order of ascending marginal gain at assignment time):
        1. Tentatively reassign w to every other target t' ≠ current_target(w).
        2. Recompute the exact expected survival value with the swap.
        3. Accept the swap if it strictly improves the global objective.
        4. Repeat until no improving swap exists (convergence).

Why this helps:
    - Weapons assigned early in the greedy pass may have been assigned to
      targets that later received multiple weapons, diluting their marginal
      impact.  Re-evaluation allows these weapons to migrate to targets that
      benefit more from extra coverage.
    - Complexity per re-evaluation pass: O(W × T) exact-objective evaluations.
    - Total complexity (with K passes until convergence): O(K × W × T) —
      typically K ≤ 5 in practice, so the overhead is modest.

Additional Modification: Tie-breaking by Target Priority
---------------------------------------------------------
Original MMR breaks ties arbitrarily (first found).  Modified MMR breaks ties
in favour of the highest-value target, concentrating early fire on the most
threatening targets.
"""

import time
import numpy as np
from typing import List, Dict, Any

from wta_utils import compute_solution_value


def _greedy_pass(
    n_weapons:     int,
    n_targets:     int,
    target_values: List[float],
    kill_prob:     List[List[float]],
) -> List[int]:
    """Original MMR greedy pass — returns allocation list."""
    current_values      = list(map(float, target_values))
    unallocated_weapons = list(range(n_weapons))
    allocation          = [-1] * n_weapons
    allocated_count     = 0

    while allocated_count < n_weapons:
        max_decrease = float("-inf")
        best_weapon  = -1
        best_target  = -1

        for k in unallocated_weapons:
            for i in range(n_targets):
                decrease = current_values[i] * kill_prob[k][i]
                # Modification: break ties by target value (higher = preferred)
                if (decrease > max_decrease) or (
                    abs(decrease - max_decrease) < 1e-12
                    and target_values[i] > target_values[best_target] if best_target >= 0 else False
                ):
                    max_decrease = decrease
                    best_target  = i
                    best_weapon  = k

        unallocated_weapons.remove(best_weapon)
        allocation[best_weapon]      = best_target
        current_values[best_target] -= max_decrease
        allocated_count             += 1

    return allocation


def _local_search(
    allocation:    List[int],
    target_values: List[float],
    kill_prob:     List[List[float]],
    n_targets:     int,
    max_passes:    int = 20,
) -> tuple:
    """
    Iterative re-evaluation local-search phase.

    Returns (improved_allocation, improved_value, passes_done).
    """
    best_alloc = list(allocation)
    best_val   = compute_solution_value(best_alloc, target_values, kill_prob, n_targets)
    n_weapons  = len(allocation)

    for pass_no in range(max_passes):
        improved = False
        for w in range(n_weapons):
            current_target = best_alloc[w]
            for t in range(n_targets):
                if t == current_target:
                    continue
                # Tentative swap
                trial = list(best_alloc)
                trial[w] = t
                trial_val = compute_solution_value(trial, target_values, kill_prob, n_targets)
                if trial_val < best_val:            # lower survival = better for attacker
                    best_val   = trial_val
                    best_alloc = trial
                    improved   = True
        if not improved:
            return best_alloc, best_val, pass_no + 1

    return best_alloc, best_val, max_passes


def mmr_modified(instance: Dict[str, Any], max_ls_passes: int = 20) -> Dict[str, Any]:
    """
    Run the Modified MMR (MMR-IR) algorithm on a WTA instance.

    Parameters
    ----------
    instance     : dict — same format as mmr_original
    max_ls_passes: int  — maximum local-search passes (default 20)

    Returns
    -------
    dict with keys
        allocation      : list[int]
        value           : float
        time_sec        : float
        greedy_value    : float  — value after greedy phase (before LS)
        ls_passes       : int    — number of local-search passes executed
        improvement_pct : float  — % improvement from local search
    """
    n_weapons    = instance["n_weapons"]
    n_targets    = instance["n_targets"]
    target_values = list(instance["target_values"])
    kill_prob    = instance["kill_prob"]

    t0 = time.perf_counter()

    # Phase 1: greedy pass (same as original MMR, with tie-break enhancement)
    greedy_alloc = _greedy_pass(n_weapons, n_targets, target_values, kill_prob)
    greedy_value = compute_solution_value(greedy_alloc, target_values, kill_prob, n_targets)

    # Phase 2: iterative local-search re-evaluation
    final_alloc, final_value, ls_passes = _local_search(
        greedy_alloc, target_values, kill_prob, n_targets, max_ls_passes
    )

    elapsed = time.perf_counter() - t0

    improvement_pct = (
        100.0 * (greedy_value - final_value) / greedy_value
        if greedy_value > 0 else 0.0
    )

    return {
        "allocation":      final_alloc,
        "value":           final_value,
        "time_sec":        elapsed,
        "greedy_value":    greedy_value,
        "ls_passes":       ls_passes,
        "improvement_pct": improvement_pct,
    }


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from mmr_original import mmr_original

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
    orig = mmr_original(inst)
    mod  = mmr_modified(inst)

    print("MMR Original  value:", round(orig["value"], 4))
    print("MMR Modified  value:", round(mod["value"], 4))
    print(f"  Greedy value      : {mod['greedy_value']:.4f}")
    print(f"  LS passes         : {mod['ls_passes']}")
    print(f"  Improvement       : {mod['improvement_pct']:.2f}%")
