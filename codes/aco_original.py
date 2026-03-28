"""
Original ACO (Ant Colony Optimization) Algorithm for WTA
=========================================================
Faithfully implements the pseudocode from:
    Hasan & Barua, "Weapon-Target Assignment Problem", wta.pdf, pp. 1-29.

ACO Parameters (standard literature values):
    α (alpha)  = 1.0   — pheromone weight
    β (beta)   = 2.0   — heuristic weight
    ρ (rho)    = 0.1   — evaporation rate
    Q          = 100   — pheromone deposit constant
    noOfAnts   = max(noOfWeapons, noOfTargets)   [from pseudocode line 5-9]

Heuristic value η_ij:
    η_ij = V_j * p_ij   (expected value destroyed by assigning weapon i to target j)

Pheromone update (iteration-best):
    τ_ij ← (1 - ρ) * τ_ij          (evaporation for all)
    τ_ij ← τ_ij + Q / F_best       (deposit on iteration-best path)

Termination: time-budget in seconds (default 5 s for small, scales with size).
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional

from wta_utils import compute_solution_value


# ── ACO hyper-parameters ─────────────────────────────────────────────────────
ALPHA       = 1.0     # pheromone exponent
BETA        = 2.0     # heuristic exponent
RHO         = 0.1     # evaporation rate
Q           = 100.0   # pheromone deposit constant
TAU_INIT    = 1.0     # initial pheromone value
TAU_MIN     = 1e-6    # floor to avoid zero


def _calc_heuristic(
    target_values: List[float],
    kill_prob:     List[List[float]],
    n_weapons:     int,
    n_targets:     int,
) -> np.ndarray:
    """
    η_ij = V_j * p_ij
    Shape: (n_weapons, n_targets)
    """
    V  = np.array(target_values, dtype=float)
    P  = np.array(kill_prob,     dtype=float)       # shape (W, T)
    return P * V[np.newaxis, :]                      # broadcast


def _init_pheromone(n_weapons: int, n_targets: int) -> np.ndarray:
    return np.full((n_weapons, n_targets), TAU_INIT, dtype=float)


def _construct_solution(
    tau:          np.ndarray,
    eta:          np.ndarray,
    n_weapons:    int,
    n_targets:    int,
    rng:          np.random.Generator,
) -> List[int]:
    """
    Each weapon independently selects a target using the ACO transition
    probability rule:
        P_ij = [τ_ij]^α * [η_ij]^β  /  Σ_j [τ_ij]^α * [η_ij]^β
    """
    allocation = [-1] * n_weapons
    tau_a = np.power(np.maximum(tau, TAU_MIN), ALPHA)   # (W, T)
    eta_b = np.power(np.maximum(eta, TAU_MIN), BETA)    # (W, T)
    probs = tau_a * eta_b                               # (W, T)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    probs /= row_sums                                   # normalise per weapon

    for w in range(n_weapons):
        allocation[w] = int(rng.choice(n_targets, p=probs[w]))

    return allocation


def _update_pheromone(
    tau:           np.ndarray,
    best_alloc:    List[int],
    best_value:    float,
) -> np.ndarray:
    """
    Evaporate all; deposit on iteration-best path.
    line 27: UpdatePheromoneValues(iterationBestSolAlloc, bestSolValue)
    """
    tau = (1.0 - RHO) * tau                            # evaporation
    tau = np.maximum(tau, TAU_MIN)

    if best_value > 0:
        deposit = Q / best_value
        for w, t in enumerate(best_alloc):
            if t >= 0:
                tau[w, t] += deposit

    return tau


def aco_original(
    instance:         Dict[str, Any],
    time_budget_sec:  Optional[float] = None,
    seed:             int = 42,
) -> Dict[str, Any]:
    """
    Run the original ACO algorithm on a WTA instance.

    Parameters
    ----------
    instance        : WTA instance dict
    time_budget_sec : wall-clock budget (auto-scaled if None)
    seed            : RNG seed for reproducibility

    Returns
    -------
    dict with keys
        allocation    : list[int]
        value         : float
        time_sec      : float
        iterations    : int   — number of ACO iterations completed
    """
    n_weapons    = instance["n_weapons"]
    n_targets    = instance["n_targets"]
    target_values = instance["target_values"]
    kill_prob    = instance["kill_prob"]

    # Auto-scale time budget proportional to problem size (but cap for 100×100)
    if time_budget_sec is None:
        size = n_weapons * n_targets
        budget = min(2.0 + size / 1000.0, 30.0)
    else:
        budget = time_budget_sec
        
    # line 5-9: noOfAnts = max(noOfWeapons, noOfTargets)
    n_ants = max(n_weapons, n_targets)

    rng = np.random.default_rng(seed)

    # line 10-11: initialise heuristic and pheromone values
    eta = _calc_heuristic(target_values, kill_prob, n_weapons, n_targets)
    tau = _init_pheromone(n_weapons, n_targets)

    best_alloc = None
    best_value = float("inf")
    iterations = 0

    t0       = time.perf_counter()
    end_time = t0 + budget

    # line 12: while endTime < Now
    while time.perf_counter() < end_time:
        iter_best_alloc = None
        iter_best_value = float("inf")     # line 13: minSolutionValue

        # line 14-26: iterate over ants
        for _ in range(n_ants):            # line 15
            sol_alloc = _construct_solution(tau, eta, n_weapons, n_targets, rng)
            sol_value = compute_solution_value(
                sol_alloc, target_values, kill_prob, n_targets
            )

            if sol_value < iter_best_value:    # line 17
                iter_best_value = sol_value    # line 18
                iter_best_alloc = sol_alloc    # line 19

                if sol_value < best_value:     # line 20
                    best_value = sol_value     # line 21 (global best)
                    best_alloc = list(sol_alloc)

            # line 24: CalculateHeuristicValues() — static in this formulation,
            # so this is a no-op (η does not change between ants)

        # line 27: UpdatePheromoneValues
        if iter_best_alloc is not None:
            tau = _update_pheromone(tau, iter_best_alloc, iter_best_value)

        iterations += 1

    elapsed = time.perf_counter() - t0

    if best_alloc is None:
        # Fallback: at least one construction (shouldn't happen with budget ≥ 1s)
        best_alloc = _construct_solution(tau, eta, n_weapons, n_targets, rng)
        best_value = compute_solution_value(
            best_alloc, target_values, kill_prob, n_targets
        )

    return {
        "allocation": best_alloc,
        "value":      best_value,
        "time_sec":   elapsed,
        "iterations": iterations,
    }


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
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
    result = aco_original(inst, time_budget_sec=2.0)
    print("ACO Original — smoke test")
    print(f"  Allocation : {result['allocation']}")
    print(f"  Value      : {result['value']:.4f}")
    print(f"  Iterations : {result['iterations']}")
    print(f"  Time       : {result['time_sec']*1000:.1f} ms")
