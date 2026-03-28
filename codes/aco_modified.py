"""
Modified ACO Algorithm — ACO with Adaptive Pheromone Bounds & MMR Seeding
==========================================================================
Proposed improvement over the original ACO (Hasan & Barua, wta.pdf).

Motivation / Rationale
-----------------------
Two well-known weaknesses of the original ACO formulation are addressed:

1. **Premature Convergence**
   Without explicit bounds on pheromone levels, strong early paths can
   dominate τ so completely that ants stop exploring alternative solutions.
   This is especially severe on large WTA instances where one greedy path
   can accumulate disproportionate pheromone.

   Fix — Max-Min Ant System (MMAS) bounds:
       τ_max = Q / (ρ * f_best)          (upper bound derived from best found)
       τ_min = τ_max / (2 * n_targets)   (lower bound to force exploration)
   Pheromone is clipped to [τ_min, τ_max] after every update.

2. **Weak Initialisation**
   The original algorithm starts from a uniform pheromone matrix, so early
   iterations are essentially random walks.  This wastes the time budget.

   Fix — MMR-Seeded Initialisation:
       Run the greedy MMR pass once to obtain a good initial solution.
       Deposit Q/f_greedy pheromone on the MMR solution edges before the
       main ACO loop, giving ants a "warm start" without sacrificing
       diversity (τ_min floor still applies).

Additional parameter change:
   β = 3.0  (increased heuristic weight) — empirically improves convergence
   on WTA because η_ij already encodes problem structure well.

Reference for MMAS: Stützle & Hoos (2000), "MAX-MIN Ant System",
Future Generation Computer Systems 16(8), 889-914.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional

from wta_utils import compute_solution_value
from mmr_original import mmr_original


# ── Modified ACO hyper-parameters ─────────────────────────────────────────────
ALPHA    = 1.0    # pheromone exponent (unchanged)
BETA     = 3.0    # heuristic exponent (increased from 2.0)
RHO      = 0.1    # evaporation rate
Q        = 100.0  # pheromone deposit constant
TAU_INIT = 1.0    # initial pheromone (overridden by MMR-seed)
TAU_ABS_MIN = 1e-6


def _calc_heuristic(
    target_values: List[float],
    kill_prob:     List[List[float]],
    n_weapons:     int,
    n_targets:     int,
) -> np.ndarray:
    V = np.array(target_values, dtype=float)
    P = np.array(kill_prob,     dtype=float)
    return P * V[np.newaxis, :]


def _mmas_bounds(best_value: float, n_targets: int) -> tuple:
    """Compute MMAS τ_max and τ_min."""
    if best_value <= 0:
        return TAU_INIT * 10, TAU_ABS_MIN
    tau_max = Q / (RHO * best_value)
    tau_min = max(tau_max / (2.0 * n_targets), TAU_ABS_MIN)
    return tau_max, tau_min


def _init_pheromone_mmr_seed(
    instance:  Dict[str, Any],
    n_weapons: int,
    n_targets: int,
) -> tuple:
    """
    Warm-start: run MMR once, deposit pheromone on its solution, then clip.
    Returns (tau_matrix, mmr_value).
    """
    mmr_result  = mmr_original(instance)
    mmr_alloc   = mmr_result["allocation"]
    mmr_value   = mmr_result["value"]

    tau         = np.full((n_weapons, n_targets), TAU_INIT, dtype=float)
    if mmr_value > 0:
        deposit = Q / mmr_value
        for w, t in enumerate(mmr_alloc):
            if t >= 0:
                tau[w, t] += deposit

    return tau, mmr_alloc, mmr_value


def _construct_solution(
    tau:       np.ndarray,
    eta:       np.ndarray,
    n_weapons: int,
    n_targets: int,
    rng:       np.random.Generator,
) -> List[int]:
    allocation = [-1] * n_weapons
    tau_a = np.power(np.maximum(tau, TAU_ABS_MIN), ALPHA)
    eta_b = np.power(np.maximum(eta, TAU_ABS_MIN), BETA)
    probs = tau_a * eta_b
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    probs /= row_sums

    for w in range(n_weapons):
        allocation[w] = int(rng.choice(n_targets, p=probs[w]))
    return allocation


def _update_pheromone_mmas(
    tau:         np.ndarray,
    best_alloc:  List[int],
    best_value:  float,
    n_targets:   int,
) -> np.ndarray:
    """MMAS pheromone update with adaptive bounds."""
    tau_max, tau_min = _mmas_bounds(best_value, n_targets)

    # Evaporation
    tau = (1.0 - RHO) * tau

    # Deposit on iteration-best path
    if best_value > 0:
        deposit = Q / best_value
        for w, t in enumerate(best_alloc):
            if t >= 0:
                tau[w, t] += deposit

    # Clip to [τ_min, τ_max]
    tau = np.clip(tau, tau_min, tau_max)
    return tau


def aco_modified(
    instance:        Dict[str, Any],
    time_budget_sec: Optional[float] = None,
    seed:            int = 42,
) -> Dict[str, Any]:
    """
    Run the Modified ACO (ACO-MMAS+MMR) algorithm on a WTA instance.

    Returns
    -------
    dict with keys
        allocation      : list[int]
        value           : float
        time_sec        : float
        iterations      : int
        mmr_seed_value  : float  — value of MMR warm-start solution
        improvement_pct : float  — % improvement over MMR seed
    """
    n_weapons    = instance["n_weapons"]
    n_targets    = instance["n_targets"]
    target_values = instance["target_values"]
    kill_prob    = instance["kill_prob"]

    if time_budget_sec is None:
        size = n_weapons * n_targets
        time_budget_sec = min(2.0 + size / 1000.0, 30.0)

    n_ants = max(n_weapons, n_targets)
    rng    = np.random.default_rng(seed)

    eta = _calc_heuristic(target_values, kill_prob, n_weapons, n_targets)

    # Modified initialisation: MMR-seeded warm start
    tau, mmr_alloc, mmr_value = _init_pheromone_mmr_seed(instance, n_weapons, n_targets)

    best_alloc = list(mmr_alloc)
    best_value = mmr_value

    iterations = 0
    t0         = time.perf_counter()
    end_time   = t0 + time_budget_sec

    while time.perf_counter() < end_time:
        iter_best_alloc = None
        iter_best_value = float("inf")

        for _ in range(n_ants):
            sol_alloc = _construct_solution(tau, eta, n_weapons, n_targets, rng)
            sol_value = compute_solution_value(
                sol_alloc, target_values, kill_prob, n_targets
            )

            if sol_value < iter_best_value:
                iter_best_value = sol_value
                iter_best_alloc = sol_alloc

                if sol_value < best_value:
                    best_value = sol_value
                    best_alloc = list(sol_alloc)

        if iter_best_alloc is not None:
            # Modified update: MMAS with adaptive bounds
            tau = _update_pheromone_mmas(tau, iter_best_alloc, iter_best_value, n_targets)

        iterations += 1

    elapsed = time.perf_counter() - t0

    improvement_pct = (
        100.0 * (mmr_value - best_value) / mmr_value if mmr_value > 0 else 0.0
    )

    return {
        "allocation":      best_alloc,
        "value":           best_value,
        "time_sec":        elapsed,
        "iterations":      iterations,
        "mmr_seed_value":  mmr_value,
        "improvement_pct": improvement_pct,
    }


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from aco_original import aco_original

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
    orig = aco_original(inst, time_budget_sec=2.0)
    mod  = aco_modified(inst, time_budget_sec=2.0)

    print("ACO Original  value:", round(orig["value"], 4))
    print("ACO Modified  value:", round(mod["value"], 4))
    print(f"  MMR seed value    : {mod['mmr_seed_value']:.4f}")
    print(f"  Iterations        : {mod['iterations']}")
    print(f"  Improvement       : {mod['improvement_pct']:.2f}%")
