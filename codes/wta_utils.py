"""
Shared utilities for WTA algorithms.

Weapon-type model (from slide 7 of Checkpoint 1):
    W = {1,...,m}   : weapon TYPES  (not individual weapons)
    w_i             : quantity (count) of weapons of type i available
    p_ij            : kill probability of weapon TYPE i against target j
    x_ij in {0,1,2,...} : how many weapons of type i are assigned to target j

    Constraint : sum_j x_ij <= w_i   for each type i
    Objective  : minimize  sum_j  V_j * product_i (1-p_ij)^x_ij

expand_instance() converts the typed representation into a flat
"individual weapon" representation that all algorithms consume unchanged:
    - Type i with w_i weapons becomes w_i identical rows in kill_prob
    - x_ij then naturally means "row repeated w_i times"
    - The algorithms (MMR, ACO) assign each token to one target
"""

import numpy as np
from typing import List, Dict, Any


# ── Typed-model objective ─────────────────────────────────────────────────────

def compute_solution_value_typed(
    x:             List[List[int]],
    target_values: List[float],
    kill_prob:     List[List[float]],
    n_weapon_types: int,
    n_targets:     int,
) -> float:
    """
    Exact objective for the typed model.

    x[i][j] = number of weapons of type i assigned to target j.
    survival_j = product_i (1 - p_ij)^x_ij
    objective  = sum_j V_j * survival_j
    """
    survival = np.ones(n_targets)
    for i in range(n_weapon_types):
        for j in range(n_targets):
            if x[i][j] > 0:
                survival[j] *= (1.0 - kill_prob[i][j]) ** x[i][j]
    return float(np.dot(target_values, survival))


# ── Instance expansion: typed -> flat ────────────────────────────────────────

def expand_instance(inst: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand a typed WTA instance (with weapon_quantities) into a flat
    individual-weapon instance that all algorithms can consume directly.

    Type i with quantity w_i produces w_i identical rows in kill_prob.
    The flat instance has n_weapons = sum(weapon_quantities) rows.

    If weapon_quantities is absent (legacy instances), defaults to all-ones
    (i.e., one weapon per type — the binary special case).

    Parameters
    ----------
    inst : typed instance dict with keys:
        n_weapon_types   (or n_weapons for legacy)
        n_targets
        weapon_quantities  (optional; default = all-ones)
        target_values
        kill_prob          shape: (n_weapon_types x n_targets)

    Returns
    -------
    flat instance dict with keys:
        n_weapons       = sum(weapon_quantities)  total weapon tokens
        n_targets
        target_values
        kill_prob       shape: (n_weapons x n_targets)  rows repeated per qty
        weapon_quantities   kept for reference
        n_weapon_types      kept for reference
    """
    n_types     = inst.get("n_weapon_types", inst.get("n_weapons", len(inst["kill_prob"])))
    quantities  = inst.get("weapon_quantities", [1] * n_types)
    quantities  = [int(q) for q in quantities]

    expanded_kp = []
    for i, qty in enumerate(quantities):
        expanded_kp.extend([list(inst["kill_prob"][i])] * qty)

    return {
        "n_weapons":        sum(quantities),
        "n_targets":        inst["n_targets"],
        "target_values":    inst["target_values"],
        "kill_prob":        expanded_kp,
        # metadata
        "n_weapon_types":   n_types,
        "weapon_quantities": quantities,
    }


# ── Flat-model objective (used by all algorithms) ─────────────────────────────

def compute_solution_value(
    allocation:    List[int],
    target_values: List[float],
    kill_prob:     List[List[float]],
    n_targets:     int,
) -> float:
    """
    Expected survival value for the flat individual-weapon representation.

    allocation[i] = target index weapon-token i is assigned to (-1 = unused).
    survival_j    = product_{i: allocation[i]==j} (1 - p_ij)
    objective     = sum_j V_j * survival_j

    Works correctly for the typed model after expand_instance(), because
    multiple tokens of the same type assigned to the same target give:
        (1-p_ij) * (1-p_ij) * ... = (1-p_ij)^count
    which matches x_ij weapons of type i hitting target j.
    """
    survival = np.ones(n_targets)
    for weapon_idx, target_idx in enumerate(allocation):
        if target_idx < 0:
            continue
        survival[target_idx] *= (1.0 - kill_prob[weapon_idx][target_idx])
    return float(np.dot(target_values, survival))


def survival_prob_array(
    allocation: List[int],
    kill_prob:  List[List[float]],
    n_targets:  int,
) -> np.ndarray:
    """Return per-target survival probability array given a flat allocation."""
    survival = np.ones(n_targets)
    for w, t in enumerate(allocation):
        if t >= 0:
            survival[t] *= (1.0 - kill_prob[w][t])
    return survival
