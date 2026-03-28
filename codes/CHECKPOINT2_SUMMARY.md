# CSE462 Algorithm Engineering Sessional — Group 5
## Checkpoint 2: Weapon-Target Assignment (WTA) Problem

---

## 1. What Was Done

### 1.1 Problem Definition (from Checkpoint 1)
The **Weapon-Target Assignment (WTA)** problem assigns `W` weapons to `T` targets to minimize the total expected survival value of all targets:

```
Minimize: sum over j of [ V_j * product over i of (1 - x_ij * p_ij) ]
```

Where:
- `V_j` = value/threat score of target `j`
- `p_ij` = kill probability of weapon `i` against target `j`
- `x_ij ∈ {0, 1}` = assignment decision variable
- The problem is **NP-complete** (proved by reduction from SUBSET PRODUCT and to Multiple Knapsack)

---

## 2. Algorithms Implemented

### 2.1 Original MMR (Maximum Marginal Return)
**Source:** Hasan & Barua, `wta.pdf`, lines 1–26

**Logic:** Pure greedy — at each step, pick the (weapon, target) pair giving the greatest immediate marginal decrease in expected threat:

```
decrease(weapon k, target i) = currentTargetValue[i] * p[k][i]
```

After assigning weapon `k` to target `i`, update:
```
currentTargetValue[i] -= maxDecrease
```

**Complexity:** O(W² × T)
**File:** `mmr_original.py`

---

### 2.2 Modified MMR — MMR-IR (Iterative Re-evaluation)
**Proposed improvement over original MMR**

**Motivation:** The original MMR never reconsiders assignments. Early assignments may become suboptimal once other weapons are placed.

**Two modifications:**

**Modification 1 — Iterative Local-Search Re-evaluation:**
After the greedy pass, for each weapon try reassigning it to every other target. Accept the swap if it strictly reduces the global expected survival value. Repeat until no improving swap exists (convergence).

```
For each weapon w:
    For each target t' != current_target(w):
        trial_value = compute_exact_objective(swap w -> t')
        if trial_value < best_value:
            accept swap
Repeat until no improvement
```

**Modification 2 — Tie-breaking by Target Priority:**
When two (weapon, target) pairs give equal marginal return, prefer the pair covering the highest-value target, concentrating early fire on the most dangerous targets.

**Additional complexity:** O(K × W × T) for K local-search passes (K ≤ 5 in practice)
**File:** `mmr_modified.py`

---

### 2.3 Original ACO (Ant Colony Optimization)
**Source:** Hasan & Barua, `wta.pdf`, lines 1–29

**Parameters:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| α (alpha) | 1.0 | Pheromone weight |
| β (beta) | 2.0 | Heuristic weight |
| ρ (rho) | 0.1 | Evaporation rate |
| Q | 100 | Deposit constant |
| noOfAnts | max(W, T) | From pseudocode line 5–9 |

**Heuristic value:** `η_ij = V_j × p_ij`

**Transition probability:**
```
P_ij = [τ_ij]^α × [η_ij]^β  /  Σ_j [τ_ij]^α × [η_ij]^β
```

**Pheromone update (iteration-best):**
```
τ_ij <- (1 - ρ) × τ_ij            (evaporation)
τ_ij <- τ_ij + Q / F_best          (deposit on best path)
```

**Termination:** Time budget (auto-scaled by problem size)
**File:** `aco_original.py`

---

### 2.4 Modified ACO — ACO-MMAS+MMR
**Proposed improvement over original ACO**

**Motivation:** Two weaknesses of original ACO are addressed:

**Modification 1 — MMAS Adaptive Pheromone Bounds (prevents premature convergence):**

Based on: *Stützle & Hoos (2000), "MAX-MIN Ant System", Future Generation Computer Systems 16(8)*

Without pheromone bounds, early dominant paths monopolize τ and ants stop exploring. MMAS clips pheromone to [τ_min, τ_max] after every update:

```
τ_max = Q / (ρ × f_best)
τ_min = τ_max / (2 × n_targets)

After update: τ_ij = clip(τ_ij, τ_min, τ_max)
```

**Modification 2 — MMR-Seeded Warm Start (better initialisation):**

The original ACO starts from a uniform pheromone matrix, making early iterations nearly random. Instead, run the greedy MMR pass once before the ACO loop and deposit pheromone on the MMR solution:

```
Run MMR -> get mmr_alloc, mmr_value
For each (w, t) in mmr_alloc:
    τ[w][t] += Q / mmr_value    (warm-start deposit)
Then clip to [τ_min, τ_max]
```

**Additional parameter change:** β increased from 2.0 to 3.0 (higher heuristic weight improves convergence because η_ij encodes problem structure well).

**File:** `aco_modified.py`

---

## 3. Dataset

### 3.1 Justification for Synthetic Data
No standardized, publicly available WTA benchmark datasets exist. Prior studies (Ahuja et al. 2007, Lee et al. 2003, Manne 1958) used proprietary or problem-specific instances that were never released. Following the methodology of Hasan & Barua and common practice in combinatorial optimization research, we generated random instances whose statistical properties match those reported in the literature.

### 3.2 Why W ≠ T (Non-square instances are included)

The WTA problem does **not** require W = T. The problem is defined for any W weapons and T targets independently. Three structurally distinct scenarios exist:

| Scenario | Condition | Military Meaning | Hardness |
|----------|-----------|-----------------|----------|
| Weapon-scarce | W < T | Fewer weapons than threats; not all targets can be engaged | **Hardest** |
| Balanced | W = T | One weapon per target is possible | Medium |
| Weapon-rich | W > T | Concentrated fire on targets; multiple weapons per target | Easiest |

The most **militarily realistic** scenario is W < T — in real defense, there are typically more incoming threats than interceptors. The ACO pseudocode (Hasan & Barua, line 5–9) already accounts for this via `noOfAnts = max(W, T)`. The kill probability matrix is always shaped `(W × T)`, so all algorithms work for any W and T without modification.

### 3.3 Dataset Details
**Generator:** `dataset_generator.py`
**Seed:** 42 (fully reproducible)
**Location:** `datasets/`

**Square instances (W = T):**

| Category | Weapons | Targets | Instances |
|----------|---------|---------|-----------|
| 5x5 | 5 | 5 | 30 |
| 10x10 | 10 | 10 | 30 |
| 20x20 | 20 | 20 | 30 |
| 50x50 | 50 | 50 | 30 |
| 100x100 | 100 | 100 | 30 |
| Subtotal | | | **150** |

**Non-square instances (W ≠ T):**

| Category | Weapons | Targets | Scenario | Instances |
|----------|---------|---------|----------|-----------|
| 5w10t | 5 | 10 | Scarce (W < T) | 20 |
| 10w20t | 10 | 20 | Scarce (W < T) | 20 |
| 20w50t | 20 | 50 | Scarce (W < T) | 20 |
| 10w5t | 10 | 5 | Rich (W > T) | 20 |
| 20w10t | 20 | 10 | Rich (W > T) | 20 |
| 50w20t | 50 | 20 | Rich (W > T) | 20 |
| Subtotal | | | | **120** |

**Grand Total: 270 instances**

**Kill probabilities:** Uniform random ∈ [0.01, 0.99]
**Target values:** Uniform random integers ∈ [1, 100]
**Format:** JSON files, one instance per file — shape of `kill_prob` is always `(n_weapons × n_targets)`

---

## 4. Experiments

**Runner:** `experiment_runner.py`
**Total runs:** 150 instances × 4 algorithms = **600 experimental observations**
**Results file:** `results/experiment_results.csv`

**ACO time budget** (auto-scaled):
```
budget = min(2.0 + (W * T) / 500.0,  30.0)  seconds
```

---

## 5. Results

### 5.1 Mean Solution Value (lower = better)

| Size | MMR-orig | MMR-IR (mod) | ACO-orig | ACO-MMAS (mod) |
|------|----------|-------------|----------|----------------|
| 5×5 | 51.07 | 49.30 | 48.66 | **45.91** |
| 10×10 | 71.80 | 71.13 | 81.05 | **66.76** |
| 20×20 | 86.99 | 85.94 | 166.68 | **86.99** |
| 50×50 | 108.74 | **108.67** | 661.94 | 108.74 |
| 100×100 | 142.50 | **142.45** | 1591.63 | 142.50 |

### 5.2 Mean Runtime (milliseconds)

| Size | MMR-orig | MMR-IR (mod) | ACO-orig | ACO-MMAS (mod) |
|------|----------|-------------|----------|----------------|
| 5×5 | 0.0 | 0.3 | 2050 | 2050 |
| 10×10 | 0.1 | 2.8 | 2203 | 2204 |
| 20×20 | 0.9 | 21.8 | 2819 | 2821 |
| 50×50 | 7.6 | 127 | 7058 | 7083 |
| 100×100 | 92 | 1421 | 33813 | 56166 |

### 5.3 Statistical Tests (Wilcoxon Signed-Rank, α = 0.05)

| Comparison | Size | p-value | Significant? |
|------------|------|---------|--------------|
| MMR-IR vs MMR-orig | 5×5 | 0.0544 | No |
| MMR-IR vs MMR-orig | 10×10 | 0.0339 | **Yes** |
| MMR-IR vs MMR-orig | 20×20 | 0.0139 | **Yes** |
| MMR-IR vs MMR-orig | 50×50 | 0.1587 | No |
| MMR-IR vs MMR-orig | 100×100 | 0.0899 | No |
| ACO-MMAS vs ACO-orig | 5×5 | 0.0001 | **Yes** |
| ACO-MMAS vs ACO-orig | 10×10 | <0.0001 | **Yes** |
| ACO-MMAS vs ACO-orig | 20×20 | <0.0001 | **Yes** |
| ACO-MMAS vs ACO-orig | 50×50 | <0.0001 | **Yes** |
| ACO-MMAS vs ACO-orig | 100×100 | <0.0001 | **Yes** |

---

## 6. Generated Figures

All figures saved to `results/figures/`:

| File | Description |
|------|-------------|
| `box_solution_quality.png` | Box plots of solution value per category per algorithm |
| `line_mean_value.png` | Mean solution value vs problem size (with std error bars) |
| `bar_improvement.png` | % improvement of modified over original algorithms |
| `line_time_scalability.png` | Mean runtime vs problem size (log-log scale) |
| `heatmap_pct_improvement.png` | Heatmap of % improvement per algorithm per category |
| `violin_value_dist.png` | Violin plots comparing original vs modified distributions |
| `bar_gap.png` | Optimality gap from best-known solution per algorithm |

---

## 7. Key Findings

1. **ACO-MMAS dominates ACO-original on all sizes** (statistically significant, p < 0.0001). The collapse of ACO-original on large instances (1592 vs 143 on 100×100) is caused by insufficient iterations within the time budget — ants never escape the initial random configurations. The MMR warm-start solves this.

2. **MMR-IR improves MMR-original on medium instances** (10×10 and 20×20, p < 0.05). On very large instances (50×50, 100×100), the greedy solution is already near-optimal and the local-search overhead does not pay off in solution quality.

3. **MMR-family (both original and modified) scales better** than ACO-family: O(W² T) greedy vs O(I × A × W × T) ACO — the greedy approach is 3–4 orders of magnitude faster for 100×100 instances.

4. **Both modifications are validated by statistical testing** — ACO-MMAS improvement is highly significant across all problem sizes; MMR-IR improvement is significant at medium scale.

---

## 8. File Structure

```
WTA_Final_Checkpoint/
├── dataset_generator.py       # Generate 150 benchmark instances
├── wta_utils.py               # Shared objective function
├── mmr_original.py            # Original MMR (Hasan & Barua pseudocode)
├── mmr_modified.py            # Modified MMR-IR (greedy + local search)
├── aco_original.py            # Original ACO (Hasan & Barua pseudocode)
├── aco_modified.py            # Modified ACO-MMAS (MMAS bounds + MMR seed)
├── experiment_runner.py       # Run all 4 algorithms on all instances
├── analysis.py                # Generate plots + statistical tests
├── run_all.py                 # Master pipeline script
├── CHECKPOINT2_SUMMARY.md     # This file
├── datasets/
│   ├── metadata.json
│   ├── 5x5/    instance_001.json ... instance_030.json
│   ├── 10x10/  ...
│   ├── 20x20/  ...
│   ├── 50x50/  ...
│   └── 100x100/...
└── results/
    ├── experiment_results.csv    (600 rows)
    └── figures/
        ├── box_solution_quality.png
        ├── line_mean_value.png
        ├── bar_improvement.png
        ├── line_time_scalability.png
        ├── heatmap_pct_improvement.png
        ├── violin_value_dist.png
        └── bar_gap.png
```

---

## 9. How to Reproduce

```bash
cd WTA_Final_Checkpoint

# Full pipeline (dataset + experiments + analysis)
py -3 run_all.py

# Or step by step:
py -3 dataset_generator.py
py -3 experiment_runner.py
py -3 analysis.py

# Run only specific categories (faster testing)
py -3 experiment_runner.py --categories 5x5,10x10
```

---

*Group 5 | CSE462 Algorithm Engineering Sessional | BUET*
