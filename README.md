# CSE 462 Algorithm Engineering Sessional

## Weapon-Target Assignment (WTA) Problem — Group 5 — Checkpoint 2

---

## Problem Overview

The **Weapon-Target Assignment (WTA)** problem is an NP-complete combinatorial
optimization problem: given W weapons and T enemy targets, assign weapons to targets
to **minimize the total expected survival value** of all targets.

**Objective (minimize):**

```
Z = sum_j  V_j * prod_i (1 - p_ij)^x_ij
```

- `V_j` = military value of target j
- `p_ij` = kill probability of weapon i against target j
- `x_ij` = weapons i assigned to target j (integer >= 0)

---

## Algorithms

### MMR — Maximum Marginal Return (Original)

Greedy heuristic from Hasan & Barua. At each step assigns the (weapon, target)
pair with greatest marginal decrease in the objective. O(W²T), deterministic.

### MMR-IR — Iterative Re-evaluation (Our Modification)

Three independently-designed enhancements:

1. **Tie-breaking by target value** — prefer highest-value target on greedy ties
2. **1-opt incremental local search** — after greedy, try reassigning each weapon
   to every other target; accept if objective improves; repeat until convergence
3. **2-opt swap search** — try swapping targets of two weapons to escape 1-opt optima

### GA — Genetic Algorithm (Original)

Standard GA faithfully implementing Hasan & Barua:

- Encoding: `chromosome[i]` = target assigned to weapon i
- Population: max(W, T), Tournament selection (k=3)
- Uniform crossover (rate=0.8), random mutation (rate=1/W)
- Time-based termination

### Hybrid GA — Our Modification

Three novel modifications:

1. **MMR-Seeded Initialization** — seed one individual with MMR greedy solution
2. **Elitism + Stagnation Detection** — preserve top 10%; re-diversify after 5 stagnant generations
3. **Threat-Proportional Mutation** _(WTA-specific, novel)_ — bias mutation toward
   high-value poorly-covered targets using `remaining_threat[j] = V_j * survival_j`
   as selection weights

---

## Dataset

180 synthetic instances across 9 categories (3 tiers × 3 scenarios).

| Tier   | Scenario     | Weapons (W) | Targets (T) | Instances |
| ------ | ------------ | ----------- | ----------- | --------- |
| small  | balanced     | 20          | 20          | 20        |
| small  | scarce (W<T) | 10          | 20          | 20        |
| small  | rich (W>T)   | 30          | 20          | 20        |
| medium | balanced     | 100         | 100         | 20        |
| medium | scarce       | 50          | 100         | 20        |
| medium | rich         | 150         | 100         | 20        |
| large  | balanced     | 250         | 250         | 20        |
| large  | scarce       | 125         | 250         | 20        |
| large  | rich         | 375         | 250         | 20        |

Parameters: `V_j ~ U(10,100)` integers, `p_ij ~ U(0.05, 0.90)`, RNG seed=42.

---

## Repository Structure

```
CSE_462-Algorithm-Engineering-Sessional/
├── README.md
├── codes/
│   ├── run_all.py              # Master script — full pipeline
│   ├── dataset_generator.py    # Generates 180 benchmark instances
│   ├── experiment_runner.py    # Runs all 4 algorithms, saves CSV
│   ├── analysis.py             # Generates 8 plots + stats
│   ├── mmr_original.py         # MMR (Hasan & Barua)
│   ├── mmr_modified.py         # MMR-IR (our modification)
│   ├── ga_original.py          # GA (Hasan & Barua)
│   ├── ga_modified.py          # Hybrid GA (our modification)
│   └── wta_utils.py            # Shared utilities
├── datasets/                   # Auto-generated JSON instances
├── results/
│   ├── experiment_results.csv  # 720-row results (180 instances x 4 algorithms)
│   └── figures/                # 8 PNG plots
├── checkpoint-1/               # Checkpoint 1 materials
├── checkpoint-2/               # Checkpoint 2 guidelines & deep research
└── files/                      # Reference papers
```

---

## Setup

```bash
# From WTA root directory (one-time setup)
cd "d:/Term Files/l4-t2/CSE 462/WTA"
py -3 -m venv .venv

# Activate (Git Bash / MSYS2)
source .venv/Scripts/activate

# Activate (Windows CMD/PowerShell)
.venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy
```

---

## Running

### Full pipeline (single command)

```bash
cd codes
python run_all.py
```

Estimated runtime: ~3.5 hours (large tier dominates).

---

### Run tiers separately (recommended — monitor progress)

```bash
cd codes

# Step 1: Generate dataset (~5 seconds)
python dataset_generator.py

# Step 2a: Small tier — 60 instances, ~10 min
python experiment_runner.py --categories small_balanced,small_scarce,small_rich

# Step 2b: Medium tier — 60 instances, ~50 min
python experiment_runner.py --categories medium_balanced,medium_scarce,medium_rich --append

# Step 2c: Large tier — 60 instances, ~2 hours
python experiment_runner.py --categories large_balanced,large_scarce,large_rich --append

# Step 3: Generate all 8 plots + statistical tests
python analysis.py
```

---

### Quick smoke test (5 instances)

```bash
cd codes
python experiment_runner.py --categories small_balanced --instances 1-5
python analysis.py
```

---

### Per-algorithm smoke tests

```bash
cd codes
python mmr_original.py    # 3x3 example
python mmr_modified.py    # MMR-O vs MMR-IR comparison
python ga_original.py     # 3x3 example
python ga_modified.py     # all 4 algorithms on 3x3
```

---

## Output Plots (`results/figures/`)

| File                              | Description                                         |
| --------------------------------- | --------------------------------------------------- |
| `01_box_solution_quality.png`     | Box plots per tier x algorithm (balanced instances) |
| `02_bar_mean_objective.png`       | Mean objective value grouped by tier                |
| `03_bar_improvement.png`          | % improvement: modified vs original per tier        |
| `04_line_time_scalability.png`    | Runtime scalability (log-log scale)                 |
| `05_bar_gap.png`                  | Optimality gap from best-known solution             |
| `06_scenario_comparison.png`      | Performance: balanced vs scarce vs rich             |
| `07_violin_value_dist.png`        | Value distribution: original vs modified            |
| `08_bar_scenario_improvement.png` | Improvement % by scenario type                      |

Statistical significance: Wilcoxon signed-rank test (alpha=0.05) printed to console.

---

## References

1. Hasan, M. & Barua, A. — _Weapon-Target Assignment Problem_ (IntechOpen)
2. Ahuja, R. K. et al. (2007) — _Exact and Heuristic Algorithms for the WTA_
3. Lee, Z. J. et al. (2003) — _Efficiently Solving General WTA Problems_
4. Manne, A. S. (1958) — _A Target-Assignment Problem_ (Operations Research)
