"""
Analysis & Visualisation
=========================
Reads experiment_results.csv and produces all publication-quality plots
for the Checkpoint 2 presentation.

Tested with: seaborn 0.13.2, matplotlib 3.10.8, pandas 3.0.1

Plots generated (saved to results/figures/):
  Square instances (W = T):
    01_box_solution_quality.png     -- box plots per category x algorithm
    02_line_mean_value.png          -- mean solution value vs problem size
    03_bar_improvement.png          -- % improvement modified over original
    04_line_time_scalability.png    -- mean runtime vs problem size
    05_heatmap_pct_improvement.png  -- improvement heatmap
    06_violin_value_dist.png        -- violin plots (seaborn 0.13 compatible)
    07_bar_gap.png                  -- optimality gap

  Non-square instances (W != T):
    08_scenario_comparison.png      -- scarce vs balanced vs rich per algorithm
    09_bar_scenario_improvement.png -- % improvement per scenario type
    10_scarce_vs_rich.png           -- side-by-side: scarce vs rich
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

RESULTS_CSV = "results/experiment_results.csv"
FIGURES_DIR = "results/figures"
DPI         = 150

# ── category definitions ───────────────────────────────────────────────────────
SQUARE_CATS = ["5x5", "10x10", "20x20", "50x50", "100x100"]
SCARCE_CATS = ["5w10t", "10w20t", "20w50t"]   # W < T
RICH_CATS   = ["10w5t", "20w10t", "50w20t"]   # W > T
ALL_CATS    = SQUARE_CATS + SCARCE_CATS + RICH_CATS

# max(W, T) used as problem-complexity proxy for x-axes
CAT_SIZES = {
    "5x5": 5,   "10x10": 10,  "20x20": 20,  "50x50": 50,  "100x100": 100,
    "5w10t": 10, "10w20t": 20, "20w50t": 50,
    "10w5t": 10, "20w10t": 20, "50w20t": 50,
}

ALGO_ORDER  = ["MMR_Original", "MMR_Modified", "ACO_Original", "ACO_Modified"]
ALGO_LABELS = {
    "MMR_Original": "MMR (orig)",
    "MMR_Modified": "MMR-IR (mod)",
    "ACO_Original": "ACO (orig)",
    "ACO_Modified": "ACO-MMAS (mod)",
}
ALGO_COLORS = {
    "MMR_Original": "#4878CF",
    "MMR_Modified": "#1F3A6E",
    "ACO_Original": "#D65F5F",
    "ACO_Modified": "#8B0000",
}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": DPI, "savefig.bbox": "tight"})


# ── helpers ────────────────────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    present = [c for c in ALL_CATS if c in df["category"].unique()]
    df["category"] = pd.Categorical(df["category"], categories=present, ordered=True)
    df["label"]    = df["algorithm"].map(ALGO_LABELS)

    def _scenario(cat: str) -> str:
        if cat in SQUARE_CATS: return "square"
        if cat in SCARCE_CATS: return "scarce"
        return "rich"

    df["scenario"] = df["category"].astype(str).map(_scenario)
    return df


def best_known(df: pd.DataFrame) -> pd.DataFrame:
    bk = (
        df.groupby(["category", "instance_id"])["value"]
        .min().reset_index()
        .rename(columns={"value": "best_known"})
    )
    merged = df.merge(bk, on=["category", "instance_id"])
    # avoid div-by-zero: cap gap where best_known == 0
    merged["best_known"] = merged["best_known"].replace(0, np.nan)
    return merged


def _safe_pct_improvement(orig: pd.Series, mod: pd.Series) -> pd.Series:
    """% improvement = 100*(orig - mod)/orig; returns NaN where orig <= 0."""
    orig = orig.replace(0, np.nan)
    return 100 * (orig - mod) / orig


def save_fig(name: str):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"  saved -> {path}")


def _cat_to_str(series: pd.Series) -> pd.Series:
    """Cast Categorical or any series to str before using .map()."""
    return series.astype(str)


# ── Square instance plots ──────────────────────────────────────────────────────

def plot_box_solution_quality(df: pd.DataFrame):
    sq      = df[df["category"].isin(SQUARE_CATS)]
    present = [c for c in SQUARE_CATS if c in sq["category"].astype(str).unique()]

    fig, axes = plt.subplots(1, len(present), figsize=(18, 5), sharey=False)
    if len(present) == 1:
        axes = [axes]

    for ax, cat in zip(axes, present):
        sub    = sq[sq["category"].astype(str) == cat]
        labels = [ALGO_LABELS[a] for a in ALGO_ORDER if a in sub["algorithm"].values]
        pal    = {ALGO_LABELS[a]: ALGO_COLORS[a] for a in ALGO_ORDER}
        sns.boxplot(data=sub, x="label", y="value",
                    order=labels, palette=pal, ax=ax,
                    linewidth=1.2, fliersize=3)
        ax.set_title(cat, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Expected Survival Value" if cat == present[0] else "")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Solution Quality Distribution — Square Instances (W = T)",
                 fontsize=14, y=1.02)
    save_fig("01_box_solution_quality.png")


def plot_mean_value(df: pd.DataFrame):
    sq      = df[df["category"].isin(SQUARE_CATS)].copy()
    sq["cat_str"] = _cat_to_str(sq["category"])
    summary = sq.groupby(["cat_str", "algorithm"])["value"].agg(["mean", "std"]).reset_index()
    summary["size"] = summary["cat_str"].map(CAT_SIZES)

    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGO_ORDER:
        sub = summary[summary["algorithm"] == algo].sort_values("size")
        ax.errorbar(sub["size"], sub["mean"], yerr=sub["std"],
                    label=ALGO_LABELS[algo], color=ALGO_COLORS[algo],
                    marker="o", linewidth=2, capsize=4)

    ax.set_xlabel("Problem Size  n  (W = T = n)", fontsize=12)
    ax.set_ylabel("Mean Expected Survival Value", fontsize=12)
    ax.set_title("Mean Solution Quality vs Problem Size (Square Instances)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    save_fig("02_line_mean_value.png")


def plot_improvement_bar(df: pd.DataFrame):
    sq    = df[df["category"].isin(SQUARE_CATS)]
    pivot = sq.pivot_table(
        index=["category", "instance_id"], columns="algorithm", values="value"
    ).reset_index()

    pivot["mmr_imp"] = _safe_pct_improvement(pivot["MMR_Original"], pivot["MMR_Modified"])
    pivot["aco_imp"] = _safe_pct_improvement(pivot["ACO_Original"], pivot["ACO_Modified"])
    pivot["cat_str"] = _cat_to_str(pivot["category"])

    present = [c for c in SQUARE_CATS if c in pivot["cat_str"].values]
    agg     = pivot.groupby("cat_str")[["mmr_imp", "aco_imp"]].mean().reindex(present)

    x, width = np.arange(len(present)), 0.35
    fig, ax  = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, agg["mmr_imp"], width,
           label="MMR-IR vs MMR", color=ALGO_COLORS["MMR_Modified"], alpha=0.85)
    ax.bar(x + width/2, agg["aco_imp"], width,
           label="ACO-MMAS vs ACO", color=ALGO_COLORS["ACO_Modified"], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_xlabel("Problem Size Category", fontsize=12)
    ax.set_ylabel("Mean % Improvement over Original", fontsize=12)
    ax.set_title("Solution Quality Improvement of Modified Algorithms (Square Instances)",
                 fontsize=13)
    ax.legend(fontsize=10)
    save_fig("03_bar_improvement.png")


def plot_time_scalability(df: pd.DataFrame):
    sq       = df[df["category"].isin(SQUARE_CATS)].copy()
    sq["cat_str"] = _cat_to_str(sq["category"])
    summary  = sq.groupby(["cat_str", "algorithm"])["time_sec"].mean().reset_index()
    summary["size"] = summary["cat_str"].map(CAT_SIZES)

    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGO_ORDER:
        sub = summary[summary["algorithm"] == algo].sort_values("size")
        ax.plot(sub["size"], sub["time_sec"] * 1000,
                label=ALGO_LABELS[algo], color=ALGO_COLORS[algo],
                marker="o", linewidth=2)

    ax.set_xlabel("Problem Size  n  (W = T = n)", fontsize=12)
    ax.set_ylabel("Mean Runtime (ms)", fontsize=12)
    ax.set_title("Runtime Scalability (Square Instances)", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend(fontsize=10)
    save_fig("04_line_time_scalability.png")


def plot_heatmap_improvement(df: pd.DataFrame):
    sq    = df[df["category"].isin(SQUARE_CATS)]
    pivot = sq.pivot_table(
        index=["category", "instance_id"], columns="algorithm", values="value"
    ).reset_index()

    pivot["MMR_imp"] = _safe_pct_improvement(pivot["MMR_Original"], pivot["MMR_Modified"])
    pivot["ACO_imp"] = _safe_pct_improvement(pivot["ACO_Original"], pivot["ACO_Modified"])
    pivot["cat_str"] = _cat_to_str(pivot["category"])

    heat    = pivot.groupby("cat_str")[["MMR_imp", "ACO_imp"]].mean()
    present = [c for c in SQUARE_CATS if c in heat.index]
    heat    = heat.reindex(present)
    heat.columns = ["MMR-IR vs MMR", "ACO-MMAS vs ACO"]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(heat.T, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, cbar_kws={"label": "% improvement"})
    ax.set_title("Mean % Improvement Heatmap (Square Instances)", fontsize=13)
    ax.set_xlabel("Problem Size", fontsize=11)
    ax.set_ylabel("")
    save_fig("05_heatmap_pct_improvement.png")


def plot_violin(df: pd.DataFrame):
    """
    Violin plots — seaborn 0.13 compatible.
    split=True was removed in 0.13; we use side-by-side violins with hue instead.
    inner="quartile" replaces the old "quart" spelling in 0.13.
    """
    sq      = df[df["category"].isin(SQUARE_CATS)]
    present = [c for c in SQUARE_CATS if c in sq["category"].astype(str).unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, orig, mod, title in [
        (axes[0], "MMR_Original", "MMR_Modified", "MMR: Original vs Modified"),
        (axes[1], "ACO_Original", "ACO_Modified", "ACO: Original vs Modified"),
    ]:
        sub = sq[sq["algorithm"].isin([orig, mod])].copy()
        sub["cat_str"] = _cat_to_str(sub["category"])
        pal = {ALGO_LABELS[orig]: ALGO_COLORS[orig],
               ALGO_LABELS[mod]:  ALGO_COLORS[mod]}
        sns.violinplot(data=sub, x="cat_str", y="value", hue="label",
                       order=present, hue_order=[ALGO_LABELS[orig], ALGO_LABELS[mod]],
                       palette=pal, ax=ax,
                       inner="quartile", linewidth=1.0)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Problem Size", fontsize=11)
        ax.set_ylabel("Expected Survival Value", fontsize=11)
        ax.legend(fontsize=9, title="")

    fig.suptitle("Solution Value Distributions — Violin Plots (Square Instances)",
                 fontsize=14, y=1.01)
    save_fig("06_violin_value_dist.png")


def plot_optimality_gap(df: pd.DataFrame):
    sq  = df[df["category"].isin(SQUARE_CATS)]
    df2 = best_known(sq).copy()
    df2["gap"] = (df2["value"] - df2["best_known"]) / df2["best_known"] * 100
    df2["gap"] = df2["gap"].clip(lower=0)          # non-negative gap by definition

    summary = df2.groupby(["category", "algorithm"])["gap"].mean().reset_index()
    summary["cat_str"] = _cat_to_str(summary["category"])
    pivot   = summary.pivot(index="cat_str", columns="algorithm", values="gap")
    present = [c for c in SQUARE_CATS if c in pivot.index]
    pivot   = pivot.reindex(index=present, columns=ALGO_ORDER)

    x, width, offsets = np.arange(len(present)), 0.2, [-1.5, -0.5, 0.5, 1.5]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, algo in enumerate(ALGO_ORDER):
        ax.bar(x + offsets[i] * width, pivot[algo].fillna(0), width,
               label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_xlabel("Problem Size Category", fontsize=12)
    ax.set_ylabel("Mean Gap from Best-Known (%)", fontsize=12)
    ax.set_title("Optimality Gap — lower is better (Square Instances)", fontsize=13)
    ax.legend(fontsize=10)
    save_fig("07_bar_gap.png")


# ── Non-square instance plots ──────────────────────────────────────────────────

def plot_scenario_comparison(df: pd.DataFrame):
    """Compare solution quality across Scarce / Balanced / Rich at matched scales."""
    triplets = [
        ("~n=10", "10x10", "5w10t",  "10w5t"),
        ("~n=20", "20x20", "10w20t", "20w10t"),
        ("~n=50", "50x50", "20w50t", "50w20t"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, (label, sq_cat, sc_cat, ri_cat) in zip(axes, triplets):
        rows = []
        for cat, sc_label in [(sq_cat, "Balanced\n(W=T)"),
                               (sc_cat, "Scarce\n(W<T)"),
                               (ri_cat, "Rich\n(W>T)")]:
            sub = df[df["category"].astype(str) == cat]
            if sub.empty:
                continue
            for algo in ALGO_ORDER:
                vals = sub[sub["algorithm"] == algo]["value"]
                rows.append({"scenario_label": sc_label,
                              "algorithm":      ALGO_LABELS[algo],
                              "value":          vals.mean()})
        if not rows:
            continue

        rdf      = pd.DataFrame(rows)
        sc_order = ["Balanced\n(W=T)", "Scarce\n(W<T)", "Rich\n(W>T)"]
        pal      = {ALGO_LABELS[a]: ALGO_COLORS[a] for a in ALGO_ORDER}
        sns.barplot(data=rdf, x="scenario_label", y="value", hue="algorithm",
                    order=[s for s in sc_order if s in rdf["scenario_label"].values],
                    palette=pal, ax=ax, errorbar=None)
        ax.set_title(f"Scale: {label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Scenario Type", fontsize=11)
        ax.set_ylabel("Mean Survival Value" if label == "~n=10" else "")
        ax.legend(fontsize=8, title="", loc="upper left")

    fig.suptitle("Effect of W vs T Ratio on Solution Quality\n"
                 "(Scarce = W<T, Balanced = W=T, Rich = W>T)",
                 fontsize=13, y=1.03)
    save_fig("08_scenario_comparison.png")


def plot_scenario_improvement(df: pd.DataFrame):
    """% improvement of modified over original, grouped by scenario type."""
    records = []
    for scenario, cats in [("scarce", SCARCE_CATS),
                            ("square", SQUARE_CATS),
                            ("rich",   RICH_CATS)]:
        sub = df[df["category"].isin(cats)]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index=["category", "instance_id"], columns="algorithm", values="value"
        ).reset_index()

        if "MMR_Original" in pivot.columns and "MMR_Modified" in pivot.columns:
            imp = _safe_pct_improvement(pivot["MMR_Original"], pivot["MMR_Modified"])
            records.append({"scenario": scenario,
                             "algorithm_pair": "MMR-IR vs MMR",
                             "improvement": imp.mean()})

        if "ACO_Original" in pivot.columns and "ACO_Modified" in pivot.columns:
            imp = _safe_pct_improvement(pivot["ACO_Original"], pivot["ACO_Modified"])
            records.append({"scenario": scenario,
                             "algorithm_pair": "ACO-MMAS vs ACO",
                             "improvement": imp.mean()})

    rdf = pd.DataFrame(records)
    if rdf.empty:
        return

    sc_ord = ["scarce", "square", "rich"]
    sc_lbl = ["Scarce (W<T)", "Balanced (W=T)", "Rich (W>T)"]
    x, width = np.arange(3), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, pair in enumerate(["MMR-IR vs MMR", "ACO-MMAS vs ACO"]):
        vals   = [rdf[(rdf["scenario"] == s) &
                      (rdf["algorithm_pair"] == pair)]["improvement"].values
                  for s in sc_ord]
        vals   = [float(v[0]) if len(v) else 0.0 for v in vals]
        offset = -width/2 if i == 0 else width/2
        color  = ALGO_COLORS["MMR_Modified"] if i == 0 else ALGO_COLORS["ACO_Modified"]
        ax.bar(x + offset, vals, width, label=pair, color=color, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(sc_lbl)
    ax.set_xlabel("Scenario Type", fontsize=12)
    ax.set_ylabel("Mean % Improvement over Original", fontsize=12)
    ax.set_title("Improvement of Modified Algorithms by Scenario Type", fontsize=13)
    ax.legend(fontsize=10)
    save_fig("09_bar_scenario_improvement.png")


def plot_scarce_vs_rich(df: pd.DataFrame):
    """Box plots comparing algorithm performance on scarce vs rich instances."""
    ns_df = df[df["category"].isin(SCARCE_CATS + RICH_CATS)].copy()
    if ns_df.empty:
        print("  (skipping scarce_vs_rich — no non-square data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cats, title in [
        (axes[0], SCARCE_CATS, "Weapon-Scarce (W < T)"),
        (axes[1], RICH_CATS,   "Weapon-Rich   (W > T)"),
    ]:
        sub = ns_df[ns_df["category"].isin(cats)]
        if sub.empty:
            ax.set_visible(False)
            continue
        labels = [ALGO_LABELS[a] for a in ALGO_ORDER if a in sub["algorithm"].values]
        pal    = {ALGO_LABELS[a]: ALGO_COLORS[a] for a in ALGO_ORDER}
        sns.boxplot(data=sub, x="label", y="value",
                    order=labels, palette=pal, ax=ax,
                    linewidth=1.2, fliersize=3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Expected Survival Value", fontsize=11)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Algorithm Performance: Weapon-Scarce vs Weapon-Rich Scenarios",
                 fontsize=13, y=1.01)
    save_fig("10_scarce_vs_rich.png")


# ── summary & stats ────────────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame):
    sq = df[df["category"].isin(SQUARE_CATS)]
    pv = sq.pivot_table(index="category", columns="algorithm",
                         values="value", aggfunc="mean")
    pv.index = pv.index.astype(str)
    present  = [c for c in SQUARE_CATS if c in pv.index]
    print("\n=== Mean Solution Value (Square Instances) ===")
    print(pv.reindex(index=present, columns=ALGO_ORDER).to_string(float_format="%.2f"))

    tp = sq.pivot_table(index="category", columns="algorithm",
                         values="time_sec", aggfunc="mean")
    tp.index = tp.index.astype(str)
    print("\n=== Mean Runtime ms (Square Instances) ===")
    print((tp.reindex(index=present, columns=ALGO_ORDER) * 1000)
          .to_string(float_format="%.1f"))

    ns = df[df["category"].isin(SCARCE_CATS + RICH_CATS)]
    if not ns.empty:
        ns_pv = ns.pivot_table(index=["scenario", "category"], columns="algorithm",
                                values="value", aggfunc="mean")
        print("\n=== Mean Solution Value (Non-Square Instances) ===")
        print(ns_pv.reindex(columns=ALGO_ORDER).to_string(float_format="%.2f"))


def run_statistical_tests(df: pd.DataFrame):
    sq = df[df["category"].isin(SQUARE_CATS)]
    print("\n=== Statistical Tests: Square Instances (Wilcoxon signed-rank, a=0.05) ===")
    for orig, mod, name in [
        ("MMR_Original", "MMR_Modified", "MMR"),
        ("ACO_Original", "ACO_Modified", "ACO"),
    ]:
        for cat in SQUARE_CATS:
            o = sq[(sq["algorithm"] == orig) &
                   (sq["category"].astype(str) == cat)]["value"].values
            m = sq[(sq["algorithm"] == mod) &
                   (sq["category"].astype(str) == cat)]["value"].values
            if len(o) < 5:
                continue
            try:
                stat, pval = stats.wilcoxon(o, m, alternative="greater")
                sig = "**" if pval < 0.05 else "  "
                print(f"  {name} {cat:8s}: W={stat:.1f}  p={pval:.4f} {sig}")
            except Exception as e:
                print(f"  {name} {cat:8s}: {e}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(RESULTS_CSV):
        print(f"Results file not found: {RESULTS_CSV}")
        print("Run experiment_runner.py first.")
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = load_results()

    print(f"Loaded {len(df)} rows from {RESULTS_CSV}")
    print(f"Algorithms : {sorted(df['algorithm'].unique())}")
    print(f"Categories : {df['category'].astype(str).unique().tolist()}\n")

    print("Generating plots ...")
    plot_box_solution_quality(df)
    plot_mean_value(df)
    plot_improvement_bar(df)
    plot_time_scalability(df)
    plot_heatmap_improvement(df)
    plot_violin(df)
    plot_optimality_gap(df)
    plot_scenario_comparison(df)
    plot_scenario_improvement(df)
    plot_scarce_vs_rich(df)

    print_summary_table(df)
    run_statistical_tests(df)
    print("\nAll done.")


if __name__ == "__main__":
    main()
