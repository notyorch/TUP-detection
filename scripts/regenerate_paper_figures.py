#!/usr/bin/env python3
"""Regenerate paper figures from frozen JSON (no HF API)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "notebooks" / "data" / "external" / "results"
FIGURES = REPO / "notebooks" / "figures"
LATEX_FIGURES = REPO / "latex" / "figures"


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def stack_df(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(payload["results"]).set_index("stack")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)

    deepset_ablation = load_json("stack-ablation-deepset.json")
    synthetic_ablation = load_json("stack-ablation-synthetic.json")
    crescendo = load_json("crescendo-sentinel-hf.json")
    deepset_deberta = load_json("deepset-full.json")

    STACK_COLORS = {
        "TUP Layer 1 only": "#6366F1",
        "Sentinel v2 only": "#0891B2",
        "TUP + Sentinel v2": "#0D9488",
    }
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.family": "serif", "font.size": 10})

    df_deep = stack_df(deepset_ablation)
    order = ["TUP Layer 1 only", "Sentinel v2 only", "TUP + Sentinel v2"]
    plot_df = df_deep.loc[order]

    # fig01
    metrics = [
        ("pint_balanced_pct", "PINT balanced"),
        ("attack_detection_pct", "Attack recall"),
        ("benign_pass_pct", "Benign pass"),
        ("overall_accuracy_pct", "Overall accuracy"),
    ]
    x = np.arange(len(metrics))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, stack in enumerate(order):
        vals = [plot_df.loc[stack, m] for m, _ in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=stack, color=STACK_COLORS[stack],
                      edgecolor="black", linewidth=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f"{val:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%)")
    ax.set_title("Figure 1 (Primary). deepset Tier-B — TUP vs Sentinel vs hybrid stack")
    ax.axhline(90, color="#94A3B8", linestyle="--", linewidth=0.8, label="90% target")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    p1 = FIGURES / "fig01_deepset_stack_ablation.png"
    plt.savefig(p1, bbox_inches="tight", facecolor="white")
    plt.close()

    # fig02
    deberta_row = next(r for r in deepset_deberta["results"] if r["engine"] == "TUP Layer 1+Classifier")
    combined_row = df_deep.loc["TUP + Sentinel v2"]
    baselines = pd.DataFrame([
        {"system": "TUP + Sentinel v2\n(ours, measured)", "pint_balanced_pct": combined_row["pint_balanced_pct"], "color": "#0D9488"},
        {"system": "TUP + DeBERTa v2\n(ours, measured)", "pint_balanced_pct": deberta_row["pint_balanced_pct"], "color": "#F59E0B"},
        {"system": "Sentinel v2\n(paper / model card)", "pint_balanced_pct": 88.0, "color": "#0891B2"},
        {"system": "ProtectAI DeBERTa\n(literature)", "pint_balanced_pct": 77.6, "color": "#D97706"},
    ]).sort_values("pint_balanced_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(baselines["system"], baselines["pint_balanced_pct"],
                   color=baselines["color"], edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, baselines["pint_balanced_pct"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 102)
    ax.set_xlabel("PINT balanced accuracy (%)")
    ax.set_title("Figure 2. deepset — TUP+Sentinel vs baselines")
    ax.axvline(90, color="#94A3B8", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    p2 = FIGURES / "fig02_deepset_vs_baselines.png"
    plt.savefig(p2, bbox_inches="tight", facecolor="white")
    plt.close()

    # fig03
    comp = deepset_ablation["complementarity"]
    labels = ["TUP only", "Sentinel only", "Both", "Missed (FN)"]
    vals = [comp["tup_only_catches"], comp["sentinel_only_catches"], comp["both_catch"], comp["neither_fn"]]
    colors = ["#6366F1", "#0891B2", "#0D9488", "#DC2626"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, vals, color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_ylabel("Attack samples (n)")
    axes[0].set_title("Attack detection by layer (deepset)")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 1, str(v), ha="center", fontsize=9)
    sentinel_tp = comp["sentinel_only_catches"] + comp["both_catch"]
    combined_tp = comp["combined_tp"]
    axes[1].bar(["Sentinel alone", "TUP + Sentinel"], [sentinel_tp, combined_tp],
                color=["#0891B2", "#0D9488"], edgecolor="black", linewidth=0.4)
    axes[1].set_ylabel("True positives (attacks)")
    axes[1].set_title(f"Hybrid adds {combined_tp - sentinel_tp} TP via TUP L1")
    axes[1].set_ylim(0, deepset_ablation["attacks"] + 10)
    for i, v in enumerate([sentinel_tp, combined_tp]):
        axes[1].text(i, v + 2, str(v), ha="center")
    plt.tight_layout()
    p3 = FIGURES / "fig03_complementarity_deepset.png"
    plt.savefig(p3, bbox_inches="tight", facecolor="white")
    plt.close()

    # fig04
    cre_views = pd.DataFrame([
        {"view": "Turn-only\n(stateless guard)", **crescendo["turn_only"]},
        {"view": "Cumulative\n(multi-turn context)", **crescendo["cumulative"]},
        {"view": "Full transcript\n(STCA-style)", **crescendo["full_transcript"]},
    ])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cre_views))
    w = 0.25
    for i, col, c in [(0, "attack_detection_pct", "#DC2626"), (1, "benign_pass_pct", "#2563EB"), (2, "pint_balanced_pct", "#0D9488")]:
        ax.bar(x + (i - 1) * w, cre_views[col], w, label=col.replace("_pct", "").replace("_", " "), color=c, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cre_views["view"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title("Figure 4. Crescendo multi-turn — Sentinel v2 scoring views")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    p4 = FIGURES / "fig04_crescendo_multiturn.png"
    plt.savefig(p4, bbox_inches="tight", facecolor="white")
    plt.close()

    # fig05
    row = df_deep.loc["TUP + Sentinel v2"]
    cm = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred benign", "Pred attack"],
                yticklabels=["True benign", "True attack"], cbar=False)
    ax.set_title("Figure 5. Confusion matrix — deepset, TUP + Sentinel v2")
    plt.tight_layout()
    p5 = FIGURES / "fig05_deepset_confusion_combined.png"
    plt.savefig(p5, bbox_inches="tight", facecolor="white")
    plt.close()

    # figA synthetic
    df_syn = stack_df(synthetic_ablation)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    syn_order = order
    x = np.arange(2)
    w = 0.25
    for i, stack in enumerate(syn_order):
        vals = [df_syn.loc[stack, "pint_balanced_pct"], df_syn.loc[stack, "attack_detection_pct"]]
        ax.bar(x + (i - 1) * w, vals, w, label=stack, color=STACK_COLORS[stack], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["PINT balanced", "Attack recall"])
    ax.set_ylim(0, 108)
    ax.set_ylabel("PINT balanced (%)")
    ax.set_title("Appendix A. Synthetic dataset — stack ablation")
    ax.legend(fontsize=8)
    plt.tight_layout()
    pa = FIGURES / "figA_synthetic_ablation.png"
    plt.savefig(pa, bbox_inches="tight", facecolor="white")
    plt.close()

    for name in ["fig01_deepset_stack_ablation.png", "fig02_deepset_vs_baselines.png"]:
        shutil.copy2(FIGURES / name, LATEX_FIGURES / name)

    print("Saved figures:")
    for p in sorted(FIGURES.glob("fig*.png")):
        print(f"  {p.relative_to(REPO)}")
    print(f"\nCopied paper figures to {LATEX_FIGURES.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
