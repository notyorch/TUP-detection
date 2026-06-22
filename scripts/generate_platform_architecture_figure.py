#!/usr/bin/env python3
"""Generate fig00_tup_platform_architecture.png for the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
OUT_PATHS = [
    REPO / "notebooks" / "figures" / "fig00_tup_platform_architecture.png",
    REPO / "latex" / "figures" / "fig00_tup_platform_architecture.png",
]

C_SOURCE = "#E0E7FF"
C_ENGINE = "#DBEAFE"
C_L1 = "#99F6E4"
C_SENT = "#0D9488"
C_JUDGE = "#F0FDFA"
C_FRAME = "#CCFBF1"
C_SIEM = "#1E293B"
C_TEXT = "#0F172A"
C_MUTED = "#64748B"
C_ARROW = "#475569"
C_EDGE = "#0F766E"


def rounded_box(ax, x, y, w, h, text, face, edge="#334155", fs=10, bold=False, tc=C_TEXT):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs,
        color=tc, weight="bold" if bold else "normal", zorder=3,
    )


def v_arrow(ax, x, y1, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x, y1), (x, y2),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.5,
            color=C_ARROW, shrinkA=6, shrinkB=6, zorder=1,
        )
    )


def h_arrow(ax, x1, x2, y):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y), (x2, y),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.5,
            color=C_ARROW, shrinkA=6, shrinkB=6, zorder=1,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.3, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(5, 6.85, "TUP / AIGSMP Platform Architecture",
            ha="center", fontsize=15, weight="bold", color=C_TEXT)
    ax.text(5, 6.45,
            "Production stack: TUP Detection (Layer 1 + Sentinel v2 + optional judge)",
            ha="center", fontsize=9.5, color=C_MUTED)

    # --- Sources ---
    sw, sh, sy = 2.55, 0.75, 5.35
    sx = [0.55, 3.725, 6.9]
    labels = ["Chat / Copilots", "RAG pipelines", "Autonomous agents"]
    for x, label in zip(sx, labels):
        rounded_box(ax, x, sy, sw, sh, label, C_SOURCE)
    ax.text(5, 5.12, "AI-interaction sources", ha="center", fontsize=8.5, color=C_MUTED)

    # --- Collection ---
    cx, cy, cw, ch = 1.4, 4.05, 7.2, 0.78
    rounded_box(ax, cx, cy, cw, ch,
                "TUP collection engine\n(normalize · segment · event envelope)",
                C_ENGINE, bold=True, fs=10)

    for x in [sx[0] + sw / 2, 5.0, sx[2] + sw / 2]:
        v_arrow(ax, x, sy, cy + ch)

    v_arrow(ax, 5.0, cy, 3.95)

    # --- Detection frame ---
    fx, fy, fw, fh = 0.45, 1.55, 9.1, 2.15
    ax.add_patch(
        FancyBboxPatch(
            (fx, fy), fw, fh,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=2.0, edgecolor=C_EDGE, facecolor=C_FRAME, zorder=1,
        )
    )
    ax.text(5, 3.45, "TUP Detection module  ·  TUP-detection repo (this work)",
            ha="center", fontsize=10.5, weight="bold", color=C_EDGE, zorder=3)

    # Internal pipeline (left → right)
    bw, bh, by = 2.15, 1.05, 1.85
    b1x, b2x, b3x = 0.85, 3.55, 6.55
    rounded_box(ax, b1x, by, bw, bh, "Layer 1\nregex / OWASP", C_L1, edge=C_EDGE, fs=9.5)
    rounded_box(ax, b2x, by, bw, bh, "Sentinel v2\nHF inference", C_SENT, edge=C_EDGE, fs=9.5, tc="white")
    rounded_box(ax, b3x, by, 2.35, bh, "Optional L3\nNVIDIA judge", C_JUDGE, edge=C_EDGE, fs=9)

    h_arrow(ax, b1x + bw, b2x, by + bh / 2)
    h_arrow(ax, b2x + bw, b3x, by + bh / 2)

    ax.text(b1x + bw / 2, 1.62, "fast-path alert", ha="center", fontsize=7.5, color=C_MUTED)
    ax.text(5, 1.62, "if no L1 hit", ha="center", fontsize=7.5, color=C_MUTED)

    v_arrow(ax, 5.0, fy, 1.08)

    # --- SIEM ---
    rounded_box(
        ax, 0.9, 0.45, 8.2, 0.62,
        "Alerts · correlation · TUP-fullstack SIEM  (dashboards · cases · audit)",
        C_SIEM, edge=C_SIEM, fs=10, bold=True, tc="white",
    )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    for p in OUT_PATHS:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.15, facecolor="white")
        print(f"Saved {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
