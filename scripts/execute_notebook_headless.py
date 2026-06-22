#!/usr/bin/env python3
"""Execute notebook code cells headless; save every figure to notebooks/figures/."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nbformat  # noqa: E402


def execute_notebook(nb_path: Path, cwd: Path, fig_prefix: str) -> list[Path]:
    repo = cwd.parent if cwd.name == "notebooks" else cwd
    figures_dir = repo / "notebooks" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(nb_path, as_version=4)
    saved: list[Path] = []
    fig_counter = 0
    ns: dict = {"__name__": "__main__", "__file__": str(nb_path)}

    original_show = plt.show

    def show_and_save(*args, **kwargs):
        nonlocal fig_counter
        for num in plt.get_fignums():
            fig = plt.figure(num)
            fig_counter += 1
            out = figures_dir / f"{fig_prefix}_{fig_counter:02d}.png"
            fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
            saved.append(out)
            print(f"  saved {out.relative_to(cwd.parent if cwd.name == 'notebooks' else cwd)}")
        plt.close("all")

    plt.show = show_and_save

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    print(f"\n=== {nb_path.name} ({len(code_cells)} code cells) ===")

    for i, cell in enumerate(code_cells, start=1):
        src = cell.source.strip()
        if not src or src.startswith("%pip"):
            continue
        print(f"[{i}/{len(code_cells)}] executing...")
        try:
            exec(compile(src, f"{nb_path.name}:cell{i}", "exec"), ns)
        except Exception:
            print(f"FAILED cell {i} in {nb_path.name}:")
            traceback.print_exc()
            raise

    plt.show = original_show
    return saved


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    notebooks_dir = repo / "notebooks"
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", help="Notebook paths (default: all three)")
    args = parser.parse_args()

    default = [
        notebooks_dir / "tup_detection_guard_benchmark_report.ipynb",
        notebooks_dir / "tier_b_guard_comparison.ipynb",
        notebooks_dir / "tup_pint_ai_safety_report.ipynb",
    ]
    paths = [Path(p) for p in args.notebooks] if args.notebooks else default

    prefixes = {
        "tup_detection_guard_benchmark_report.ipynb": "guard",
        "tier_b_guard_comparison.ipynb": "tierb",
        "tup_pint_ai_safety_report.ipynb": "pint",
    }

    all_saved: list[Path] = []
    for nb_path in paths:
        if not nb_path.is_file():
            print(f"SKIP missing: {nb_path}", file=sys.stderr)
            continue
        prefix = prefixes.get(nb_path.name, nb_path.stem)
        all_saved.extend(execute_notebook(nb_path, notebooks_dir, prefix))

    print(f"\nTotal figures saved: {len(all_saved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
