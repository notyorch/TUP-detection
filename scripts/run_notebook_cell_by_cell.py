#!/usr/bin/env python3
"""Execute a Jupyter notebook sequentially, saving after each cell."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbformat.validator import normalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run notebook cell-by-cell")
    p.add_argument("notebook", type=Path, help="Path to .ipynb")
    p.add_argument("--timeout", type=int, default=900, help="Per-cell timeout (seconds)")
    p.add_argument("--cwd", type=Path, default=None, help="Working directory for kernel")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    nb_path = args.notebook.resolve()
    if not nb_path.is_file():
        print(f"Not found: {nb_path}", file=sys.stderr)
        return 1

    cwd = args.cwd or nb_path.parent
    nb = nbformat.read(nb_path, as_version=4)
    normalize(nb)

    # Clear stale outputs for a clean run
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd)}},
    )

    total = sum(1 for c in nb.cells if c.cell_type == "code")
    done = 0
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            done += 1
            preview = "".join(cell.source).strip().splitlines()
            head = preview[0][:80] if preview else "(empty)"
            print(f"\n[{done}/{total}] Cell {i}: {head}")
            try:
                client.execute_cell(cell, i)
            except CellExecutionError as exc:
                print(f"FAILED at cell {i}: {exc}", file=sys.stderr)
                nbformat.write(nb, nb_path)
                return 1
            nbformat.write(nb, nb_path)
            print(f"  ✓ saved {nb_path.name}")

    print(f"\nDone — {done} code cells executed → {nb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
