"""Render the experiment CSV files into one readable PNG with tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _format(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.6g}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--output", default="outputs/csv_tables.png")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = [
        ("FlashSinkhorn toy benchmark", input_dir / "flashsinkhorn_toy.csv"),
        ("OPW solver benchmark", input_dir / "opw_solver_benchmark.csv"),
        ("OWD-style NM / k-NN classification", input_dir / "owd_style_classification.csv"),
    ]
    tables = [(title, _format(pd.read_csv(path))) for title, path in files if path.exists()]
    if not tables:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")

    height = sum(max(1.8, 0.55 + 0.42 * (len(df) + 1)) for _, df in tables)
    fig = plt.figure(figsize=(20, height), facecolor="white")
    grid = fig.add_gridspec(len(tables), 1, hspace=0.42)
    for row, (title, df) in enumerate(tables):
        ax = fig.add_subplot(grid[row])
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=10)
        table = ax.table(
            cellText=df.values,
            colLabels=list(df.columns),
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#B8C2CC")
            if r == 0:
                cell.set_facecolor("#1F4E78")
                cell.set_text_props(color="white", weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#EEF4F8")
            else:
                cell.set_facecolor("white")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()

