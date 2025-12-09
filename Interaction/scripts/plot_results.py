"""
Plot performance deltas between baseline (1D) and interaction (2D) feature sets.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "results"
PLOT_DIR = RESULT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Prefer CJK-friendly fonts on macOS; fallback to Arial Unicode MS / Noto
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang TC",
    "PingFang HK",
    "Heiti TC",
    "Arial Unicode MS",
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "Microsoft JhengHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_summary() -> pd.DataFrame:
    path = RESULT_DIR / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    df = pd.read_csv(path)
    return df


def plot_metric(df: pd.DataFrame, value_col: str, title: str, ylabel: str, filename: str):
    """
    value_col: column in df (e.g., r2_gain, rmse_delta)
    Positive r2_gain is better; negative delta for errors (rmse/mae) is better.
    """
    # Compose a readable label for each row: segment-cabin-model
    df = df.copy()
    model_short = {"RandomForest": "RF", "SVR": "SVR", "XGBoost": "XGB"}
    df["label"] = df["segment"] + "_" + df["cabin"] + " - " + df["model"].map(model_short)
    df = df.sort_values(value_col, ascending=False)

    positions = range(len(df))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(positions, df[value_col], color="#4C72B0")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(df["label"], rotation=60, ha="right")

    # Annotate values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = PLOT_DIR / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = load_summary()
    plot_metric(df, "r2_gain", "R² gain (Interaction - Baseline)", "ΔR² (higher better)", "r2_gain.png")
    plot_metric(df, "rmse_delta", "RMSE delta (Interaction - Baseline)", "ΔRMSE (negative better)", "rmse_delta.png")
    plot_metric(df, "mae_delta", "MAE delta (Interaction - Baseline)", "ΔMAE (negative better)", "mae_delta.png")


if __name__ == "__main__":
    main()

