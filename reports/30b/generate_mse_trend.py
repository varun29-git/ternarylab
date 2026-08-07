"""Render the recorded relative-MSE trajectory for the completed 30B run."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tables" / "training_timeseries.csv"
OUTPUT = ROOT / "figures" / "quantization_relative_mse_trend.png"
FINAL_TOKENS_B = 29.999996928
FINAL_RELATIVE_MSE = 0.42584320893511174


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["quant_relative_mse"]]
    tokens_b = [float(row["tokens_b"]) for row in rows]
    mse = [float(row["quant_relative_mse"]) for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    axis.plot(tokens_b, mse, color="#2457A5", linewidth=2.5, label="Recorded telemetry")
    axis.scatter([FINAL_TOKENS_B], [FINAL_RELATIVE_MSE], color="#D94A3A", s=45, zorder=3, label="Exact final scan")
    axis.annotate(
        "Final: 0.42584\n30.0B tokens",
        xy=(FINAL_TOKENS_B, FINAL_RELATIVE_MSE),
        xytext=(-108, -37),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=9,
    )
    axis.set(
        title="30B Training Run: Ternary Quantization Relative-MSE Trajectory",
        xlabel="Cumulative training tokens (billions)",
        ylabel="Relative MSE: latent weights vs. ternary reconstruction",
        xlim=(0, 30.5),
    )
    axis.legend(frameon=True)
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
