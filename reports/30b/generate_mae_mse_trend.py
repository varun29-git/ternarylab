"""Render absolute MAE and relative MSE from the completed 30B telemetry."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tables" / "training_timeseries.csv"
OUTPUT = ROOT / "figures" / "quantization_mae_mse_trend.png"
FINAL_TOKENS_B = 29.999996928
FINAL_MAE = 0.04426160319514262
FINAL_RELATIVE_MSE = 0.42584320893511174


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["quant_relative_mse"]]
    tokens_b = [float(row["tokens_b"]) for row in rows]
    maes = [float(row["quant_abs_mae"]) for row in rows]
    mses = [float(row["quant_relative_mse"]) for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, mae_axis = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    mse_axis = mae_axis.twinx()
    mae_line = mae_axis.plot(tokens_b, maes, color="#D94A3A", linewidth=2.5, label="Absolute MAE")
    mse_line = mse_axis.plot(tokens_b, mses, color="#2457A5", linewidth=2.5, label="Relative MSE")
    mae_axis.scatter([FINAL_TOKENS_B], [FINAL_MAE], color="#D94A3A", s=42, zorder=3)
    mse_axis.scatter([FINAL_TOKENS_B], [FINAL_RELATIVE_MSE], color="#2457A5", s=42, zorder=3)
    mae_axis.annotate(
        "Final MAE: 0.04426",
        xy=(FINAL_TOKENS_B, FINAL_MAE),
        xytext=(-118, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#D94A3A"},
        color="#9C2F24",
        fontsize=9,
    )
    mse_axis.annotate(
        "Final relative MSE: 0.42584",
        xy=(FINAL_TOKENS_B, FINAL_RELATIVE_MSE),
        xytext=(-145, -36),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#2457A5"},
        color="#1B427D",
        fontsize=9,
    )
    mae_axis.set_xlabel("Cumulative training tokens (billions)")
    mae_axis.set_ylabel("Absolute MAE (weight units)", color="#9C2F24")
    mse_axis.set_ylabel("Relative MSE", color="#1B427D")
    mae_axis.set_xlim(0, 30.5)
    mae_axis.set_title("30B Run: Absolute MAE vs. Relative MSE")
    lines = mae_line + mse_line
    mae_axis.legend(lines, [line.get_label() for line in lines], loc="upper left")
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
