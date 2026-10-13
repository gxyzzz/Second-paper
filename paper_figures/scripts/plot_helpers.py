from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_style import (
    paper_style, save_figure, clean_axes,
    VALIDATION, TEST, NEUTRAL_LIGHT, NEUTRAL_DARK, GRID, PAPER,
)


def load_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def pct(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}%"


def validation_test_dumbbell(
    csv_path: Path | str,
    outbase: Path | str,
    *,
    parameter_label: str,
    formal_value: float,
    value_formatter,
    figsize=(3.5, 2.55),
) -> None:
    """Paired Validation/Test comparison for one discrete hyperparameter sweep."""
    d = load_csv(csv_path).sort_values("parameter_value", ascending=False).reset_index(drop=True)
    y = np.arange(len(d))

    with paper_style():
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        for i, row in d.iterrows():
            formal = np.isclose(float(row.parameter_value), formal_value)
            ax.plot(
                [row.validation_u4, row.test_u4], [y[i], y[i]],
                color=NEUTRAL_DARK if formal else NEUTRAL_LIGHT,
                linewidth=1.45 if formal else 1.0,
                zorder=1,
            )
            marker_size = 6.0 if formal else 5.1
            edge_width = 1.25 if formal else 0.95
            ax.plot(row.validation_u4, y[i], marker="o", markersize=marker_size,
                    markerfacecolor=VALIDATION, markeredgecolor=VALIDATION,
                    markeredgewidth=edge_width, linestyle="none", zorder=3)
            ax.plot(row.test_u4, y[i], marker="s", markersize=marker_size,
                    markerfacecolor=PAPER, markeredgecolor=TEST,
                    markeredgewidth=edge_width, linestyle="none", zorder=3)

        labels = []
        for value in d.parameter_value:
            text = value_formatter(float(value))
            if np.isclose(float(value), formal_value):
                variant = d.loc[np.isclose(d.parameter_value, formal_value), "variant"].iloc[0]
                text += "*" if variant == "M25" else "†"
            labels.append(text)
        ax.set_yticks(y, labels)
        for tick, value in zip(ax.get_yticklabels(), d.parameter_value):
            if np.isclose(float(value), formal_value):
                tick.set_fontweight("semibold")

        all_values = np.r_[d.validation_u4.to_numpy(), d.test_u4.to_numpy()]
        xmin = min(-0.05, float(all_values.min()) - 0.22)
        xmax = float(all_values.max()) + 0.28
        ax.set_xlim(xmin, xmax)
        ax.axvline(0, color=GRID, linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xlabel("Gain over CoLiftRec (%)")
        ax.set_ylabel(parameter_label)
        reference = d.loc[np.isclose(d.parameter_value, formal_value), "variant"].iloc[0]
        ax.set_title("Baby · " + ("margin sweep" if reference == "M25" else "reference-config sweep"), loc="left", pad=32, fontsize=8)
        clean_axes(ax, grid_axis="x", hide_left=True)

        handles = [
            Line2D([], [], marker="o", linestyle="none", color=VALIDATION,
                   markerfacecolor=VALIDATION, markersize=5.2, label="Validation"),
            Line2D([], [], marker="s", linestyle="none", color=TEST,
                   markerfacecolor=PAPER, markeredgecolor=TEST, markersize=5.2, label="Test"),
        ]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.015),
                  ncol=2, handletextpad=0.35, columnspacing=1.0, borderaxespad=0.0)
        save_figure(fig, outbase)
        plt.close(fig)
