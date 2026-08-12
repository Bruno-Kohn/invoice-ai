"""Visualization functions for experiment results.

Generates publication-ready charts for comparing pipeline configurations,
OCR engines, parsers, and preprocessing techniques.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Default style
plt.rcParams["figure.dpi"] = 100
sns.set_style("whitegrid")

FIGURES_DIR = Path("results/figures")


def save_fig(fig, name: str, output_dir: Path = None):
    """Save figure to results/figures/."""
    out = output_dir or FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")


def plot_metric_comparison(
    labels: list[str],
    values: list[float],
    metric_name: str = "CER",
    title: str = "",
    colors: Optional[list[str]] = None,
    figsize: tuple = (10, 6),
    save_name: Optional[str] = None,
):
    """Bar chart comparing a metric across configurations.

    Args:
        labels: Configuration names (x-axis).
        values: Metric values (y-axis).
        metric_name: Name of the metric (y-label).
        title: Chart title.
        colors: Bar colors.
        figsize: Figure size.
        save_name: If set, saves to results/figures/{save_name}.png.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = colors or sns.color_palette("viridis", len(labels))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title or f"{metric_name} by Configuration", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_grouped_bar(
    categories: list[str],
    groups: dict[str, list[float]],
    title: str = "",
    ylabel: str = "Score",
    figsize: tuple = (12, 6),
    save_name: Optional[str] = None,
):
    """Grouped bar chart for comparing multiple metrics across categories.

    Args:
        categories: Category names (x-axis groups).
        groups: Dict of group_name -> values (one bar per group per category).
        title: Chart title.
        ylabel: Y-axis label.
        save_name: If set, saves figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(categories))
    n_groups = len(groups)
    width = 0.8 / n_groups
    colors = sns.color_palette("Set2", n_groups)

    for i, (name, vals) in enumerate(groups.items()):
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_heatmap(
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    cmap: str = "YlOrRd",
    fmt: str = ".3f",
    figsize: tuple = (10, 8),
    save_name: Optional[str] = None,
):
    """Heatmap for field × approach comparison.

    Args:
        data: 2D array of values.
        row_labels: Row names.
        col_labels: Column names.
        title: Chart title.
        save_name: If set, saves figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=col_labels, yticklabels=row_labels, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_box(
    data: dict[str, list[float]],
    title: str = "",
    ylabel: str = "Score",
    figsize: tuple = (10, 6),
    save_name: Optional[str] = None,
):
    """Box plot for score distributions.

    Args:
        data: Dict of group_name -> list of values.
        title: Chart title.
        save_name: If set, saves figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(data.keys())
    values = list(data.values())

    bp = ax.boxplot(values, labels=labels, patch_artist=True)
    colors = sns.color_palette("Set2", len(labels))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_line(
    x: list,
    lines: dict[str, list[float]],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (10, 6),
    save_name: Optional[str] = None,
):
    """Line chart for trends (e.g., F1 vs threshold).

    Args:
        x: X-axis values.
        lines: Dict of line_name -> y values.
        title: Chart title.
        save_name: If set, saves figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("Set1", len(lines))

    for (name, y), color in zip(lines.items(), colors):
        ax.plot(x, y, "-o", label=name, color=color, markersize=6, linewidth=2)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_radar(
    categories: list[str],
    groups: dict[str, list[float]],
    title: str = "",
    figsize: tuple = (8, 8),
    save_name: Optional[str] = None,
):
    """Radar chart for multi-dimensional comparison (e.g., Regex vs LLM).

    Args:
        categories: Metric names around the radar.
        groups: Dict of group_name -> metric values.
        title: Chart title.
        save_name: If set, saves figure.
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    colors = sns.color_palette("Set1", len(groups))

    for (name, vals), color in zip(groups.items(), colors):
        values = vals + vals[:1]
        ax.plot(angles, values, "o-", label=name, color=color, linewidth=2)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_waterfall(
    stages: list[str],
    contributions: list[float],
    title: str = "",
    ylabel: str = "F1 Score",
    figsize: tuple = (10, 6),
    save_name: Optional[str] = None,
):
    """Waterfall chart showing contribution of each pipeline stage.

    Args:
        stages: Stage names.
        contributions: Incremental contribution of each stage.
        title: Chart title.
        save_name: If set, saves figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    cumulative = np.cumsum(contributions)
    bottoms = np.concatenate([[0], cumulative[:-1]])

    colors = ["#2ecc71" if c >= 0 else "#e74c3c" for c in contributions]
    bars = ax.bar(stages, contributions, bottom=bottoms, color=colors, edgecolor="white")

    for bar, val, cum in zip(bars, contributions, cumulative):
        y = cum if val >= 0 else cum - val
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.005,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_name:
        save_fig(fig, save_name)
    return fig
