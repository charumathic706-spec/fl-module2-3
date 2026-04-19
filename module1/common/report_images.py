from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_grouped_bar_chart(
    rows: Sequence[Mapping[str, Any]],
    output_path: str,
    *,
    title: str,
    label_key: str = "label",
    series: Sequence[Tuple[str, str, str]] = (),
    ylabel: str = "Score",
    ylim: Tuple[float, float] = (0.0, 1.0),
    figsize: Tuple[float, float] = (11.0, 5.0),
    rotate_xticks: int = 20,
) -> str:
    if not rows:
        raise RuntimeError("No rows provided for chart generation")
    if not series:
        raise RuntimeError("No series configured for chart generation")

    labels = [str(row.get(label_key, "")) for row in rows]
    x = list(range(len(rows)))
    width = min(0.8 / max(len(series), 1), 0.3)

    plt.figure(figsize=figsize)
    offsets = [((index - (len(series) - 1) / 2) * width) for index in range(len(series))]

    for offset, (series_label, value_key, color) in zip(offsets, series):
        values = [float(row.get(value_key, 0.0)) for row in rows]
        plt.bar([index + offset for index in x], values, width=width, label=series_label, color=color)

    plt.xticks(x, labels, rotation=rotate_xticks, ha="right" if rotate_xticks else "center")
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_single_metric_chart(
    rows: Sequence[Mapping[str, Any]],
    output_path: str,
    *,
    title: str,
    metric_key: str,
    label_key: str = "label",
    color: str = "#3B82F6",
    ylabel: str = "Score",
    ylim: Tuple[float, float] = (0.0, 1.0),
    figsize: Tuple[float, float] = (11.0, 5.0),
    rotate_xticks: int = 20,
) -> str:
    if not rows:
        raise RuntimeError("No rows provided for chart generation")

    labels = [str(row.get(label_key, "")) for row in rows]
    values = [float(row.get(metric_key, 0.0)) for row in rows]
    x = list(range(len(rows)))

    plt.figure(figsize=figsize)
    plt.bar(x, values, color=color)
    plt.xticks(x, labels, rotation=rotate_xticks, ha="right" if rotate_xticks else "center")
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_series_line_chart(
    x_labels: Sequence[str],
    series_map: Mapping[str, Sequence[float]],
    output_path: str,
    *,
    title: str,
    ylabel: str = "Score",
    ylim: Tuple[float, float] = (0.0, 1.0),
    figsize: Tuple[float, float] = (10.0, 5.0),
) -> str:
    if not x_labels:
        raise RuntimeError("No x_labels provided for line chart")
    if not series_map:
        raise RuntimeError("No series provided for line chart")

    x = list(range(len(x_labels)))
    plt.figure(figsize=figsize)
    palette = ["#3B82F6", "#F59E0B", "#10B981", "#EF4444", "#A855F7"]

    for idx, (name, values) in enumerate(series_map.items()):
        vals = [float(v) for v in values]
        if len(vals) != len(x_labels):
            raise RuntimeError(f"Series '{name}' length mismatch with x_labels")
        plt.plot(
            x,
            vals,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=name,
            color=palette[idx % len(palette)],
        )

    plt.xticks(x, list(x_labels))
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path
