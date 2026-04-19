from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List

from module1.common.report_images import save_series_line_chart


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _derive_run_summary(run_dir: str) -> Dict[str, Any] | None:
    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    split2_log = _read_json(os.path.join(run_dir, "trust_training_log.json"))
    split1_log = _read_json(os.path.join(run_dir, "training_log.json"))

    if split2_log and isinstance(split2_log, list):
        last = split2_log[-1]
        best_f1 = max(float(r.get("global_f1", 0.0)) for r in split2_log)
        best_pr_auc = max(float(r.get("global_pr_auc", 0.0)) for r in split2_log)
        strategy = (manifest or {}).get("strategy", "TrustWeighted")
        attack_type = ((manifest or {}).get("attack", {}) or {}).get("attack_type", "unknown")
        return {
            "run_dir": run_dir,
            "strategy": strategy,
            "attack_type": attack_type,
            "final_f1": float(last.get("global_f1", 0.0)),
            "final_auc": float(last.get("global_auc", 0.0)),
            "final_pr_auc": float(last.get("global_pr_auc", 0.0)),
            "final_recall": float(last.get("global_recall", 0.0)),
            "final_precision": float(last.get("global_precision", 0.0)),
            "best_f1": best_f1,
            "best_pr_auc": best_pr_auc,
        }

    if split1_log and isinstance(split1_log, list):
        last = split1_log[-1]
        best_f1 = max(float(r.get("global_f1", 0.0)) for r in split1_log)
        strategy = (manifest or {}).get("strategy", "FedAvg")
        attack_type = ((manifest or {}).get("attack", {}) or {}).get("attack_type", "none")
        return {
            "run_dir": run_dir,
            "strategy": strategy,
            "attack_type": attack_type,
            "final_f1": float(last.get("global_f1", 0.0)),
            "final_auc": float(last.get("global_auc", 0.0)),
            "final_pr_auc": float(last.get("global_pr_auc", 0.0)),
            "final_recall": float(last.get("global_recall", 0.0)),
            "final_precision": float(last.get("global_precision", 0.0)),
            "best_f1": best_f1,
            "best_pr_auc": max(float(r.get("global_pr_auc", 0.0)) for r in split1_log),
        }

    return None


def generate_baseline_comparison_report(run_dirs: List[str], output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        summary = _derive_run_summary(rd)
        if summary is not None:
            rows.append(summary)

    if not rows:
        raise RuntimeError("No valid run logs found for comparison report.")

    trust_rows = [
        row for row in rows
        if str(row.get("strategy", "")).strip().lower() in {"trustweighted", "trust_weighted"}
    ]
    if trust_rows:
        rows = trust_rows

    csv_path = os.path.join(output_dir, "baseline_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_dir",
                "strategy",
                "attack_type",
                "final_f1",
                "final_auc",
                "final_pr_auc",
                "final_recall",
                "final_precision",
                "best_f1",
                "best_pr_auc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    def _group_key(attack_type: str) -> str:
        return "Without Attack" if str(attack_type).strip().lower() in {"", "none"} else "With Attack"

    grouped: Dict[str, List[Dict[str, Any]]] = {"Without Attack": [], "With Attack": []}
    for row in rows:
        grouped[_group_key(str(row.get("attack_type", "")))].append(row)

    def _avg(metric: str, items: List[Dict[str, Any]]) -> float:
        if not items:
            return float("nan")
        return sum(float(it.get(metric, 0.0)) for it in items) / len(items)

    x_labels = ["Final F1", "Final AUC", "Final PR-AUC"]
    series_map: Dict[str, List[float]] = {}
    for name in ("Without Attack", "With Attack"):
        if not grouped[name]:
            continue
        series_map[name] = [
            _avg("final_f1", grouped[name]),
            _avg("final_auc", grouped[name]),
            _avg("final_pr_auc", grouped[name]),
        ]

    line_path = os.path.join(output_dir, "trust_weighted_attack_vs_no_attack_line.png")
    save_series_line_chart(
        x_labels,
        series_map,
        line_path,
        title="Trust-Weighted Aggregation: With vs Without Attack",
        ylabel="Score",
        ylim=(0.0, 1.0),
        figsize=(10.0, 5.0),
    )

    json_path = os.path.join(output_dir, "baseline_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)

    return {
        "csv": csv_path,
        "plot": line_path,
        "json": json_path,
        "line_chart": line_path,
    }
