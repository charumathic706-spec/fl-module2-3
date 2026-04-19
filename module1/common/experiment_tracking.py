from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from module1.common.report_images import save_series_line_chart


def _safe_git_sha(repo_root: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        sha = (result.stdout or "").strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def _safe_git_dirty(repo_root: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        return bool((result.stdout or "").strip())
    except Exception:
        return False


def write_run_manifest(
    *,
    repo_root: str,
    log_dir: str,
    run_config: Dict[str, Any],
    dataset_meta: Dict[str, Any],
    runtime_meta: Dict[str, Any],
    governance_output_dir: Optional[str] = None,
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    manifest = {
        "schema": "batfl.run-manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit_sha": _safe_git_sha(repo_root),
            "dirty_worktree": _safe_git_dirty(repo_root),
        },
        "dataset": dataset_meta,
        "run_config": run_config,
        "runtime": runtime_meta,
    }

    path = os.path.join(log_dir, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if governance_output_dir:
        os.makedirs(governance_output_dir, exist_ok=True)
        gov_copy = os.path.join(governance_output_dir, "run_manifest.json")
        with open(gov_copy, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return path


@dataclass
class RunSummary:
    label: str
    strategy: str
    attack: str
    backend: str
    rounds: int
    best_f1: float
    final_f1: float
    best_auc: float
    final_auc: float
    best_pr_auc: float
    final_pr_auc: float
    final_recall: float
    final_precision: float


def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_round_log(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def summarize_run(log_path: str, manifest_path: Optional[str] = None, label: Optional[str] = None) -> RunSummary:
    rounds = _load_round_log(log_path)
    if not rounds:
        raise RuntimeError(f"No rounds found in log: {log_path}")

    m = _load_manifest(manifest_path) if manifest_path and os.path.exists(manifest_path) else {}
    run_cfg = m.get("run_config", {}) if isinstance(m, dict) else {}

    final = rounds[-1]
    best_f1 = max(float(r.get("global_f1", 0.0)) for r in rounds)
    best_auc = max(float(r.get("global_auc", 0.0)) for r in rounds)
    best_pr_auc = max(float(r.get("global_pr_auc", 0.0)) for r in rounds)

    out_label = label or os.path.basename(os.path.dirname(log_path))
    return RunSummary(
        label=out_label,
        strategy=str(run_cfg.get("strategy", "trust_weighted")),
        attack=str(run_cfg.get("attack", "none")),
        backend=str(run_cfg.get("blockchain", run_cfg.get("blockchain_backend", "simulation"))),
        rounds=len(rounds),
        best_f1=best_f1,
        final_f1=float(final.get("global_f1", 0.0)),
        best_auc=best_auc,
        final_auc=float(final.get("global_auc", 0.0)),
        best_pr_auc=best_pr_auc,
        final_pr_auc=float(final.get("global_pr_auc", 0.0)),
        final_recall=float(final.get("global_recall", 0.0)),
        final_precision=float(final.get("global_precision", 0.0)),
    )


def write_baseline_comparison_report(
    summaries: List[RunSummary],
    output_dir: str,
    prefix: str = "baseline_comparison",
) -> Dict[str, str]:
    if not summaries:
        raise RuntimeError("No run summaries provided")

    trust_weighted_summaries = [
        s for s in summaries
        if str(s.strategy).strip().lower() in {"trustweighted", "trust_weighted"}
    ]
    if trust_weighted_summaries:
        summaries = trust_weighted_summaries

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{prefix}.csv")
    md_path = os.path.join(output_dir, f"{prefix}.md")

    headers = [
        "label", "strategy", "attack", "backend", "rounds",
        "best_f1", "final_f1", "best_auc", "final_auc",
        "best_pr_auc", "final_pr_auc", "final_recall", "final_precision",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for s in summaries:
            writer.writerow([
                s.label, s.strategy, s.attack, s.backend, s.rounds,
                f"{s.best_f1:.6f}", f"{s.final_f1:.6f}",
                f"{s.best_auc:.6f}", f"{s.final_auc:.6f}",
                f"{s.best_pr_auc:.6f}", f"{s.final_pr_auc:.6f}",
                f"{s.final_recall:.6f}", f"{s.final_precision:.6f}",
            ])

    lines = [
        "# BATFL Baseline Comparison Report",
        "",
        "| Label | Strategy | Attack | Backend | Rounds | Best F1 | Final F1 | Best AUC | Final AUC | Best PR-AUC | Final PR-AUC | Final Recall | Final Precision |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.label} | {s.strategy} | {s.attack} | {s.backend} | {s.rounds} "
            f"| {s.best_f1:.4f} | {s.final_f1:.4f} | {s.best_auc:.4f} | {s.final_auc:.4f} "
            f"| {s.best_pr_auc:.4f} | {s.final_pr_auc:.4f} | {s.final_recall:.4f} | {s.final_precision:.4f} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    def _group_key(attack_name: str) -> str:
        return "Without Attack" if str(attack_name).strip().lower() in {"", "none"} else "With Attack"

    grouped: Dict[str, List[RunSummary]] = {"Without Attack": [], "With Attack": []}
    for s in summaries:
        grouped[_group_key(s.attack)].append(s)

    def _avg(metric: str, items: List[RunSummary]) -> float:
        if not items:
            return float("nan")
        return sum(float(getattr(it, metric, 0.0)) for it in items) / len(items)

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

    line_path = os.path.join(output_dir, f"{prefix}_attack_vs_no_attack_line.png")
    save_series_line_chart(
        x_labels,
        series_map,
        line_path,
        title="Trust-Weighted Aggregation: With vs Without Attack",
        ylabel="Score",
        ylim=(0.0, 1.0),
        figsize=(10.0, 5.0),
    )

    return {
        "csv": csv_path,
        "markdown": md_path,
        "plot": line_path,
        "line_chart": line_path,
    }
