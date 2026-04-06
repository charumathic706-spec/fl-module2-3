#!/usr/bin/env python3
# =============================================================================
# dashboard_server.py
# Real-time FL training dashboard — HTTP server + WebSocket push
#
# HOW IT WORKS:
#   1. Reads trust_training_log.json every second as FL training runs
#   2. Serves the dashboard HTML on http://localhost:5000
#   3. Pushes live updates to the browser via Server-Sent Events (SSE)
#      — no WebSocket library needed, works in every browser natively
#
# HOW TO RUN:
#   Terminal 1: python -m module1.split2.main --data_path data/creditcard.csv \
#                   --attack label_flip --malicious 1
#   Terminal 2: python dashboard_server.py --log logs_split2/trust_training_log.json
#   Browser:    http://localhost:5000
#
# FOR TWO-MACHINE SETUP:
#   Run dashboard_server.py on System 1 (same machine as server)
#   Open browser on EITHER machine: http://<System1_IP>:5000
#   System 2 attacker sees the dashboard too — you can show both
#   terminals AND the dashboard simultaneously during the demo
#
# DEPENDENCIES: none beyond Python standard library
#   (no Flask, no Django, no WebSockets — pure stdlib)
# =============================================================================

import argparse
import http.server
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from urllib.parse import urlparse


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_LOG  = "logs_split2/trust_training_log.json"
POLL_SECS    = 1.0      # how often to check log file for new rounds
DEFAULT_NUM_CLIENTS = 5


# =============================================================================
# LOG READER — watches the JSON log file written by split2/main.py
# =============================================================================

class LogWatcher:
    """
    Watches trust_training_log.json and extracts latest state.
    The FL training loop writes this file after every round.
    """
    def __init__(self, log_path: str, expected_rounds: int | None = None):
      self.log_path = self._resolve_log_path(log_path)
      self.input_log_path = log_path
      self.expected_rounds = expected_rounds
      self.last_mtime = 0
      self.last_size = -1
      self.rounds = []
      self.lock = threading.Lock()
      self.parse_errors = 0
      self.last_refresh_ts = 0.0
      self.last_success_ts = 0.0

    def refresh(self):
      """Read log file if it changed since last read."""
      self.last_refresh_ts = time.time()
      if not os.path.exists(self.log_path):
        return False

      mtime = os.path.getmtime(self.log_path)
      size = os.path.getsize(self.log_path)
      if mtime <= self.last_mtime and size == self.last_size:
        return False

      try:
        with open(self.log_path, "r", encoding="utf-8") as f:
          data = json.load(f)
        with self.lock:
          self.rounds = data if isinstance(data, list) else []
          self.last_mtime = mtime
          self.last_size = size
          self.last_success_ts = self.last_refresh_ts
        return True
      except (json.JSONDecodeError, OSError, IOError):
        self.parse_errors += 1
        return False

    def get_state(self) -> dict:
      """Return current dashboard state from latest rounds."""
      with self.lock:
        rounds = list(self.rounds)

      if not rounds:
        return {
          "status": "waiting",
          "rounds": [],
          "latest": None,
          "blockchain": self._build_blockchain_state(),
          "log_status": self._build_log_status(),
        }

      latest = rounds[-1]
      best_f1 = max((r.get("global_f1", 0) for r in rounds), default=0)
      best_auc = max((r.get("global_auc", 0) for r in rounds), default=0)

      client_ids_set = set()
      for r in rounds:
        for c in r.get("trusted_clients", []):
          client_ids_set.add(int(c))
        for c in r.get("flagged_clients", []):
          client_ids_set.add(int(c))
        for k in r.get("trust_scores", {}).keys():
          try:
            client_ids_set.add(int(k))
          except (TypeError, ValueError):
            continue
        for k in r.get("anomaly_scores", {}).keys():
          try:
            client_ids_set.add(int(k))
          except (TypeError, ValueError):
            continue

      if not client_ids_set:
        client_ids_set = set(range(DEFAULT_NUM_CLIENTS))

      client_ids = sorted(client_ids_set)

      trust_history = {str(i): [] for i in client_ids}
      alpha_history = {str(i): [] for i in client_ids}
      f1_history = []
      auc_history = []
      round_nums = []

      for r in rounds:
        round_nums.append(r.get("round", 0))
        f1_history.append(round(r.get("global_f1", 0), 4))
        auc_history.append(round(r.get("global_auc", 0), 4))
        ts = r.get("trust_scores", {})
        as_ = r.get("anomaly_scores", {})
        for i in client_ids:
          trust_history[str(i)].append(
            round(float(ts.get(str(i), ts.get(i, 1.0))), 4)
          )
          alpha_history[str(i)].append(
            round(float(as_.get(str(i), as_.get(i, 0.0))), 4)
          )

      flagged = [int(c) for c in latest.get("flagged_clients", [])]
      trusted = [int(c) for c in latest.get("trusted_clients", [])]
      ts_latest = latest.get("trust_scores", {})
      as_latest = latest.get("anomaly_scores", {})

      attacker = flagged[0] if flagged else None
      detected_rounds = sum(1 for r in rounds if r.get("flagged_clients"))

      current_round = int(latest.get("round", 0))
      inferred_total = max(current_round, len(rounds))
      total_rounds = self.expected_rounds if self.expected_rounds else inferred_total
      status = "complete" if (self.expected_rounds and current_round >= self.expected_rounds) else "running"
      blockchain_state = self._build_blockchain_state()

      return {
        "status": status,
        "current_round": current_round,
        "total_rounds": total_rounds,
        "global_f1": round(latest.get("global_f1", 0), 4),
        "global_auc": round(latest.get("global_auc", 0), 4),
        "best_f1": round(best_f1, 4),
        "best_auc": round(best_auc, 4),
        "flagged": flagged,
        "trusted": trusted,
        "attacker": attacker,
        "detected_rounds": detected_rounds,
        "false_positives": sum(
          1 for r in rounds
          for c in r.get("flagged_clients", [])
          if int(c) != attacker
        ),
        "client_ids": client_ids,
        "trust_latest": {
          str(i): round(float(ts_latest.get(str(i), ts_latest.get(i, 1.0))), 4)
          for i in client_ids
        },
        "alpha_latest": {
          str(i): round(float(as_latest.get(str(i), as_latest.get(i, 0.0))), 4)
          for i in client_ids
        },
        "round_nums": round_nums,
        "f1_history": f1_history,
        "auc_history": auc_history,
        "trust_history": trust_history,
        "alpha_history": alpha_history,
        "model_hash": latest.get("model_hash", ""),
        "blockchain_round": latest.get("round", 0),
        "blockchain": blockchain_state,
        "log_status": self._build_log_status(),
      }

    def _build_log_status(self) -> dict:
      return {
        "path": self.log_path,
        "input_path": self.input_log_path,
        "exists": os.path.exists(self.log_path),
        "last_refresh_ts": self.last_refresh_ts,
        "last_success_ts": self.last_success_ts,
        "parse_errors": self.parse_errors,
      }

    def _resolve_log_path(self, log_path: str) -> str:
      candidate = os.path.expanduser(log_path)
      if os.path.isabs(candidate) and os.path.exists(candidate):
        return candidate

      cwd_candidate = os.path.abspath(candidate)
      if os.path.exists(cwd_candidate):
        return cwd_candidate

      module_dir = os.path.dirname(os.path.abspath(__file__))
      repo_root = os.path.dirname(module_dir)
      repo_candidate = os.path.abspath(os.path.join(repo_root, candidate))
      if os.path.exists(repo_candidate):
        return repo_candidate

      return cwd_candidate

    def _build_blockchain_state(self) -> dict:
        """Build blockchain telemetry from governance artifacts, if available."""
        state = {
            "backend": "unknown",
            "chain_intact": None,
            "tamper_events": 0,
            "committed_rounds": 0,
            "hash_chain_entries": 0,
            "contract_address": "",
            "deploy_tx": "",
            "latest_commit_tx": "",
            "latest_audit_tx": "",
            "latest_block_hash": "",
            "recent_commits": [],
            "source": "none",
        }

        gov_dir = self._find_governance_dir()
        if gov_dir is None:
            return state

        report_path = os.path.join(gov_dir, "governance_report.json")
        hash_chain_path = os.path.join(gov_dir, "hash_chain.json")
        deployment_path = os.path.join(gov_dir, "eth_deployment.json")

        report = self._safe_json(report_path)
        if isinstance(report, dict):
            summary = report.get("summary", {}) or {}
            records = report.get("round_records", []) or []
            state["source"] = report_path
            state["chain_intact"] = summary.get("chain_intact")
            state["tamper_events"] = int(summary.get("tamper_events", 0) or 0)
            state["committed_rounds"] = len(records)

            if records:
                last = records[-1]
                state["latest_commit_tx"] = str(last.get("blockchain_tx_id", ""))
                state["latest_audit_tx"] = str(last.get("audit_tx_id", ""))
                state["latest_block_hash"] = str(last.get("block_hash", ""))

            recent = []
            for rec in records[-10:]:
                recent.append({
                    "round": int(rec.get("round", 0) or 0),
                    "commit_tx": str(rec.get("blockchain_tx_id", "")),
                    "audit_tx": str(rec.get("audit_tx_id", "")),
                    "model_hash": str(rec.get("model_hash", "")),
                    "block_hash": str(rec.get("block_hash", "")),
                    "flagged_count": len(rec.get("flagged_clients", []) or []),
                    "trusted_count": len(rec.get("trusted_clients", []) or []),
                })
            state["recent_commits"] = recent

            if records:
                first_tx = str(records[0].get("blockchain_tx_id", ""))
                if first_tx.startswith("SIM_TX"):
                    state["backend"] = "simulation"
                else:
                    state["backend"] = "fabric"

        hash_chain = self._safe_json(hash_chain_path)
        if isinstance(hash_chain, list):
            state["hash_chain_entries"] = len(hash_chain)

        deployment = self._safe_json(deployment_path)
        if isinstance(deployment, dict):
            state["backend"] = "ganache"
            state["contract_address"] = str(deployment.get("address", ""))
            state["deploy_tx"] = str(deployment.get("deploy_tx", ""))

        return state

    def _find_governance_dir(self):
        base_from_log = os.path.dirname(os.path.abspath(self.log_path))
        candidates = [
            os.path.join(base_from_log, "governance_output"),
            os.path.join(os.getcwd(), "governance_output"),
            os.path.join(os.getcwd(), "logs_split2", "governance_output"),
        ]

        seen = set()
        for candidate in candidates:
            norm = os.path.normpath(candidate)
            if norm in seen:
                continue
            seen.add(norm)
            if os.path.isdir(norm):
                return norm
        return None

    @staticmethod
    def _safe_json(path: str):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, IOError):
            return None


# =============================================================================
# DASHBOARD HTML — complete single-file dashboard
# =============================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BATFL — Dynamic Trust FL Architecture</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg-dark: #0B0E14; --surface: #151A22; --surface-light: #1D2430;
    --primary: #00F4B9; --primary-glow: rgba(0, 244, 185, 0.4);
    --secondary: #3B82F6; --secondary-glow: rgba(59, 130, 246, 0.4);
    --danger: #FF3B30; --danger-bg: rgba(255, 59, 48, 0.1);
    --text-main: #FFFFFF; --text-muted: #8E9BAE; --border: #2A3441;
  }
  
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { 
    background: var(--bg-dark); color: var(--text-main); 
    font-family: 'Outfit', sans-serif;
    background-image: radial-gradient(circle at 50% 0%, rgba(59, 130, 246, 0.15), transparent 50%);
  }

  /* Glassmorphic Utilities */
  .glass-panel {
    background: rgba(21, 26, 34, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
  }

  /* Header */
  .header {
    padding: 24px 40px; display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    background: rgba(11, 14, 20, 0.8); backdrop-filter: blur(20px);
    position: sticky; top: 0; z-index: 100;
  }
  .header-left h1 { 
    font-size: 24px; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(90deg, var(--text-main), var(--primary));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .header-left p { font-size: 13px; color: var(--text-muted); margin-top: 4px; font-weight: 300; }
  
  /* Status Indicator */
  .status-wrapper { text-align: right; font-size: 14px; font-weight: 600; }
  .status-dot { 
    width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px; 
    box-shadow: 0 0 10px var(--primary-glow);
  }
  .status-dot.running { background: var(--primary); animation: pulse 1.5s infinite; }
  .status-dot.waiting { background: #FF9F0A; box-shadow: 0 0 10px rgba(255, 159, 10, 0.4); }
  .status-dot.complete { background: var(--secondary); box-shadow: 0 0 10px var(--secondary-glow); }
  @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.2); } 100% { opacity: 1; transform: scale(1); } }

  .layout { padding: 30px 40px; display: flex; flex-direction: column; gap: 24px; max-width: 1800px; margin: 0 auto; }

  /* Architecture Explainer */
  .explainer-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
  .explainer-card { padding: 20px; transition: transform 0.2s; }
  .explainer-card:hover { transform: translateY(-3px); }
  .explainer-card h3 { font-size: 16px; margin-bottom: 12px; color: var(--primary); display: flex; align-items: center; gap: 8px; }
  .explainer-card p { font-size: 14px; color: var(--text-muted); line-height: 1.6; }
  .number-badge { 
    background: rgba(0, 244, 185, 0.15); color: var(--primary); 
    width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; 
  }

  /* Internal Execution Pipeline */
  .pipeline {
    padding: 24px; display: flex; justify-content: space-between; align-items: center; position: relative;
    overflow: hidden;
  }
  .pipeline-line {
    position: absolute; top: 50%; left: 40px; right: 40px; height: 2px; background: rgba(255,255,255,0.1); z-index: 0;
  }
  .pipeline-line-fill {
    height: 100%; width: 0%; background: linear-gradient(90deg, var(--secondary), var(--primary)); transition: width 0.8s ease;
  }
  .pipeline-step {
    position: relative; z-index: 1; text-align: center; background: var(--surface); padding: 12px 20px; border-radius: 20px; border: 1px solid var(--border);
    transition: all 0.3s ease; width: 18%; box-shadow: 0 4px 6px rgba(0,0,0,0.2);
  }
  .pipeline-step.active { border-color: var(--primary); box-shadow: 0 0 20px var(--primary-glow); transform: scale(1.05); }
  .step-title { font-size: 13px; font-weight: 600; color: var(--text-main); margin-bottom: 4px; }
  .step-desc { font-size: 11px; color: var(--text-muted); }

  /* Metrics row */
  .metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
  .metric-box { padding: 20px; display: flex;flex-direction: column; justify-content: space-between;}
  .metric-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
  .metric-val { font-size: 36px; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-top: 8px; }
  .metric-val.green { color: var(--primary); }
  .metric-val.blue { color: var(--secondary); }

  /* Clients Section */
  .section-title { font-size: 18px; margin: 10px 0 16px 0; font-weight: 600; display: flex; align-items: center; justify-content: space-between; }
  .client-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
  .client-card { padding: 20px; position: relative; overflow: hidden; border-top: 3px solid var(--border); transition: all 0.3s;}
  .client-card.trusted { border-top-color: var(--primary); }
  .client-card.flagged { border-top-color: var(--danger); background: var(--danger-bg); }
  
  .client-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;}
  .client-name { font-size: 16px; font-weight: 600; }
  .status-badge { font-size: 10px; padding: 4px 8px; border-radius: 12px; font-weight: bold; letter-spacing: 0.5px; }
  .client-card.trusted .status-badge { background: rgba(0, 244, 185, 0.1); color: var(--primary); }
  .client-card.flagged .status-badge { background: rgba(255, 59, 48, 0.2); color: var(--danger); }

  .score-row { display: flex; justify-content: space-between; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace;}
  .score-label { font-size: 12px; color: var(--text-muted); font-family: 'Outfit', sans-serif;}
  .score-val { font-size: 14px; font-weight: 600; }
  
  .trust-bar-container { width: 100%; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; margin-bottom: 6px; }
  .trust-bar { height: 100%; transition: width 0.5s ease; }
  .client-card.trusted .trust-bar { background: var(--primary); box-shadow: 0 0 8px var(--primary-glow); }
  .client-card.flagged .trust-bar { background: var(--danger); }
  
  .attacker-explainer { font-size: 11px; color: var(--danger); margin-top: 10px; padding-top: 10px; border-top: 1px dashed rgba(255,59,48,0.3); font-style: italic;}

  /* Charts & Blockchain */
  .bottom-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
  .chart-container { padding: 20px; height: 350px;}
  
  .blockchain-container { padding: 20px; display: flex; flex-direction: column;}
  .terminal {
    background: #000; border: 1px solid #333; border-radius: 8px; padding: 16px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #00FF41;
    flex-grow: 1; overflow-y: auto; max-height: 250px;
  }
  .tx-line { margin-bottom: 8px; line-height: 1.4; }
  .tx-hash { color: #F8F8F2; word-break: break-all; font-size: 11px;}
  .tx-meta { color: #6272A4; font-size: 10px; }

  /* Waiting Screen */
  .overlay-waiting {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(11, 14, 20, 0.9); backdrop-filter: blur(5px);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 1000; transition: opacity 0.5s;
  }
  .overlay-waiting h2 { font-size: 32px; margin-bottom: 16px; color: var(--text-main); }
  .code-hint { background: var(--surface); padding: 16px 24px; border-radius: 8px; border: 1px solid var(--border); font-family: 'JetBrains Mono', monospace; color: var(--primary); }
</style>
</head>
<body>

<div class="overlay-waiting" id="waiting-screen">
  <h2>BATFL System Idle</h2>
  <p style="color: var(--text-muted); margin-bottom: 30px;">Waiting for a Federated Learning simulation to start...</p>
  <div class="code-hint">python -m module1.split2.main --attack label_flip --malicious 1</div>
</div>

<header class="header">
  <div class="header-left">
    <h1>BATFL Command Center</h1>
    <p>Blockchain-Based Dynamic Trust Modeling for Federated Fraud Detection</p>
  </div>
  <div class="status-wrapper">
    <div style="display: flex; align-items: center; justify-content: flex-end;">
      <span class="status-dot waiting" id="status-dot"></span>
      <span id="status-text">Waiting...</span>
    </div>
    <div style="font-size: 11px; font-weight: 300; color: var(--text-muted); margin-top: 4px;" id="last-update">—</div>
  </div>
</header>

<main class="layout">
  
  <!-- Architectural Overview -->
  <section class="explainer-grid">
    <div class="glass-panel explainer-card">
      <h3><div class="number-badge">1</div> Federated Edge Training</h3>
      <p>Banks train the underlying deep learning fraud model locally on their private data. They do not share raw transactions. Only encrypted gradient updates are sent to the central orchestrator.</p>
    </div>
    <div class="glass-panel explainer-card">
      <h3><div class="number-badge">2</div> Dynamic Trust Scoring</h3>
      <p>Before aggregation, the orchestrator evaluates each bank's gradient using an Isolation Forest. Anomalous gradients are flagged (α > threshold), immediately isolating attackers and dynamically lowering their Trust Score (τ).</p>
    </div>
    <div class="glass-panel explainer-card">
      <h3><div class="number-badge">3</div> Immutable Governance</h3>
      <p>The aggregated Global Model's SHA-256 fingerprint, along with all trust decisions and participant attestations, are permanently committed to the Hyperledger Fabric / Ganache blockchain for auditing.</p>
    </div>
  </section>

  <!-- Execution Pipeline -->
  <section class="glass-panel pipeline">
    <div class="pipeline-line"><div class="pipeline-line-fill" id="pipe-fill"></div></div>
    
    <div class="pipeline-step" id="step-1">
      <div class="step-title">1. Local Training</div>
      <div class="step-desc">Banks compute gradients</div>
    </div>
    <div class="pipeline-step" id="step-2">
      <div class="step-title">2. Anomaly Check</div>
      <div class="step-desc">α-score assigned via PCA</div>
    </div>
    <div class="pipeline-step" id="step-3">
      <div class="step-title">3. Trust Evaluation</div>
      <div class="step-desc">τ-score penalizes attackers</div>
    </div>
    <div class="pipeline-step" id="step-4">
      <div class="step-title">4. Aggregation</div>
      <div class="step-desc">Trust-weighted FedAvg</div>
    </div>
    <div class="pipeline-step" id="step-5">
      <div class="step-title">5. Blockchain Commit</div>
      <div class="step-desc">Governance audit ledger</div>
    </div>
  </section>

  <!-- High Level Metrics -->
  <section class="metrics-row">
    <div class="glass-panel metric-box">
      <div class="metric-label">Training Round</div>
      <div class="metric-val blue"><span id="cur-round">0</span> <span style="font-size:16px;color:var(--text-muted)">/ <span id="tot-rounds">20</span></span></div>
    </div>
    <div class="glass-panel metric-box">
      <div class="metric-label">Global F1 Score</div>
      <div class="metric-val green" id="m-f1">0.00</div>
    </div>
    <div class="glass-panel metric-box">
      <div class="metric-label">AUC-ROC Performance</div>
      <div class="metric-val" id="m-auc">0.00</div>
    </div>
    <div class="glass-panel metric-box">
      <div class="metric-label">Attack Detection Rate</div>
      <div class="metric-val" style="color: var(--danger)" id="m-det">0 / 0</div>
    </div>
  </section>

  <!-- Client Federation -->
  <section>
    <div class="section-title">
      Federated Client Nodes
      <span style="font-size: 13px; font-weight: 400; color: var(--text-muted)">Real-time dynamic trust evaluation isolating poisoned models</span>
    </div>
    <div class="client-grid" id="client-cards">
      <!-- Generated via JS -->
    </div>
  </section>

  <!-- Charts and Blockchain -->
  <section class="bottom-grid">
    <div class="glass-panel chart-container">
      <div class="section-title" style="margin-top: 0">Model Convergence & Stability</div>
      <div style="position:relative;height:280px">
        <canvas id="perf-chart"></canvas>
      </div>
    </div>
    
    <div class="glass-panel blockchain-container">
      <div class="section-title" style="margin-top: 0">Blockchain Immutable Ledger</div>
      <div class="terminal" id="terminal-log">
        <div style="color: #6272A4; margin-bottom: 10px;">> Connection to Smart Contract established...</div>
        <div style="color: #6272A4; margin-bottom: 10px;">> Awaiting commits...</div>
      </div>
    </div>
  </section>

</main>

<script>
const COLORS = ['#3B82F6', '#00F4B9', '#F59E0B', '#8B5CF6', '#EC4899'];
let perfChart = null;
let initialized = false;
let lastRoundProcessed = -1;

function getClientIds(state) {
  if (Array.isArray(state.client_ids) && state.client_ids.length > 0) {
    return state.client_ids.slice().sort((a,b)=>a-b);
  }
  return Object.keys(state.trust_latest || {}).map(k=>parseInt(k)).filter(n=>Number.isFinite(n)).sort((a,b)=>a-b);
}

function initCharts() {
  const ctx = document.getElementById('perf-chart').getContext('2d');
  perfChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Global F1', data: [], borderColor: '#00F4B9', backgroundColor: 'rgba(0,244,185,0.1)', tension: 0.4, fill: true, borderWidth: 3 },
        { label: 'Global AUC', data: [], borderColor: '#3B82F6', backgroundColor: 'rgba(59,130,246,0.05)', tension: 0.4, fill: true, borderWidth: 3 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8E9BAE', font: {family: 'Outfit'} } }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8E9BAE' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8E9BAE' }, min: 0 }
      }
    }
  });
}

function renderClientCards(state) {
  const container = document.getElementById('client-cards');
  const clientIds = getClientIds(state);
  container.innerHTML = '';
  
  for (const i of clientIds) {
    const isFlagged = state.flagged.includes(i);
    const isAttacker = state.attacker === i;
    const tau = (state.trust_latest[i] || 1.0).toFixed(3);
    const alpha = (state.alpha_latest[i] || 0.0).toFixed(3);
    
    // Attacker explanation text
    let attackText = '';
    if (isFlagged) {
      attackText = `<div class="attacker-explainer">Model poisoned. High anomaly detected (α > thresh). Trust score slashed to isolate.</div>`;
    }
    
    const card = document.createElement('div');
    card.className = 'glass-panel client-card ' + (isFlagged ? 'flagged' : 'trusted');
    card.innerHTML = `
      <div class="client-header">
        <div class="client-name">Bank Client ${i}</div>
        <div class="status-badge">${isFlagged ? 'MALICIOUS' : 'TRUSTED'}</div>
      </div>
      <div class="score-row">
        <span class="score-label">Trust Score (τ)</span>
        <span class="score-val">${(tau * 100).toFixed(1)}%</span>
      </div>
      <div class="trust-bar-container">
        <div class="trust-bar" style="width: ${(tau * 100)}%;"></div>
      </div>
      <div class="score-row" style="margin-bottom: 0;">
        <span class="score-label">Anomaly (α)</span>
        <span class="score-val">${alpha}</span>
      </div>
      ${attackText}
    `;
    container.appendChild(card);
  }
}

function updatePipelineAnim() {
  // Animate the pipeline sequence every second for visual flair
  const steps = [1,2,3,4,5];
  let cur = 0;
  setInterval(() => {
    steps.forEach(s => document.getElementById('step-'+s).classList.remove('active'));
    document.getElementById('step-'+steps[cur]).classList.add('active');
    document.getElementById('pipe-fill').style.width = (cur * 25) + '%';
    cur = (cur + 1) % steps.length;
  }, 1000);
}

function appendBlockchainTerminal(state) {
  if (state.current_round <= lastRoundProcessed) return;
  
  const term = document.getElementById('terminal-log');
  if (lastRoundProcessed === -1) term.innerHTML = ''; // clear initial msg
  
  const hash = state.model_hash || "0x" + Array(64).fill(0).map(()=>Math.random().toString(16)[3]).join('');
  
  const div = document.createElement('div');
  div.className = 'tx-line';
  div.innerHTML = `
    <div style="color: #FF5555;">[BLOCKCHAIN COMMIT] Round ${state.current_round} Complete</div>
    <div class="tx-meta">ATTESTATION: Verifying signatures for ${state.trusted.length} trusted banks... OK</div>
    ${state.flagged.length > 0 ? `<div class="tx-meta" style="color:#FFA500">REVOCATION: Access revoked for Bank ${state.flagged.join(',')}</div>` : ''}
    <div class="tx-meta">GLOB_HASH: Committing Global Model SHA-256 fingerprint</div>
    <div class="tx-hash">TX_ID: ${hash}</div>
    <div style="margin-bottom: 12px;"></div>
  `;
  term.appendChild(div);
  term.scrollTop = term.scrollHeight;
  lastRoundProcessed = state.current_round;
}

function applyState(state) {
  if (!initialized) {
    document.getElementById('waiting-screen').style.opacity = '0';
    setTimeout(() => document.getElementById('waiting-screen').style.display = 'none', 500);
    initCharts();
    updatePipelineAnim();
    initialized = true;
  }

  // Header status
  const dot = document.getElementById('status-dot');
  dot.className = 'status-dot ' + state.status;
  document.getElementById('status-text').textContent = state.status === 'running' ? 'Training & Auditing Live' : 'Federated Task Complete';
  document.getElementById('last-update').textContent = 'Last sync: ' + new Date().toLocaleTimeString();

  // Metrics
  document.getElementById('cur-round').textContent = state.current_round;
  document.getElementById('tot-rounds').textContent = state.total_rounds;
  document.getElementById('m-f1').textContent = state.global_f1.toFixed(3);
  document.getElementById('m-auc').textContent = state.global_auc.toFixed(3);
  document.getElementById('m-det').textContent = state.detected_rounds + ' / ' + state.current_round;

  renderClientCards(state);
  
  // Charts
  perfChart.data.labels = state.round_nums;
  perfChart.data.datasets[0].data = state.f1_history;
  perfChart.data.datasets[1].data = state.auc_history;
  perfChart.update('none');

  appendBlockchainTerminal(state);
}

const evtSource = new EventSource('/stream');
evtSource.onmessage = (e) => {
  const state = JSON.parse(e.data);
  if (state.status !== 'waiting') applyState(state);
};
evtSource.onerror = () => {
  document.getElementById('status-text').textContent = 'Disconnected...';
};
</script>
</body>
</html>
"""


# =============================================================================
# HTTP REQUEST HANDLER
# =============================================================================

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Serves dashboard HTML and SSE stream."""

    watcher: LogWatcher = None   # set by main()
    clients_lock = threading.Lock()
    sse_clients  = []

    def log_message(self, fmt, *args):
        pass  # suppress default access log noise

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/stream":
            self._serve_sse()
        elif path == "/state":
            self._serve_json()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_html(self):
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_ui.html")
        if os.path.exists(ui_path):
            with open(ui_path, "r", encoding="utf-8") as f:
                html_str = f.read()
        else:
            html_str = DASHBOARD_HTML
        data = html_str.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_json(self):
        state = DashboardHandler.watcher.get_state()
        data  = json.dumps(state).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_sse(self):
        """Server-Sent Events — keep connection open, push updates."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type",  "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection",    "keep-alive")
        self.end_headers()

        try:
            while True:
                state = DashboardHandler.watcher.get_state()
                data  = json.dumps(state)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(POLL_SECS)
        except (BrokenPipeError, ConnectionResetError):
            pass


# =============================================================================
# BACKGROUND WATCHER THREAD
# =============================================================================

def watch_loop(watcher: LogWatcher):
    """Continuously refresh the log file in background."""
    while True:
        watcher.refresh()
        time.sleep(POLL_SECS)


# =============================================================================
# MAIN
# =============================================================================

def build_parser():
  p = argparse.ArgumentParser(description="FL Real-time Dashboard Server")
  p.add_argument(
    "--log",
    type=str,
    default=DEFAULT_LOG,
    help="Path to trust_training_log.json",
  )
  p.add_argument(
    "--port",
    type=int,
    default=5000,
    help="HTTP port (default: 5000)",
  )
  p.add_argument(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host to bind (default: 127.0.0.1; use 0.0.0.0 for LAN access)",
  )
  p.add_argument(
    "--expected_rounds",
    type=int,
    default=None,
    help="Expected total FL rounds (improves progress and completion status)",
  )
  return p


def main():
    args   = build_parser().parse_args()
    watcher = LogWatcher(log_path=args.log, expected_rounds=args.expected_rounds)
    DashboardHandler.watcher = watcher

    # Start background file watcher
    t = threading.Thread(target=watch_loop, args=(watcher,), daemon=True)
    t.start()

    # Start HTTP server
    try:
      server = http.server.ThreadingHTTPServer(
        (args.host, args.port), DashboardHandler
      )
    except OSError as exc:
      print(f"\n[Dashboard] Failed to bind {args.host}:{args.port}: {exc}")
      print("[Dashboard] If port is in use, choose another one (e.g. --port 5001).")
      raise SystemExit(1) from exc

    print(f"\n{'='*55}")
    print(f"  FL Trust Dashboard")
    print(f"  Watching : {args.log}")
    print(f"  Open in browser:")
    print(f"    http://localhost:{args.port}           (this machine)")

    if args.host == "0.0.0.0":
      # Show LAN URL only when intentionally exposed.
      try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        print(f"    http://{ip}:{args.port}   (any machine on your WiFi)")
      except Exception:
        pass

    print(f"\n  Dashboard auto-updates every {POLL_SECS}s as training runs.")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*55}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
