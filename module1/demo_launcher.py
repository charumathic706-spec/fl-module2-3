from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path


def _python_exe() -> str:
    return sys.executable


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("[demo]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _start_dashboard(log_dir: str, rounds: int, port: int) -> subprocess.Popen:
    cmd = [
        _python_exe(),
        "-m",
        "module1.dashboard_server",
        "--log",
        os.path.join(log_dir, "trust_training_log.json"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--expected_rounds",
        str(rounds),
    ]
    proc = subprocess.Popen(cmd)
    url = f"http://127.0.0.1:{port}"
    print(f"[demo] Dashboard started at {url}")
    webbrowser.open(url)
    return proc


def _bundle_report(run_dir: str, include_paths: list[str]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_root = Path(run_dir) / "report_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)

    for p in include_paths:
        src = Path(p)
        if not src.exists():
            continue
        dst = bundle_root / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    archive_base = str(Path(run_dir) / f"batfl_report_bundle_{ts}")
    zip_path = shutil.make_archive(archive_base, "zip", root_dir=str(bundle_root))
    print(f"[demo] Report bundle -> {zip_path}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-command BATFL demo launcher")
    parser.add_argument("--run_dir", type=str, default="logs_split2")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "dnn"])
    parser.add_argument("--attack", type=str, default="label_flip")
    parser.add_argument("--malicious", type=int, nargs="+", default=[1])
    parser.add_argument("--blockchain", type=str, default="simulation", choices=["simulation", "ganache", "fabric", "disabled"])
    parser.add_argument("--dashboard_port", type=int, default=5000)
    parser.add_argument("--event_storage", type=str, default="sqlite", choices=["jsonl", "sqlite"])
    parser.add_argument("--skip_dashboard", action="store_true")
    args = parser.parse_args()

    split2_cmd = [
        _python_exe(),
        "-m",
        "module1.split2.main",
        "--synthetic",
        "--rounds",
        str(args.rounds),
        "--num_clients",
        str(args.num_clients),
        "--model",
        args.model,
        "--attack",
        args.attack,
        "--event_storage",
        args.event_storage,
        "--log_dir",
        args.run_dir,
        "--malicious",
    ]
    split2_cmd.extend([str(cid) for cid in args.malicious])

    if args.blockchain == "disabled":
        split2_cmd.append("--no_blockchain")
    else:
        split2_cmd.extend(["--blockchain", args.blockchain])

    _run(split2_cmd)

    if args.blockchain != "disabled":
        _run([
            _python_exe(),
            "-m",
            "module1.split3.split3_main",
            "--blockchain",
            args.blockchain,
            "--audit_chain",
            "--output_dir",
            os.path.join(args.run_dir, "governance_output"),
        ])

    _run([
        _python_exe(),
        "-m",
        "module1.verify_run",
        "--run_dir",
        args.run_dir,
    ])

    dashboard_proc = None
    if not args.skip_dashboard:
        dashboard_proc = _start_dashboard(args.run_dir, args.rounds, args.dashboard_port)

    include_paths = [
        os.path.join(args.run_dir, "trust_training_log.json"),
        os.path.join(args.run_dir, "round_events.jsonl"),
        os.path.join(args.run_dir, "round_events.db"),
        os.path.join(args.run_dir, "run_manifest.json"),
        os.path.join(args.run_dir, "governance_output"),
    ]
    _bundle_report(args.run_dir, include_paths)

    if dashboard_proc is not None:
        print("[demo] Dashboard is running in background. Stop it manually when done.")


if __name__ == "__main__":
    main()
