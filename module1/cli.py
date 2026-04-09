from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


def _run_module(module_name: str, forwarded_args: List[str]) -> int:
    args = list(forwarded_args)
    if args and args[0] == "--":
        args = args[1:]
    cmd = [sys.executable, "-m", module_name, *args]
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batfl",
        description="BATFL unified CLI for training, verification, dashboard, and demo flows",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run one-command demo launcher")
    p_demo.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to module1.demo_launcher")

    p_verify = sub.add_parser("verify", help="Verify a run directory")
    p_verify.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to module1.verify_run")

    p_dash = sub.add_parser("dashboard", help="Start BATFL dashboard server")
    p_dash.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to module1.dashboard_server")

    p_s1 = sub.add_parser("train-split1", help="Run Split 1 baseline training")
    p_s1.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to module1.split1.main")

    p_s2 = sub.add_parser("train-split2", help="Run Split 2 trust-weighted training")
    p_s2.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to module1.split2.main")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    route = {
        "demo": "module1.demo_launcher",
        "verify": "module1.verify_run",
        "dashboard": "module1.dashboard_server",
        "train-split1": "module1.split1.main",
        "train-split2": "module1.split2.main",
    }

    module_name = route[args.command]
    forwarded_args = getattr(args, "args", [])
    code = _run_module(module_name, forwarded_args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()