"""
Phase 5 Step 2 - automated Locust sweep runner.

Runs load tests for user counts 1, 5, 10, 20 and records throughput trend.

Example:
    python benchmarks/load_test_runner.py --host http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "load-testing"
LOCUSTFILE = ROOT / "benchmarks" / "locustfile.py"

DEFAULT_USERS = [1, 5, 10, 20]


def _run_locust_once(
    host: str,
    users: int,
    spawn_rate: float,
    duration: str,
    out_dir: Path,
) -> dict:
    prefix = out_dir / f"users_{users}"
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE),
        "--host",
        host,
        "--users",
        str(users),
        "--spawn-rate",
        str(spawn_rate),
        "--run-time",
        duration,
        "--headless",
        "--only-summary",
        "--csv",
        str(prefix),
    ]
    print(f"[load-test] Running users={users} duration={duration} ...")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stats_path = Path(f"{prefix}_stats.csv")
    failures_path = Path(f"{prefix}_failures.csv")

    if proc.returncode != 0:
        return {
            "users": users,
            "status": "failed",
            "error": (proc.stderr or proc.stdout)[-1200:],
            "stats_csv": str(stats_path),
            "failures_csv": str(failures_path),
        }

    metrics = _extract_generate_metrics(stats_path)
    metrics.update(
        {
            "users": users,
            "status": "ok",
            "stats_csv": str(stats_path),
            "failures_csv": str(failures_path),
        }
    )
    return metrics


def _extract_generate_metrics(stats_csv: Path) -> dict:
    if not stats_csv.exists():
        return {
            "generate_rps": None,
            "generate_avg_ms": None,
            "generate_p95_ms": None,
            "generate_fail_ratio": None,
            "total_rps": None,
        }

    with open(stats_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Aggregate row contains Name="Aggregated"
    agg = next((r for r in rows if r.get("Name") == "Aggregated"), None)
    gen = next((r for r in rows if r.get("Name") == "POST /generate"), None)

    def _to_float(row: dict | None, key: str):
        if not row:
            return None
        try:
            return float(row[key])
        except Exception:
            return None

    total_reqs = _to_float(gen, "Request Count")
    total_fails = _to_float(gen, "Failure Count")
    fail_ratio = None
    if total_reqs is not None and total_reqs > 0 and total_fails is not None:
        fail_ratio = total_fails / total_reqs

    return {
        "generate_rps": _to_float(gen, "Requests/s"),
        "generate_avg_ms": _to_float(gen, "Average Response Time"),
        "generate_p95_ms": _to_float(gen, "95%"),
        "generate_fail_ratio": fail_ratio,
        "total_rps": _to_float(agg, "Requests/s"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Locust load sweep for 1/5/10/20 users")
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="Target API host")
    parser.add_argument(
        "--users",
        default="1,5,10,20",
        help="Comma-separated user counts, e.g. 1,5,10,20",
    )
    parser.add_argument("--duration", default="60s", help="Locust runtime per sweep point")
    parser.add_argument("--spawn-rate", type=float, default=2.0, help="Users per second")
    args = parser.parse_args()

    user_counts = [int(x.strip()) for x in args.users.split(",") if x.strip()]
    if not user_counts:
        user_counts = DEFAULT_USERS

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "duration": args.duration,
        "spawn_rate": args.spawn_rate,
        "users": user_counts,
        "results": [],
    }

    for users in user_counts:
        result = _run_locust_once(
            host=args.host,
            users=users,
            spawn_rate=args.spawn_rate,
            duration=args.duration,
            out_dir=out_dir,
        )
        run_manifest["results"].append(result)
        status = result.get("status")
        print(
            f"[load-test] users={users} status={status} "
            f"generate_rps={result.get('generate_rps')} "
            f"p95={result.get('generate_p95_ms')}"
        )

    # Save detailed JSON
    summary_json = out_dir / "load_test_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    # Save compact CSV trend
    trend_csv = out_dir / "throughput_trend.csv"
    with open(trend_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "users",
                "status",
                "generate_rps",
                "generate_avg_ms",
                "generate_p95_ms",
                "generate_fail_ratio",
                "total_rps",
            ],
        )
        writer.writeheader()
        for row in run_manifest["results"]:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    print(f"\n[load-test] Saved summary: {summary_json}")
    print(f"[load-test] Saved trend:   {trend_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

