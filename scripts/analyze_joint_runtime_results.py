from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def _read_csvs(root: pathlib.Path, name: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in root.rglob(name):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def _float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _write_csv(path: pathlib.Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_episodes(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("method", ""), row.get("run_id", ""), row.get("seed", ""))].append(row)
    output: List[Dict[str, object]] = []
    for (method, run_id, seed), group in sorted(grouped.items()):
        total = len(group)
        successes = sum(1 for r in group if str(r.get("success", "")).lower() in {"true", "1", "success"})
        output.append({
            "method": method,
            "run_id": run_id,
            "seed": seed,
            "total_episodes": total,
            "total_successes": successes,
            "success_rate": successes / total if total else 0.0,
            "mean_episode_wall_time_ms": _mean(_float(r, "episode_wall_time_ms") for r in group),
            "mean_task_completion_steps": _mean(_float(r, "task_completion_steps") for r in group),
            "mean_inference_count": _mean(_float(r, "inference_count") for r in group),
            "mean_action_count": _mean(_float(r, "action_count") for r in group),
            "mean_stale_action_count": _mean(_float(r, "stale_action_count") for r in group),
            "mean_selected_horizon": _mean(_float(r, "mean_selected_horizon") for r in group),
            "mean_token_keep_ratio": _mean(_float(r, "mean_token_keep_ratio") for r in group),
            "mean_consistency_error": _mean(_float(r, "mean_consistency_error") for r in group),
            "mean_uplink_payload_bytes": _mean(_float(r, "sum_uplink_payload_bytes") for r in group),
            "mean_downlink_payload_bytes": _mean(_float(r, "sum_downlink_payload_bytes") for r in group),
        })
    return output


def aggregate_profiling(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("method", "")].append(row)
    fields = [
        "image_preprocess_time_ms", "client_action_wait_time_ms", "server_policy_time_ms", "vla_forward_time_ms",
        "action_chunk_generation_time_ms", "runtime_token_count_before", "runtime_token_count_after",
        "uplink_payload_bytes", "downlink_payload_bytes",
    ]
    output: List[Dict[str, object]] = []
    for method, group in sorted(grouped.items()):
        row: Dict[str, object] = {"method": method}
        for field in fields:
            row["mean_{}".format(field)] = _mean(_float(r, field) for r in group)
        output.append(row)
    return output


def write_summary(path: pathlib.Path, episode_rows: List[Dict[str, object]]) -> None:
    headers = ["method", "success_rate", "wall_time", "inference_count", "horizon", "keep_ratio", "uplink", "downlink"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    by_method: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in episode_rows:
        by_method[str(row["method"])].append(row)
    for method, group in sorted(by_method.items()):
        lines.append("| {} | {:.4f} | {:.2f} | {:.2f} | {:.2f} | {:.3f} | {:.2f} | {:.2f} |".format(
            method,
            _mean(float(r["success_rate"]) for r in group),
            _mean(float(r["mean_episode_wall_time_ms"]) for r in group),
            _mean(float(r["mean_inference_count"]) for r in group),
            _mean(float(r["mean_selected_horizon"]) for r in group),
            _mean(float(r["mean_token_keep_ratio"]) for r in group),
            _mean(float(r["mean_uplink_payload_bytes"]) for r in group),
            _mean(float(r["mean_downlink_payload_bytes"]) for r in group),
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.root
    episode_rows = aggregate_episodes(_read_csvs(args.root, "episode_summary.csv"))
    profiling_rows = aggregate_profiling(_read_csvs(args.root, "profiling_steps.csv"))
    _write_csv(output_dir / "episode_summary_aggregate.csv", episode_rows)
    _write_csv(output_dir / "profiling_summary_aggregate.csv", profiling_rows)
    write_summary(output_dir / "summary.md", episode_rows)


if __name__ == "__main__":
    main()
