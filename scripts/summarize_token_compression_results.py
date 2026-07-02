#!/usr/bin/env python3
"""Summarize StreamingVLA token-compression LIBERO timing files.

The LIBERO runner writes one timing text file per experiment.  This helper parses
those files and emits a compact CSV/Markdown table with success, episode-time,
action-count, and AEO telemetry so fixed and AEO-aware compression runs can be
compared consistently.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import re
import sys


@dataclass(frozen=True)
class TimingSummary:
    experiment: str
    success_rate: float | None
    successful_episodes: int | None
    total_episodes: int | None
    avg_episode_time_success: float | None
    avg_episode_time_all: float | None
    avg_actions_success: float | None
    avg_actions_all: float | None
    norm_exceeded_count: int | None
    skipped_denoise_count: int | None
    avg_client_e2e_ms: float | None
    avg_server_policy_ms: float | None
    avg_uplink_payload_bytes: float | None
    avg_downlink_payload_bytes: float | None
    path: str


PATTERNS = {
    "success": re.compile(r"Overall Success Rate \(All Trials\):\s*([0-9.]+)\s*\((\d+)/(\d+)\)"),
    "time_success": re.compile(r"Overall Avg Episode Time \(Success Only\):\s*([0-9.]+)"),
    "time_all": re.compile(r"Overall Avg Episode Time \(All Trials\):\s*([0-9.]+)"),
    "actions_success": re.compile(r"Overall Avg Actions/Episode \(Success Only\):\s*([0-9.]+)"),
    "actions_all": re.compile(r"Overall Avg Actions/Episode \(All Trials\):\s*([0-9.]+)"),
    "norm": re.compile(r"Overall Norm Exceeded Count:\s*(\d+)"),
    "skip": re.compile(r"Overall Skipping Denoise Count:\s*(\d+)"),
    "client_e2e": re.compile(r"Avg Client E2E Action Latency:\s*([0-9.]+)"),
    "server_policy": re.compile(r"Avg Server Policy Time:\s*([0-9.]+)"),
    "uplink_payload": re.compile(r"Avg Uplink Payload:\s*([0-9.]+)"),
    "downlink_payload": re.compile(r"Avg Downlink Payload:\s*([0-9.]+)"),
}


def _search_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    return None if match is None else float(match.group(1))


def _search_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return None if match is None else int(match.group(1))


def parse_timing_file(path: Path) -> TimingSummary:
    text = path.read_text(encoding="utf-8")
    success_match = PATTERNS["success"].search(text)
    success_rate = float(success_match.group(1)) if success_match else None
    successful_episodes = int(success_match.group(2)) if success_match else None
    total_episodes = int(success_match.group(3)) if success_match else None
    return TimingSummary(
        experiment=path.stem,
        success_rate=success_rate,
        successful_episodes=successful_episodes,
        total_episodes=total_episodes,
        avg_episode_time_success=_search_float(PATTERNS["time_success"], text),
        avg_episode_time_all=_search_float(PATTERNS["time_all"], text),
        avg_actions_success=_search_float(PATTERNS["actions_success"], text),
        avg_actions_all=_search_float(PATTERNS["actions_all"], text),
        norm_exceeded_count=_search_int(PATTERNS["norm"], text),
        skipped_denoise_count=_search_int(PATTERNS["skip"], text),
        avg_client_e2e_ms=_search_float(PATTERNS["client_e2e"], text),
        avg_server_policy_ms=_search_float(PATTERNS["server_policy"], text),
        avg_uplink_payload_bytes=_search_float(PATTERNS["uplink_payload"], text),
        avg_downlink_payload_bytes=_search_float(PATTERNS["downlink_payload"], text),
        path=str(path),
    )


def _print_markdown(rows: list[TimingSummary]) -> None:
    headers = [
        "experiment",
        "success",
        "episodes",
        "time_success",
        "time_all",
        "actions_success",
        "actions_all",
        "norm_exceeded",
        "skipped_denoise",
        "e2e_ms",
        "server_policy_ms",
        "uplink_bytes",
        "downlink_bytes",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        episodes = "" if row.successful_episodes is None else f"{row.successful_episodes}/{row.total_episodes}"
        values = [
            row.experiment,
            "" if row.success_rate is None else f"{row.success_rate:.4f}",
            episodes,
            "" if row.avg_episode_time_success is None else f"{row.avg_episode_time_success:.2f}",
            "" if row.avg_episode_time_all is None else f"{row.avg_episode_time_all:.2f}",
            "" if row.avg_actions_success is None else f"{row.avg_actions_success:.2f}",
            "" if row.avg_actions_all is None else f"{row.avg_actions_all:.2f}",
            "" if row.norm_exceeded_count is None else str(row.norm_exceeded_count),
            "" if row.skipped_denoise_count is None else str(row.skipped_denoise_count),
            "" if row.avg_client_e2e_ms is None else f"{row.avg_client_e2e_ms:.2f}",
            "" if row.avg_server_policy_ms is None else f"{row.avg_server_policy_ms:.2f}",
            "" if row.avg_uplink_payload_bytes is None else f"{row.avg_uplink_payload_bytes:.0f}",
            "" if row.avg_downlink_payload_bytes is None else f"{row.avg_downlink_payload_bytes:.0f}",
        ]
        print("| " + " | ".join(values) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Timing .txt files or directories containing timing .txt files.")
    parser.add_argument("--format", choices=("csv", "markdown"), default="markdown")
    args = parser.parse_args()

    timing_files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            timing_files.extend(sorted(path.glob("*.txt")))
        else:
            timing_files.append(path)

    rows = [parse_timing_file(path) for path in timing_files]
    if args.format == "markdown":
        _print_markdown(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(asdict(rows[0]).keys()) if rows else list(TimingSummary.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


if __name__ == "__main__":
    main()
