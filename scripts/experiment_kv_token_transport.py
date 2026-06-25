#!/usr/bin/env python3
"""Estimate token-uplink / KV-cache-downlink transport costs.

This is a protocol-planning tool for the proposed split transport experiment:
client/edge uploads compressed visual tokens, while the server optionally sends
prefix KV-cache telemetry/cache payload back.  It does not require a checkpoint
and intentionally reports payload estimates so we can decide whether KV-cache
transport is likely to improve or hurt end-to-end latency before implementing a
full edge/server split runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class TransportSummary:
    mode: str
    keep_ratio: float
    uplink_mb: float
    downlink_mb: float
    total_mb: float
    uplink_ms: float
    downlink_ms: float
    total_transfer_ms: float
    note: str


def _mb_from_values(num_values: float, bytes_per_value: int) -> float:
    return num_values * bytes_per_value / (1024 * 1024)


def _transfer_ms(payload_mb: float, bandwidth_mbps: float) -> float:
    if bandwidth_mbps <= 0:
        raise ValueError(f"bandwidth_mbps must be positive, got {bandwidth_mbps}")
    return payload_mb * 8 * 1000 / bandwidth_mbps


def _visual_token_payload_mb(tokens: int, hidden_dim: int, bytes_per_value: int, keep_ratio: float) -> float:
    kept_tokens = max(1, round(tokens * keep_ratio))
    return _mb_from_values(kept_tokens * hidden_dim, bytes_per_value)


def _kv_cache_payload_mb(
    prefix_tokens: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    bytes_per_value: int,
    keep_ratio: float,
) -> float:
    kept_tokens = max(1, round(prefix_tokens * keep_ratio))
    # K and V are both transmitted.
    return _mb_from_values(layers * 2 * kept_tokens * kv_heads * head_dim, bytes_per_value)


def summarize_transport(args: argparse.Namespace) -> list[TransportSummary]:
    summaries: list[TransportSummary] = []
    image_uplink_mb = args.image_uplink_mb
    action_downlink_mb = args.action_downlink_mb

    for keep_ratio in args.keep_ratios:
        token_uplink_mb = _visual_token_payload_mb(
            args.visual_tokens,
            args.hidden_dim,
            args.token_bytes_per_value,
            keep_ratio,
        )
        kv_downlink_mb = _kv_cache_payload_mb(
            args.prefix_tokens,
            args.layers,
            args.kv_heads,
            args.head_dim,
            args.kv_bytes_per_value,
            keep_ratio,
        )
        rows = [
            (
                "image_up/action_down",
                image_uplink_mb,
                action_downlink_mb,
                "current websocket baseline: client uploads images/obs; server sends actions",
            ),
            (
                "token_up/action_down",
                token_uplink_mb,
                action_downlink_mb,
                "edge vision encoder uploads compressed visual tokens; server sends actions",
            ),
            (
                "token_up/kv_down",
                token_uplink_mb,
                kv_downlink_mb,
                "proposed split: token uplink plus prefix KV-cache downlink; useful only if KV is reused downstream",
            ),
        ]
        for mode, uplink_mb, downlink_mb, note in rows:
            summaries.append(
                TransportSummary(
                    mode=mode,
                    keep_ratio=keep_ratio,
                    uplink_mb=uplink_mb,
                    downlink_mb=downlink_mb,
                    total_mb=uplink_mb + downlink_mb,
                    uplink_ms=_transfer_ms(uplink_mb, args.uplink_mbps),
                    downlink_ms=_transfer_ms(downlink_mb, args.downlink_mbps),
                    total_transfer_ms=_transfer_ms(uplink_mb, args.uplink_mbps)
                    + _transfer_ms(downlink_mb, args.downlink_mbps),
                    note=note,
                )
            )
    return summaries


def _parse_keep_ratios(value: str) -> list[float]:
    ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
    for ratio in ratios:
        if not 0 < ratio <= 1:
            raise ValueError(f"keep ratios must be in (0, 1], got {ratio}")
    return ratios


def _print_markdown(rows: list[TransportSummary]) -> None:
    headers = ["mode", "keep", "uplink_MB", "downlink_MB", "total_MB", "transfer_ms"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row.mode,
                    f"{row.keep_ratio:.3f}",
                    f"{row.uplink_mb:.4f}",
                    f"{row.downlink_mb:.4f}",
                    f"{row.total_mb:.4f}",
                    f"{row.total_transfer_ms:.2f}",
                ]
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-ratios", type=_parse_keep_ratios, default=_parse_keep_ratios("1.0,0.75,0.5"))
    parser.add_argument("--visual-tokens", type=int, default=256)
    parser.add_argument("--prefix-tokens", type=int, default=456, help="Visual + language prefix tokens used for KV-cache estimates.")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=18)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--token-bytes-per-value", type=int, default=4)
    parser.add_argument("--kv-bytes-per-value", type=int, default=2)
    parser.add_argument("--image-uplink-mb", type=float, default=0.3000)
    parser.add_argument("--action-downlink-mb", type=float, default=0.0010)
    parser.add_argument("--uplink-mbps", type=float, default=100.0)
    parser.add_argument("--downlink-mbps", type=float, default=100.0)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    rows = summarize_transport(args)
    if args.format == "markdown":
        _print_markdown(rows)
    else:
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
