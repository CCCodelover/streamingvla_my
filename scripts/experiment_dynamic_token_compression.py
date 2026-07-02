#!/usr/bin/env python3
"""Compare StreamingVLA visual-token compression schedules.

The script is checkpoint-free and intended for ablation planning.  It reports
real token/attention savings for each schedule and a calibrated *proxy* success
rate that estimates how much task success may degrade when high-urgency steps
lose visual context.  For real LIBERO success rate, run examples/libero/streamingvla.py
against a served policy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
import math
import random
from statistics import mean


@dataclass(frozen=True)
class ScenarioSummary:
    scenario: str
    schedule: str
    saliency: str
    samples: int
    avg_keep_ratio: float
    min_observed_keep_ratio: float
    max_observed_keep_ratio: float
    avg_prefix_tokens_before: float
    avg_prefix_tokens_after: float
    avg_total_tokens_before: float
    avg_total_tokens_after: float
    prefix_compression_pct: float
    total_compression_pct: float
    attention_cost_reduction_pct: float
    proxy_success_rate_pct: float
    proxy_success_delta_pct: float


def _action_norm(sample: list[float]) -> float:
    return math.sqrt(sum(value * value for value in sample))


def _urgency(sample: list[float], norm_scale: float) -> float:
    return min(_action_norm(sample) / norm_scale, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _dynamic_keep_ratio(
    sample: list[float],
    *,
    schedule: str,
    min_keep_ratio: float,
    max_keep_ratio: float,
    norm_scale: float,
    fixed_ratio: float | None = None,
) -> float:
    if schedule == "fixed":
        if fixed_ratio is None:
            raise ValueError("fixed schedule requires fixed_ratio")
        return _clamp(fixed_ratio, min_keep_ratio, max_keep_ratio)

    action_urgency = _urgency(sample, norm_scale)
    if schedule == "action_norm":
        schedule_value = action_urgency
    elif schedule == "inverse_action_norm":
        schedule_value = 1.0 - action_urgency
    elif schedule == "two_stage":
        schedule_value = 1.0 if action_urgency >= 0.65 else 0.0
    elif schedule == "mid_band":
        schedule_value = 1.0 - abs(action_urgency - 0.5) * 2.0
    elif schedule in {"aeo_dynamic", "aeo_risk"}:
        return 0.75 if action_urgency > 0.85 else 0.50
    elif schedule == "aeo_conservative":
        return 1.0 if action_urgency > 0.75 else 0.50
    elif schedule == "aeo_three_stage":
        if action_urgency > 0.90:
            return 1.0
        if action_urgency > 0.70:
            return 0.75
        return 0.50
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return min_keep_ratio + (max_keep_ratio - min_keep_ratio) * schedule_value


def _proxy_success_rate(
    samples: list[list[float]],
    keep_ratios: list[float],
    *,
    baseline_success_rate: float,
    max_success_drop: float,
    norm_scale: float,
    saliency: str,
) -> float:
    """Estimate success degradation from urgency-weighted visual token removal.

    The proxy intentionally penalizes compression more when action residuals are
    large.  Saliency-aware pruning receives a smaller penalty than uniform
    baselines because it preferentially keeps high-activation tokens.
    """
    saliency_factor = {
        "l2": 0.72,
        "abs_mean": 0.78,
        "uniform_stride": 1.0,
    }[saliency]
    risks = [(_urgency(sample, norm_scale) ** 1.5) * (1.0 - ratio) * saliency_factor for sample, ratio in zip(samples, keep_ratios, strict=True)]
    expected_drop = max_success_drop * mean(risks)
    return _clamp(baseline_success_rate - expected_drop, 0.0, 1.0) * 100.0


def _summarize(
    samples: list[list[float]],
    keep_ratios: list[float],
    *,
    scenario: str,
    schedule: str,
    saliency: str,
    args: argparse.Namespace,
) -> ScenarioSummary:
    kept_image_tokens = [math.ceil(args.image_tokens_per_camera * ratio) * args.cameras for ratio in keep_ratios]
    prefix_before = args.cameras * args.image_tokens_per_camera + args.language_tokens
    total_before = prefix_before + args.suffix_tokens
    prefix_after = [tokens + args.language_tokens for tokens in kept_image_tokens]
    total_after = [tokens + args.suffix_tokens for tokens in prefix_after]
    avg_prefix_after = mean(prefix_after)
    avg_total_after = mean(total_after)
    proxy_success = _proxy_success_rate(
        samples,
        keep_ratios,
        baseline_success_rate=args.baseline_success_rate,
        max_success_drop=args.max_success_drop,
        norm_scale=args.norm_scale,
        saliency=saliency,
    )
    return ScenarioSummary(
        scenario=scenario,
        schedule=schedule,
        saliency=saliency,
        samples=len(samples),
        avg_keep_ratio=mean(keep_ratios),
        min_observed_keep_ratio=min(keep_ratios),
        max_observed_keep_ratio=max(keep_ratios),
        avg_prefix_tokens_before=float(prefix_before),
        avg_prefix_tokens_after=avg_prefix_after,
        avg_total_tokens_before=float(total_before),
        avg_total_tokens_after=avg_total_after,
        prefix_compression_pct=(1.0 - avg_prefix_after / prefix_before) * 100.0,
        total_compression_pct=(1.0 - avg_total_after / total_before) * 100.0,
        attention_cost_reduction_pct=(1.0 - (avg_total_after / total_before) ** 2) * 100.0,
        proxy_success_rate_pct=proxy_success,
        proxy_success_delta_pct=proxy_success - args.baseline_success_rate * 100.0,
    )


def _make_samples(args: argparse.Namespace) -> list[list[float]]:
    rng = random.Random(args.seed)
    return [
        [rng.gammavariate(args.gamma_concentration, 1.0 / args.gamma_rate) * rng.choice((-1.0, 1.0)) for _ in range(args.action_dim)]
        for _ in range(args.samples)
    ]


def _parse_fixed_ratios(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def run_experiment(args: argparse.Namespace) -> list[ScenarioSummary]:
    samples = _make_samples(args)
    summaries: list[ScenarioSummary] = []

    for ratio in _parse_fixed_ratios(args.fixed_keep_ratios):
        keep_ratios = [
            _dynamic_keep_ratio(
                sample,
                schedule="fixed",
                min_keep_ratio=args.min_keep_ratio,
                max_keep_ratio=args.max_keep_ratio,
                norm_scale=args.norm_scale,
                fixed_ratio=ratio,
            )
            for sample in samples
        ]
        summaries.append(
            _summarize(samples, keep_ratios, scenario=f"fixed_{ratio:.3f}", schedule="fixed", saliency=args.fixed_saliency, args=args)
        )

    for schedule in [item.strip() for item in args.dynamic_schedules.split(",") if item.strip()]:
        for saliency in [item.strip() for item in args.saliency_modes.split(",") if item.strip()]:
            keep_ratios = [
                _dynamic_keep_ratio(
                    sample,
                    schedule=schedule,
                    min_keep_ratio=args.min_keep_ratio,
                    max_keep_ratio=args.max_keep_ratio,
                    norm_scale=args.norm_scale,
                )
                for sample in samples
            ]
            summaries.append(_summarize(samples, keep_ratios, scenario=f"{schedule}_{saliency}", schedule=schedule, saliency=saliency, args=args))

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--cameras", type=int, default=3)
    parser.add_argument("--image-tokens-per-camera", type=int, default=256)
    parser.add_argument("--language-tokens", type=int, default=200)
    parser.add_argument("--suffix-tokens", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--min-keep-ratio", type=float, default=0.5)
    parser.add_argument("--max-keep-ratio", type=float, default=1.0)
    parser.add_argument("--norm-scale", type=float, default=4.0)
    parser.add_argument("--fixed-keep-ratios", default="1.0,0.875,0.75,0.625,0.5")
    parser.add_argument("--fixed-saliency", choices=("uniform_stride", "l2", "abs_mean"), default="uniform_stride")
    parser.add_argument("--dynamic-schedules", default="action_norm,two_stage,aeo_dynamic,aeo_conservative,aeo_three_stage,inverse_action_norm,mid_band")
    parser.add_argument("--saliency-modes", default="l2,abs_mean,uniform_stride")
    parser.add_argument("--baseline-success-rate", type=float, default=0.85)
    parser.add_argument("--max-success-drop", type=float, default=0.35)
    parser.add_argument("--gamma-concentration", type=float, default=1.2)
    parser.add_argument("--gamma-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pretty", action="store_true", help="Print a compact table before the JSON payload.")
    args = parser.parse_args()
    summaries = run_experiment(args)

    if args.pretty:
        header = f"{'scenario':<34} {'keep':>7} {'tok%':>8} {'attn%':>8} {'succ%*':>8}"
        print(header)
        print("-" * len(header))
        for summary in summaries:
            print(
                f"{summary.scenario:<34} {summary.avg_keep_ratio:7.3f} "
                f"{summary.total_compression_pct:8.2f} {summary.attention_cost_reduction_pct:8.2f} "
                f"{summary.proxy_success_rate_pct:8.2f}"
            )
        print("\n*succ% is a checkpoint-free proxy; use LIBERO rollout for measured task success.\n")

    print(json.dumps([asdict(summary) for summary in summaries], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
