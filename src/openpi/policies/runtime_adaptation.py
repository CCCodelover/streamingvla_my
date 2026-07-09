from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class RuntimePolicyConfig:
    policy_mode: str = "fixed"
    bootstrap_horizon: int = 10
    min_horizon: int = 2
    mid_horizon: int = 4
    high_horizon: int = 10
    max_horizon: int = 10
    low_threshold: float = 0.08
    mid_threshold: float = 0.18
    gripper_change_threshold: float = 0.20
    token_policy: str = "none"
    fixed_keep_ratio: float = 1.0
    keep_low: float = 0.50
    keep_mid: float = 0.75
    keep_high: float = 1.00
    base_vision_tokens: int = 256
    token_hidden_dim: int = 2048
    token_bytes_per_value: int = 2


@dataclasses.dataclass(frozen=True)
class RuntimeDecision:
    selected_horizon: int
    token_keep_ratio: float
    horizon_reason: str
    token_reason: str
    consistency_error: Optional[float]
    action_variation: Optional[float]
    gripper_change: bool
    estimated_vision_tokens: int
    estimated_token_payload_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _as_action_array(actions: Any) -> Optional[np.ndarray]:
    if actions is None:
        return None
    try:
        arr = np.asarray(actions, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr


def compute_chunk_consistency(prev_chunk: Any, new_chunk: Any, executed_offset: int = 0) -> Optional[float]:
    prev_arr = _as_action_array(prev_chunk)
    new_arr = _as_action_array(new_chunk)
    if prev_arr is None or new_arr is None:
        return None
    offset = max(0, int(executed_offset))
    if offset >= prev_arr.shape[0]:
        return None
    prev_remaining = prev_arr[offset:]
    overlap = min(prev_remaining.shape[0], new_arr.shape[0])
    if overlap <= 0:
        return None
    pose_dims = min(6, prev_remaining.shape[1], new_arr.shape[1])
    if pose_dims <= 0:
        return None
    diff = prev_remaining[:overlap, :pose_dims] - new_arr[:overlap, :pose_dims]
    return float(np.linalg.norm(diff, axis=1).mean())


def detect_gripper_change(actions: Any, threshold: float) -> bool:
    arr = _as_action_array(actions)
    if arr is None or arr.shape[1] < 7 or arr.shape[0] < 2:
        return False
    gripper = arr[:, 6]
    return bool(np.max(np.abs(np.diff(gripper))) > float(threshold))


def summarize_action_chunk(actions: Any, gripper_threshold: float = 0.20) -> Dict[str, Any]:
    arr = _as_action_array(actions)
    if arr is None:
        return {
            "chunk_length": 0,
            "action_variation": None,
            "gripper_change": False,
            "mean_action_norm": None,
            "max_action_norm": None,
        }
    pose_dims = min(6, arr.shape[1])
    norms = np.linalg.norm(arr[:, :pose_dims], axis=1) if pose_dims > 0 else np.zeros(arr.shape[0], dtype=np.float32)
    variation = None
    if arr.shape[0] >= 2 and pose_dims > 0:
        variation = float(np.linalg.norm(np.diff(arr[:, :pose_dims], axis=0), axis=1).mean())
    return {
        "chunk_length": int(arr.shape[0]),
        "action_variation": variation,
        "gripper_change": detect_gripper_change(arr, gripper_threshold),
        "mean_action_norm": float(norms.mean()) if norms.size else None,
        "max_action_norm": float(norms.max()) if norms.size else None,
    }


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _clamp_float(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def select_horizon(
    consistency_error: Optional[float],
    gripper_change: bool,
    fixed_horizon: int,
    config: RuntimePolicyConfig,
) -> Tuple[int, str]:
    if config.policy_mode == "fixed":
        horizon = fixed_horizon
        reason = "fixed"
    elif consistency_error is None:
        horizon = config.bootstrap_horizon
        reason = "bootstrap_no_consistency"
    elif gripper_change:
        horizon = config.min_horizon
        reason = "gripper_change"
    elif consistency_error < config.low_threshold:
        horizon = config.high_horizon
        reason = "low_consistency_error"
    elif consistency_error < config.mid_threshold:
        horizon = config.mid_horizon
        reason = "mid_consistency_error"
    else:
        horizon = config.min_horizon
        reason = "high_consistency_error"
    return _clamp_int(horizon, 1, max(1, config.max_horizon)), reason


def select_token_keep_ratio(
    consistency_error: Optional[float],
    gripper_change: bool,
    config: RuntimePolicyConfig,
) -> Tuple[float, str]:
    if config.token_policy == "none":
        keep = 1.0
        reason = "token_none"
    elif config.token_policy == "fixed":
        keep = config.fixed_keep_ratio
        reason = "token_fixed"
    elif config.token_policy in {"action_aware", "joint"}:
        if gripper_change:
            keep = config.keep_high
            reason = "token_gripper_change"
        elif consistency_error is None:
            keep = config.keep_high
            reason = "token_bootstrap"
        elif consistency_error < config.low_threshold:
            keep = config.keep_low
            reason = "token_low_error"
        elif consistency_error < config.mid_threshold:
            keep = config.keep_mid
            reason = "token_mid_error"
        else:
            keep = config.keep_high
            reason = "token_high_error"
    else:
        keep = 1.0
        reason = "token_unknown_policy"
    return _clamp_float(keep, 0.01, 1.0), reason


def estimate_token_payload_bytes(keep_ratio: float, config: RuntimePolicyConfig) -> Tuple[int, int]:
    estimated_vision_tokens = max(1, int(config.base_vision_tokens * _clamp_float(keep_ratio, 0.01, 1.0)))
    estimated_payload = int(estimated_vision_tokens * config.token_hidden_dim * config.token_bytes_per_value)
    return estimated_vision_tokens, estimated_payload


def make_runtime_decision(
    prev_chunk: Any,
    new_chunk: Any,
    executed_offset: int,
    fixed_horizon: int,
    config: RuntimePolicyConfig,
) -> RuntimeDecision:
    consistency_error = compute_chunk_consistency(prev_chunk, new_chunk, executed_offset)
    chunk_summary = summarize_action_chunk(new_chunk, config.gripper_change_threshold)
    gripper_change = bool(chunk_summary["gripper_change"])
    horizon, horizon_reason = select_horizon(consistency_error, gripper_change, fixed_horizon, config)
    keep_ratio, token_reason = select_token_keep_ratio(consistency_error, gripper_change, config)
    est_tokens, est_payload = estimate_token_payload_bytes(keep_ratio, config)
    return RuntimeDecision(
        selected_horizon=horizon,
        token_keep_ratio=keep_ratio,
        horizon_reason=horizon_reason,
        token_reason=token_reason,
        consistency_error=consistency_error,
        action_variation=chunk_summary["action_variation"],
        gripper_change=gripper_change,
        estimated_vision_tokens=est_tokens,
        estimated_token_payload_bytes=est_payload,
    )
