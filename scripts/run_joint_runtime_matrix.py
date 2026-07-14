from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
from typing import Dict, List


STAGES: Dict[str, List[Dict[str, object]]] = {
    "A": [
        {"method": "baseline_h10", "runtime_policy": "fixed", "token_policy": "none", "h": 10},
        {"method": "fixed_h1", "runtime_policy": "fixed", "token_policy": "none", "h": 1},
        {"method": "fixed_h4", "runtime_policy": "fixed", "token_policy": "none", "h": 4},
        {"method": "fixed_h8", "runtime_policy": "fixed", "token_policy": "none", "h": 8},
        {"method": "fixed_h10", "runtime_policy": "fixed", "token_policy": "none", "h": 10},
        {"method": "fixed_h16", "runtime_policy": "fixed", "token_policy": "none", "h": 16},
    ],
    "B": [
        {"method": "token_none_h10", "runtime_policy": "fixed", "token_policy": "none", "h": 10},
        {"method": "token_fixed075_h10", "runtime_policy": "fixed", "token_policy": "fixed", "h": 10, "keep": 0.75},
        {"method": "token_fixed050_h10", "runtime_policy": "fixed", "token_policy": "fixed", "h": 10, "keep": 0.50},
        {"method": "token_action_h10", "runtime_policy": "fixed", "token_policy": "action_aware", "h": 10},
    ],
    "C": [
        {"method": "fixed_h8", "runtime_policy": "fixed", "token_policy": "none", "h": 8},
        {"method": "fixed_h10", "runtime_policy": "fixed", "token_policy": "none", "h": 10},
        {"method": "adaptive_horizon_no_token", "runtime_policy": "adaptive", "token_policy": "none", "h": 10},
    ],
    "D": [
        {"method": "token_joint_fixed_h10", "runtime_policy": "fixed", "token_policy": "joint", "h": 10},
        {"method": "adaptive_horizon_no_token", "runtime_policy": "adaptive", "token_policy": "none", "h": 10},
        {"method": "joint_runtime", "runtime_policy": "joint", "token_policy": "joint", "h": 10},
        {"method": "joint_runtime_fixed075", "runtime_policy": "joint", "token_policy": "fixed", "h": 10, "keep": 0.75},
    ],
}


def _parse_seeds(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _build_command(args: argparse.Namespace, exp: Dict[str, object], seed: int, out_dir: pathlib.Path) -> List[str]:
    method = str(exp["method"])
    horizon = int(exp["h"])
    cmd = [
        "python", "examples/libero/joint_runtime_eval.py",
        "--args.host", args.host,
        "--args.port", str(args.port),
        "--args.task-suite-name", args.task_suite_name,
        "--args.num-trials-per-task", str(args.num_trials_per_task),
        "--args.seed", str(seed),
        "--args.method", method,
        "--args.run-id", "{}_seed{}".format(method, seed),
        "--args.runtime-policy", str(exp["runtime_policy"]),
        "--args.token-policy", str(exp["token_policy"]),
        "--args.replan-steps", str(horizon),
        "--args.max-actions-per-inference", str(horizon),
        "--args.horizon-label", "h{}".format(horizon),
        "--args.timing-output-path", str(out_dir / "timing.txt"),
        "--args.profiling-output-path", str(out_dir / "profiling_steps.csv"),
        "--args.episode-summary-output-path", str(out_dir / "episode_summary.csv"),
        "--args.video-out-path", str(out_dir / "videos"),
        "--args.no-save-videos",
    ]
    if "keep" in exp:
        cmd.extend(["--args.fixed-token-keep-ratio", str(exp["keep"])])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("7,13,21"))
    parser.add_argument("--output-root", default="runs/joint_runtime")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--num-trials-per-task", type=int, default=10)
    parser.add_argument("--run", action="store_true", help="Execute commands. Default is dry-run print only.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    for exp in STAGES[args.stage]:
        method = str(exp["method"])
        for seed in args.seeds:
            out_dir = pathlib.Path(args.output_root) / args.stage / method / "seed{}".format(seed)
            cmd = _build_command(args, exp, seed, out_dir)
            printable = "MUJOCO_GL=egl PYOPENGL_PLATFORM=egl " + " ".join(cmd)
            print(printable)
            if args.run:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "videos").mkdir(parents=True, exist_ok=True)
                subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
