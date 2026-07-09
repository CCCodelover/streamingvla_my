from __future__ import annotations

import csv
import dataclasses
import logging
import math
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi.policies import runtime_adaptation
from openpi_client import image_tools
from openpi_client import streaming_websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8192
    resize_size: int = 224
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    seed: int = 7

    replan_steps: int = 10
    max_actions_per_inference: int = 10
    action_timeout: float = 20.0
    clear_action_queue_on_replan: bool = False
    max_timeouts_per_episode: int = 1
    strict_request_id_matching: bool = False
    enable_runtime_protocol: bool = False

    method: str = "joint_runtime"
    run_id: str = "run_001"
    runtime_policy: str = "fixed"
    token_policy: str = "none"
    horizon_label: str = ""

    adaptive_min_horizon: int = 2
    adaptive_mid_horizon: int = 4
    adaptive_high_horizon: int = 10
    adaptive_low_threshold: float = 0.08
    adaptive_mid_threshold: float = 0.18
    adaptive_gripper_change_threshold: float = 0.20

    fixed_token_keep_ratio: float = 1.0
    keep_low: float = 0.50
    keep_mid: float = 0.75
    keep_high: float = 1.00

    timing_output_path: str = "runs/joint_runtime/timing.txt"
    profiling_output_path: str = ""
    episode_summary_output_path: str = ""
    video_out_path: str = "runs/joint_runtime/videos"
    save_videos: bool = True


PROFILING_FIELDS = [
    "episode_id", "task_id", "task_name", "seed", "run_id", "step",
    "method", "runtime_policy", "token_policy", "selected_horizon", "horizon_reason",
    "token_keep_ratio", "token_reason", "consistency_error", "action_variation", "gripper_change",
    "request_id", "image_preprocess_time_ms", "client_infer_call_time_ms", "client_action_wait_time_ms",
    "total_action_latency_ms", "server_unpack_time_ms", "server_policy_time_ms", "server_pack_time_ms",
    "image_encode_time_ms", "tokenization_time_ms", "vla_forward_time_ms", "action_chunk_generation_time_ms",
    "runtime_token_count_before", "runtime_token_count_after", "uplink_payload_bytes", "downlink_payload_bytes",
    "inference_count", "action_count", "stale_action_count", "success",
]

EPISODE_FIELDS = [
    "episode_id", "task_id", "task_name", "seed", "run_id", "method", "runtime_policy", "token_policy",
    "success", "inference_count", "action_count", "stale_action_count", "task_completion_steps",
    "episode_wall_time_ms", "mean_selected_horizon", "mean_token_keep_ratio", "mean_consistency_error",
    "sum_uplink_payload_bytes", "sum_downlink_payload_bytes",
]


def _max_steps_for_suite(name: str) -> int:
    if name == "libero_spatial":
        return 220
    if name == "libero_object":
        return 280
    if name == "libero_goal":
        return 300
    if name == "libero_10":
        return 520
    if name == "libero_90":
        return 400
    raise ValueError("Unknown task suite: {}".format(name))


def _get_libero_env(task: Any, resolution: int, seed: int) -> Tuple[Any, str]:
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _get_timing(action_data: Dict[str, Any], key: str) -> float:
    server = action_data.get("server_timing", {}) if isinstance(action_data, dict) else {}
    transport = action_data.get("transport_timing", {}) if isinstance(action_data, dict) else {}
    value = server.get(key, transport.get(key, transport.get(key.replace("_time", ""))))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_runtime_config(args: Args) -> runtime_adaptation.RuntimePolicyConfig:
    return runtime_adaptation.RuntimePolicyConfig(
        policy_mode=args.runtime_policy,
        bootstrap_horizon=args.max_actions_per_inference,
        min_horizon=args.adaptive_min_horizon,
        mid_horizon=args.adaptive_mid_horizon,
        high_horizon=args.adaptive_high_horizon,
        max_horizon=max(args.max_actions_per_inference, args.adaptive_high_horizon, 1),
        low_threshold=args.adaptive_low_threshold,
        mid_threshold=args.adaptive_mid_threshold,
        gripper_change_threshold=args.adaptive_gripper_change_threshold,
        token_policy=args.token_policy,
        fixed_keep_ratio=args.fixed_token_keep_ratio,
        keep_low=args.keep_low,
        keep_mid=args.keep_mid,
        keep_high=args.keep_high,
    )


def _write_rows(path: str, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    if not path:
        return
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    pathlib.Path(args.timing_output_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    if not args.profiling_output_path:
        args.profiling_output_path = str(pathlib.Path(args.timing_output_path).with_name("profiling_steps.csv"))
    if not args.episode_summary_output_path:
        args.episode_summary_output_path = str(pathlib.Path(args.timing_output_path).with_name("episode_summary.csv"))

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = _max_steps_for_suite(args.task_suite_name)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    runtime_config = _build_runtime_config(args)

    profiling_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []
    suite_total_episodes = 0
    suite_total_successes = 0
    suite_total_time_success = 0.0
    suite_total_actions_success = 0
    suite_total_episode_time = 0.0
    suite_total_episode_actions = 0
    suite_total_norm_exceeded = 0
    suite_total_skipped_denoise = 0

    with open(args.timing_output_path, "w", encoding="utf-8") as timing_file:
        timing_file.write("--- Joint Runtime Timing Results for Task Suite: {} ---\n".format(args.task_suite_name))
        timing_file.write("Task | Episode | Success | Total Time (s) | Actions | Inferences\n")
        timing_file.write("-" * 80 + "\n")

        for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Overall Task Suite"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            task_results: List[Tuple[bool, float, int]] = []
            task_success_data: List[Tuple[float, int]] = []
            task_transport: Dict[str, List[float]] = {}
            task_norm_exceeded = 0
            task_skipped_denoise = 0
            task_total_time = 0.0
            task_total_actions = 0

            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Task {}".format(task_id)):
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                episode_id = "{}_task{}_ep{}".format(args.run_id, task_id, episode_idx)
                t = 0
                steps_since_replan = 0
                request_seq = 0
                new_task = True
                replay_images: List[np.ndarray] = []
                action_states = np.zeros(7, dtype=np.float32)
                chunk_action_states = np.zeros(7, dtype=np.float32)
                prior_chunk: Optional[np.ndarray] = None
                last_chunk: Optional[np.ndarray] = None
                current_chunk_actions: List[np.ndarray] = []
                executed_offset = 0
                inference_count = 0
                action_count = 0
                stale_action_count = 0
                timeout_count = 0
                episode_success = False
                episode_start_time = 0.0
                selected_horizons: List[float] = []
                keep_ratios: List[float] = []
                consistency_values: List[float] = []
                uplink_payloads: List[float] = []
                downlink_payloads: List[float] = []

                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    if episode_start_time == 0.0:
                        episode_start_time = time.monotonic()

                    preprocess_start = time.monotonic()
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
                    image_preprocess_ms = (time.monotonic() - preprocess_start) * 1000
                    replay_images.append(img)

                    should_replan = steps_since_replan == 0
                    if should_replan:
                        if current_chunk_actions:
                            prior_chunk = last_chunk
                            last_chunk = np.asarray(current_chunk_actions, dtype=np.float32)
                            current_chunk_actions = []
                        if args.clear_action_queue_on_replan and not new_task:
                            dropped = client.clear_action_queue()
                            stale_action_count += dropped
                            logging.info("Cleared %d stale actions before replan.", dropped)
                        provisional = runtime_adaptation.make_runtime_decision(
                            prev_chunk=prior_chunk,
                            new_chunk=last_chunk,
                            executed_offset=0,
                            fixed_horizon=args.max_actions_per_inference,
                            config=runtime_config,
                        )
                        selected_horizon = min(provisional.selected_horizon, args.max_actions_per_inference)
                        token_keep_ratio = provisional.token_keep_ratio
                        current_request_id = "{}_task{}_ep{}_t{}_req{}".format(args.run_id, task_id, episode_idx, t, request_seq)
                        request_seq += 1
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                            "prompt": str(task_description),
                            "observation/action_left_sum": np.zeros(6, dtype=np.float32),
                            "observation/action_states": action_states.astype(np.float32),
                            "observation/threshold": np.asarray(100000.0, dtype=np.float32),
                        }
                        use_runtime_protocol = (
                            args.enable_runtime_protocol
                            or args.runtime_policy in {"adaptive", "joint"}
                            or args.token_policy in {"fixed", "action_aware", "joint"}
                        )
                        if use_runtime_protocol:
                            element.update({
                                "__request_id__": current_request_id,
                                "execution_horizon": selected_horizon,
                                "vision_keep_ratio": np.asarray(token_keep_ratio, dtype=np.float32),
                                "runtime_token_keep_ratio": np.asarray(token_keep_ratio, dtype=np.float32),
                                "runtime_policy": provisional.to_dict(),
                            })
                        infer_start = time.monotonic()
                        client.infer(element, new_task)
                        client_infer_ms = (time.monotonic() - infer_start) * 1000
                        new_task = False
                        inference_count += 1
                        selected_horizons.append(float(selected_horizon))
                        keep_ratios.append(float(token_keep_ratio))
                        if provisional.consistency_error is not None:
                            consistency_values.append(float(provisional.consistency_error))
                        steps_since_replan = 0
                        executed_offset = 0
                    else:
                        current_request_id = ""
                        selected_horizon = args.max_actions_per_inference
                        token_keep_ratio = 1.0
                        provisional = runtime_adaptation.make_runtime_decision(prior_chunk, last_chunk, 0, selected_horizon, runtime_config)
                        client_infer_ms = 0.0

                    wait_start = time.monotonic()
                    expected_request_id = current_request_id if (should_replan and args.strict_request_id_matching and use_runtime_protocol) else None
                    while True:
                        action_data = client.get_next_action(timeout=args.action_timeout, request_id=expected_request_id)
                        if action_data is not None:
                            break
                        timeout_count += 1
                        logging.warning(
                            "No action received yet for request_id=%s; keep waiting without re-sending inference.",
                            current_request_id,
                        )
                    client_wait_ms = (time.monotonic() - wait_start) * 1000
                    total_action_latency_ms = image_preprocess_ms + client_infer_ms + client_wait_ms

                    if "server_error" in action_data:
                        logging.error("Server error while waiting for request_id=%s: %s", current_request_id, action_data.get("server_error"))
                        stale_action_count += 1
                        break

                    if "norm_exceeded" in action_data:
                        stale_action_count += 1
                        task_norm_exceeded += 1
                        task_skipped_denoise += 1
                        suite_total_norm_exceeded += 1
                        suite_total_skipped_denoise += 1
                        steps_since_replan = 0
                        continue

                    if "actions" not in action_data:
                        stale_action_count += 1
                        steps_since_replan = 0
                        continue

                    action = np.asarray(action_data["actions"], dtype=np.float32).reshape(-1)[:7]
                    if action.size < 7 or not np.all(np.isfinite(action)):
                        stale_action_count += 1
                        steps_since_replan = 0
                        continue

                    transport = action_data.get("transport_timing", {})
                    server = action_data.get("server_timing", {})
                    for key, value in list(transport.items()) + list(server.items()):
                        if isinstance(value, (int, float)):
                            task_transport.setdefault(key, []).append(float(value))
                    uplink_payloads.append(float(transport.get("client_uplink_payload_bytes", transport.get("uplink_payload_bytes", 0.0)) or 0.0))
                    downlink_payloads.append(float(transport.get("client_downlink_payload_bytes", transport.get("downlink_payload_bytes", 0.0)) or 0.0))

                    action_states += action
                    chunk_action_states += action
                    current_chunk_actions.append(action.copy())
                    obs, reward, done, info = env.step(action.tolist())
                    action_count += 1
                    task_total_actions += 1
                    executed_offset += 1
                    steps_since_replan = (steps_since_replan + 1) % max(1, selected_horizon)
                    t += 1

                    profiling_rows.append({
                        "episode_id": episode_id, "task_id": task_id, "task_name": task_description, "seed": args.seed,
                        "run_id": args.run_id, "step": t, "method": args.method, "runtime_policy": args.runtime_policy,
                        "token_policy": args.token_policy, "selected_horizon": selected_horizon, "horizon_reason": provisional.horizon_reason,
                        "token_keep_ratio": token_keep_ratio, "token_reason": provisional.token_reason,
                        "consistency_error": provisional.consistency_error, "action_variation": provisional.action_variation,
                        "gripper_change": provisional.gripper_change, "request_id": action_data.get("request_id", current_request_id),
                        "image_preprocess_time_ms": image_preprocess_ms, "client_infer_call_time_ms": client_infer_ms,
                        "client_action_wait_time_ms": client_wait_ms, "total_action_latency_ms": total_action_latency_ms,
                        "server_unpack_time_ms": _get_timing(action_data, "server_unpack_time_ms"),
                        "server_policy_time_ms": _get_timing(action_data, "server_policy_time_ms"),
                        "server_pack_time_ms": _get_timing(action_data, "server_pack_time_ms"),
                        "image_encode_time_ms": _get_timing(action_data, "image_encode_time_ms"),
                        "tokenization_time_ms": _get_timing(action_data, "tokenization_time_ms"),
                        "vla_forward_time_ms": _get_timing(action_data, "vla_forward_time_ms"),
                        "action_chunk_generation_time_ms": _get_timing(action_data, "action_chunk_generation_time_ms"),
                        "runtime_token_count_before": server.get("runtime_token_count_before", ""),
                        "runtime_token_count_after": server.get("runtime_token_count_after", ""),
                        "uplink_payload_bytes": uplink_payloads[-1] if uplink_payloads else 0.0,
                        "downlink_payload_bytes": downlink_payloads[-1] if downlink_payloads else 0.0,
                        "inference_count": inference_count, "action_count": action_count, "stale_action_count": stale_action_count,
                        "success": bool(done),
                    })
                    if done:
                        episode_success = True
                        break

                episode_end = time.monotonic()
                episode_total_time = episode_end - episode_start_time if episode_start_time > 0 else 0.0
                suffix = "success" if episode_success else "failure"
                if args.save_videos:
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / "{}_{}_{}_{}.mp4".format(task_description, task_id, episode_idx % 3, suffix),
                        [np.asarray(x) for x in replay_images],
                        fps=20,
                    )
                task_results.append((episode_success, episode_total_time, action_count))
                task_total_time += episode_total_time
                if episode_success:
                    task_success_data.append((episode_total_time, action_count))
                    suite_total_time_success += episode_total_time
                    suite_total_actions_success += action_count
                suite_total_episodes += 1
                suite_total_successes += int(episode_success)
                suite_total_episode_time += episode_total_time
                suite_total_episode_actions += action_count

                episode_rows.append({
                    "episode_id": episode_id, "task_id": task_id, "task_name": task_description, "seed": args.seed,
                    "run_id": args.run_id, "method": args.method, "runtime_policy": args.runtime_policy,
                    "token_policy": args.token_policy, "success": bool(episode_success), "inference_count": inference_count,
                    "action_count": action_count, "stale_action_count": stale_action_count, "task_completion_steps": t,
                    "episode_wall_time_ms": episode_total_time * 1000, "mean_selected_horizon": _mean(selected_horizons),
                    "mean_token_keep_ratio": _mean(keep_ratios), "mean_consistency_error": _mean(consistency_values),
                    "sum_uplink_payload_bytes": sum(uplink_payloads), "sum_downlink_payload_bytes": sum(downlink_payloads),
                })
                timing_file.write("{:4d} | {:7d} | {} | {:.2f} | {:7d} | {:7d}\n".format(task_id, episode_idx, "SUCCESS" if episode_success else "FAILURE", episode_total_time, action_count, inference_count))
                timing_file.flush()

            trials = len(task_results)
            successes = len(task_success_data)
            task_success_rate = successes / trials if trials else 0.0
            task_avg_time_success = sum(x[0] for x in task_success_data) / successes if successes else 0.0
            task_avg_actions_success = sum(x[1] for x in task_success_data) / successes if successes else 0.0
            task_avg_time_all = task_total_time / trials if trials else 0.0
            task_avg_actions_all = task_total_actions / trials if trials else 0.0

            def avg_transport(key: str) -> float:
                return _mean(task_transport.get(key, []))

            timing_file.write("-" * 80 + "\n")
            timing_file.write("Task Summary: {}\n".format(task_description))
            timing_file.write("  Avg Success Rate (All Trials): {:.4f} ({}/{})\n".format(task_success_rate, successes, trials))
            timing_file.write("  Avg Episode Time (Success Only): {:.2f} seconds\n".format(task_avg_time_success))
            timing_file.write("  Avg Episode Time (All Trials): {:.2f} seconds\n".format(task_avg_time_all))
            timing_file.write("  Avg Actions/Episode (Success Only): {:.2f} actions\n".format(task_avg_actions_success))
            timing_file.write("  Avg Actions/Episode (All Trials): {:.2f} actions\n".format(task_avg_actions_all))
            timing_file.write("  Norm Exceeded Count: {}\n".format(task_norm_exceeded))
            timing_file.write("  Skipping Denoise Count: {}\n".format(task_skipped_denoise))
            timing_file.write("  Avg Server Policy Time: {:.4f} ms\n".format(avg_transport("server_policy_time_ms")))
            timing_file.write("  Avg Client E2E Action Latency: {:.4f} ms\n".format(avg_transport("client_e2e_ms")))
            timing_file.write("  Avg Uplink Payload: {:.2f} bytes\n".format(avg_transport("client_uplink_payload_bytes")))
            timing_file.write("  Avg Downlink Payload: {:.2f} bytes\n".format(avg_transport("client_downlink_payload_bytes")))
            timing_file.write("=" * 80 + "\n\n")

        suite_avg_success = suite_total_successes / suite_total_episodes if suite_total_episodes else 0.0
        suite_avg_time_success = suite_total_time_success / suite_total_successes if suite_total_successes else 0.0
        suite_avg_actions_success = suite_total_actions_success / suite_total_successes if suite_total_successes else 0.0
        suite_avg_time_all = suite_total_episode_time / suite_total_episodes if suite_total_episodes else 0.0
        suite_avg_actions_all = suite_total_episode_actions / suite_total_episodes if suite_total_episodes else 0.0
        timing_file.write("\n\n################################################################################\n")
        timing_file.write("### OVERALL SUITE SUMMARY ###\n")
        timing_file.write("################################################################################\n")
        timing_file.write("Total Tasks Completed: {}\n".format(num_tasks_in_suite))
        timing_file.write("Total Episodes Attempted: {}\n".format(suite_total_episodes))
        timing_file.write("Total Successful Episodes: {}\n".format(suite_total_successes))
        timing_file.write("Overall Success Rate (All Trials): {:.4f} ({}/{})\n".format(suite_avg_success, suite_total_successes, suite_total_episodes))
        timing_file.write("Overall Avg Episode Time (Success Only): {:.2f} seconds\n".format(suite_avg_time_success))
        timing_file.write("Overall Avg Episode Time (All Trials): {:.2f} seconds\n".format(suite_avg_time_all))
        timing_file.write("Overall Avg Actions/Episode (Success Only): {:.2f} actions\n".format(suite_avg_actions_success))
        timing_file.write("Overall Avg Actions/Episode (All Trials): {:.2f} actions\n".format(suite_avg_actions_all))
        timing_file.write("Overall Norm Exceeded Count: {}\n".format(suite_total_norm_exceeded))
        timing_file.write("Overall Skipping Denoise Count: {}\n".format(suite_total_skipped_denoise))

    _write_rows(args.profiling_output_path, PROFILING_FIELDS, profiling_rows)
    _write_rows(args.episode_summary_output_path, EPISODE_FIELDS, episode_rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
