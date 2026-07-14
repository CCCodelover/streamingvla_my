# StreamingVLA 视觉 Token 压缩与传输实验说明文档

本文档用于下一阶段办公讨论和 A100 实验排期，集中整理目前已经跑出的实验结论、后续需要安排的实验组、测评参数指标、动态压缩策略，以及端云上下行传输策略。

## 1. 总体研究目标

当前工作围绕三个递进模块展开：

1. **SVLA 推理链路中的视觉 token 压缩**：在 visual token 进入 LLM / action expert prefix 前进行压缩，观察 token 数量、payload、成功率、AEO 触发次数和 episode time 的变化。
2. **AEO / action-aware 动态 token 保留**：利用动作上下文和 AEO predictor 的 `delta_embedding_norm / threshold` 风险信号，在关键动作阶段保留更多视觉 token，在低风险阶段压缩更多 token。
3. **端云协同传输拆分**：评估 image up / action down、token up / action down、token up / KV down 等传输形态，判断传输瓶颈到底来自上行、下行、序列化，还是云端主计算。

最终每组实验都需要同时报告：

- 任务成功率；
- 平均保留 visual tokens；
- token-level payload proxy；
- AEO telemetry：`norm_exceeded`、`skipped_denoise`；
- 每段 action latency breakdown；
- 上行 / 下行 payload bytes；
- LIBERO-10 / 长程任务失败诊断结果。

## 2. 目前已经整理出的实验结果

### 2.1 固定压缩主曲线

| 方法 | Keep ratio | Visual tokens | Token-level payload proxy | Success rate | Episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.00 | 256 | 2.0000 MB | 93.3% | 30 |
| 1D fixed | 0.75 | 192 | 1.5000 MB | 66.7%--80.0% | 30 |
| 1D fixed | 0.50 | 128 | 1.0000 MB | 66.7% | 30 |
| 1D fixed | 0.40 | 102 | 0.7969 MB | 53.3% | 30 |
| 1D fixed | 0.25 | 64 | 0.5000 MB | 13.3% | 30 |

**主要结论：**固定压缩存在明确的 token-saving / success trade-off。0.50 到 0.75 是下一阶段最值得对照的压缩区间；0.25 压缩过强，主要作为失败下界参考。

### 2.2 非单调失效现象

| 方法 | Visual tokens | Success rate | 现象 |
| --- | ---: | ---: | --- |
| 1D fixed 0.90 | 230 | 0% | 轻压缩但异常失败 |
| 1D fixed 0.875 | 224 | 3.3% | 轻压缩但异常失败 |
| 1D fixed 0.80 | 205 | 46.7% | 轻压缩不稳定 |
| 2D-area 0.75 | 196 | 0% | 2D area pooling 异常 |
| 2D-select 0.75 | 196 | 0% | 2D selection 异常 |

**主要结论：**StreamingVLA 的 visual-token 压缩不是简单的“保留 token 越多越好”。压缩算子会改变 visual embedding 分布、token 顺序、prefix 长度和 action generation 稳定性。因此后续主线不再继续做 2D pooling，而是保留固定 1D pooling 作为 baseline，并重点发展 AEO-aware 与 action-sensitive 的 token 保留策略。

## 3. 后续实验组安排

所有实验尽量使用同一个 checkpoint，只切换 `--policy.config`，这样成功率、时延和 AEO 事件才可直接对比。

| 实验组 | Policy config | 目的 |
| --- | --- | --- |
| baseline | `streamingvla_pi05_libero` | 无压缩性能上限；记录原始成功率和端到端时延 |
| fixed 0.75 | `streamingvla_pi05_libero_token_fixed075` | 轻压缩固定 baseline |
| fixed 0.50 | `streamingvla_pi05_libero_token_fixed050` | 中等压缩固定 baseline |
| AEO three-stage | `streamingvla_pi05_libero_token_aeo_three_stage` | 主动态压缩方法，按 AEO risk 分档 |
| action-sensitive | `streamingvla_pi05_libero_token_action_sensitive` | 选择对当前动作推理最敏感的 token 保留 |

## 4. 动态视觉 token 压缩策略

### 4.1 固定压缩策略

- `vision_compression_strategy="fixed"`
- `dynamic_token_fixed_keep_ratio=0.75` 或 `0.50`
- 压缩算子：deterministic 1D average pooling。

该策略不使用 AEO risk，只用于建立固定压缩对照曲线。

### 4.2 AEO-aware 动态策略

风险信号定义为：

```text
risk = delta_embedding_norm / threshold
```

其中 `delta_embedding_norm` 来自 AEO predictor，`threshold` 是 AEO 判断高风险 / 是否跳过 denoise 的阈值。

| 策略 | keep-ratio 规则 | 用途 |
| --- | --- | --- |
| `aeo_dynamic` / `aeo_risk` | 普通观测：1.0；risk > 0.85：0.75；否则：0.50 | 简单动态 baseline |
| `aeo_conservative` | 普通观测：1.0；risk > 0.70：1.0；否则：0.50 | 如果动态压缩导致成功率下降，用该版本保守恢复成功率 |
| `aeo_three_stage` | 普通观测：1.0；risk > 0.90：1.0；0.70 < risk <= 0.90：0.75；否则：0.50 | 当前推荐主方法 |

**推荐主跑：**`aeo_three_stage`。它比 fixed 0.75 更可能降低平均 token 数，又比 fixed 0.50 更可能保护关键动作阶段。

### 4.3 Action-sensitive token selection

`action_sensitive` 策略先根据 AEO risk 选择 keep ratio，然后为每个 visual token 计算重要性分数，并按分数保留 top-k token，最后恢复原始顺序送入后续 prefix。

Token sensitivity score 由三类信息组成：

1. token activation norm；
2. per-token AEO delta norm；
3. 可选的 `action_left_sum` 动作上下文对齐项。

该策略和 1D pooling 的区别是：1D pooling 是局部平均压缩，可能把关键 token 与非关键 token 混在一起；action-sensitive 是直接保留估计最影响当前动作推理的 token，更适合作为下一阶段方法创新点。

## 5. 上下行传输策略

### 5.1 当前可运行 baseline：image up / action down

```text
端侧 / client: image, wrist_image
上行: 原始观测 / 图像输入
云端 / server: vision encoder + LLM/action expert
下行: action
```

这是当前系统最稳定、最接近原始 StreamingVLA 的部署方式，所有压缩实验都应该和它对比。

### 5.2 更合理的后续方向：token up / action down

```text
端侧 / edge: image -> vision encoder -> compressed / quantized visual tokens
上行: 压缩且量化后的 visual tokens
云端 / server: LLM/action expert
下行: action only
```

该方案的核心判断是：如果端侧可以承担 vision encoder 和 token compression，则上行不再传图像，而是传 compressed / quantized visual token；云端只负责计算密集的 LLM / action expert；下行只传 action，不传完整 KV cache。

在 0.3000 MB image uplink proxy、100 Mbps 上行带宽假设下，估算结果如下：

| Token precision | Keep ratio | token_up/action_down total | Transfer estimate | 相对 image_up/action_down |
| --- | ---: | ---: | ---: | --- |
| int8 | 0.75 | 0.3760 MB | 30.01 ms | 略差 |
| int8 | 0.50 | 0.2510 MB | 20.01 ms | 更好 |
| int8 | 0.40 | 0.2002 MB | 15.95 ms | 更好 |
| int8 | 0.25 | 0.1260 MB | 10.01 ms | 传输最好，但成功率风险大 |
| fp16 | 0.75 | 0.7510 MB | 60.01 ms | 更差 |
| fp16 | 0.50 | 0.5010 MB | 40.01 ms | 更差 |
| fp16 | 0.40 | 0.3994 MB | 31.88 ms | 更差 |
| fp16 | 0.25 | 0.2510 MB | 20.01 ms | 传输变好，但压缩过强 |

**主要结论：**token up / action down 必须配合量化和足够压缩才有传输收益。下一阶段最合理的系统方向是：端侧生成 int8 compressed visual token，上行 token，云端执行 LLM / action expert，下行动作。

### 5.3 仅作为探索：token up / KV down

```text
端侧 / edge: compressed visual tokens
上行: token payload
云端 / server: LLM/action expert prefill
下行: optional KV cache
```

KV-cache 下行通常不是主线，因为它的 payload 随以下因素线性放大：

```text
layers × 2(K,V) × prefix_tokens × kv_heads × head_dim × bytes
```

估算脚本显示，在常见设定下 token_up / KV_down 的下行量会远大于 action down。除非 KV cache 能被端侧多步复用，或者做强量化 / pruning，否则不建议把“下行完整 KV cache”作为主方案。

## 6. 每组实验必须记录的测评指标

### 6.1 成功率 / 任务指标

| 指标 | 含义 |
| --- | --- |
| `success_rate` | 成功 episodes / 总 episodes |
| `total_episodes` | rollout episode 数量 |
| `avg_episode_time_success` | 成功 episodes 的平均耗时 |
| `avg_episode_time_all` | 所有 episodes 的平均耗时，包括失败 |
| `avg_actions_success` | 成功 episodes 的平均动作步数 |
| `avg_actions_all` | 所有 episodes 的平均动作步数 |

### 6.2 AEO / 稳定性指标

| 指标 | 含义 |
| --- | --- |
| `norm_exceeded_count` | AEO 高风险触发次数 |
| `skipped_denoise_count` | 因 AEO 风险跳过 denoise 的次数 |
| `delta_embedding_norm` | AEO predictor 输出的 delta magnitude |
| `risk` | `delta_embedding_norm / threshold` |

### 6.3 时延 / 传输指标

| 指标 | 偏高时说明的瓶颈 |
| --- | --- |
| `client_pack_ms` | client 序列化 / 打包开销 |
| `client_send_ms` | websocket send / 上行传输开销 |
| `client_uplink_payload_bytes` | 上行 payload 过大 |
| `server_unpack_ms` | server 反序列化开销 |
| `server_policy_to_action_ms` | 云端 policy / action generation 主计算瓶颈 |
| `server_pack_ms` | server 序列化开销 |
| `client_downlink_payload_bytes` | 下行 payload 过大，尤其用于判断 KV down 是否不可行 |
| `client_unpack_ms` | client 反序列化开销 |
| `client_e2e_ms` | 单步 action 端到端时延 |

### 6.4 瓶颈判断规则

- `server_policy_to_action_ms` 高：瓶颈在模型主计算；优先做 token compression、prefix/KV/attention reduction。
- `client_uplink_payload_bytes` 和 `client_send_ms` 高：瓶颈在上行；优先做端侧 token compression / quantization。
- `client_downlink_payload_bytes` 高：瓶颈在下行；KV down 可能不可行。
- `client_pack_ms` 或 `client_unpack_ms` 高：瓶颈在序列化 / 反序列化。
- `client_e2e_ms` 高但 `server_policy_to_action_ms` 不高：瓶颈更可能来自网络、队列或序列化，而不是模型计算。

## 7. A100 实验流程与命令

### 7.1 基础环境

```bash
cd /home/ubuntu/streamingvla_my
export CUDA_VISIBLE_DEVICES=0
export CKPT=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO/58300
export AEO_PREDICTOR=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO_Predictor
export SVLA_AEO_PREDICTOR_PATH=$AEO_PREDICTOR/svla_predictor.pth
export PORT=8000

mkdir -p outputs/token_compression/timing
mkdir -p outputs/token_compression/videos
mkdir -p outputs/token_compression/logs
```

LIBERO 环境：

```bash
git submodule update --init --recursive
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=glx
```

### 7.2 不加载 checkpoint 的规划脚本

动态压缩 proxy：

```bash
python scripts/experiment_dynamic_token_compression.py --samples 1024 --pretty
```

上下行传输估算：

```bash
python scripts/experiment_kv_token_transport.py \
  --keep-ratios 0.75,0.5,0.4,0.25 \
  --token-bytes-per-value 1 \
  --kv-bytes-per-value 1 \
  --downlink-mbps 1000 \
  --format markdown
```

### 7.3 真实 LIBERO rollout 模板

Server：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=<POLICY_CONFIG> \
  --policy.dir=$CKPT \
  --port=$PORT 2>&1 | tee outputs/token_compression/logs/server_<RUN_NAME>.log
```

Evaluator：

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=glx

python examples/libero/streamingvla.py \
  --host 0.0.0.0 \
  --port $PORT \
  --task-suite-name <TASK_SUITE> \
  --num-trials-per-task <N> \
  --timing-output-path outputs/token_compression/timing/<RUN_NAME>.txt \
  --video-out-path outputs/token_compression/videos/<RUN_NAME> \
  2>&1 | tee outputs/token_compression/logs/eval_<RUN_NAME>.log
```

结果汇总：

```bash
python scripts/summarize_token_compression_results.py \
  outputs/token_compression/timing \
  --format markdown | tee outputs/token_compression/summary.md
```

## 8. A100 上逐步执行的完整命令

下面命令假设：

- 仓库路径：`/home/ubuntu/streamingvla_my`；
- StreamingVLA 权重：`/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO/58300`；
- AEO predictor：`/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO_Predictor/svla_predictor.pth`；
- 单卡 A100：`CUDA_VISIBLE_DEVICES=0`；
- policy server 端口：`8000`。

### 8.1 先跑不加载 checkpoint 的规划脚本

```bash
cd /home/ubuntu/streamingvla_my
export CUDA_VISIBLE_DEVICES=0
export CKPT=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO/58300
export AEO_PREDICTOR=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO_Predictor
export SVLA_AEO_PREDICTOR_PATH=$AEO_PREDICTOR/svla_predictor.pth
export PORT=8000

mkdir -p outputs/token_compression/timing \
         outputs/token_compression/videos \
         outputs/token_compression/logs

python scripts/experiment_dynamic_token_compression.py --samples 1024 --pretty | \
  tee outputs/token_compression/dynamic_proxy.txt

python scripts/experiment_kv_token_transport.py \
  --keep-ratios 0.75,0.5,0.4,0.25 \
  --token-bytes-per-value 1 \
  --kv-bytes-per-value 1 \
  --downlink-mbps 1000 \
  --format markdown | tee outputs/token_compression/transport_proxy.md
```

### 8.2 Smoke test：baseline / fixed050 / action-sensitive

每次只启动一个 server。启动 server 后，在第二个终端跑 evaluator；当前实验结束后，停止 server，再启动下一组 config。

Baseline server：

```bash
cd /home/ubuntu/streamingvla_my
export CUDA_VISIBLE_DEVICES=0
export CKPT=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO/58300
export AEO_PREDICTOR=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO_Predictor
export SVLA_AEO_PREDICTOR_PATH=$AEO_PREDICTOR/svla_predictor.pth
export PORT=8000

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=streamingvla_pi05_libero \
  --policy.dir=$CKPT \
  --port=$PORT 2>&1 | tee outputs/token_compression/logs/server_smoke_baseline.log
```

Baseline evaluator：

```bash
cd /home/ubuntu/streamingvla_my
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=glx
export PORT=8000

python examples/libero/streamingvla.py \
  --host 0.0.0.0 \
  --port $PORT \
  --task-suite-name libero_object \
  --num-trials-per-task 2 \
  --timing-output-path outputs/token_compression/timing/smoke_baseline.txt \
  --video-out-path outputs/token_compression/videos/smoke_baseline \
  2>&1 | tee outputs/token_compression/logs/eval_smoke_baseline.log
```

Fixed 0.50 server：

```bash
cd /home/ubuntu/streamingvla_my
export CUDA_VISIBLE_DEVICES=0
export CKPT=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO/58300
export AEO_PREDICTOR=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO_Predictor
export SVLA_AEO_PREDICTOR_PATH=$AEO_PREDICTOR/svla_predictor.pth
export PORT=8000

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=streamingvla_pi05_libero_token_fixed050 \
  --policy.dir=$CKPT \
  --port=$PORT 2>&1 | tee outputs/token_compression/logs/server_smoke_fixed050.log
```

Fixed 0.50 evaluator：

```bash
cd /home/ubuntu/streamingvla_my
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=glx
export PORT=8000

python examples/libero/streamingvla.py \
  --host 0.0.0.0 \
  --port $PORT \
  --task-suite-name libero_object \
  --num-trials-per-task 2 \
  --timing-output-path outputs/token_compression/timing/smoke_fixed050.txt \
  --video-out-path outputs/token_compression/videos/smoke_fixed050 \
  2>&1 | tee outputs/token_compression/logs/eval_smoke_fixed050.log
```

Action-sensitive server：

```bash
cd /home/ubuntu/streamingvla_my
export CUDA_VISIBLE_DEVICES=0
export CKPT=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO/58300
export AEO_PREDICTOR=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO_Predictor
export SVLA_AEO_PREDICTOR_PATH=$AEO_PREDICTOR/svla_predictor.pth
export PORT=8000

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=streamingvla_pi05_libero_token_action_sensitive \
  --policy.dir=$CKPT \
  --port=$PORT 2>&1 | tee outputs/token_compression/logs/server_smoke_action_sensitive.log
```

Action-sensitive evaluator：

```bash
cd /home/ubuntu/streamingvla_my
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=glx
export PORT=8000

python examples/libero/streamingvla.py \
  --host 0.0.0.0 \
  --port $PORT \
  --task-suite-name libero_object \
  --num-trials-per-task 2 \
  --timing-output-path outputs/token_compression/timing/smoke_action_sensitive.txt \
  --video-out-path outputs/token_compression/videos/smoke_action_sensitive \
  2>&1 | tee outputs/token_compression/logs/eval_smoke_action_sensitive.log
```

### 8.3 主实验：libero_object，10 次 / 30 次 trials

把下面的 `N=10` 先跑完；如果没有 server/evaluator 错误，再改成 `N=30` 重跑主结果。每个 config 仍然需要单独启动对应 server。

```bash
cd /home/ubuntu/streamingvla_my
export CUDA_VISIBLE_DEVICES=0
export CKPT=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO/58300
export AEO_PREDICTOR=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO_Predictor
export SVLA_AEO_PREDICTOR_PATH=$AEO_PREDICTOR/svla_predictor.pth
export PORT=8000
export SUITE=libero_object
export N=10
```

对每个 config 启动 server，例如 baseline：

```bash
export CONFIG=streamingvla_pi05_libero
export RUN=object_${N}_baseline
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=$CONFIG \
  --policy.dir=$CKPT \
  --port=$PORT 2>&1 | tee outputs/token_compression/logs/server_${RUN}.log
```

第二个终端跑对应 evaluator：

```bash
cd /home/ubuntu/streamingvla_my
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=glx
export PORT=8000
export SUITE=libero_object
export N=10
export RUN=object_${N}_baseline

python examples/libero/streamingvla.py \
  --host 0.0.0.0 \
  --port $PORT \
  --task-suite-name $SUITE \
  --num-trials-per-task $N \
  --timing-output-path outputs/token_compression/timing/${RUN}.txt \
  --video-out-path outputs/token_compression/videos/${RUN} \
  2>&1 | tee outputs/token_compression/logs/eval_${RUN}.log
```

主实验需要依次替换为以下 config / run name：

```bash
# baseline
CONFIG=streamingvla_pi05_libero
RUN=object_${N}_baseline

# fixed 0.75
CONFIG=streamingvla_pi05_libero_token_fixed075
RUN=object_${N}_fixed075

# fixed 0.50
CONFIG=streamingvla_pi05_libero_token_fixed050
RUN=object_${N}_fixed050

# AEO-aware three-stage
CONFIG=streamingvla_pi05_libero_token_aeo_three_stage
RUN=object_${N}_aeo_three_stage

# action-sensitive
CONFIG=streamingvla_pi05_libero_token_action_sensitive
RUN=object_${N}_action_sensitive
```

### 8.4 长程任务诊断：libero_10

长程任务先只跑 baseline、AEO three-stage、action-sensitive。

```bash
cd /home/ubuntu/streamingvla_my
export CUDA_VISIBLE_DEVICES=0
export CKPT=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO/58300
export AEO_PREDICTOR=/home/ubuntu/streamingvla_my/checkpoints/StreamingVLA_LIBERO_Predictor
export SVLA_AEO_PREDICTOR_PATH=$AEO_PREDICTOR/svla_predictor.pth
export PORT=8000
export SUITE=libero_10
export N=10
```

长程诊断 config / run name：

```bash
CONFIG=streamingvla_pi05_libero
RUN=libero10_${N}_baseline

CONFIG=streamingvla_pi05_libero_token_aeo_three_stage
RUN=libero10_${N}_aeo_three_stage

CONFIG=streamingvla_pi05_libero_token_action_sensitive
RUN=libero10_${N}_action_sensitive
```

server / evaluator 命令与 8.3 相同，只需要替换 `$SUITE`、`$N`、`$CONFIG`、`$RUN`。

### 8.5 汇总所有结果

```bash
cd /home/ubuntu/streamingvla_my
python scripts/summarize_token_compression_results.py \
  outputs/token_compression/timing \
  --format markdown | tee outputs/token_compression/summary.md

python scripts/summarize_token_compression_results.py \
  outputs/token_compression/timing \
  --format csv | tee outputs/token_compression/summary.csv
```

## 9. 推荐实验顺序

### Stage A：smoke test

任务集：`libero_object`  
每个任务：`num_trials_per_task=2`

依次跑：

1. baseline；
2. fixed 0.50；
3. action-sensitive。

目标是先确认 server / evaluator / timing output / transport telemetry 都能正常记录。

### Stage B：短程任务主实验

任务集：`libero_object`  
先 `num_trials_per_task=10`，稳定后扩展到 30。

依次跑：

1. baseline；
2. fixed 0.75；
3. fixed 0.50；
4. AEO three-stage；
5. action-sensitive。

目标是回答：动态压缩是否能在接近 fixed 0.75 成功率的同时，把平均 token 数降到接近 fixed 0.50。

### Stage C：长程任务诊断实验

任务集：`libero_10` 或当前长程失败较明显的任务集  
每个任务：`num_trials_per_task=10`

依次跑：

1. baseline；
2. AEO three-stage；
3. action-sensitive。

如果失败集中在高 `risk`、高 `norm_exceeded_count` 或 episode 后半段，下一步应该增加 phase-aware replanning / 长程状态识别模块。

## 10. 办公讨论时建议准备的结果表

### 10.1 成功率与稳定性表

| Method | Suite | Success | Episodes | Norm exceeded | Skipped denoise | Avg episode time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | libero_object | | | | | |
| fixed075 | libero_object | | | | | |
| fixed050 | libero_object | | | | | |
| aeo_three_stage | libero_object | | | | | |
| action_sensitive | libero_object | | | | | |

### 10.2 时延与 payload 表

| Method | E2E ms | Server policy ms | Uplink bytes | Downlink bytes | Client pack ms | Client send ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | | | | | | |
| fixed075 | | | | | | |
| fixed050 | | | | | | |
| aeo_three_stage | | | | | | |
| action_sensitive | | | | | | |

### 10.3 长程任务诊断表

| Method | Suite | Success | Failure phase | Avg risk | Norm exceeded | Main observation |
| --- | --- | ---: | --- | ---: | ---: | --- |
| baseline | libero_10 | | | | | |
| aeo_three_stage | libero_10 | | | | | |
| action_sensitive | libero_10 | | | | | |

## 11. 当前阶段的预期判断

1. 如果 `action_sensitive` 的成功率接近 baseline，同时降低 `server_policy_to_action_ms` 和平均 visual tokens，它就是最有价值的模型侧方法。
2. 如果 `fixed050` 明显降时延但成功率下降，它仍然是中等压缩下界 baseline。
3. 如果 `client_uplink_payload_bytes` / `client_send_ms` 是瓶颈，下一步优先做端侧 int8 token uplink。
4. 如果 `server_policy_to_action_ms` 是瓶颈，下一步优先优化 prefix token 数、KV cache 和 attention 计算。
5. 如果长程任务失败伴随高 AEO risk 或高 `norm_exceeded_count`，下一步应从长程状态识别、阶段切换和 replanning 入手，而不是继续盲目压缩 token。
