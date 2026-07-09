# StreamingVLA 实验现状与后续办公测评说明

> 目的：把当前已经在跑/已经完成的实验、接下来建议安排的实验、测评参数指标、动态策略，以及上行/下行通信策略统一整理成一份可直接用于后续办公推进和复现实验的说明文档。

## 1. 当前实验主线

本阶段实验围绕 StreamingVLA 的在线推理效率展开，核心问题不是重新训练模型，而是在保持 LIBERO 任务成功率的前提下，降低视觉 token、传输 payload 和动作生成延迟。

当前主线分为四类：

1. **固定视觉 token 压缩实验**：验证保留固定比例视觉 token 时，成功率和 token/payload 的变化曲线。
2. **AEO-aware 动态视觉 token 压缩实验**：用 AEO predictor 的风险信号动态选择保留比例，避免在高风险动作阶段过度压缩。
3. **Action-sensitive token selection 实验**：在动态保留比例基础上，不再简单平均池化，而是按动作敏感性选择关键视觉 token。
4. **上下行通信/传输策略估算实验**：比较图像上行、视觉 token 上行、KV-cache 下行等协议方案的 payload 和传输时间。

## 2. 已有固定压缩实验结果

### 2.1 主固定压缩曲线

| 实验方法 | Keep ratio | 视觉 token 数 | Token-level payload proxy | LIBERO 成功率 | Episode 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1.00 | 256 | 2.0000 MB | 93.3% | 30 |
| 1D fixed | 0.75 | 192 | 1.5000 MB | 66.7%--80.0% | 30 |
| 1D fixed | 0.50 | 128 | 1.0000 MB | 66.7% | 30 |
| 1D fixed | 0.40 | 102 | 0.7969 MB | 53.3% | 30 |
| 1D fixed | 0.25 | 64 | 0.5000 MB | 13.3% | 30 |

结论：

- 视觉 token 数下降会明显降低 token-level payload proxy。
- 成功率不是只由 token 数决定，0.75 和 0.50 的成功率差距并不总是线性。
- 0.25 过于激进，不适合作为主推策略，只适合做下界/极限压缩对照。

### 2.2 非单调异常实验

| 实验方法 | 视觉 token 数 | 成功率 | 观察 |
| --- | ---: | ---: | --- |
| 1D fixed 0.90 | 230 | 0% | 轻压缩反而崩溃，存在非单调异常 |
| 1D fixed 0.875 | 224 | 3.3% | 轻压缩异常 |
| 1D fixed 0.80 | 205 | 46.7% | 轻压缩不稳定 |
| 2D-area 0.75 | 196 | 0% | 2D area pooling 异常 |
| 2D-select 0.75 | 196 | 0% | 2D selection 异常 |

办公讨论时需要强调：**VLA 视觉 token 压缩不能只看 token 数和带宽，压缩算子对 embedding 分布、token 顺序、位置编码和 action prefix 的扰动也会影响成功率。** 因此后续实验应避免继续大规模投入 2D pooling 异常路径，优先推进运行时稳定的 1D deterministic pooling 和 action-sensitive selection。

## 3. 当前代码中可直接运行的实验配置

以下配置均基于 `streamingvla_pi05_libero`，checkpoint、数据配置和训练超参数保持一致，只切换模型配置中的视觉压缩策略。

| 实验名称 | policy config | 策略类型 | 关键参数 | 办公用途 |
| --- | --- | --- | --- | --- |
| Baseline | `streamingvla_pi05_libero` | 不压缩 | `dynamic_token_compression=False` | 成功率和延迟上界/基线 |
| Fixed 0.75 | `streamingvla_pi05_libero_token_fixed075` | 固定压缩 | `keep_ratio=0.75` | 轻压缩对照 |
| Fixed 0.50 | `streamingvla_pi05_libero_token_fixed050` | 固定压缩 | `keep_ratio=0.50` | 中等压缩对照，可能是传输收益拐点 |
| AEO three-stage | `streamingvla_pi05_libero_token_aeo_three_stage` | AEO 动态三阶段 | 低风险 0.50 / 中风险 0.75 / 高风险 1.00 | 当前主推动态策略 |
| Action-sensitive | `streamingvla_pi05_libero_token_action_sensitive` | 动作敏感 token 选择 | norm 0.25 / delta 0.65 / action 0.10 | 当前主推创新策略候选 |

## 4. 动态视觉 token 策略说明

### 4.1 风险信号

动态策略使用 AEO predictor 相关信号估计当前 observation/action 阶段的风险：

```text
risk = delta_embedding_norm / threshold
```

风险越高，说明提前观测或压缩视觉上下文更可能影响后续动作生成，因此应该保留更多视觉 token。

### 4.2 已实现策略

| 策略 | 决策规则 | 建议用途 |
| --- | --- | --- |
| `fixed` | 全程固定 keep ratio | 基础对照组 |
| `aeo_dynamic` / `aeo_risk` | 正常 observation 保留 1.0；若 `risk > 0.85` 保留 0.75；否则保留 0.50 | 简单动态基线，偏效率 |
| `aeo_conservative` | 正常 observation 保留 1.0；若 `risk > 0.70` 保留 1.0；否则保留 0.50 | 成功率兜底，偏保守 |
| `aeo_three_stage` | `risk > 0.90` 保留 1.0；`0.70 < risk <= 0.90` 保留 0.75；否则保留 0.50 | 主推平衡策略 |
| `action_sensitive` | 先按 AEO 风险确定 keep ratio，再按 token 对动作的敏感性选 top-k token，并保持原始序列顺序 | 主推创新策略，避免平均池化破坏关键 token |

### 4.3 Action-sensitive scoring

`action_sensitive` 的 token 重要性分数由三部分组成：

1. **token activation norm**：视觉 token 本身的激活强度，作为通用视觉显著性。
2. **per-token AEO delta norm**：当前 token 对 AEO 预测变化的影响，是主要敏感性指标。
3. **action-context alignment**：与当前 action context 的一致性，辅助反映动作相关性。

当前权重安排：

```text
score = 0.25 * token_norm + 0.65 * token_delta + 0.10 * action_alignment
```

该策略与 1D average pooling 的区别是：它不平均邻近 token，而是显式保留对当前动作更关键的 token，并按原始序列顺序送入后续 LLM/action expert。

## 5. 推荐的下一轮 LIBERO 实测安排

### 5.1 最小必跑矩阵

为了节省办公和算力成本，下一轮真实 LIBERO sweep 建议先跑下表五组：

| 优先级 | 实验 | policy config | 每任务 trial 数 | 目的 |
| ---: | --- | --- | ---: | --- |
| P0 | Baseline | `streamingvla_pi05_libero` | 30 | 确认 checkpoint 和环境基线 |
| P0 | Fixed 0.75 | `streamingvla_pi05_libero_token_fixed075` | 30 | 对照轻压缩成功率与 latency |
| P0 | Fixed 0.50 | `streamingvla_pi05_libero_token_fixed050` | 30 | 对照中压缩收益和成功率 |
| P0 | AEO three-stage | `streamingvla_pi05_libero_token_aeo_three_stage` | 30 | 主动态策略 |
| P1 | Action-sensitive | `streamingvla_pi05_libero_token_action_sensitive` | 30 | 创新策略，与 three-stage 对比 |

### 5.2 可选扩展矩阵

如果 P0 结果显示动态策略稳定，再追加：

| 扩展实验 | 参数 | 目的 |
| --- | --- | --- |
| Fixed 0.40 | keep ratio 0.40 | 检查更激进压缩的传输收益上限 |
| AEO dynamic | low 0.50 / high 0.75 | 与 three-stage 对比是否过激 |
| AEO conservative | low 0.50 / high 1.00 | 成功率兜底对照 |
| int8 token uplink simulation | token bytes/value = 1 | 判断 token 上行是否真正优于 image 上行 |

## 6. 测评指标与记录格式

### 6.1 必报指标

每个实验必须汇报：

| 指标 | 含义 | 用途 |
| --- | --- | --- |
| Success rate | 成功 episode / 总 episode | 最核心效果指标 |
| Average visual tokens | 平均保留视觉 token 数 | 压缩强度 |
| Average token-level payload proxy | 视觉 embedding 级 payload 代理 | 模型内部传输/计算 proxy |
| Norm exceeded count | AEO norm 超阈值次数 | 风险触发频率 |
| Skipping denoise count | 跳过 denoise 次数 | StreamingVLA 执行节奏指标 |
| Avg episode time, success only | 成功 episode 的平均耗时 | 真实完成效率 |
| Avg episode time, all trials | 全部 episode 平均耗时 | 包含失败情况的整体效率 |
| Avg actions/episode | 每 episode 动作数 | 判断是否更快完成或卡住 |
| Client E2E action latency | 客户端端到端单动作延迟 | 用户感知延迟 |
| Server policy time | 服务端 policy/action generation 耗时 | 计算瓶颈定位 |
| Avg uplink payload bytes | 平均上行 payload | 上行通信压力 |
| Avg downlink payload bytes | 平均下行 payload | 下行通信压力 |

### 6.2 结果文件汇总

LIBERO runner 每个实验输出一个 timing txt，跑完后统一汇总：

```bash
python scripts/summarize_token_compression_results.py outputs/token_compression --format markdown
```

汇总表会包含：success、episodes、成功/全部 episode time、actions、`norm_exceeded`、`skipped_denoise`、`e2e_ms`、`server_policy_ms`、`uplink_bytes`、`downlink_bytes`。

## 7. 服务端与评测命令模板

### 7.1 启动 policy server

只需要替换 `--policy.config` 和 `--policy.dir`：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=streamingvla_pi05_libero_token_aeo_three_stage \
  --policy.dir=/path/to/your/checkpoint \
  --port=8000
```

### 7.2 运行 LIBERO evaluator

```bash
uv run examples/libero/streamingvla.py \
  --host 0.0.0.0 \
  --port 8000 \
  --task-suite-name libero_object \
  --num-trials-per-task 30 \
  --timing-output-path outputs/token_compression/aeo_three_stage.txt \
  --video-out-path outputs/token_compression/videos/aeo_three_stage
```

建议命名规则：

```text
outputs/token_compression/
  baseline.txt
  fixed075.txt
  fixed050.txt
  aeo_three_stage.txt
  action_sensitive.txt
  videos/<experiment_name>/
```

## 8. 上行/下行通信策略

### 8.1 当前 baseline：image_up/action_down

```text
client: 采集 RGB 图像 + observation
uplink: 图像/observation 上行
server: vision encoder + LLM/action expert + policy
Downlink: action 下行
```

特点：

- 实现简单，是当前 websocket 运行基线。
- 上行受图像 payload 影响；下行 action payload 很小。
- 适合先作为系统基线，不需要 edge 侧 vision encoder。

### 8.2 Edge token uplink：token_up/action_down

```text
client/edge: image -> vision encoder -> 压缩/量化视觉 token
uplink: compressed visual tokens
server: LLM/action expert + policy
Downlink: action only
```

判断原则：

- 只有当视觉 token 被足够压缩和量化后，token 上行才可能比图像上行更划算。
- fp32/fp16 token 在 0.75 或 0.50 keep ratio 下通常仍大于 0.3000 MB 图像上行 proxy。
- 更实际的方向是 **int8 token uplink + keep ratio 0.50 左右**，或者动态策略平均 keep ratio 约 0.50--0.60。

### 8.3 Token uplink + KV-cache downlink：token_up/kv_down

```text
client/edge: image -> vision encoder -> compressed visual tokens
uplink: compressed visual tokens
server: LLM/action expert prefill
Downlink: optional prefix KV cache / telemetry payload
```

当前估算结论：

- KV-cache 下行一般远大于视觉 token，因为它随 layers、K/V、prefix length、heads、head dim 同时放大。
- 如果没有明确的 downstream KV reuse，`token_up/kv_down` 很可能比 baseline 更慢。
- 该方案应作为 ablation 明确测试，不应默认作为主系统方案。

### 8.4 默认传输估算参数

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `visual_tokens` | 256 | 单次视觉 token 数 |
| `hidden_dim` | 2048 | token hidden dimension |
| `prefix_tokens` | 456 | visual + language prefix token 数 |
| `layers` | 18 | KV-cache 层数 |
| `kv_heads` | 8 | KV heads |
| `head_dim` | 256 | head dimension |
| `token_bytes_per_value` | 4 | 默认 token fp32 bytes/value |
| `kv_bytes_per_value` | 2 | 默认 KV fp16 bytes/value |
| `image_uplink_mb` | 0.3000 MB | 图像上行 proxy |
| `action_downlink_mb` | 0.0010 MB | action 下行 proxy |
| `uplink_mbps` / `downlink_mbps` | 100 Mbps | 默认网络带宽 |

默认估算命令：

```bash
python scripts/experiment_kv_token_transport.py --keep-ratios 1.0,0.75,0.5 --format markdown
```

默认估算结果：

| mode | keep | uplink_MB | downlink_MB | total_MB | transfer_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| image_up/action_down | 1.000 | 0.3000 | 0.0010 | 0.3010 | 24.08 |
| token_up/action_down | 1.000 | 2.0000 | 0.0010 | 2.0010 | 160.08 |
| token_up/kv_down | 1.000 | 2.0000 | 64.1250 | 66.1250 | 5290.00 |
| image_up/action_down | 0.750 | 0.3000 | 0.0010 | 0.3010 | 24.08 |
| token_up/action_down | 0.750 | 1.5000 | 0.0010 | 1.5010 | 120.08 |
| token_up/kv_down | 0.750 | 1.5000 | 48.0938 | 49.5938 | 3967.50 |
| image_up/action_down | 0.500 | 0.3000 | 0.0010 | 0.3010 | 24.08 |
| token_up/action_down | 0.500 | 1.0000 | 0.0010 | 1.0010 | 80.08 |
| token_up/kv_down | 0.500 | 1.0000 | 32.0625 | 33.0625 | 2645.00 |

## 9. 办公推进建议

### 9.1 本周优先级

1. 先确认 baseline、fixed075、fixed050 在同一 checkpoint、同一 LIBERO suite、同一 trial 数下可以稳定复现。
2. 再跑 `aeo_three_stage`，重点看成功率是否接近 baseline，同时平均 payload 是否接近 fixed050--fixed075 之间。
3. 最后跑 `action_sensitive`，重点比较它是否在同等平均 token 数下优于 1D pooling。
4. 通信方向先不实现完整 edge runtime，先用 estimator + timing telemetry 判断是否值得投入。

### 9.2 决策标准

| 决策问题 | 推荐判断标准 |
| --- | --- |
| 动态压缩是否值得作为主方法？ | 成功率接近 baseline，且平均 token/payload 明显低于 baseline |
| Action-sensitive 是否值得写进主线？ | 同等 token 数下成功率高于 fixed/aeo pooling，或延迟不增加明显 |
| token 上行是否值得做系统实现？ | int8 + keep ratio 0.50--0.60 后 total uplink 明显低于 image baseline |
| KV downlink 是否值得继续？ | 只有存在 KV reuse 并能显著减少后续 server compute 时才继续，否则降级为 ablation |

### 9.3 会议汇报建议结构

1. **先讲现象**：固定压缩出现非单调失败，说明不能只按 token 数评估。
2. **再讲方法**：用 AEO risk 做动态 keep ratio；用 action-sensitive scoring 保留关键 token。
3. **再讲指标**：success、payload、latency、AEO norm、denoise skip、上下行 payload。
4. **最后讲安排**：五组最小 LIBERO sweep + transport estimator，不先投入大规模 edge runtime。

## 10. 风险与注意事项

- 当前 token-level payload proxy 是模型内部视觉 embedding 级 proxy，不等同于真实 websocket 图像上行带宽。
- 真实上行/下行策略需要结合序列化开销、压缩格式、网络抖动和 edge GPU/CPU 能力。
- 轻压缩异常说明算子稳定性很关键；不要只报告 keep ratio，一定要报告具体压缩方式。
- 动态策略需要记录每个 episode 的平均 keep ratio、风险触发分布和高风险动作阶段的成功/失败案例。
- 如果 checkpoint 或 norm stats 不一致，LIBERO 成功率会失真；所有对比必须使用同一 checkpoint 和同一资产配置。
