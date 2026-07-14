# StreamingVLA Dynamic Visual Token Compression Experiments

## Current experimental results

### Main fixed-compression curve

| Method | Keep ratio | Visual tokens | Token-level payload proxy | Success rate | Episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1.00 | 256 | 2.0000 MB | 93.3% | 30 |
| 1D fixed | 0.75 | 192 | 1.5000 MB | 66.7%--80.0% | 30 |
| 1D fixed | 0.50 | 128 | 1.0000 MB | 66.7% | 30 |
| 1D fixed | 0.40 | 102 | 0.7969 MB | 53.3% | 30 |
| 1D fixed | 0.25 | 64 | 0.5000 MB | 13.3% | 30 |

### Non-monotonic failure cases

| Method | Visual tokens | Success rate | Observation |
| --- | ---: | ---: | --- |
| 1D fixed 0.90 | 230 | 0% | Non-monotonic anomaly |
| 1D fixed 0.875 | 224 | 3.3% | Non-monotonic anomaly |
| 1D fixed 0.80 | 205 | 46.7% | Light compression is unstable |
| 2D-area 0.75 | 196 | 0% | 2D area pooling anomaly |
| 2D-select 0.75 | 196 | 0% | 2D selection anomaly |

## Innovation point 1: visual token compression validation framework

The PyTorch StreamingVLA inference path now has a visual token compression insertion point before visual tokens enter the LLM/action expert prefix:

```text
image / wrist_image
  -> vision encoder
  -> visual token embedding
  -> token compression
  -> LLM / action expert prefix
  -> action generation
```

The framework reports token count and a token-level payload proxy, and the LIBERO evaluator records task success rate, AEO `norm_exceeded` count, denoise-skip count, episode time, and action count.  The payload is a token-level proxy for visual embedding traffic inside the server; it is not yet client-server image transmission bandwidth.

## Innovation point 2: non-monotonic fixed-compression failure

The fixed-compression baseline shows that success does not degrade monotonically with the number of retained tokens.  Some light-compression points, especially 0.90/0.875/0.80 and untrained 2D pooling variants, fail more severely than lower-token 1D baselines.

This suggests that VLA visual token compression cannot be evaluated only by token count.  The compression operator can perturb visual embedding distribution, token order, position IDs, and the action-generation prefix in ways that affect VLA stability.

## Innovation point 3: AEO-aware dynamic visual token compression

The next method uses the AEO predictor's delta norm and threshold as a risk signal:

```text
risk = delta_embedding_norm / threshold
```

Implemented strategies:

| Strategy | Rule | Intended use |
| --- | --- | --- |
| `aeo_dynamic` / `aeo_risk` | normal observation: 1.0; risk > 0.85: 0.75; otherwise: 0.50 | Simple conservative/efficient baseline |
| `aeo_conservative` | normal observation: 1.0; risk > 0.70: 1.0; otherwise: 0.50 | Recover success if `aeo_risk` is too aggressive |
| `aeo_three_stage` | normal observation: 1.0; risk > 0.90: 1.0; 0.70 < risk <= 0.90: 0.75; otherwise: 0.50 | Main dynamic method |

## Recommended next experiments

Run only these four groups for the next real LIBERO sweep:

| Method | Purpose |
| --- | --- |
| Baseline 1.0 | Upper-bound success rate |
| Fixed 0.75 | Light-compression control |
| Fixed 0.50 | Medium-compression control |
| AEO-aware dynamic | Proposed method |

Metrics to report:

- Success rate
- Average visual tokens
- Average token-level payload proxy
- Norm exceeded count
- Skipping denoise count
- Average episode time
- VLA forward / action generation latency

## Concrete LIBERO command matrix

Use the same checkpoint directory for every row and only switch `--policy.config`:

| Experiment | Policy config |
| --- | --- |
| Baseline | `streamingvla_pi05_libero` |
| Fixed 0.75 | `streamingvla_pi05_libero_token_fixed075` |
| Fixed 0.50 | `streamingvla_pi05_libero_token_fixed050` |
| AEO-aware three-stage | `streamingvla_pi05_libero_token_aeo_three_stage` |
| Action-sensitive token selection | `streamingvla_pi05_libero_token_action_sensitive` |

All three compression configs use the `vision_compression_strategy` field and apply deterministic 1D average pooling after the keep ratio is selected. No 2D pooling is used in the runtime path.

Example server command:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=streamingvla_pi05_libero_token_aeo_three_stage \
  --policy.dir=/path/to/your/checkpoint \
  --port=8000
```

Example evaluator command:

```bash
uv run examples/libero/streamingvla.py \
  --host 0.0.0.0 \
  --port 8000 \
  --task-suite-name libero_object \
  --num-trials-per-task 30 \
  --timing-output-path outputs/token_compression/aeo_three_stage.txt \
  --video-out-path outputs/token_compression/videos/aeo_three_stage
```

After running all variants, summarize timing files with:

```bash
python scripts/summarize_token_compression_results.py outputs/token_compression --format markdown
```

## KV-cache / token split-transport design

A separate transport ablation is planned for cases where the vision encoder can run on the client/edge side:

```text
client/edge: image -> vision encoder -> compressed visual tokens
uplink:      compressed visual tokens
server:      LLM/action expert prefill -> prefix KV cache / actions
downlink:    optional prefix KV cache telemetry/cache payload
```

The current implementation should be treated as a planning/estimation step rather than a full edge-runtime implementation.  The key question is whether moving from image upload to token upload, or adding KV-cache downlink, improves end-to-end latency under realistic bandwidth assumptions.

Important boundary:

- Token uplink only helps if image upload is expensive or if visual token payload is quantized/compressed below the image payload.
- KV-cache downlink is usually much larger than visual tokens because it scales with layers, K/V tensors, prefix length, heads, and head dimension.
- Therefore `token_up/kv_down` should be tested as an explicit ablation; it is not assumed to be better.

Run the estimator with:

```bash
python scripts/experiment_kv_token_transport.py --keep-ratios 1.0,0.75,0.5 --format markdown
```

The estimator reports uplink MB, downlink MB, total MB, and transfer-time estimates for:

1. `image_up/action_down` current baseline;
2. `token_up/action_down` edge-token upload;
3. `token_up/kv_down` split token/KV-cache transport.

## Edge compressed/quantized token uplink result

For the more practical split direction, the experiment keeps action generation on the cloud/server and only changes the uplink object:

```text
client/edge: image -> vision encoder -> 1D-compressed + quantized visual tokens
uplink:      compressed/quantized visual tokens
server:      LLM/action expert
Downlink:    action only
```

Under the default image uplink proxy of 0.3000 MB and 100 Mbps uplink, the useful region is:

| Token precision | Keep ratio | token_up/action_down total | transfer estimate | Compared with image_up/action_down |
| --- | ---: | ---: | ---: | --- |
| int8 | 0.75 | 0.3760 MB | 30.01 ms | worse than 0.3010 MB / 24.01 ms |
| int8 | 0.50 | 0.2510 MB | 20.01 ms | better |
| int8 | 0.40 | 0.2002 MB | 15.95 ms | better |
| int8 | 0.25 | 0.1260 MB | 10.01 ms | best transfer, but likely hurts success |
| fp16 | 0.75 | 0.7510 MB | 60.01 ms | worse |
| fp16 | 0.50 | 0.5010 MB | 40.01 ms | worse |
| fp16 | 0.40 | 0.3994 MB | 31.88 ms | worse |
| fp16 | 0.25 | 0.2510 MB | 20.01 ms | better transfer, but too aggressive for success |

Conclusion: token uplink becomes transport-positive only when the visual tokens are quantized and compressed enough.  The most plausible next system experiment is int8 token uplink with keep ratio around 0.50 or AEO-aware dynamic compression whose average keep ratio is near 0.50--0.60, followed by cloud LLM/action expert and action-only downlink.


## Action-sensitive visual token selection module

A new action-sensitive token selector scores visual tokens before compression and keeps the most important tokens for the current action inference step.  The score combines:

1. token activation norm, as a generic visual saliency signal;
2. per-token AEO delta norm, as the main sensitivity signal for how much AEO changes each visual token;
3. optional action-context alignment from `action_left_sum`.

The `action_sensitive` strategy first computes `risk = delta_embedding_norm / threshold`, chooses a keep ratio with the same low/high rule as `aeo_dynamic`, and then keeps top-scoring sensitive tokens in original sequence order.  This differs from 1D pooling: it does not average neighboring tokens, but explicitly preserves tokens estimated to matter most for the current action generation.

## Latency breakdown telemetry

The streaming websocket path now records per-action timing and payload fields so each rollout can separate compute and transport bottlenecks:

- client pack time;
- client websocket send time;
- uplink payload bytes;
- server unpack time;
- server policy/action-generation time;
- server pack time;
- downlink payload bytes;
- client unpack time;
- client end-to-end action latency.

These fields are written into the LIBERO timing file as task averages and parsed by `scripts/summarize_token_compression_results.py`.  Use them to decide whether a run is bottlenecked by uplink transport, cloud policy compute, downlink transport, or client-side serialization.
