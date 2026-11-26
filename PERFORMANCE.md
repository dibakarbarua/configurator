## `llm-optimizer estimate`

- The `llm-optimizer estimate` command compares total model compute with memory movement
  to estimate roofline TTFT (time to first token) and TPOT (time per output token,
  called ITL in the code).

### Code flow and metadata map

```text
cli.py: estimate_performance()
  -> performance.py: run_performance_estimation()
       -> common.py: get_model_config_and_precision_from_hf()
       -> performance.py: find_best_performance()
            -> estimate_llm_performance() for each candidate concurrency
                 -> GPUResourceManager.get_total_resources(): load and aggregate hardware specifications
                 -> ModelMemoryCalculator: check weights + KV cache fit for supplied lengths and concurrency
                 -> calculate_transformer_flops(): estimate compute ceiling
                 -> calculate_memory_access_bytes(): estimate memory movement ceiling
                 -> calculate_arithmetic_intensity(): calculate FLOPs per byte moved
                 -> determine_performance_bound(): compare arithmetic intensity with hardware specifications
                      -> use the selected bound to calculate roofline TTFT and TPOT
                 -> PerformanceResult: return latency, throughput, capacity, and bound diagnostics
       -> calculate_concurrency_limits(): estimate KV, prefill, and decode concurrency limits
       -> find_optimal_concurrency_threshold(): sweep concurrency for output-throughput gains
       -> estimate_performance_under_constraints(): select best throughput satisfying supplied SLOs
       -> tuning/: generate suggested benchmark configurations and commands
  -> performance.py: display_performance_estimation_results()
```

| File | Metadata or calculation it owns |
| --- | --- |
| `src/llm_optimizer/cli.py` | `estimate_performance()`: CLI inputs, GPU normalization, parameter collection, estimator invocation. |
| `src/llm_optimizer/cli_utils.py` | GPU detection and interactive parameter collection. |
| `src/llm_optimizer/common.py` | `ModelConfig`, Hugging Face model metadata loading, parameter-count approximation, precision inference, bytes per parameter. |
| `src/llm_optimizer/predefined/gpus.py` | `GPU_SPECS`: compute, memory bandwidth/capacity, SM count, shared memory, architecture, memory type; also SIMT instruction ceilings and clock assumptions. This is the hardware metadata source. |
| `src/llm_optimizer/resources/gpu_manager.py` | `get_total_resources()`: hardware metadata conversion; `get_compute_memory_ratio()`: roofline crossover. |
| `src/llm_optimizer/resources/types.py` | `GPUResources` stores per-GPU compute, capacity, bandwidth, SM count, and shared memory. Its properties multiply compute, capacity, bandwidth, and SM count by GPU count; shared memory per GPU is SM count times bytes per SM. `MemoryBreakdown` stores model, KV-per-token, activation, and overhead bytes; `MemoryLimits` stores capacity budgets and remaining KV capacity. |
| `src/llm_optimizer/resources/memory_calculator.py` | Weight and KV-cache capacity calculations used by the estimator; additional activation/overhead helpers exist separately. |
| `src/llm_optimizer/performance.py` | FLOP and traffic equations, defaults, bottleneck choice, latency, throughput, concurrency searches, SLO heuristics, output formatting. |
| `src/llm_optimizer/tuning/` | Suggested serving configurations and benchmark commands derived after estimation. |
| `tests/test_resources.py` | Checks resource aggregation, GPU metadata, and memory calculations. |
| `tests/test_simt_roofline.py` | Checks SIMT units, architecture differences, unknown ceilings, and isolation of returned metadata. |

> NOTE: Weights are fixed for the model and precision and activations are not included in the memory-footprint in this mode.

### Inputs and notation

| Symbol | Meaning |
| --- | --- |
| `G` | Number of identical GPUs. |
| `B` | Concurrency, treated as a batch of identical requests. |
| `I`, `O` | Input and output token lengths per request. |
| `L`, `d`, `H`, `K` | Layer count, hidden dimension, query-head count, KV-head count. |
| `h = d // H` | Head dimension, using integer division. |
| `d_kv = K * h` | KV projection width. |
| `f = 4*d` | Assumed feed-forward width in the FLOP/traffic helpers. |
| `P` | Parameter count stored in `ModelConfig.num_params`. |
| `s` | Bytes per element: 2 for FP16/BF16, 1 for FP8. |
| `C` | Aggregate peak compute in FLOP/s. |
| `W` | Aggregate device-memory bandwidth in bytes/s. |
| `V` | Aggregate device-memory capacity in bytes. |

### Hardware ceilings and units

`GPUResourceManager.get_total_resources()` and `GPUResources` compute:
```text
C = G * selected_TFLOPS * 10^12
W = G * int(Memory_Bandwidth_GBs * 1024^3)
V = G * int(VRAM_GB * 1024^3)
R = C / W // hardware crossover, FLOP/byte
```

> NOTE:
> - There is no interconnect, collective, topology, replication, or sharding cost in these equations.
- > Consequently, `R` is independent of `G` for identical GPUs.

The `estimate` mode allows **"de-rating"** to account for efficiency losses which are configurable:

| Parameter | Default | Effect |
| --- | --- | --- |
| `mfu_prefill` | 0.45 | Multiplies compute capacity only when prefill is classified compute-bound. |
| `mfu_decode` | 0.30 | Multiplies compute capacity only when decode is classified compute-bound. |
| `vram_util_factor` | 0.90 | Fraction of aggregate capacity allowed for weights plus KV cache. |

> NOTE: There is no explicit bandwidth utilization factor.

### Capacity Gating

- `estimate_llm_performance()` calls `ModelMemoryCalculator` for:

```text
model_bytes = P * s
kv_bytes = 2 * L * K * h * (I + O) * B * s
needed_bytes = model_bytes + kv_bytes
usable_bytes = 0.90 * V            // or caller's vram_util_factor
```

> NOTE: If needed capacity exceeds usable capacity, the function returns infinite TTFT, ITL,
  and end-to-end latency, zero throughput, and the requested/usable memory sizes.

### Work and traffic equations

#### Transformer work: `calculate_transformer_flops()`

Per-sequence model computation FLOPS `F(T)` are calculated as follows:

| Component | FLOPs |
| --- | --- |
| Q/K/V projections | `2 * T * (d*d + 2*d*d_kv) * L` |
| Attention scores, QK transpose | `2 * H * T*T * h * L` |
| Softmax approximation | `3 * H * T*T * L` |
| Attention times V | `2 * H * T*T * h * L` |
| Output projection | `2 * T * d*d * L` |
| Two-matrix FFN approximation | `2 * T * (d*f + f*d) * L` |


> NOTE: The caller multiplies it by concurrency `B` to get total concurrent FLOPs
NOTE: Norms, positional operations, embeddings, and the final vocabulary projection are not
  separately included in this work breakdown.

#### Memory traffic: `calculate_memory_access_bytes()`

Per-sequence memory footprint `D(T,B)` includes:
```
weight_bytes = L * s * [d*(d + 2*d_kv) + d*d + 2*d*f]
kv_access_bytes = 2 * L * K * h * s * T * B
activation_bytes = B*T*d*s + B*H*T*T*s
D(T, B) = weight_bytes + kv_access_bytes + activation_bytes
```

### Prefill and decode estimates

#### Prefill (TTFT)

```text
F_prefill = B * F(I)
D_prefill = D(I, B)
AI_prefill = F_prefill / D_prefill
```

#### Decode (ITL)

- The estimator models one generation step at `Q = I + 1`:

```text
context_attention = L * (4*H*Q*h + 3*H*Q)
F_decode = B * [F(1) + context_attention]
D_decode = D(Q, B)
AI_decode = F_decode / D_decode
```

#### Bound selection and phase time

For either phase, with that phase's `F`, `D`, and MFU `u`:
```text
AI = F / D                         # infinity if D <= 0
memory_bound = AI < R              # strict comparison; equality selects compute

if memory_bound:
    time_seconds = D / W
else:
    time_seconds = F / (C * u)
```

### Aggregate outputs

```
input_throughput_tps  = I * B / TTFT
output_throughput_tps = B / ITL
e2e_latency_s        = TTFT + O * ITL
requests_per_sec     = B / e2e_latency_s
```

### Concurrency search and SLOs

All functions in this section live in `performance.py`.

##### `calculate_concurrency_limits()`
```text
KV limit = max(1, int((usable_bytes - P*s) / KV_bytes_per_request))
prefill limit = clamp(int(C*0.45*0.5 / (2*P*I)), 1, 512)
decode token capacity = min(C*0.30/(2*P), W/(s*P))
decode limit = clamp(int(decode_token_capacity*0.05), 1, 1024)
overall limit = min(KV limit, prefill limit, decode limit)
```
- The 0.5-second TTFT and 0.05-second ITL targets are hardcoded heuristics.
- The MFU and usable-capacity factors can be overridden in direct calls.
- The KV limit uses 1024 as a fallback if per-request KV size is zero.
- Clamping limits to at least one does not prove that one request fits; the core
  capacity gate does.

##### `find_optimal_concurrency_threshold()` 
- Evaluates the detailed estimator up to the overall limit, but adds candidates through 16 when that limit is below 16.
- It tracks best output throughput and stops when relative improvement falls below 5% at
  concurrency 8 or higher.
- This search is theoretical despite the function's docstring referring to "empirical"
  performance.

##### `estimate_performance_under_constraints()` 
- Runs its own powers-of-two search and chooses maximum output throughput among candidates meeting all SLOs.
- For comparisons only, it multiplies estimated latency by a statistical factor: mean
  1.0, median 1.1, p95 1.3, p99 1.6.
- These are fixed heuristics, not percentiles computed from a modeled or sampled latency
  distribution; returned base metrics are not replaced with these adjusted values.

### NEXT: Boundaries to review before hierarchical/agentic extensions

- Compute and device-memory bandwidth are pooled across identical GPUs with
  ideal linear scaling. There are no per-rank capacity checks, TP/DP-specific
  timing, communication costs, or network ceilings.
- There are no separate register, shared-memory, L1, L2, host-memory, or storage
  traffic/bandwidth models. Shared-memory capacity metadata alone does not
  provide a shared-memory roofline.
- A workload is one model, one input length, one output length, and a synchronous
  concurrency value. Arrival processes, queues, mixed distributions, scheduling,
  agent branches, tool latency, and cross-model dependencies are absent.
- The work, traffic, and capacity models use different approximations. Their
  differences above should be resolved explicitly when adding architecture or
  kernel-specific models.
- Fixed utilization and percentile factors are not fitted by this estimation
  flow. Benchmarking is a separate execution path, not an automatic calibration
  step for these predictions..
