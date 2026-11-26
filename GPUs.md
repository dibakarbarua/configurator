# GPU Hardware Specifications Glossary

This file describes fine-grained per-cycle estimations on GPUs for a given algorithm for Speed-of-Light estimations.

### `predefined/gpus.py` 
- defines `_SIMT_ARITHMETIC_RESULTS_PER_SM_CYCLE`, `_SIMT_CLOCKS`, and `_simt_roofline()`.
- At module import, these populate `GPU_SPECS[gpu]["SIMT"]` with:

- `SM_Clock_MHz` and `Clock_Basis`: chosen clock and its provenance/assumption.
- `OpsPerSMCycle`: per-operation ceilings by FP32, packed FP16, and packed BF16.
- `TeraOpsPerSecond`: corresponding whole-GPU rates.
- For each operation:
```tex
TeraOpsPerSecond = OpsPerSMCycle * Num_SMs * SM_Clock_MHz / 10^6
isolated_time_ms = operation_count / (TeraOpsPerSecond * 10^9)
```
- FMA rates already count two FLOPs per scalar element.
- Packed rates already count both elements; neither requires another factor of two.
- For exp, reciprocal, conversion, and shuffle, the counts refer to scalar elements
  under the operation definitions in `_simt_roofline()`.

- `Exp` denotes native approximate base-two exponentiation, not full `expf`; `Rcpf`
  denotes approximate FP32 reciprocal, not general division.
- Packed exp/reciprocal ceilings are `None`.
- Unknown B100/L20 clocks produce `None` whole-GPU rates rather than zero throughput.
- Some other clocks are inferred equivalents; `Clock_Basis` records this distinction.

- These independent instruction ceilings cannot simply be added: operations can share
  execution resources.
- `Num_SMs` and `SMEM_KB` are copied into `GPUResources`, but neither they nor the SIMT
  rates affect the current TTFT/ITL calculations.
- In particular, the softmax operation approximation is charged against the selected
  tensor-core compute ceiling, not a separate SIMT ceiling.

### Example: H100 SIMT throughput dictionary

- Retrieve the dictionary with `get_gpu_specs("H100")["SIMT"]` from
  `src/llm_optimizer/predefined/gpus.py`.
- The following excerpt shows selected fields; the full dictionary also includes
  conversion rates, additional operations, and packed BF16 (`bf16x2`) rates.
- Values describe the repository's hardware assumptions, not measured kernel performance.

```python
from llm_optimizer.predefined.gpus import get_gpu_specs

specs = get_gpu_specs("H100")
num_sms = specs["Num_SMs"]  # 132 SMs on this GPU
simt = specs["SIMT"]

# Selected contents of simt (whole-GPU rates rounded here for readability):
{
    "SM_Clock_MHz": 1980.0,
    "Clock_Basis": "Published H100 SXM SIMT boost clock",
    "OpsPerSMCycle": {
        "fp32": {
            "FMA": 256,             # FLOPs per SM per cycle
            "AddOrMultiply": 128,   # FLOPs per SM per cycle
            "Exp": 16,              # scalar elements per SM per cycle
            "Rcpf": 16,
            "Shuffle": 32,
        },
        "fp16x2": {
            "FMA": 512,             # both packed elements and both FLOPs counted
            "AddOrMultiply": 256,
            "Exp": None,            # no native packed ceiling specified
            "Rcpf": None,
            "Shuffle": 64,          # two elements moved together per register
        },
    },
    "TeraOpsPerSecond": {
        "fp32": {
            "FMA": 66.90816,         # whole-GPU TFLOP/s
            "AddOrMultiply": 33.45408,
            "Exp": 4.18176,         # trillion scalar elements per second
            "Rcpf": 4.18176,
            "Shuffle": 8.36352,
        },
        "fp16x2": {
            "FMA": 133.81632,
            "AddOrMultiply": 66.90816,
            "Exp": None,
            "Rcpf": None,
            "Shuffle": 16.72704,
        },
    },
}
```