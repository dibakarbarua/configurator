"""
GPU specifications for LLM performance estimation.
Data compiled from official NVIDIA specifications and technical documentation.

SIMT usage (separate from the existing tensor-core TFLOPS fields)::

    rates = get_gpu_specs("H100")["SIMT"]["TeraOpsPerSecond"]["fp32"]
    fma_time_ms = num_fma_flops / (rates["FMA"] * 1e9)

Use fp16x2/bf16x2 for packed arithmetic, counting individual elements rather
than pairs. FMA rates include both FLOPs per element. See _simt_roofline for
operation definitions and model limits.
"""

from copy import deepcopy


# Native instruction ceilings, in scalar ELEMENT results per SM cycle. FMA
# produces one element result (two FLOPs). Packed rates already include both
# elements of each half2/bfloat162; do not multiply them by two again.
# Source: CUDA Best Practices Guide, Table 5 and footnotes 4, 5, 7, 10:
# https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#arithmetic-instructions
# In particular, Blackwell's packed arithmetic rate differs from Hopper's.
# A100 (CC 8.0) has 128 packed FP16 results/SM/cycle, but footnote 4 limits
# BF16 to 64. Each packed result contains two scalar elements, hence 256 vs
# 128 below. This is an Ampere SIMT limitation; it does not describe tensor
# cores, nor apply to Hopper's equal FP16/BF16 SIMT rates.
_SIMT_ARITHMETIC_RESULTS_PER_SM_CYCLE = {
    "Ampere": {"fp32": 64, "fp16x2": 256, "bf16x2": 128},
    "Ada Lovelace": {"fp32": 128, "fp16x2": 128, "bf16x2": 128},
    "Hopper": {"fp32": 128, "fp16x2": 256, "bf16x2": 256},
    "Blackwell": {"fp32": 128, "fp16x2": 128, "bf16x2": 128},
    "Blackwell Ultra": {"fp32": 128, "fp16x2": 128, "bf16x2": 128},
}


def _simt_roofline(architecture, num_sms, clock_mhz, clock_basis):
    """Build independent native instruction ceilings, not measured latency.

    OpsPerSMCycle counts scalar operations per SM cycle. FMA includes BOTH
    multiply and add (two FLOPs/element); AddOrMultiply counts one FLOP per
    element. Packed rates include both elements. TeraOpsPerSecond scales
    these rates to 1e12 operations/s across the GPU: FMA and AddOrMultiply
    are directly in TFLOP/s, with no client-side multiplier. Other operations
    count one exp, reciprocal, conversion or shuffle per scalar element.
    For an isolated operation, time_ms = operation_count / (rate * 1e9),
    where operation_count is FLOPs for FMA/AddOrMultiply, elements otherwise.

    Exp means ex2.approx.f32 (base TWO), not a full expf implementation;
    Rcpf means approximate FP32 reciprocal, not correctly rounded division.
    None for packed Exp/Rcpf means no single native packed ceiling is given:
    model the actual unpack/convert/FP32/pack sequence instead. None for a
    GPU clock means its whole-GPU rates remain unknown, not zero throughput.

    Conv means floating-point CONVERSION. FromFP32/ToFP32 are directional
    element rates for FP16/BF16, not a blanket 2x packed speedup. These use
    the guide's scalar conversion rates; packed lowering can differ.
    Shuffle assumes converged threads and transfers a whole 32-bit register:
    the packed rate counts two elements moved TOGETHER with one source lane.

    Ceilings sharing execution resources cannot be added as independent
    capacities. Dependencies, occupancy, launch cost and memory traffic must
    be modeled separately. All rates scale linearly with the chosen SM clock.
    """
    per_cycle = {}
    for precision, arithmetic in (
        _SIMT_ARITHMETIC_RESULTS_PER_SM_CYCLE[architecture].items()
    ):
        packed = precision != "fp32"
        per_cycle[precision] = {
            "FMA": 2 * arithmetic,
            "AddOrMultiply": arithmetic,
            "Exp": None if packed else 16,
            "Rcpf": None if packed else 16,
            "Shuffle": 64 if packed else 32,
        }
    # The same conversion is accessible from either endpoint precision.
    per_cycle["fp32"]["ConvToFP16"] = 16
    per_cycle["fp32"]["ConvFromFP16"] = 64
    per_cycle["fp32"]["ConvToBF16"] = 16
    per_cycle["fp32"]["ConvFromBF16"] = 32
    for precision, to_fp32 in (("fp16x2", 64), ("bf16x2", 32)):
        per_cycle[precision]["ConvFromFP32"] = 16
        per_cycle[precision]["ConvToFP32"] = to_fp32

    scale = None if clock_mhz is None else num_sms * clock_mhz / 1e6
    return {
        "SM_Clock_MHz": clock_mhz,
        "Clock_Basis": clock_basis,
        "OpsPerSMCycle": per_cycle,
        "TeraOpsPerSecond": {
            precision: {
                op: None if rate is None or scale is None else rate * scale
                for op, rate in rates.items()
            }
            for precision, rates in per_cycle.items()
        },
    }


# GPU specifications from official NVIDIA sources and datasheets
GPU_SPECS = {
    # NVIDIA Hopper Architecture GPUs
    "H100": {
        "FP16_TFLOPS": 989.5,  # Official NVIDIA specification for SXM5 variant
        "FP8_TFLOPS": 1978.0,  # 4th gen Tensor Cores with FP8 support
        "Memory_Bandwidth_GBs": 3350,  # 3.35 TB/s HBM3 memory bandwidth
        "VRAM_GB": 80,
        "Num_SMs": 132,
        "SMEM_KB": 228,  # Max configurable shared memory per SM (256 KB L1/SMEM)
        "Architecture": "Hopper",
        "Memory_Type": "HBM3",
    },
    "H200": {
        "FP16_TFLOPS": 989.0,  # Official NVIDIA specification
        "FP8_TFLOPS": 1978.0,  # Official NVIDIA specification
        "Memory_Bandwidth_GBs": 4800,  # 4.8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 141,
        "Num_SMs": 132,
        "SMEM_KB": 228,  # Max configurable shared memory per SM (256 KB L1/SMEM)
        "Architecture": "Hopper",
        "Memory_Type": "HBM3e",
    },
    # NVIDIA Ampere Architecture GPUs
    "A100": {
        "FP16_TFLOPS": 312.0,  # Official NVIDIA specification for 80GB variant
        "FP8_TFLOPS": None,  # Not supported on Ampere architecture
        "Memory_Bandwidth_GBs": 2039,  # 2.039 TB/s HBM2e memory bandwidth
        "VRAM_GB": 80,
        "Num_SMs": 108,
        "SMEM_KB": 164,  # Max configurable shared memory per SM (192 KB L1/SMEM)
        "Architecture": "Ampere",
        "Memory_Type": "HBM2e",
    },
    "A100-40GB": {
        "FP16_TFLOPS": 312.0,  # Same compute as 80GB variant
        "FP8_TFLOPS": None,  # Not supported on Ampere architecture
        "Memory_Bandwidth_GBs": 1555,  # 1.555 TB/s HBM2 memory bandwidth
        "VRAM_GB": 40,
        "Num_SMs": 108,
        "SMEM_KB": 164,  # Max configurable shared memory per SM (192 KB L1/SMEM)
        "Architecture": "Ampere",
        "Memory_Type": "HBM2",
    },
    # NVIDIA Ada Lovelace Architecture GPUs
    "L20": {
        "FP16_TFLOPS": 119.5,  # Official NVIDIA specification
        "FP8_TFLOPS": 239.0,  # Ada Lovelace supports FP8
        "Memory_Bandwidth_GBs": 864,  # GDDR6 memory bandwidth
        "VRAM_GB": 48,
        "Num_SMs": 92,
        "SMEM_KB": 100,  # Max configurable shared memory per SM (128 KB L1/SMEM)
        "Architecture": "Ada Lovelace",
        "Memory_Type": "GDDR6",
    },
    "L40": {
        "FP16_TFLOPS": 181.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 362.0,  # Ada Lovelace 4th gen Tensor Cores
        "Memory_Bandwidth_GBs": 864,  # GDDR6 memory bandwidth
        "VRAM_GB": 48,
        "Num_SMs": 142,
        "SMEM_KB": 100,  # Max configurable shared memory per SM (128 KB L1/SMEM)
        "Architecture": "Ada Lovelace",
        "Memory_Type": "GDDR6",
    },
    # NVIDIA Blackwell Architecture GPUs
    "B100": {
        "FP16_TFLOPS": 1800.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 3500.0,  # 5th gen Tensor Cores (dense)
        "Memory_Bandwidth_GBs": 8000,  # 8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 192,  # 8x24GB HBM3e stacks
        "Num_SMs": 144,
        "SMEM_KB": 228,  # Max configurable shared memory per SM (256 KB L1/SMEM)
        "Architecture": "Blackwell",
        "Memory_Type": "HBM3e",
    },
    "B200": {
        "FP16_TFLOPS": 2250.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 4500.0,  # 5th gen Tensor Cores (dense)
        "Memory_Bandwidth_GBs": 8000,  # 8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 192,  # 8x24GB HBM3e stacks
        "Num_SMs": 148,
        "SMEM_KB": 228,  # Max configurable shared memory per SM (256 KB L1/SMEM)
        "Architecture": "Blackwell",
        "Memory_Type": "HBM3e",
    },
    "B300": {
        "FP16_TFLOPS": 2250.0,  # Official NVIDIA Blackwell Ultra spec (dense)
        "FP8_TFLOPS": 4500.0,  # 5th gen Tensor Cores (dense)
        "Memory_Bandwidth_GBs": 8000,  # 8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 288,  # 8x36GB (12-Hi) HBM3e stacks
        "Num_SMs": 160,
        "SMEM_KB": 228,  # Max configurable shared memory per SM (256 KB L1/SMEM)
        "Architecture": "Blackwell Ultra",
        "Memory_Type": "HBM3e",
    },
}


# Clock assumptions are separate from tensor-core throughput. H100/H200 SXM:
# https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance
# A100 SXM (1410 MHz): https://resources.nvidia.com/en-us-hopper-architecture
# Blackwell clocks below are EQUIVALENTS inferred from rounded HGX FP32 peaks,
# not verified boost clocks. HGX lists 600 TFLOP/s across eight GPUs:
# https://www.nvidia.com/en-us/data-center/hgx/
# L40 uses its 90.5 TFLOP/s datasheet peak to infer an equivalent clock:
# https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/support-guide/NVIDIA-L40-Datasheet-January-2023.pdf
# B100/L20 are deliberately left uncalibrated pending a verified SKU clock.
_SIMT_CLOCKS = {
    "H100": (1980.0, "Published H100 SXM SIMT boost clock"),
    "H200": (1980.0, "Published H200 SXM boost clock"),
    "A100": (1410.0, "Published A100 SXM boost clock"),
    "A100-40GB": (1410.0, "Published A100 SXM boost clock"),
    "L20": (None, "Unverified SKU clock; per-SM ceilings only"),
    "L40": (90.5e6 / (142 * 128 * 2), "Inferred from 90.5 FP32 TFLOP/s"),
    "B100": (None, "Unverified SKU clock; per-SM ceilings only"),
    "B200": (75e6 / (148 * 128 * 2), "Inferred from HGX 75 FP32 TFLOP/s"),
    "B300": (75e6 / (160 * 128 * 2), "Inferred from HGX 75 FP32 TFLOP/s"),
}
for _gpu_name, (_clock_mhz, _clock_basis) in _SIMT_CLOCKS.items():
    _specs = GPU_SPECS[_gpu_name]
    _specs["SIMT"] = _simt_roofline(
        _specs["Architecture"], _specs["Num_SMs"], _clock_mhz, _clock_basis
    )


def get_gpu_specs(gpu_name: str) -> dict:
    """
    Get GPU specifications by name.

    Args:
        gpu_name: Name of the GPU (e.g., "H100", "h100", "A100", "a100")

    Returns:
        Dictionary containing GPU specifications

    Raises:
        ValueError: If GPU name is not found
    """
    # Normalize to uppercase for lookup
    normalized_name = gpu_name.upper()

    if normalized_name not in GPU_SPECS:
        available = ", ".join(GPU_SPECS.keys())
        available_lower = ", ".join([name.lower() for name in GPU_SPECS.keys()])
        raise ValueError(f"GPU '{gpu_name}' not found. Available GPUs: {available} (case-insensitive: {available_lower})")

    return deepcopy(GPU_SPECS[normalized_name])


def list_available_gpus() -> list[str]:
    """Return list of available GPU names."""
    return list(GPU_SPECS.keys())


def list_available_gpus_with_lowercase() -> list[str]:
    """Return list of available GPU names including lowercase variants."""
    gpus = list(GPU_SPECS.keys())
    # Add lowercase variants for better tab completion
    gpus.extend([name.lower() for name in GPU_SPECS.keys()])
    return sorted(gpus)


def get_precision_tflops(gpu_name: str, precision: str) -> float:
    """
    Get TFLOPS for a specific precision.

    Args:
        gpu_name: Name of the GPU (case-insensitive)
        precision: "bf16", "fp16" or "fp8"

    Returns:
        TFLOPS value for the precision

    Raises:
        ValueError: If GPU or precision not supported
    """
    specs = get_gpu_specs(gpu_name)

    if precision in ("fp16", "bf16"):
        return specs["FP16_TFLOPS"]
    elif precision == "fp8":
        if specs["FP8_TFLOPS"] is None:
            raise ValueError(f"FP8 precision not supported on {gpu_name}")
        return specs["FP8_TFLOPS"]
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 'fp16' or 'fp8'")
