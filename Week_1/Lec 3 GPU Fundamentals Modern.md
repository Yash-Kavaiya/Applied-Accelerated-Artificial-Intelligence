# Study Notes: GPU Fundamentals & Modern AI Accelerators

## 1. Learning Objectives

By the end of this segment, you should be able to:

1. Describe the **SIMT** (Single Instruction, Multiple Threads) execution model of GPUs
2. Explain the **CUDA thread hierarchy**: threads → warps → blocks → grids
3. Identify the role of **Tensor Cores** in matrix-multiplication acceleration
4. Compare **NVIDIA H100, AMD MI300X, Google TPU v4, and Graphcore IPU** architectures
5. Select appropriate accelerators for **training vs. inference** workloads based on specifications

## 2. Why This Segment Matters

The previous segment covered CPU performance metrics, workload classification, and the roofline model. This segment shifts focus to **how GPUs actually achieve the performance numbers seen earlier**, and surveys the broader landscape of AI accelerators (GPUs, ASICs, custom chips) so you know **which device to pick for which workload** — e.g., CNNs, RNNs/reinforcement learning, Transformers, diffusion models, and other generative AI models.


## 3. SIMT: Single Instruction, Multiple Threads

- GPUs extract parallelism the way CPUs use **SIMD** (Single Instruction, Multiple Data) — but "at scale," across **threads** rather than just data lanes.
- **SIMT = Single Instruction, Multiple Threads.** The GPU executes **thousands of threads simultaneously**, grouped into lockstep units called **warps**.
- A **warp = 32 threads** executing the *same instruction* on *different data* simultaneously. This is essentially "SIMD at scale."
- **Warp divergence:** If a kernel contains `if-else` branches, threads within a warp may take different paths, forcing **serialization** (some threads idle while others execute). This hurts performance, so GPU kernels are written to **avoid heavy control flow / branching** — GPUs favor a data-flow-like execution style.
- **Latency hiding:** If one warp is stalled waiting on data (e.g., memory access), the warp scheduler/dispatch unit immediately switches to and executes another *ready* warp. This hides memory latency instead of wasting cycles.

<img width="1448" height="1086" alt="image" src="https://github.com/user-attachments/assets/efdece0f-0fb8-495a-a38b-0c665a5910dc" />

## 4. Streaming Multiprocessor (SM) Architecture — NVIDIA H100 (Hopper/SXM)

The H100 SM is the fundamental building block of the GPU. Key numbers:

| Property | Value |
|---|---|
| SMs per H100 GPU | **132** |
| CUDA cores per SM | **128** (FP32) |
| Total CUDA cores | 132 × 128 = **16,896** |
| Tensor Core units per SM | **4** |
| Total Tensor Cores | **528** |
| Register file per SM | **256 KB** = 65,536 × 32-bit registers |
| Shared memory / L1 per SM | **228 KB** (configurable split) |
| L2 cache (shared across all SMs) | **~50 MB**, ~30-cycle latency |
| HBM3 memory | **80 GB**, **3,350 GB/s (~3.35 TB/s)** bandwidth |



### Inside one SM
Each SM contains **4 execution units** (partitions), each with its own **warp scheduler + dispatch unit**, so an SM can issue **4 warp instructions simultaneously**. Each of these 4 units contains:

- **32× FP32 units** (single precision)
- **16× FP64 units** (double precision)
- → 32 × 4 = **128 CUDA cores** (FP32) per SM
- **16× INT32 units** — combined with a Special Functional Unit (SFU) and Load/Store (LD/ST) unit
  - **INT32:** specialized for integer operations
  - **SFU:** specialized for transcendental math functions (sine, log, etc.)
  - **LD/ST:** handles memory read/write operations
- **1 Tensor Core** (per execution unit → 4 per SM, 528 total per GPU)

> **CUDA Cores vs. Tensor Cores — key distinction:**
> - **CUDA cores** are simple, general-purpose ALUs (not as specialized as a CPU core) that execute basic arithmetic operations.
> - **Tensor Cores** are highly specialized fixed-function units built to do **only** matrix multiply-accumulate (MAC) operations — the core computation of AI workloads.

<img width="1448" height="1086" alt="image" src="https://github.com/user-attachments/assets/e9eb9bc1-ad98-4fd1-befc-d8bd6ad77a7e" />


## 5. CUDA Thread Hierarchy

CUDA (NVIDIA's GPU programming framework/software stack) organizes execution in a hierarchy:

```
Thread  →  Warp (32 threads)  →  Block (up to 1024 threads)  →  Grid (millions of threads)
```

- **Thread:** smallest unit of execution.
- **Warp:** group of 32 threads executing in lockstep (the SIMT unit).
- **Block:** collection of warps, up to 1024 threads, scheduled together on one SM.
- **Grid:** collection of blocks — can represent millions of threads total, mapped across the whole GPU.

### Memory visibility across the hierarchy
- **Shared memory:** per-block; very fast (no cache-miss behavior), and **programmer-managed** (explicitly allocated/controlled in code).
- **Global memory (HBM):** accessible by *all* threads across the GPU; has **high latency** but **high bandwidth** — this is the "off-chip-relative" memory tier discussed relative to the on-chip caches.

This hierarchy determines how many threads you offload per SM and shapes how a CUDA kernel is written and tuned.

<img width="1448" height="1086" alt="image" src="https://github.com/user-attachments/assets/8f74d9ae-56c8-4aa5-ac22-a47d00f4722e" />


## 6. Tensor Cores: Accelerating the GEMM at the Heart of AI

### What is a Tensor Core?
A **Tensor Core** is a specialized fixed-function unit that performs:

**D = A × B + C**

(a matrix multiply-accumulate / MAC operation on small "tiny" matrices) — **entirely within one clock cycle**. This is the operation underlying nearly all deep learning compute (GEMMs — General Matrix Multiplications).

- **4th Generation Tensor Core (H100):** 1 Tensor Core performs **256 FP16 MACs per clock cycle**.
- **H100 SXM5 peak FP16 performance:**
  528 Tensor Cores × 256 ops/cycle × 1.98 GHz ≈ **270 TFLOPS (FP16)**

> **Important limitation:** Tensor Cores are only effective for **large matrix shapes**. Small GEMM sizes **underutilize** them.

### Supported Precisions and Use Cases (H100)

| Precision | Peak Performance | Use Case |
|---|---|---|
| **FP64** | ~67 TFLOPS | Scientific simulation (needs high accuracy); **not used** in DL training |
| **TF32** (19-bit "TensorFloat-32") | ~989 TFLOPS | Drop-in replacement for FP32 in training — **no code change needed** (hardware auto-drops FP32 → TF32) |
| **BF16 / FP16** | ~1979 TFLOPS | Standard for **mixed-precision training** (paired with FP32 master weights) |
| **FP8** (E4M3/E5M2) | ~3958 TFLOPS | Emerging standard for **LLM training** (via Transformer Engine) |
| **INT8 / INT4** | INT4 ≈ 7.9 PFLOPS (7,900+ TFLOPS-equivalent) | **Quantized inference** — extreme throughput, generally not used for training |

**Comparison — A100 vs H100 Peak TFLOPS/TOPS by precision:**

| Precision | A100 SXM4 | H100 SXM5 |
|---|---|---|
| FP64 | 20 | 67 |
| TF32 | 156 | 989 |
| BF16/FP16 | 312 | 1979 |
| FP8 (E4M3/E5M2) | N/A (no comparable mode) | 3958 |
| INT8 | 624 | 3958 |
| INT4 | 1248 | 7916 |

→ Roughly an **8–10× generational gain** from Ampere (A100) to Hopper (H100), depending on precision.

**Notes on precision types:**
- **TF32:** 19-bit tensor float; if your code uses FP32, the hardware automatically drops to TF32 internally — you get a performance boost **without changing any code**.
- **BF16 (bfloat16):** introduced by Google Brain; same exponent range as FP32 but reduced mantissa — good for training stability.
- Lower precision (FP8, INT8, INT4) = higher throughput but reduced numerical accuracy, so it's used differently for training vs. inference (see below).

### Mixed Precision Training (AMP)
- **Weights are stored in FP32** (the "master copy") to preserve numerical stability.
- The **forward and backward passes** run in **FP16 (or BF16/TF32)** via Tensor Cores for speed.
- **Loss scaling** is used to prevent FP16 underflow (gradients can become too small to represent in 16-bit).
- After gradients are computed, the **optimizer step** updates the FP32 master copy from the 16-bit gradient copies.
- In PyTorch, this is enabled with a single line: `torch.cuda.amp.autocast()` — it automatically handles the FP32 ↔ lower-precision conversions during forward/backward passes and returns results in full precision.

### Training vs. Inference precision (general rule of thumb)
- **Training:** typically uses FP32 (master) + FP16/BF16/TF32 (compute) via AMP; FP8 is emerging for LLM training.
- **Inference:** typically uses **INT8 or INT4** (quantized inference) for maximum throughput and lower memory footprint.

---

## 7. Beyond NVIDIA: The AI Accelerator Landscape

Not all AI compute runs on NVIDIA GPUs. Key alternative players:

### AMD MI300X (2023)
- Chiplet-based architecture: **13 chiplets** (3nm CDNA3 compute chiplets + 6× HBM3 stacks)
- **1.307 PFLOPS FP16** — surpasses the H100's FP16 performance
- **192 GB HBM3 memory @ 5.3 TB/s** — the **largest GPU VRAM available**, more than double the H100's 80GB
- Software stack: **ROCm**, with PyTorch/JAX support via **HIP** (a CUDA portability/translation layer)
- **Best for LLM inference** — a huge model can fit entirely within a single GPU's VRAM (no need to shard across GPUs)

### Google TPU v4 (2021)
- **Custom ASIC** — no general-purpose CUDA cores; built almost purely around matrix multiplication (only Tensor-Core-like units, no equivalent of general CUDA cores)
- **275 TFLOPS (BF16)**, **32 GB HBM2e @ 1.2 TB/s**
- **TPU Pod v4:** ~4,096 chips interconnected via a **custom 3D torus network**
- **Systolic array architecture:** data flows through a mesh of MAC units without control overhead
- Optimal for **JAX/XLA workloads** (TensorFlow, Flax) — not a general-purpose accelerator
- Accessed via **Google Cloud**

### Graphcore IPU (Intelligence Processing Unit)
- **1,472 independent processor tiles**, each with local SRAM
- Execution model: **BSP (Bulk Synchronous Parallel)** — a compute phase alternates with a communication phase
- **Ideal for sparse, irregular workloads**: Graph Neural Networks (GNNs), symbolic AI
- **Poor for dense GEMM** / standard matrix-multiplication-heavy workloads (e.g., Transformers/LLMs are not a good fit)

### AWS Trainium & Inferentia (Custom ASICs)
- **Trainium2:** ~190.7 TFLOPS (FP16) — optimized for AWS SageMaker training jobs
- **Inferentia2:** ~190 TOPS (INT8) — purpose-built for low-latency inference serving
- Both come with a ready-to-use software stack on AWS Cloud

### Quick Comparison Table

| Accelerator | Type | Peak Perf. (approx.) | Memory | Best For |
|---|---|---|---|---|
| NVIDIA H100 SXM5 | GPU | 3958 TFLOPS (FP8) | 80 GB HBM3 @ 3.35 TB/s | General AI training/inference, LLMs |
| AMD MI300X | GPU (chiplet) | 1.307 PFLOPS (FP16) | 192 GB HBM3 @ 5.3 TB/s | LLM inference (fits huge models in 1 GPU) |
| Google TPU v4 | Custom ASIC | 275 TFLOPS (BF16) | 32 GB HBM2e @ 1.2 TB/s | JAX/XLA/TensorFlow workloads |
| Graphcore IPU | Custom (tile-based) | — | Local SRAM per tile | Sparse/irregular workloads: GNNs, symbolic AI |
| AWS Trainium2 | Custom ASIC | 190.7 TFLOPS (FP16) | — | AWS-native training |
| AWS Inferentia2 | Custom ASIC | 190 TOPS (INT8) | — | Low-latency inference serving |

### Accelerator Selection Guide (rules of thumb from the lecture)
- **Large-model training:** H100 SXM or MI300X (highest HBM bandwidth)
- **Cost-efficient training:** A100 80GB or H100 PCIe clusters (Ampere-generation for cost efficiency)
- **Low-latency inference:** Inferentia2, L40S, or quantized RTX 4090 / RTX-series GPUs
- **Batch inference (best performance/dollar):** A10G, L4, T4 (commonly on Google Cloud)
- **Sparse/graph workloads:** Graphcore IPU
- **JAX/TensorFlow-centric workloads:** Google TPU

---

## 8. GPU Memory Management & VRAM Optimization

### Why VRAM is the binding constraint
- Example: **GPT-3 (175B parameters)** — FP16 weights alone require **~350 GB of VRAM**.
- A single H100 has only **80 GB** → you'd need **5+ GPUs** just to hold the weights.
- Total VRAM usage during training = **weights + gradients + optimizer states + activations** — all of these must fit *simultaneously* in memory.
- **Adam optimizer** stores **3 copies per weight** (parameter, momentum, variance) → **3× the weight size** just for optimizer state.

### ZeRO (Zero Redundancy Optimizer) — Microsoft DeepSpeed
A family of techniques to shard/partition memory-heavy components across multiple GPUs instead of duplicating them on every GPU:

| Stage | What it shards | Memory reduction |
|---|---|---|
| **ZeRO-1** | Optimizer states | ~4× reduction |
| **ZeRO-2** | Optimizer states + gradients | ~8× reduction |
| **ZeRO-3** | Parameters + gradients + optimizer states | Linear scaling with GPU count |
| **ZeRO-Offload** | Spills optimizer states to **CPU RAM** | Enables ~10× larger models on the same GPU |

### Practical VRAM Budgeting Formulas
- **Training memory (bytes) ≈ 16 × model_parameters** (for FP16 + AMP + Adam optimizer)
- **Inference memory (bytes) ≈ 2 × model_parameters** (FP16) **or ≈ 1 × model_parameters** (INT8)
- **Activation memory per layer** is proportional to: **batch_size × sequence_length × hidden_dimension**

> **Intuition:** Inference is cheaper than training memory-wise because there are no gradients or optimizer states to store — only the model weights (and activations) need to fit.

### Useful diagnostic tools
- `nvidia-smi --query-gpu=memory.used,memory.free --format=csv` — quick GPU memory usage check
- `torch.cuda.memory_summary()` — detailed per-tensor memory breakdown in PyTorch

---

## 9. Segment Summary (Key Takeaways)

- ✅ GPUs execute thousands of threads in **warps of 32** via the **SIMT** model; **occupancy** and **warp scheduling** hide memory latency.
- ✅ **Tensor Cores** perform matrix MACs per cycle — the H100 reaches **3,958 TFLOPS at FP8**, roughly **600× a single CPU core**.
- ✅ **HBM3 (3.35 TB/s)** is the defining bandwidth advantage of the H100; the **AMD MI300X** offers even more — **192 GB / 5.3 TB/s**.
- ✅ **TPUs** (systolic arrays), **IPUs** (tile-based), and **cloud custom ASICs** (Trainium/Inferentia) provide architectural diversity suited to specific workload types.
- ✅ **ZeRO-3 sharding** and related memory-optimization techniques are essential for training models that exceed single-GPU VRAM limits.

---

## 10. Quick-Reference Glossary

| Term | Meaning |
|---|---|
| **SIMT** | Single Instruction, Multiple Threads — GPU execution model |
| **SIMD** | Single Instruction, Multiple Data — CPU-side parallelism concept |
| **Warp** | Group of 32 threads executing in lockstep |
| **SM** | Streaming Multiprocessor — core compute block of an NVIDIA GPU |
| **CUDA core** | General-purpose ALU inside an SM (FP32/FP64/INT32 ops) |
| **Tensor Core** | Fixed-function unit specialized for matrix multiply-accumulate (MAC) |
| **GEMM** | General Matrix Multiplication — the core AI compute operation |
| **HBM** | High Bandwidth Memory — on-package GPU memory (e.g., HBM3) |
| **TF32 / BF16 / FP8** | Reduced-precision floating-point formats used to trade accuracy for speed |
| **AMP** | Automatic Mixed Precision — training technique combining FP32 master weights with FP16/BF16 compute |
| **ZeRO** | Zero Redundancy Optimizer — memory-sharding technique (DeepSpeed) |
| **ASIC** | Application-Specific Integrated Circuit (e.g., TPU, Trainium) |
| **Systolic array** | Data flows through a fixed mesh of MAC units without control overhead (used in TPUs) |
| **BSP** | Bulk Synchronous Parallel — Graphcore IPU's compute/communicate execution model |

---

## 11. References (as cited in lecture slides)

- NVIDIA H100 GPU Architecture Whitepaper (2022)
- Kirk & Hwu, *"Programming Massively Parallel Processors,"* 4th Ed.
- NVIDIA Hopper Architecture In-Depth (2022)
- Micikevicius et al., *"Mixed Precision Training,"* ICLR 2018
- AMD MI300X Architecture Whitepaper (2023)
- Google TPU v4, Jouppi et al., ISCA 2023
- Rajbhandari et al., *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models,"* SC'20
