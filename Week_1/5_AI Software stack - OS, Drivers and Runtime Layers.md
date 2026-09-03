# NPTEL — Applied Accelerated Artificial Intelligence

> *The layered software architecture that connects your Python code to silicon transistors.*

Previously, the course covered the **hardware** side (CPUs, GPUs, accelerators, performance metrics). This segment shifts focus to the **software stack** — the layers of code, libraries, drivers, and OS configuration that sit between a Python/PyTorch program and the physical transistors executing it.

## Learning Objectives

1. Map the full AI software stack from user-space frameworks down to kernel drivers.
2. Explain the role of **CUDA Runtime**, **cuDNN**, **cuBLAS**, and **NCCL** in the NVIDIA stack.
3. Describe GPU driver management, kernel modules, and CUDA version compatibility.
4. Understand container runtimes (Docker, containerd) and the **NVIDIA Container Toolkit**.
5. Identify key Linux kernel features leveraged by AI workloads (huge pages, cgroups, IOMMU).

## 1. The Full AI Software Stack — Layer by Layer

*(Fig 13: The Complete AI Software Stack, Ref: NVIDIA CUDA Programming Guide v12 (2024); CUDA C++ Best Practices Guide)*

The stack runs from **User → Hardware** (top to bottom):

| # | Layer | Examples / Components | What Happens Here |
|---|-------|------------------------|--------------------|
| 8 | **User Applications & Research Code** | Hugging Face Transformers, PyTorch Lightning, Keras, Fast.ai | The actual application/research code a developer writes, often built on top of higher-level APIs. |
| 7 | **ML Frameworks** | PyTorch, TensorFlow, JAX/Flax | Define the ML workflow — forward pass, backward pass, model definitions. |
| 6 | **Compilers & Graph Optimisers** | `torch.compile` (Triton/Inductor), XLA, TensorRT, TVM | The framework code is compiled into a **computation graph**. Operations are **fused, simplified, or eliminated** if unnecessary. This is where graph-level optimization happens. |
| 5 | **CUDA Middleware Libraries** | cuDNN, cuBLAS, NCCL, cuSPARSE, cuFFT, Thrust | Thousands of pre-implemented algorithms for specific operation types (convolution, attention, matmul, communication, sparse ops, FFT). Each op in the computation graph is mapped to a library routine. |
| 4 | **CUDA Runtime** (`libcudart.so`) | Memory management, kernel launch, CUDA Streams, CUDA Graphs | Manages memory allocation/release, kernel launch pipelines, and stream overlap. |
| 3 | **CUDA Driver API** (`libcuda.so`) | Stable ABI, persistent across CUDA versions, context management | Bridges the Runtime and the hardware-facing driver. Handles context creation/switching between streams. Rarely changes because it's a stable ABI. |
| 2 | **NVIDIA Kernel Driver** (`nvidia.ko`) | HW control, PCIe DMA, interrupt handling, power management | Interface between the API layers above and the physical hardware. Controls DMA, interrupts, clock scaling, and power management. Must version-match the CUDA runtime/toolkit installed. |
| 1 | **GPU Hardware** | Streaming Multiprocessors, Tensor Cores, HBM3, NVLink | The actual silicon executing the workload — tensor cores/CUDA cores for compute, HBM for memory. |

**Key idea:** Every layer maps cleanly onto where it executes and how it can be optimized — from Python code at the top to transistors at the bottom.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/ddac3c6b-6140-4a8c-ba1f-6e71face02f1" />


## 2. CUDA Middleware, Runtime, and the Algorithm Selection Process

### How an operation gets executed (conceptual flow)
1. A model is defined in PyTorch/TensorFlow/JAX.
2. The compiler (`torch.compile`, XLA, TensorRT, TVM) extracts and optimizes the **computation graph** — fusing/simplifying/removing operations.
3. Each operation in the optimized graph is mapped to an algorithm inside a **middleware library** (cuDNN, cuBLAS, NCCL, etc.).
4. **Heuristic algorithm selection:** On the *first run*, the library heuristically picks an algorithm it estimates will be fastest for the target hardware and tensor shapes — this makes the first run of a workload noticeably slower.
5. Benchmarking happens during this first pass, and the chosen/cached algorithm is reused on subsequent runs, which run faster.
6. `cudnnFind` can be used to explicitly benchmark **all** algorithm options once and cache the result — useful for models with fixed input shapes.

### CUDA Runtime (`libcudart.so`) — Memory & Execution Management
- `cudaMalloc` / `cudaFree` — allocate/release GPU global memory (HBM).
- `cudaMemcpy` / `cudaMemcpyAsync` — host–device data transfers (sync or async via streams).
- **CUDA Streams** — logical queues that order kernel execution and memory transfers; enable overlap between data transfer and compute.
- **CUDA Graphs** — capture and replay sequences of graph operations with reduced kernel-launch overhead.

### cuDNN — The Convolution and Attention Engine
- Provides conv2d, batch norm, pooling, activation, dropout, RNN, and multi-head attention (MHA) ops for deep learning.
- Uses **heuristic algorithm selection** to pick the fastest kernel for given tensor shapes and hardware.
- **Flash Attention** is integrated into cuDNN **8.9+** for memory-efficient attention (relevant for transformer/LLM workloads).
- `cudnnFind` benchmarks all algorithm options — call once, cache the result; ideal for models with fixed input shapes.

### cuBLAS — The GEMM Backbone
- SGEMM / DGEMM / HGEMM — single/double/half-precision matrix–matrix multiplication.
- `cublasGemmEx` — mixed-precision GEMM with **Tensor Core** acceleration.
- `torch.matmul()` in PyTorch ultimately calls cuBLAS GEMM routines internally.
- cuBLAS is described as the "backbone" of virtually all modern AI algorithms, since most operations reduce to matrix multiplication.

### NCCL and Other Libraries
- **NCCL** — communication library; defines algorithms for multi-GPU/multi-node communication.
- **cuSPARSE** — algorithms for sparse computation.
- **cuFFT** — FFT-based algorithms.
- **Thrust** — general parallel algorithm primitives.

### CUDA Version Compatibility
- CUDA **12.x** toolkit requires driver version **≥ 525.85.12** (example given in lecture).
- Toolkit version and driver version must match/be compatible — if the runtime can't communicate with the driver, the hardware effectively can't be reached, breaking the whole optimization chain.
- **Check with:**
  - `nvidia-smi` → shows installed **driver** version.
  - `nvcc --version` → shows installed **CUDA toolkit** version.

<img width="2752" height="1536" alt="image" src="https://github.com/user-attachments/assets/23b77143-26b8-4d24-a92a-27a5cdada39c" />


## 3. Linux OS Features for AI Workloads

Most developers focus on writing/compiling code but overlook OS-level tuning — which becomes critical when running **large** AI workloads (hundreds of MB to GBs of parameters).
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/66ae1407-23a8-4aa0-9eb9-36c46a102a1b" />


### Huge Pages
- Default Linux page size = **4 KB**.
- Large AI workloads (GBs of parameters) accessed with 4 KB pages cause excessive **TLB misses** (Translation Lookaside Buffer — used for virtual→physical address translation).
- **Huge Pages** (2 MB or 1 GB per TLB entry) reduce the number of TLB entries needed, cutting TLB misses and improving data-transfer/overall performance.
- **Transparent Huge Pages (THP)** — kernel automatically promotes pages to huge-page size; often enabled by default.
- CUDA allocates GPU memory as large contiguous blocks, which benefits from huge pages.
- Checklist setting: `vm.nr_hugepages=1024`.

### cgroups v2 (Control Groups)
- Hierarchical resource controller for **CPU, memory, I/O, and device access**.
- Kubernetes uses cgroups to enforce **GPU memory limits** and **CPU quotas per pod** — i.e., quota-per-container in containerized deployments.
- `memory.limit_in_bytes` — prevents an out-of-memory (OOM) event in one training job from propagating across others.

### IOMMU (Input–Output Memory Management Unit)
- Translates **device virtual addresses → physical addresses** for **DMA safety**.
- **Intel VT-d / AMD-Vi** — required for GPU passthrough in virtualization (e.g., KVM).
- **GPUDirect** requires IOMMU in passthrough mode for peer **RDMA** (relevant for virtualized/pass-through GPU access scenarios).

### CPU Governor and Power Management
- Set CPU governor to **`performance`**: `cpupower frequency-set -g performance`.
- **Disable CPU C-states** for lowest latency — avoids wake-up delay, e.g., for DataLoader worker threads.
- **NUMA (Non-Uniform Memory Access)** balancing — disable when NUMA affinity is manually pinned via `numactl`, for best performance.

### NVIDIA Driver Management
- `nvidia-smi` — monitor GPU utilization, power draw, temperature, ECC errors; also shows how much of tensor cores/CUDA cores/functional units are being used and where bottlenecks occur (compute vs. I/O/data offloading).
- `nvidia-smi -pm 1` — enable **persistence mode** (keeps the GPU initialized between jobs).
- `nvidia-smi --auto-boost-default=0` — disable auto-boost for reproducible benchmarks.
- **DCGM** (Data Center GPU Manager) — health checks, profiling, and field injection for testing.

### System Configuration Checklist
1. Set performance governor & disable C-states.
2. Enable huge pages (`vm.nr_hugepages=1024`).
3. Set GPU persistence mode (`nvidia-smi -pm 1`).
4. Verify NVLink topology (`nvidia-smi topo -m`).

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/2ee0d613-d521-4dfc-a121-03d91f4e38e6" />


## 4. Container Runtimes and the NVIDIA Container Toolkit

Production-grade AI workloads in modern organizations mostly run inside **containers**.

### Why Containers for AI?
- **Reproducible environments** — exact CUDA, cuDNN, Python, and framework versions pinned inside the image.
- **Isolation** — multiple research groups/teams can share the same hardware without dependency conflicts (each container has its own filesystem, dependencies, libraries).
- **Portability** — the same container runs on a local DGX box, a cloud A100 instance, or a CI/CD pipeline.

### NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- Because the **host OS** owns the GPU driver, containers need a bridge to access it — this is the **NVIDIA Container Toolkit**.
- Works as an **OCI hook** — intercepts container creation and injects GPU device files and driver libraries.
- Mounts `/dev/nvidia*` devices, `libcuda.so`, and GPU management daemons into the container.
- `docker run --gpus all` — exposes all GPUs to the container; `--gpus 'device=0,1'` — exposes specific GPUs.
- **No need to install CUDA inside the container** — the driver stays on the host, and the CUDA runtime lives in the image.

### NGC (NVIDIA GPU Cloud) Container Registry
- URL: `https://catalog.ngc.nvidia.com/`
- Pre-built, benchmarked, optimized containers for PyTorch, TensorFlow, TensorRT, Triton, etc.
- Always prefer NGC containers for production — they come with **FlashAttention** and **TransformerEngine** pre-installed.

### Container Runtime Overhead
- Docker/Podman overhead is **< 1%** for GPU compute workloads (host drivers pass through directly).
- Container **import/startup** may take **10–30 seconds** for large NGC images (>20 GB) — acceptable for first import, then runs normally.
- **Always mount datasets as volumes** (`-v /data:/workspace/data`) rather than baking them into the image.

> *Next week's segment: building and running GPU-enabled containers end-to-end, in practice.*

---

## 5. Segment Summary

- The AI software stack spans **7 (logical) layers**: Frameworks → Compilers → cuDNN/cuBLAS (Middleware) → CUDA Runtime → Driver → Kernel → Hardware.
- **cuDNN** provides convolution/attention primitives; **cuBLAS** provides GEMM (matrix multiply) — PyTorch calls both transparently under the hood (e.g., `torch.matmul` → cuBLAS).
- **CUDA Streams** enable asynchronous data transfer + kernel execution overlap; **CUDA Graphs** reduce kernel-launch overhead by capturing/replaying operation sequences.
- **Linux performance tuning** (huge pages, cgroups, CPU governor, IOMMU) is a prerequisite for consistent, reproducible AI benchmarks — especially for large models.
- The **NVIDIA Container Toolkit** injects GPU access into Docker containers; **NGC images** are the standard for reproducible, production-grade AI environments.
- **Overall takeaway:** To get the most performance out of an AI workload, you need to know exactly *which layer is doing what* — from the CUDA driver/runtime version match, to the algorithm heuristics in cuDNN/cuBLAS, to OS-level settings like huge pages and CPU governor.

![Uploading image.png…]()


## 6. Quick Command Reference

| Command | Purpose |
|---|---|
| `nvidia-smi` | Monitor GPU utilization, power, temperature, ECC errors; shows driver version |
| `nvcc --version` | Shows installed CUDA toolkit version |
| `nvidia-smi -pm 1` | Enable GPU persistence mode |
| `nvidia-smi --auto-boost-default=0` | Disable GPU boost for reproducible benchmarks |
| `nvidia-smi topo -m` | Verify NVLink/GPU topology |
| `cpupower frequency-set -g performance` | Set CPU governor to performance mode |
| `numactl` | Manually pin NUMA affinity |
| `docker run --gpus all` | Expose all GPUs to a container |
| `docker run --gpus 'device=0,1'` | Expose specific GPUs to a container |
| `docker run -v /data:/workspace/data` | Mount dataset as a volume (not baked into image) |

---

## References (as cited in slides)
- NVIDIA CUDA Programming Guide v12 (2024); CUDA C++ Best Practices Guide
- NVIDIA cuDNN Developer Guide v8.9; cuBLAS Library User Guide; CUDA C++ Programming Guide
- Red Hat Enterprise Linux Performance Tuning Guide; NVIDIA DCGM Documentation (2023)
- NVIDIA Container Toolkit Documentation; NGC Catalog (catalog.ngc.nvidia.com)

  <img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/7155e69b-6dc1-4fc9-96ba-cb9142133239" />
