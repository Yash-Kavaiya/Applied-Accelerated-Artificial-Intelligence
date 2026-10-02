# Week 2 · Session 3 of 4 — CUDA in Containers & CI Validation

**Course:** Containerized AI Systems
**Instructor:** Satyadhyan Chickerur, Ph.D., Director, Centre for AI Research, KLE Technological University

---

## Table of Contents

1. [Learning objectives](#1-learning-objectives)
2. [Why touch CUDA in this course?](#2-why-touch-cuda-in-this-course)
3. [The CUDA execution model](#3-the-cuda-execution-model)
4. [Thread hierarchy](#4-thread-hierarchy-grid--block--warp--thread)
5. [GPU memory hierarchy](#5-gpu-memory-hierarchy)
6. [Unified memory vs explicit transfers](#6-unified-memory-vs-explicit-transfers)
7. [Occupancy](#7-occupancy--keeping-the-sm-busy)
8. [CUDA streams](#8-cuda-streams--overlapping-compute-and-transfer)
9. [Error handling](#9-error-handling--silent-failures)
10. [The nvcc build pipeline](#10-the-nvcc-compilation-pipeline)
11. [Container images: devel vs runtime](#11-container-images-devel-vs-runtime)
12. [Hands-on lab: vector add](#12-hands-on-lab-vector-add)
13. [Lecture demo walkthrough](#13-lecture-demo-walkthrough)
14. [The 3-level probe](#14-localizing-failures-the-3-level-probe)
15. [CI for AI images](#15-ci-for-ai-images)
16. [Smoke tests](#16-smoke-tests-that-catch-real-breakage)
17. [Corrections and gotchas](#17-corrections-and-gotchas)
18. [Cheat sheet and review questions](#18-cheat-sheet-and-review-questions)

---

## 1. Learning objectives

After this session you should be able to:

- **Sketch the CUDA execution model:** host vs device, kernels, and grids of blocks of threads.
- **Compile and run a vector-add kernel** inside a *devel* container (hands-on).
- **Use a 3-level probe** (driver → toolkit → framework) to localize GPU stack failures.
- **Design a CI pipeline:** build → smoke test → scan → push, with GPU-aware caveats.

---

## 2. Why touch CUDA in this course?

Frameworks such as PyTorch hide CUDA, but every deep-learning or massively parallel program ends up as **kernels** running on a stack of *driver + toolkit + GPU*. If that stack disagrees with itself, you get errors no matter which framework you use.

| # | Reason | Explanation |
|---|--------|-------------|
| 1 | **Ground truth** | A ~30-line kernel that compiles and runs proves the driver, toolkit and GPU agree, independent of any framework. |
| 2 | **Vocabulary** | Logs speak CUDA: `invalid device function`, `sm_100`, `illegal memory access`. You need to be able to read them. |
| 3 | **Architecture targets** | Wheels and kernels are compiled **per GPU architecture** (e.g. Blackwell = `sm_100`). Mismatches surface as **runtime errors**. |

**Key idea:** when the GPU architecture changes (new generation), your drivers, wheels and toolkit must change too, or at least be compatible with it.

> **SM** = Streaming Multiprocessor. `sm_XY` is the compute-capability target (e.g. `sm_70` Volta, `sm_80` Ampere, `sm_86` RTX 30-series, `sm_90` Hopper, `sm_100` Blackwell).

---

## 3. The CUDA execution model

### 3.1 Host vs device

- **Host** = the CPU (and its RAM).
- **Device** = the GPU (and its global memory / VRAM).
- Heavy, massively parallel work is **offloaded** to the GPU. The CPU orchestrates; the GPU computes.

### 3.2 The standard flow

```
HOST (CPU)                              DEVICE (GPU)
-----------                             -------------
allocate + copy data  --------------->  global memory
launch kernel<<<blocks,threads>>>       grid = blocks x threads
                                        each thread: i = blockIdx.x * blockDim.x + threadIdx.x
copy results back     <---------------  global memory
```

1. Allocate memory and copy input data host → device (or use unified memory).
2. Launch the kernel with `<<<blocks, threads>>>`.
3. The GPU executes the kernel across thousands of threads.
4. Copy results back device → host.

### 3.3 Kernels

- A **kernel** is a function run by **thousands of threads at once**, each on its own index.
- The GPU should be used to run kernels, i.e. the compute-intensive parallel part of the program.
- The `<<<blocks, threads>>>` launch syntax sizes the grid to cover the data.
- **Frameworks like PyTorch do exactly this under every tensor op**, with pre-written, tuned kernels.

### 3.4 Unified memory note

Modern CUDA lets the same pointer be accessed from both host and device (unified memory, see §6), so explicit copies are optional, but the model above is still what happens underneath.

---

## 4. Thread hierarchy: Grid → Block → Warp → Thread

| Level | Size | What it is |
|-------|------|------------|
| **Grid** | 1 per kernel launch | All blocks launched by one kernel call |
| **Block** | up to **1024 threads** | Runs on **one SM**; threads share `__shared__` memory |
| **Warp** | **32 threads** | Hardware scheduling unit; **SIMT** (single instruction, multiple threads) |
| **Thread** | 1 | Executes on a CUDA core; has private registers |

**Key rules**

- Threads in a warp execute the **same instruction simultaneously**.
- **Divergent branches** (`if/else` on `threadIdx`) **serialize**: one path runs at a time. Minimize divergence.
- Blocks should be independent (serializable). The scheduler can run them in any order, which is what makes the model scale across GPUs.
- Block size should be a **multiple of 32** (the warp size).

**Launch configuration example**

```cpp
// 1D grid of blocks, 256 threads each
int blocks = (N + 255) / 256;               // ceil(N / 256)
kernel<<<blocks, 256>>>(d_a, d_b, d_c, N);  // gridDim.x = blocks, blockDim.x = 256
```

**Global thread index**

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i] + b[i];   // bounds guard: last block may overshoot n
```

---

## 5. GPU memory hierarchy

| Level | Approx. bandwidth | Scope / notes |
|-------|-------------------|---------------|
| **Registers** | ~10 TB/s | Per thread; private, fastest, limited (~255 per thread) |
| **Shared memory** (`__shared__`) | ~10 TB/s | Per block; on-chip, programmer-managed scratchpad |
| **L2 cache** | ~8 TB/s | Chip-wide; automatic, no programmer control |
| **HBM3e (global memory)** | ~8 TB/s | All threads; main VRAM, `cudaMalloc` target (B200: 8 TB/s) |
| **PCIe / system RAM** | ~32–64 GB/s | Host ↔ device transfers; **125–250× slower than HBM** |

**Rule:** keep hot data in shared memory, **minimise PCIe transfers**, and **coalesce global-memory accesses** (adjacent threads touch adjacent addresses).

**Other points from the lecture**

- The **PCI Express bus** is the communication medium between host and device. Results must come back across it too, so reducing transfers matters.
- Each GPU has a **fixed memory size**. You can only `cudaMalloc` what fits (or rely on unified memory migration).

---

## 6. Unified memory vs explicit transfers

### 6.1 `cudaMallocManaged` (Unified Memory)

- **One pointer** accessible from both host and device.
- The CUDA runtime **migrates pages on demand**.
- Simple to write.
- Under the hood it is the **same PCIe traffic**, but triggered by **page faults**, so timing is unpredictable and hard to overlap with compute.

### 6.2 Explicit transfers

- `cudaMalloc` + `cudaMemcpy` host↔device.
- Predictable and overlap-friendly.
- Use **pinned (page-locked) host memory** (`cudaMallocHost`) for maximum PCIe bandwidth and async copies.

### 6.3 When to use unified memory

- **Prototyping and teaching** (e.g. `vector_add.cu`).
- **Irregular access patterns** where you don't know what to prefetch.
- **CUDA 12+ prefetch hints** (`cudaMemPrefetchAsync`) close much of the performance gap. Before that, you had no good way to hint.

### 6.4 Side-by-side code

```cpp
// Unified Memory: simple
float *a;
cudaMallocManaged(&a, N*sizeof(float));
kernel<<<blocks,256>>>(a, N);
cudaDeviceSynchronize();   // wait before CPU reads

// Explicit: predictable, overlap-friendly
float *d_a, *h_a = new float[N];
cudaMalloc(&d_a, N*sizeof(float));
// Pin host buffer for max PCIe BW:
float *h_pinned;
cudaMallocHost(&h_pinned, N*sizeof(float));
cudaMemcpy(d_a, h_pinned, N*sizeof(float), cudaMemcpyHostToDevice);
kernel<<<blocks,256>>>(d_a, N);
cudaMemcpy(h_pinned, d_a, N*sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(d_a); cudaFreeHost(h_pinned);
```

> ⚠️ With unified memory you **must synchronize** (`cudaDeviceSynchronize()`) before the CPU reads results, because kernel launches are asynchronous.

---

## 7. Occupancy: keeping the SM busy

### 7.1 What it means

**Occupancy** = active warps ÷ maximum warps an SM can host.

- Higher occupancy → more warps ready to run when one **stalls waiting for memory**, which **hides latency**.
- **Target:** ≥ 50% for memory-bound kernels.
- Goal in one line: **keep the SMs busy**.

### 7.2 What limits occupancy

- **Registers per thread** (cap ≈ 65,536 per SM).
- **Shared memory per block** (cap ≈ 48–228 KB per SM, depending on GPU).
- **Block size**: must be a multiple of 32; **128 or 256 usually best**.

### 7.3 Compute-bound vs memory-bound

- **Compute-intensive** (HPC-style) workloads use the GPU's strengths.
- **Data/memory-intensive** workloads (lots of host→GPU transfer) spend time waiting on memory.
- CUDA is most valuable for compute-bound work. Don't leave SMs idle waiting on data.

### 7.4 Code

```cpp
// Query occupancy at runtime
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

// Check register usage:
//   nvcc --ptxas-options=-v kernel.cu
//   ptxas info: Used 32 registers, 0 bytes smem

// Compile-time limit (rarely needed):
__global__ void __launch_bounds__(256, 4) myKernel(...) { ... }
// max 256 threads/block, min 4 blocks/SM
```

### 7.5 Rule of thumb

Start with `blockDim = 256`. If **register spill** warnings appear, reduce to **128**. **Profile with Nsight** before micro-optimising.

> ⚠️ The lecture said "maximum is 256 threads per block". The hardware **maximum is 1024**; **256 is the recommended starting point** (and the `__launch_bounds__` example value).

---

## 8. CUDA streams: overlapping compute and transfer

### 8.1 What a stream is

- A **sequence of CUDA operations that execute in order**.
- **Multiple streams can run concurrently.**
- The **default stream (stream 0)** is synchronous with all other streams. (The lecturer's analogy: like the main thread in MPI, or thread 0 in OpenMP.)

### 8.2 Why it matters

- **Without streams:** H2D transfer → kernel → D2H transfer, all sequential.
- **With 2 streams:** overlap the H2D copy of batch *N+1* while the kernel processes batch *N*. The **theoretical** upper bound is ~2× throughput on data-parallel workloads.
- **PyTorch uses streams internally** for this.
- The bus is a shared medium, so streams don't make it faster. They keep the bus, compute units and copy engines **all busy at once**.

### 8.3 Code

```cpp
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

// Overlap: transfer next batch while computing current
cudaMemcpyAsync(d_in1, h_in1, sz, H2D, s1);
cudaMemcpyAsync(d_in2, h_in2, sz, H2D, s2);

kernel<<<grid, block, 0, s1>>>(d_in1, d_out1);
kernel<<<grid, block, 0, s2>>>(d_in2, d_out2);

cudaMemcpyAsync(h_out1, d_out1, sz, D2H, s1);
cudaMemcpyAsync(h_out2, d_out2, sz, D2H, s2);

cudaStreamSynchronize(s1);
cudaStreamSynchronize(s2);
cudaStreamDestroy(s1); cudaStreamDestroy(s2);
```

> `cudaMemcpyAsync` only overlaps when the host buffers are **pinned** (`cudaMallocHost`). Pageable memory silently serializes.

---

## 9. Error handling: silent failures

### 9.1 The problem

- **Kernel launches are asynchronous.** The launch call returns immediately.
- An out-of-memory or **invalid memory access inside a kernel only becomes visible at the next synchronizing CUDA call**, which can be many lines later.
- Debugging errors across thousands of threads is tedious, so you need tooling.

> ⚠️ Precision: the slide says launches "NEVER return an error code". Strictly, **launch-configuration errors** (bad grid/block size, no such kernel) *are* caught by `cudaGetLastError()` right after the launch. **Execution errors** are what surface only at the next sync.

### 9.2 The `CUDA_CHECK` macro (use everywhere)

```cpp
#define CUDA_CHECK(err) do { \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_a, N*sizeof(float)));
CUDA_CHECK(cudaMemcpy(d_a, h_a, N*sizeof(float), H2D));
kernel<<<grid, block>>>(d_a, N);
CUDA_CHECK(cudaGetLastError());        // catch launch errors
CUDA_CHECK(cudaDeviceSynchronize());   // flush async errors
```

### 9.3 `compute-sanitizer`

```bash
compute-sanitizer --tool memcheck ./vector_add
```

- Catches **out-of-bounds, unaligned and use-after-free** errors at runtime.
- **Successor to `cuda-memcheck`.**
- Adds roughly **10× overhead**, so use it **only for debugging**.

---

## 10. The nvcc compilation pipeline

`nvcc` is NVIDIA's compiler driver (a modified C++ toolchain front-end). Pipeline from `.cu` to binary:

| Step | Stage | What happens |
|------|-------|--------------|
| 1 | **Preprocessing** | `cpp` resolves `#include`, `#define`, `#ifdef` |
| 2 | **Device code → PTX** | CUDA C++ compiled to **PTX** (Parallel Thread eXecution), a virtual GPU assembly / IR |
| 3 | **PTX → SASS** | `ptxas` assembles PTX into **SASS** (native machine code) for the target SM architecture |
| 4 | **Host code → object** | `nvcc` calls the system C++ compiler (`g++`) for host functions |
| 5 | **Link** | Host objects + device **cubin** linked into the final **ELF** binary |

### Commands

```bash
# Compile for the current GPU (requires devel image, not runtime):
nvcc -arch=native vector_add.cu -o vector_add

# Generate PTX for a virtual arch and SASS for specific SMs:
nvcc -arch=compute_80 -code=sm_80,sm_90 add.cu -o add

# Inspect PTX (readable GPU IR):
nvcc -ptx add.cu -o add.ptx && cat add.ptx

# Cross-compile for a specific arch (e.g. Blackwell):
nvcc -gencode arch=compute_100,code=sm_100 vector_add.cu -o vector_add
```

### Makefile pattern used in the demo

```make
# falls back to sm_86 if sm_100 unavailable
ARCH := $(shell nvcc --query-gpu-name 2>/dev/null | grep -q "B200" && echo sm_100 || echo sm_86)
NVCCFLAGS = -arch=$(ARCH) -O2
```

> ⚠️ **Bug in the slide:** `nvcc --query-gpu-name` is **not a real nvcc option**. `2>/dev/null` hides the error, `grep` never matches, so the build **always falls back to `sm_86`**. Use `nvidia-smi` instead:
>
> ```make
> ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | sed 's/^/sm_/')
> ```
> Or simply use `-arch=native`.

**PTX vs SASS:** PTX is forward-compatible (the driver can JIT it for newer GPUs). SASS is fast but tied to a specific SM. That is why an architecture mismatch yields `invalid device function`.

---

## 11. Container images: devel vs runtime

| Component | `runtime` | `devel` |
|-----------|:---------:|:-------:|
| CUDA runtime (`libcudart`) | ✓ | ✓ |
| cuBLAS, cuDNN, cuSPARSE | ✓ (separate image tags) | ✓ |
| NCCL | ✗ (separate image) | ✓ |
| `nvcc` compiler | ✗ | ✓ |
| CUDA headers (`.h`) | ✗ | ✓ |
| Static libraries (`.a`) | ✗ | ✓ |
| `ptxas`, `fatbinary` tools | ✗ | ✓ |
| Image size (approx., CUDA 13) | ~3 GB | ~8 GB |

**Multi-stage pattern:** compile in **devel** → `COPY` the binary → run in **runtime**. *Ship the smallest image that can actually run your model.*

```dockerfile
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build
COPY vector_add.cu .
RUN nvcc -arch=native vector_add.cu -o vector_add
# or, for a portable image: -gencode arch=compute_80,code=sm_80 ... plus PTX

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
COPY --from=build /vector_add /usr/local/bin/vector_add
CMD ["vector_add"]
```

> ⚠️ `-arch=native` at **image build time** targets the GPU of the *build machine* (which in CI may have no GPU). For shippable images, specify explicit `-gencode` targets.

---

## 12. Hands-on lab: vector add

### 12.1 The code

```cpp
// vector_add.cu
#include <cstdio>

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20;                       // 1M floats
    float *a, *b, *c;
    cudaMallocManaged(&a, n * 4);
    cudaMallocManaged(&b, n * 4);
    cudaMallocManaged(&c, n * 4);
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    add<<<(n + 255) / 256, 256>>>(a, b, c, n);
    cudaDeviceSynchronize();

    printf("c[0]=%.1f c[n-1]=%.1f\n", c[0], c[n - 1]);   // 3.0 3.0
}
```

**Walkthrough**

- `__global__` marks a function that runs on the device and is launched from the host.
- Each thread computes **one element**: `c[i] = a[i] + b[i]`.
- `if (i < n)` guards the extra threads in the last block.
- `cudaMallocManaged` gives one pointer valid on both host and device.
- `<<<(n+255)/256, 256>>>` is a ceiling division, so enough blocks cover `n` elements.
- `cudaDeviceSynchronize()` waits for the GPU before the CPU reads `c`.

### 12.2 Compile and run in a container

```bash
# devel image = nvcc included. Mount the code, compile, run.
docker run --rm --gpus all -v "$PWD":/src -w /src \
  nvidia/cuda:13.0.0-devel-ubuntu24.04 bash -c '
  nvcc -arch=native vector_add.cu -o vector_add && ./vector_add'
# Expected: c[0]=3.0 c[n-1]=3.0
```

- **If this prints 3.0, the entire stack (driver, runtime, toolkit, GPU) is healthy.**
- Keep this as your permanent **"is it the cluster or my code?" probe**.

---

## 13. Lecture demo walkthrough

The instructor's demo folder contained two programs, `vector_add.cu` and `bandwidth_test.cu`, plus `smoke_test.py`.

### 13.1 Programs

- **`vector_add.cu`**: the "hello world" of CUDA. Demonstrates host vs device memory allocation, **explicit** H2D / D2H copies (the demo version uses `cudaMemcpy`, while the slide version uses unified memory on purpose, to compare), a kernel launch with grid of blocks of threads, and basic error checking. Output shows the GPU name, `C = A + B` for N elements, kernel time, and a verification pass.
- **`bandwidth_test.cu`**: measures the **device memory bandwidth** of the GPU with a copy kernel plus sanity check.

Steps in the explicit version: allocate host buffers → fill with static data (e.g. `A[i] = i`, `B[i] = 2*i`) → `cudaMalloc` device buffers → copy in → launch (typically 256 threads/block) → time it → copy out → verify → `cudaFree` all device buffers.

### 13.2 Two ways to run

**A. One-shot with Docker Compose**

```bash
docker compose up --build
```

Builds the image (the Dockerfile compiles every `.cu` in the folder with nvcc) and runs both programs.

**B. Interactive debugging inside the container**

```bash
docker run --gpus all -it -v "$PWD":/programs -w /programs <image> bash
```

You get a root prompt in the Ubuntu-based container, like a dev notebook. Inside:

```bash
nvcc --version                         # showed CUDA 13.0 toolchain
nvcc vector_add.cu -o va && ./va       # compile and run
nvidia-smi                             # shows the GPU (demo: RTX 3050 laptop GPU)
```

- Demo GPU: **RTX 3050, compute capability 8.6 (`sm_86`)**, run with N = 20 elements.
- The container had **no `nano` or `vi`**. Install an editor in the image if you want to edit inside the container.
- Laptops with RTX GPUs are enough for the course. A couple of later examples will use **DGX H100 / DGX V100** servers.

### 13.3 Assignment

Understand what **`sm_70`** (and architecture targets in general) means. Ask in the interactive session if unclear.

> ⚠️ The lecturer mentioned `SM70` for the compose build but ran on an `sm_86` GPU. That is fine only if the binary includes compatible code (e.g. embedded PTX). Check your actual `-arch` flags.

---

## 14. Localizing failures: the 3-level probe

| Level | Probe | Question it answers |
|-------|-------|--------------------|
| **1. Driver** | `nvidia-smi` inside the container | Is the GPU visible at all? |
| **2. Toolkit** | The vector-add probe | Can CUDA code compile **and** execute? |
| **3. Framework** | `python -c 'import torch; print(torch.cuda.is_available())'` | Does the framework see CUDA? |

**Rules**

- **Whichever level fails first names the layer to fix.**
- **Never debug level 3 before levels 1–2 pass.**
- Typical level-1 failure: driver/container mismatch, GPU not passed through (`--gpus all` missing, NVIDIA Container Toolkit not installed).
- Typical level-2 failure: toolkit/driver version mismatch, wrong `-arch`.
- Typical level-3 failure: CPU-only wheel, wheel built for a different CUDA/arch.
- Set up driver + toolkit **once, carefully**. It makes everything after much easier.

---

## 15. CI for AI images

### 15.1 Why CI

- **Environments rot quietly:** a base-tag update or a yanked wheel can break Tuesday's build of Monday's code.
- CI makes the image a **tested artifact**: every commit rebuilds it and proves it still works.
- **The pipeline is documentation:** the workflow file *is* the build procedure.
- **Tag and version carefully** (git SHA + semver). Without correct tags you can't tell which versions were built or updated.
- Commit in small steps, test each, improve iteratively. That is the "continuous" in CI.

### 15.2 GPU caveat

- Most **hosted runners are CPU-only**.
- Smoke tests must **degrade gracefully**: import checks + CPU ops everywhere, **full GPU tests on self-hosted runners**.
- Frameworks may **silently fall back to CPU** when no GPU is present. You must make sure the **GPU-enabled packages/wheels** are what's installed.

### 15.3 Four stages

| # | Stage | What it does |
|---|-------|--------------|
| 1 | **Build** | `docker build` with BuildKit + **registry cache** (keep builds fast) |
| 2 | **Smoke test** | Run the image: imports, versions, a tiny op. **Under a minute.** |
| 3 | **Scan** | CVE scan (e.g. **trivy**); **fail on critical findings** |
| 4 | **Push** | Tag with **git SHA + semver**; push **only after stages 1–3 pass** |

### 15.4 Example workflow skeleton (GitHub Actions)

```yaml
name: image-ci
on: [push, pull_request]
jobs:
  build-test-scan-push:
    runs-on: ubuntu-latest            # CPU-only hosted runner
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Build
        uses: docker/build-push-action@v6
        with:
          context: .
          load: true
          tags: myimage:${{ github.sha }}
          cache-from: type=registry,ref=ghcr.io/ORG/myimage:cache
          cache-to: type=registry,ref=ghcr.io/ORG/myimage:cache,mode=max
      - name: Smoke test (CPU tier)
        run: docker run --rm myimage:${{ github.sha }} python smoke_test.py
      - name: Scan
        uses: aquasecurity/trivy-action@0.28.0
        with:
          image-ref: myimage:${{ github.sha }}
          severity: CRITICAL
          exit-code: '1'
      - name: Push
        if: github.ref == 'refs/heads/main'
        run: |
          # login, tag with git SHA + semver, push
          echo "push here"
```

(Add a second job on a **self-hosted GPU runner** that runs the same smoke test with `--gpus all`.)

---

## 16. Smoke tests that catch real breakage

**Catches in seconds:** missing libraries, broken wheels, version mismatches, CUDA-less wheels.

**Same script, two tiers:** CPU assertions everywhere, GPU assertions where hardware exists.

### `smoke_test.py` (tiers)

| Tier | Runs where | What it checks |
|------|-----------|----------------|
| 1 | Always | `import torch` (fatal on failure), `import transformers` (optional, warn only), print versions |
| 2 | Always | 64×64 CPU matmul + shape assertion |
| 3 | Only if `torch.cuda.is_available()` | GPU matmul, `torch.cuda.synchronize()`, device name, total memory |
| 4 | Always, non-fatal | `nvidia-smi --query-gpu=driver_version,name` (handles `FileNotFoundError`) |

```python
# Tier 3 core
if torch.cuda.is_available():
    y = x.cuda() @ x.cuda()
    torch.cuda.synchronize()
    print(f"gpu ok  {torch.cuda.get_device_name(0)}")
else:
    print("gpu  cpu-only runner: gpu checks skipped")
```

### ⚠️ Weakness and fix

On a **GPU machine with a CPU-only torch wheel**, `is_available()` is `False`, so the GPU tier is **skipped** and the script still prints "All checks passed." That is exactly the failure it claims to catch. Add an opt-in strict mode:

```python
import os
REQUIRE_GPU = os.getenv("REQUIRE_GPU") == "1"

if torch.cuda.is_available():
    ...
elif REQUIRE_GPU:
    print("FAIL: REQUIRE_GPU=1 but CUDA is not available (CPU-only wheel or driver issue?)")
    sys.exit(1)
else:
    print("gpu  cpu-only runner: gpu checks skipped")
```

Set `REQUIRE_GPU=1` on the self-hosted GPU runner only.

> Also: the slide version of the script calls `sys.exit(0)` **inside the `else` branch**, ending the script early. The `.py` file version does not have this problem.

---

## 17. Corrections and gotchas

| Topic | What was said | Correct / clarified |
|-------|---------------|---------------------|
| Max threads per block | "Maximum is 256" (lecture) | Hardware max is **1024**. **256** is the recommended default. |
| Blackwell target | "SM_00" (transcript) | Transcription error. It is **`sm_100`**. |
| Demo arch | `sm_70` in compose, RTX 3050 in demo | The RTX 3050 is **`sm_86`**. Make sure the build targets match the hardware. |
| Launch errors | "Launches NEVER return an error" | Config errors are caught via `cudaGetLastError()`. **Execution** errors surface at the next sync. |
| Makefile probe | `nvcc --query-gpu-name` | Not a valid flag, so it always falls back to `sm_86`. Use `nvidia-smi --query-gpu=compute_cap` or `-arch=native`. |
| Smoke test | "All checks passed" on GPU hosts | Can pass with a CPU-only wheel. Add `REQUIRE_GPU=1`. |
| Streams | "2× throughput" | A **theoretical** best case. Needs pinned memory and enough independent work. |
| Unified memory | "Page fault not occurring if detected where data is" (garbled) | Page faults trigger on-demand migration. **Prefetch hints** (`cudaMemPrefetchAsync`, CUDA 12+) avoid the faults. |
| `-arch=native` in Dockerfile | Used in labs | Targets the **build machine's** GPU. Use explicit `-gencode` for shippable images. |

---

## 18. Cheat sheet and review questions

### Cheat sheet

```text
Thread → Warp (32) → Block (≤1024, one SM, shared mem) → Grid (one per launch)
i = blockIdx.x * blockDim.x + threadIdx.x
blocks = (N + threads - 1) / threads
Memory speed:  registers ≈ shared > L2 ≈ HBM >> PCIe
Debug stack:   nvidia-smi → vector_add → torch.cuda.is_available()
Error checks:  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
Sanitizer:     compute-sanitizer --tool memcheck ./app
Compile:       nvcc -arch=native app.cu -o app      (devel image only)
Images:        compile in devel → COPY → run in runtime
CI:            build → smoke test → scan → push (SHA + semver)
```

### Review questions

1. What is the difference between host and device, and what data moves between them?
2. Define grid, block, warp and thread. Which one is the hardware scheduling unit?
3. Why does warp divergence hurt performance?
4. What are the trade-offs of `cudaMallocManaged` vs `cudaMalloc` + `cudaMemcpy`?
5. What does occupancy measure, and what limits it?
6. How do two streams give overlap, and what must be true of the host memory?
7. Why can a kernel's error show up several lines after the launch? How do you catch it?
8. Walk through the nvcc stages. What is the difference between PTX and SASS?
9. What does a `devel` image have that a `runtime` image doesn't? What is the multi-stage pattern?
10. In the 3-level probe, why must you not debug the framework level first?
11. Why must CI smoke tests degrade gracefully, and how can a CPU-only wheel slip past one?
12. List the four CI stages and the condition for pushing.

### Practice tasks

- Compile and run `vector_add.cu` in a devel container. Confirm `c[0]=3.0 c[n-1]=3.0`.
- Rewrite it with `cudaMalloc`/`cudaMemcpy` and add `CUDA_CHECK`.
- Run it under `compute-sanitizer` (try deliberately removing the `if (i < n)` guard to see the report).
- Compile with `nvcc -ptx` and read the PTX.
- Add `REQUIRE_GPU` mode to `smoke_test.py` and run it on both CPU and GPU machines.
