# Segment 2: Memory Hierarchy, RAM, Storage & Data Movement

## Learning Objectives

1. Map the full data movement path from NVMe storage to GPU compute cores
2. Quantify bandwidth and capacity at each level of the memory hierarchy
3. Explain HBM (High Bandwidth Memory) and why it's critical for AI accelerators
4. Identify bottlenecks in DataLoader pipelines and apply pinned memory optimization
5. Describe NUMA (Non-Uniform Memory Access) effects in multi-socket AI servers

## 1. Big Picture: Why This Topic Matters

- AI system performance isn't just about raw compute (FLOPS) — it's about **how fast data can reach the compute units**.
- The lecture builds a full mental map: **disk → NVMe → DDR5 → HBM → GPU core**, with the bandwidth and latency at each hop.
- Core theme repeated throughout: **the gap between compute speed and data movement speed is enormous (~30,000x)**, so modern AI systems must be optimized for data movement, not just compute.

## 2. DRAM Fundamentals: DDR, LPDDR, and HBM

### 2.1 DRAM Basics (Why It's "Dynamic")
- Each DRAM cell stores **1 bit as charge in a capacitor**.
- Capacitors **leak charge over time**, so a **refresh controller** must periodically "top up" the charge.
- Refresh cycle occurs roughly every **~64 milliseconds**.
- This refresh process happens invisibly in the background but **adds latency** to memory access — a fundamental limitation of DRAM technology.

### 2.2 DDR (Double Data Rate) Technology
- **DDR = Double Data Rate** — data is transferred on **both the rising and falling edges** of the clock cycle (instead of just one edge).
- This effectively **doubles bandwidth** and helps hide/amortize the refresh-induced latency.
- **DDR5** (introduced 2021):
  - Speed range: **4800–6400 MT/s** (mega-transfers/sec)
  - Bus width: 64-bit per channel
  - Per-channel bandwidth: **~38.4 GB/s** (at 4800) up to **~51.2 GB/s** (at 6400)
- **Multi-channel scaling example:** AMD EPYC 9654 server CPU uses a **12-channel DDR5** configuration → achieves a **peak system bandwidth of ~614 GB/s**.

  <img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/2e9dd816-2997-4edd-be35-a9ca828b5c26" />


### 2.3 LPDDR (Low-Power DDR — Mobile/Edge AI)
- Used in mobile and edge devices where power efficiency matters more than raw peak bandwidth.
- **LPDDR5X**: ~8533 MT/s, lower voltage — used in NVIDIA Jetson devices and Apple's M-series chips.
- **Apple M2 Ultra** example: achieves **~800 GB/s of unified memory bandwidth** — almost **2x** a typical DDR5 server (460 GB/s), which is one reason Apple Silicon can be attractive for certain LLM training/inference workloads.

### 2.4 ECC (Error Correcting Code) Memory
- AI training involves **huge datasets and long-running jobs** (days to weeks) — a **silent bit flip** can silently corrupt training data or model state.
- ECC memory:
  - **Corrects single-bit errors automatically**
  - **Detects double-bit errors**
- Overhead: **~2–4% performance cost**
- Verdict: **non-negotiable** for production AI systems, despite the overhead.

### 2.5 HBM (High Bandwidth Memory) — "Game Changer for Accelerators"
- HBM **stacks multiple DRAM dies vertically** on a **silicon interposer**, connected via **TSVs (Through-Silicon Vias)** — essentially tiny vertical metal connections linking each stacked die.
- This 3D stacking next to the GPU die (rather than on a separate DIMM across a bus) is what enables extreme bandwidth.
- Bandwidth comparison:

  <img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/eb4b467b-9bae-4d7b-8c48-a2bcca58147e" />

  | Technology | Bandwidth | Capacity | Notes |
  |---|---|---|---|
  | HBM3 (NVIDIA H100, Hopper) | **3.35 TB/s** | 80 GB | ~10x DDR5, ~4x LPDDR5 unified memory |
  | HBM3e (NVIDIA H200) | **4.8 TB/s** | 141 GB | Next-gen HBM |
  | GDDR6X (RTX 4090) | 1.008 TB/s | — | Still ~3x less than HBM3 |
  | AMD MI300X (HBM3) | **5.3 TB/s** | — | Highest in this comparison set |
- Why it matters: HBM is now **essential for AI training and inference**, since LLMs and deep learning models are extremely bandwidth-hungry.
- Important distinction: HBM's advantage is in **bandwidth**, not necessarily capacity — it answers "how fast can I move data," not "how much data can I store."

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/72a0491a-85c0-49de-a3c2-93b2a98da3f2" />

## 3. Storage Tiers & the Data Ingestion Pipeline

### 3.1 The Storage Hierarchy (from slow/cheap to fast/expensive)

| Storage Tier | Bandwidth | Best Use Case |
|---|---|---|
| HDD (7200 RPM, spinning disk) | ~150 MB/s | **Cold storage** — cheap, terabyte-scale archived/rarely-used data |
| SATA SSD | ~550 MB/s | **Warm data staging buffers** (sequential access) |
| NVMe SSD (PCIe 4.0) | **~7 GB/s**, 1M+ IOPS | **AI training scratch space** (preferred tier for active datasets) |
| NVMe RAID-0 (8 drives) | **~56 GB/s** | Matches PCIe 4.0 x16 bandwidth (64 GB/s) — for very high-throughput needs |

- Key insight: NVMe reuses the same underlying SSD tech as SATA SSD, but a faster **interface (PCIe)** boosts throughput from ~550 MB/s to ~7 GB/s.

### 3.2 Worked Example: ImageNet Training I/O

- Dataset: **1.2 million images, ~150 GB total**
- Every **epoch**, the model must read through the *entire* dataset once.
- At **7 GB/s** (single NVMe SSD on PCIe 4.0): **150 GB ÷ 7 GB/s ≈ 21 seconds per epoch** just for I/O.
- **Verdict:** This is *fine* if compute time per epoch is in the range of minutes (typical for large models) — I/O is not the bottleneck.
- **But:** if you scale to **8 GPUs** all pulling from the same storage, you need **more than 3 GB/s aggregate** just to keep all GPUs fed — a single NVMe drive may become a bottleneck when many GPUs compete for the same data source.

### 3.3 DataLoader Pipeline Optimization (PyTorch example)
- `num_workers`: controls how many parallel worker processes prefetch data in the background.
  - **Rule of thumb: `num_workers` = 2 to 4× the number of GPUs.**
- `pin_memory=True`: pins host memory so it can be transferred to the GPU faster over CUDA (avoids extra copy through pageable memory). This is a system-level concept; exact coding details are covered in later lectures.


<img width="1055" height="1491" alt="image" src="https://github.com/user-attachments/assets/75f927ba-4d8f-4e9b-b581-5a17e7dddb33" />

### 3.4 Network / Distributed Storage for Large-Scale AI
- Once you scale beyond a single machine, you move from a "unified file system" to **distributed storage**:
  - **NFS, GPFS, Lustre** — traditional parallel file systems used in HPC clusters and SLURM-managed environments.
  - **WekaFS, VAST Data, DDN EXAScaler** — purpose-built, high-throughput storage systems specifically designed for AI training workloads.
- **GPUDirect Storage (GDS)** — an NVIDIA technology that:
  - **Bypasses the CPU entirely**
  - Uses **DMA (Direct Memory Access)** to transfer data **directly from NVMe to GPU memory**
  - This avoids the extra hop through CPU/system memory, reducing latency and CPU load.

---

## 4. NUMA Architecture in Multi-Socket AI Servers

### 4.1 What is NUMA?
- **NUMA = Non-Uniform Memory Access.**
- In multi-socket servers, **each CPU socket has its own locally-attached DRAM**.
- Accessing memory that belongs to a **different (remote) socket** takes noticeably longer — cross-socket ("cross-NUMA") access adds **~2x latency** vs local access.
- Even a **single AMD EPYC socket** internally behaves like a NUMA system due to its **CCX/CCD chiplet topology** (cores are grouped into clusters with varying distances to memory).

### 4.2 Inter-Socket Interconnects
- **AMD**: calls this **Infinity Fabric**
- **Intel**: calls this **UPI (Ultra Path Interconnect)**
- These are the high-speed links connecting multiple CPU sockets together.

### 4.3 Worked Example: NVIDIA DGX H100
- Topology: **2 CPU sockets**, each with **96 cores** and its own dedicated DDR5 (6 channels each).
- Each socket hosts **4 GPUs** (GPU 0–3 → Socket 0; GPU 4–7 → Socket 1), connected via **PCIe switches**.
- GPUs within the same socket's group are connected via **NVLink 4** (~900 GB/s bidirectional).
- **NUMA affinity matters for performance:**
  - If your data-loading process (on CPU cores in Socket 0) feeds a GPU **also attached to Socket 0** (e.g., GPU 0–3) → data stays within the same local DDR5 → **no extra cross-socket transfer needed.**
  - If instead you pin data loading to Socket 0 but training runs on a GPU in Socket 1 (e.g., GPU 5) → **data must cross sockets**, adding latency.
  - **Practical implication:** you must consciously **pin** your data-loading and training workloads to the same NUMA node/socket at "system design" and coding time to avoid this penalty.

### 4.4 Bandwidth Summary Table (from slides)

| Link / Component | Bandwidth |
|---|---|
| DDR5-4800 (1 channel) | 38.4 GB/s |
| 12-channel DDR5 (EPYC, per socket) | ~460 GB/s |
| Cross-socket (Infinity Fabric / UPI) | ~400 GB/s |
| NVLink 4 (GPU–GPU, H100) | 900 GB/s (bidirectional) |
| PCIe (CPU–GPU) | ~128 GB/s (bidirectional) |
| HBM3 (H100, on-chip) | 3.35 TB/s (~3,350 GB/s) |

**Rough overall hierarchy (slowest → fastest):** HDD → SATA SSD → NVMe SSD → NVMe RAID → DDR5 (1 channel) → DDR5 (multi-channel/socket) → cross-socket link → NVLink (GPU-GPU) → HBM (on-chip GPU memory).

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/a0e76802-f6d4-427f-8f4e-8ea3dee61277" />


## 5. Data Movement Costs: The Hidden Performance Tax

### 5.1 Energy Cost of Moving Data vs. Computing
- Moving **1 byte across the PCIe bus** costs **~700 pJ (picojoules)** of energy.
- Performing **1 FP32 MAC (multiply-accumulate) operation on-chip** costs only **~1 pJ**.
- → **Moving data is ~700x more energy-expensive than computing on it.**

### 5.2 The Compute–Bandwidth Gap
- **H100 peak compute:** ~3.958 PFLOPS (at FP8 precision)
- **H100 system bandwidth (PCIe):** ~128 GB/s
- → There is roughly a **30,000x gap** between how fast the GPU can compute vs. how fast data can move in/out over PCIe.
- **Conclusion (key takeaway of the whole segment):** Optimal AI system design should focus on **minimizing data movement and maximizing data reuse in fast memory (e.g., on-chip HBM/SRAM)** — not just on adding more compute.

### 5.3 Techniques to Reduce Data Movement
| Technique | What it does |
|---|---|
| **Operator Fusion** | Combine multiple ops (e.g., LayerNorm + Dropout + Activation) into a single GPU kernel, avoiding repeated round-trips to memory. Used in Flash Attention. |
| **Tiling** | Partition large matrices into smaller chunks that fit into fast on-chip memory (L2/SRAM), enabling data reuse without repeated DRAM accesses. |
| **Gradient Checkpointing** | Instead of storing all activations from the forward pass, recompute them on-the-fly during the backward pass — trades compute for reduced memory footprint/movement. |
| **Prefetching** | Overlap data transfer with compute (e.g., via CUDA streams and asynchronous copy), hiding data-movement latency behind useful computation. |

### 5.4 Concrete Example: Attention Mechanism
- Vanilla (standard) attention in Transformers requires **O(n²)** data fetches.
  - Example: for a sequence length of 2048, this means **2048 × 2048 tiles** need to be fetched for each attention computation.
- **Flash Attention** reduces data movement through HBM by **3–5x** via tiled/fused computation — it's a landmark example of an algorithm redesigned specifically to reduce memory traffic, not FLOPs.

### 5.5 Other Memory-Efficient Techniques Mentioned
- **Gradient Accumulation:** accumulate gradients over multiple mini-batches to simulate a large effective batch size without needing large VRAM.
- **Mixed Precision (BF16/FP16):** halves the memory bandwidth requirement compared to FP32, often yielding a **1.5–2x speedup**.

### 5.6 Guiding Philosophy
> "The free lunch is over for compute. The next frontier is memory-efficient algorithms."
— an idea adapted (originally from Herb Sutter's 2005 "free lunch" argument about the end of easy CPU clock-speed scaling) and applied here to modern AI systems: further gains will increasingly come from **smarter memory/data-movement strategies**, not just faster chips.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/015c57dc-2ecd-4d6f-bdc7-a9ce5d26742d" />

## 6. Segment Summary (Key Takeaways)

1. **Memory hierarchy** spans from **registers (<1 ns)** through **DRAM (~60 ns)** to **NVMe (~100 μs)** — latency at each level governs how you design an AI pipeline.
2. **HBM3 (3.35 TB/s) vs. DDR5 (~460 GB/s per socket)** — on-chip bandwidth is roughly **7x higher**, which is why you want to keep "hot" (frequently accessed) data resident in GPU VRAM/HBM as much as possible.
3. **NUMA topology** determines which CPU cores and memory channels have the lowest-latency path to a given GPU — always be deliberate about **pinning affinity** between data-loading processes and the GPUs they feed.
4. **DataLoader optimization** (pinned memory, correct `num_workers`, prefetching) is essential to prevent **GPU starvation** (GPU sitting idle waiting for data).
5. **Data movement's energy/time cost dwarfs the arithmetic cost** — techniques like **operator fusion, tiling, and mixed precision** are first-order optimizations that matter more than raw compute upgrades.

<img width="2752" height="1536" alt="image" src="https://github.com/user-attachments/assets/0c1963c4-3b8a-481c-9597-2c9cfcecc58a" />

