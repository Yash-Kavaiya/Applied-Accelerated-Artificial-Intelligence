Let me pull together a full walkthrough of this segment, with a couple of diagrams to make the harder parts (the ring algorithm, the interconnect hierarchy) easier to follow.This is a dense but well-organized segment. Here's the full walkthrough.

## The core problem: why interconnects matter at all

A single GPU is fine until your model doesn't fit in one. As parameter counts, precision choices, and optimizer states grow, you need multiple GPUs — sometimes multiple machines' worth of GPUs. Once you split work across devices, the question becomes: how fast can those devices talk to each other? The answer is the "communication fabric," and it's often the actual bottleneck in training speed, not compute. This segment covers three layers of that fabric — PCIe, NVLink, and NVSwitch — plus the algorithms and topologies that use them.

## PCIe: the universal but limited interconnect

PCIe (PCI Express) is a point-to-point serial link found in essentially every computer. Each "lane" carries one bit per direction *simultaneously* — so it's full-duplex, not half-duplex. A standard x16 slot has 16 transmit lanes and 16 receive lanes (32 wire pairs total), and bandwidth has climbed steadily with each generation:

- **PCIe 3.0 x16**: 16 GB/s per direction (2017-era GPUs)
- **PCIe 4.0 x16**: 32 GB/s per direction (A100-era, 2020)
- **PCIe 5.0 x16**: 64 GB/s per direction (H100-era, 2022)
- **PCIe 6.0**: 128 GB/s per direction (uses PAM-4 signaling)

In an AI server, the CPU's root complex connects downstream to PCIe switches, and multiple GPUs share that switch's uplink bandwidth — which creates contention as you add more GPUs. PCIe is what you use for CPU↔GPU transfers, and it can also do peer-to-peer GPU↔GPU transfers if the GPUs sit behind the same switch. But is it *enough* for GPU-to-GPU traffic at training scale? That's where the numbers get interesting.

**The bandwidth gap.** An H100's HBM3 memory delivers roughly 3,350 GB/s — about 52x faster than PCIe 5.0's 64 GB/s. That mismatch matters a lot once you start doing distributed training.

## Why NVLink was invented: the AllReduce bottleneck

Consider training a 7-billion-parameter model in FP16 across 8 GPUs using **data parallelism** — each GPU holds the full model but trains on a different mini-batch of data. After each backward pass, every GPU has computed different gradients (because each saw different data), and those gradients need to be averaged across all 8 GPUs before anyone updates their weights.

The standard algorithm for this is **AllReduce**, and its communication cost works out to:

```
data volume = 2 × (N-1)/N × params × bytes_per_param
            = 2 × (7/8) × 7B × 2 bytes
            ≈ 24.5 GB
```

At PCIe 5.0's 64 GB/s, that's about **380 ms** just to synchronize gradients — every single training step. That's the bottleneck the slides are pointing at: PCIe is a fine universal interconnect, but it chokes under the specific communication pattern that distributed training demands.

**NVLink was the fix**, developed starting around 2006 and first shipping in 2016 (NVLink 1.0, on the P100) at 160 GB/s bidirectional — already 5x faster than the PCIe of that era. By NVLink 4.0 (2022, H100), it reached **900 GB/s bidirectional per GPU** — 28x faster than PCIe 4.0. Technically, each NVLink 4.0 connection is 18 individual links × 25 GB/s × 2 directions = 900 GB/s. It uses differential signaling with sub-microsecond latency and lets a GPU access a remote GPU's HBM memory almost as if it were local. Redo that same AllReduce math at NVLink speeds and 380 ms drops to roughly **20 ms**.

So the division of labor becomes: **PCIe handles CPU↔GPU**, **NVLink replaces PCIe for GPU↔GPU**.## NVSwitch: solving the multi-hop bottleneck

NVLink solves point-to-point GPU pairs, but it creates a new problem once you have more than two GPUs. If GPU 0 and GPU 1 are directly linked, and GPU 1 and GPU 2 are directly linked, but GPU 0 and GPU 2 are *not*, then GPU 0 talking to GPU 2 has to "hop" through GPU 1 — and every hop degrades effective bandwidth. With 8 GPUs, point-to-point NVLink connections alone would create serious bottlenecks for some GPU pairs.

**NVSwitch** solves this by acting as an all-to-all crossbar switch. In a DGX H100, 8 GPUs connect through 4 NVSwitches (4th generation, each with 57.6 TB/s of total switching bandwidth and 256 NVLink ports) in an all-to-all fat-tree arrangement. The result: **any GPU can talk to any other GPU at the full 900 GB/s, with no bandwidth penalty regardless of which pair you pick.** This is what makes 8-GPU nodes practical for large-scale training — no GPU is ever "far" from any other GPU. NVLink Network extends this same fabric across multiple physical nodes using external NVLink switches.

## Two ways to split work: data parallel vs. model parallel

Before getting into the communication algorithms, it's worth being clear on *why* GPUs need to talk to each other in the first place. There are two basic strategies for spreading a model across GPUs:

- **Model parallel**: the model itself is chunked — e.g., layers 1–16 on GPU 0, layers 17–32 on GPU 1, and so on. Every GPU sees *all* the data, but only computes a slice of the network. Results get aggregated across the layer boundaries.
- **Data parallel**: every GPU holds a *complete copy* of the model, but each one trains on a different mini-batch of data. Since each copy sees different data, each computes different gradients after the backward pass — and those gradients must be synchronized (averaged) before the next update, which is exactly the AllReduce operation discussed above.

## Collective communication primitives

Distributed training relies on a small set of standard "collective" operations, most commonly provided by NVIDIA's **NCCL** library:

- **AllReduce**: combine (e.g., sum/average) values across all GPUs; every GPU ends up with the full result.
- **ReduceScatter**: partial reduce + scatter — each GPU ends up with just 1/N of the reduced tensor.
- **AllGather**: each GPU collects the missing pieces from all others to reconstruct the complete tensor.
- **Broadcast**: one GPU sends data to everyone else (e.g., for model initialization or checkpoint loading).

**Ring AllReduce**, the most widely used algorithm for gradient synchronization, works in two phases:

1. **ReduceScatter** (N−1 steps): the gradient tensor is split into N chunks. Each GPU passes its chunk to its ring neighbor, who adds it to the same chunk it's accumulating. After N−1 steps, each GPU holds one *fully reduced* chunk (its 1/N piece of the final answer).
2. **AllGather** (N−1 steps): those fully-reduced chunks are then passed around the ring again so that every GPU ends up with all N chunks — i.e., the complete, fully-reduced gradient.

The elegant part: total data moved per GPU is `2 × (N-1)/N × gradient_size`, regardless of ring size — this is provably near-optimal for the available bandwidth.Each GPU only ever talks to its two ring neighbors — it never needs a direct connection to every other GPU, which is exactly why this algorithm scales well even as N grows.

## NCCL vs. MPI vs. Gloo

These are the software libraries that actually implement the collectives above on top of the hardware interconnects:

- **NCCL** (NVIDIA): GPU-native, uses NVLink and RDMA directly, gives the best performance — but only on NVIDIA hardware.
- **MPI** (via OpenMPI): CPU-based, the traditional HPC standard, works across heterogeneous clusters.
- **Gloo**: a CPU fallback used inside PyTorch's DistributedDataParallel (DDP), mainly for CPU-only or debugging setups.

## Network topologies for multi-node clusters

Once you go beyond a single 8-GPU node, the *topology* connecting nodes matters as much as the per-link bandwidth:

- **Fat-tree** (standard data center): a 3-tier Clos design (top-of-rack → aggregation → core switches) that aims for full bisection bandwidth. Oversubscription ratios (2:1, 4:1) save cost but introduce congestion at scale.
- **Rail-optimized** (NVIDIA Quantum-2 InfiniBand): each GPU connects to a different spine switch; intra-node traffic uses NVSwitch, inter-node traffic spreads across multiple spine paths for load balancing. Quantum-2 offers 400 Gb/s per link, 7,500 ports per fabric, and sub-100ns latency.
- **Torus networks** (Google TPU pods, the Frontier supercomputer): nodes arranged in a multi-dimensional grid with wrap-around edges. A TPU v4 pod uses a 3D torus across 4,096 chips — excellent for collective operations. AMD's Frontier system uses a 3D torus with a Slingshot-11 network.

Practical rule of thumb from the slides: single-node 8-GPU setups are well served by NVSwitch alone; multi-node clusters under ~64 GPUs typically use 200 Gb/s InfiniBand or RoCEv2 with RDMA; HPC-scale clusters (1000+ GPUs) lean on InfiniBand HDR/NDR or proprietary fabrics like NVLink Network.

## Tying it together

- PCIe is universal and fine for CPU↔GPU, but its ~64 GB/s (Gen 5) is a serious bottleneck for GPU-to-GPU gradient synchronization — roughly 52x slower than an H100's own memory bandwidth.
- NVLink was built specifically to fix that GPU-to-GPU link, reaching 900 GB/s per GPU in its 4th generation (28x PCIe 4.0).
- NVSwitch turns point-to-point NVLink connections into a full mesh, so every GPU pair gets the same 900 GB/s regardless of hop distance — critical once you have more than 2–3 GPUs.
- Ring AllReduce (reduce-scatter + all-gather) is the standard bandwidth-optimal algorithm for synchronizing gradients in data-parallel training, and it's what NCCL implements on top of this hardware.
- Beyond a single node, the choice of network topology (fat-tree, rail-optimized, torus) determines whether that same efficiency holds at cluster scale.

The instructor's framing at the end is worth keeping in mind: none of this is inherently more "complicated" — it's the same principles applied at increasing scale, and the right topology/interconnect choice depends entirely on how big a workload you're actually trying to run.
