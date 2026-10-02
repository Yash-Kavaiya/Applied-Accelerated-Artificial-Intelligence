# Week 2 · Session 4 of 4: Schedulers & GPU Allocation in Multi-User Environments

**Course:** Containerized AI Systems
**Instructor:** Satyadhyan Chickerur, Ph.D., Director, Centre for AI Research, KLE Technological University

---

## Table of Contents

1. [Learning objectives](#1-learning-objectives)
2. [Why schedulers exist](#2-why-schedulers-exist)
3. [Scheduler taxonomy: Slurm vs Kubernetes](#3-scheduler-taxonomy-slurm-vs-kubernetes)
4. [How Slurm hands out GPUs](#4-how-slurm-hands-out-gpus)
5. [Kubernetes and GPUs (preview of Week 3)](#5-kubernetes-and-gpus-preview-of-week-3)
6. [The device plugin](#6-the-device-plugin)
7. [Requests, limits and QoS](#7-requests-limits-and-qos)
8. [Fractional GPUs: time-slicing, MPS, MIG](#8-fractional-gpus-time-slicing-mps-mig)
9. [CUDA_VISIBLE_DEVICES](#9-cuda_visible_devices)
10. [Time-slicing in detail](#10-time-slicing-in-detail)
11. [MPS in detail](#11-mps-in-detail)
12. [MIG in detail](#12-mig-in-detail)
13. [Choosing a sharing mechanism](#13-choosing-a-sharing-mechanism)
14. [Slurm fair-share scheduling](#14-slurm-fair-share-scheduling)
15. [Preemption and checkpointing](#15-preemption-and-checkpointing)
16. [Policy: queues, priorities, quotas, backfill](#16-policy-queues-priorities-quotas-backfill)
17. [GPU accounting and chargeback](#17-gpu-accounting-and-chargeback)
18. [Kubernetes quotas and namespace isolation](#18-kubernetes-quotas-and-namespace-isolation)
19. [Multi-node jobs: gang scheduling](#19-multi-node-jobs-gang-scheduling)
20. [Topology-aware scheduling](#20-topology-aware-scheduling)
21. [Debugging: pods stuck in Pending](#21-debugging-pods-stuck-in-pending)
22. [Cost-efficiency: utilization is the number that matters](#22-cost-efficiency-utilization-is-the-number-that-matters)
23. [Lab 2 and lecture demo](#23-lab-2-and-lecture-demo-watch-a-gpu-work)
24. [Week 2 recap and bridge to Week 3](#24-week-2-recap-and-bridge-to-week-3)
25. [Corrections and gotchas](#25-corrections-and-gotchas)
26. [Cheat sheet and review questions](#26-cheat-sheet-and-review-questions)

---

## 1. Learning objectives

After this session you should be able to:

- **Explain** why ad-hoc GPU sharing collapses beyond a handful of users.
- **Compare Slurm and Kubernetes** as GPU schedulers and say when each fits.
- **Distinguish the three fractional-GPU mechanisms:** time-slicing, MPS, MIG.
- **Inspect and reason about MIG partitions** with `nvidia-smi` (hands-on).
- **Measure GPU utilization** and spot the classic waste patterns (hands-on).

> **Why this matters to a developer:** most of this is the system administrator's job, but you need to understand the admin's pain points. It shapes how you write jobs (resource requests, checkpointing, honest time limits).

---

## 2. Why schedulers exist

**The multi-user problem:** 8 GPUs, 30 people.

| | Without a scheduler | With a scheduler |
|---|---|---|
| Access | SSH + a spreadsheet + "is anyone using GPU 3?" on chat | Jobs **declare needs** (GPUs, memory, time) |
| Behaviour | Collisions, jobs OOM-killing each other, idle GPUs at night | Scheduler **queues, places, isolates, preempts, accounts** |
| Visibility | Zero accounting, no accountability | Utilization becomes a **managed number** |

> **Key sentence:** a scheduler converts hardware from **territory** into a **metered utility**.

---

## 3. Scheduler taxonomy: Slurm vs Kubernetes

| | **Slurm** (HPC lineage) | **Kubernetes** (cloud lineage) |
|---|---|---|
| Workload model | **Batch jobs** with a start and an end | **Long-running services + jobs** |
| Key features | Queues/partitions, priorities, fair-share, gang scheduling for multi-node training | Declarative desired state, self-healing, rich ecosystem (operators, autoscaling) |
| Default for | Research clusters, supercomputers, **training farms** | Serving / inference, cloud-native ML platforms |

**Common pattern:** many organizations run **both**: Slurm for training, Kubernetes for serving. Alternatively, Kubernetes with batch add-ons (Volcano, etc.) does double duty.

---

## 4. How Slurm hands out GPUs

```bash
# Ask for resources; the scheduler finds them
$ srun --gres=gpu:2 --mem=64G --time=02:00:00 nvidia-smi

# Batch script header
#SBATCH --gres=gpu:b200:4     # 4 B200s on one node
#SBATCH --partition=train     # which queue
#SBATCH --time=08:00:00       # hard wall-clock limit

$ squeue -u $USER                                    # where is my job?
$ sacct -j 12345 --format=Elapsed,AllocGRES,State    # accounting
```

- **GRES** = Generic RESource (GPUs are requested via `--gres`).
- Slurm sets **`CUDA_VISIBLE_DEVICES`** so each job sees only its granted GPUs.
- **Time limits + fair-share priority** keep one team from "squatting" on the cluster.
- You declare: how many GPUs, how much memory, which queue, and for how long.

---

## 5. Kubernetes and GPUs (preview of Week 3)

```yaml
# pod.yaml (fragment)
resources:
  limits:
    nvidia.com/gpu: 1     # whole GPUs, surfaced by the device plugin
    memory: "32Gi"
    cpu: "8"
```

- The **NVIDIA device plugin / GPU Operator** advertises GPUs as schedulable resources.
- The scheduler places pods **only on nodes with free GPUs**; the container runtime wires the GPUs in (Session 1's NVIDIA Container Toolkit again).
- GPUs are requested as **whole units by default**. Fractionalization is an **add-on** (§8).
- A **pod** is the smallest schedulable unit (one or more containers running your app).

> ⚠️ The lecture said "I cannot request half a GPU, I can request a quarter GPU". That is garbled. Intended meaning: **you can't request half *or* a quarter GPU** by default. The admin has to enable time-slicing or MIG to make fractions available.

---

## 6. The device plugin

**What it is:** a **DaemonSet** pod running on every GPU node. It discovers GPUs, registers each as an allocatable resource (`nvidia.com/gpu`) and exposes them to the **kubelet**.

**Allocation flow**

1. The **scheduler** selects a node with free GPUs.
2. The **kubelet** calls the device plugin's `Allocate` RPC.
3. The plugin returns **mount paths + env vars** (`CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES`, device nodes).
4. The **container runtime** wires them into the container.

```bash
# What has the device plugin advertised?
kubectl get nodes -o json | \
  jq '.items[].status.capacity | select(."nvidia.com/gpu")'
# -> {"nvidia.com/gpu": "8"}

# GPU Operator installs driver + plugin + DCGM:
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace

# With MIG: the plugin exposes slices as separate resources
#   nvidia.com/mig-2g.45gb: "2"
#   nvidia.com/mig-3g.90gb: "1"
kubectl describe node gpu-node-01 | grep nvidia
```

- **DCGM** = Data Center GPU Manager (metrics/health).
- **Helm** = the Kubernetes package manager.
- MIG slices appear as **separate named resources**. They are only visible after the admin has enabled MIG partitioning.

---

## 7. Requests, limits and QoS

| Concept | Meaning |
|---|---|
| **requests** | Minimum the pod needs. The scheduler uses it to find a node that fits. For GPUs, **request must equal limit**. |
| **limits** | Maximum the pod may use. CPU over limit is **throttled**; memory over limit is **OOM-killed**. GPU limit = GPU request, always. |
| **QoS classes** | **Guaranteed** (request = limit), **Burstable** (request < limit), **BestEffort** (nothing set). |

**The "correct pattern" (Guaranteed):**

```yaml
resources:
  requests:
    nvidia.com/gpu: "2"
    memory: "64Gi"
    cpu: "16"
  limits:
    nvidia.com/gpu: "2"     # must match requests
    memory: "64Gi"
    cpu: "16"
```

**Lecture emphasis:** stay within the limits your admin gave you. Exceeding them means throttling or being killed, and your request must be satisfiable within your allotment.

> ⚠️ **Precision:**
> - A pod's QoS class is determined by its **CPU and memory** requests/limits. The GPU entry doesn't change the class. Setting CPU and memory request = limit is what makes it Guaranteed.
> - For extended resources like `nvidia.com/gpu` you may specify **only `limits`**. The request then defaults to the limit.
> - **GPUs are never "throttled".** Only CPU is throttled.

---

## 8. Fractional GPUs: three sharing mechanisms

A whole GPU is often too much for one user (notebooks, small inference). Admins can **logically split** a GPU.

| Mechanism | How it works | Isolation | Notes |
|---|---|---|---|
| **Time-slicing** | Contexts take turns on the GPU | **None** (a tenant can OOM another) | Fine for bursty notebooks, risky for production |
| **MPS** | Kernels from several processes run **concurrently**, sharing SMs | **Soft** | Better utilization |
| **MIG** | **Hardware partitions** (e.g. up to 7 instances) with dedicated memory and SMs | **Hard** | Partition sizes are a fixed menu |

> **Isolation strength:** time-slicing < MPS < MIG.
> **Flexibility** runs the other way.

**Analogy from the lecture:** time-slicing is like CPU multiprogramming and preemption, where tasks take turns after a time quantum.

---

## 9. CUDA_VISIBLE_DEVICES

**What it does:** an environment variable that tells the CUDA runtime **which physical GPU indices are visible** to a process.

```bash
# Slurm injects this before your job starts:
export CUDA_VISIBLE_DEVICES=2,3
```

```python
import torch
print(torch.cuda.device_count())      # -> 2
print(torch.cuda.get_device_name(0))  # -> physical GPU 2
```

- **The scheduler sets it, not you.** Index re-mapping is automatic and invisible to your code: `cuda:0` is physical GPU 2.
- **Kubernetes** does the equivalent through the container's environment, but with a **UUID, not an index**:

```yaml
env:
  - name: NVIDIA_VISIBLE_DEVICES
    value: "GPU-abc123"     # UUID, not index
```

> ⚠️ The slide calls this "isolation without hardware partitioning". It is **visibility control, not a security boundary**. A process can still override the variable. Real enforcement comes from the scheduler/runtime (Slurm's cgroup device constraints, or the container runtime only mounting the allocated device nodes).

---

## 10. Time-slicing in detail

```
| Job A | Job B |    Job A    | Job A | Job B |     <- GPU time ->
```

- The GPU **context-switches** between processes. It is **not controlled by user code**.
- **No memory isolation:** one tenant can OOM another.

| When to use | When NOT to use |
|---|---|
| Bursty notebooks | Latency-sensitive inference |
| Dev/test instances | Multi-process training |
| Small, non-continuous inference jobs | Any job that OOMs if a neighbour takes more than its share |
| Many low-utilization jobs on one GPU | |

**Enable in K8s (device plugin config):**

```yaml
time-slicing:
  resources:
    - name: nvidia.com/gpu
      replicas: 4      # exposes 4x logical GPUs per physical GPU
```

> ⚠️ The slide says switches happen at "microsecond granularity". Time-slice quanta are typically on the order of **milliseconds**. Either way, the point is that you don't control them. Also note that `replicas: 4` creates 4 *schedulable* units, but nothing guarantees each tenant a fixed quarter of memory or compute.

---

## 11. MPS in detail

**MPS (Multi-Process Service):** a server (one per GPU) that lets kernels from **multiple processes** share the GPU's SMs **concurrently**.

```
Process A (PID 101) --\
                       >-- MPS Server (one per GPU) --- GPU SMs (shared)
Process B (PID 202) --/
```

| Aspect | Detail |
|---|---|
| **Advantage over time-slicing** | Kernels execute **simultaneously** instead of taking turns. Slide quotes **30–50%** throughput gain for many small inference requests. |
| **Isolation limitation** | Memory still shared (per the slide). A crashing client can take down the other clients of the MPS server. |
| **When to use** | Inference serving with many small models; same-user multi-process training (several workers); latency-constrained but budget-constrained setups |

> ⚠️ **Nuance:** on Volta and newer, MPS clients get **separate GPU address spaces**, so "a buggy process can corrupt another's memory" is overstated for modern GPUs. What *is* true is that **fatal GPU faults** can propagate to all MPS clients, and there is **no resource-quota isolation** by default. The "30–50%" figure is workload-dependent, not a guarantee.

---

## 12. MIG in detail

**MIG (Multi-Instance GPU)** partitions one physical GPU into **up to 7 isolated instances in hardware**. Each slice gets **dedicated SMs, dedicated HBM and dedicated L2 cache**.

**Why it matters**

- One team's runaway job **cannot OOM-kill** another's.
- Memory errors in one instance don't corrupt a neighbour.
- **Preemption of other tenants doesn't affect your slice.**
- Hard isolation gives **predictable SLAs**.

**Example layout: NVIDIA B200 (180 GB HBM3e), MIG enabled**

| Slice | Resources | Tenant (example) |
|---|---|---|
| `3g.90gb` | 3 SM slices, 90 GB HBM | Team A: LLM training |
| `2g.45gb` | 2 slices, 45 GB | Team B: inference |
| `2g.45gb` | 2 slices, 45 GB | Team C: fine-tuning |

- Each slice gets a **UUID**. Containers and schedulers target it like a full GPU.
- To both the OS and the scheduler, a MIG slice looks like a **smaller, fully isolated GPU**.
- **Partition sizes are a fixed menu** (profiles like `1g`, `2g`, `3g`, ...). You can't carve arbitrary sizes.

**Hardware requirement:** Ampere **A100 or newer** (A100, H100, B200, ...).
**Your laptop (RTX 3050)** does not support MIG, so `nvidia-smi` reports MIG mode as **[N/A]**.

> ⚠️ "Consumer GPUs can never do MIG" is true for GeForce/RTX *laptop* cards like the demo GPU. Some newer workstation/server-class cards do support it, so check the model rather than assuming "data-center only".

---

## 13. Choosing a sharing mechanism

| Attribute | Time-slicing | MPS | MIG |
|---|---|---|---|
| Memory isolation | None (shared pool) | None (shared pool, per slide) | **Hard**, per-slice HBM |
| SM isolation | None | Soft (concurrent) | **Hard**, dedicated SMs |
| Fault isolation | No: OOM spreads | Partial: crash spreads | **Yes**: slices independent |
| Latency impact | High (context switch) | Low (concurrent) | **None** (dedicated HW) |
| GPU requirement | Any NVIDIA | Volta+ (CC 7.0+) | Ampere A100+ (CC 8.0+) |
| Max tenants | Unlimited (soft) | 48 clients | Up to 7 instances |
| **Best for** | Dev notebooks | Inference serving | Multi-team production clusters |

---

## 14. Slurm fair-share scheduling

- **Priority = f(shares, recent_usage).**
- Teams with more allocated **shares** start higher.
- The more GPU-hours you **consumed recently**, the **lower** your current priority.
- **Heavy users sink; idle teams float up**, so usage self-balances.
- **Half-life decay:** usage weight decays exponentially (configurable, typically 7–14 days). Idle time restores priority.
- **Design long jobs to checkpoint.** Short gaps don't hurt you.

Illustrative curve from the slide: Team A (heavy) starts high, sinks to ~18 by Day 7, then recovers. Team B (idle then active) does the opposite.

> ⚠️ After one half-life, past usage counts for **half** as much. It decays gradually and isn't fully "restored" after a week of idleness.

---

## 15. Preemption and checkpointing

| Step | What happens |
|---|---|
| 1 | A **higher-priority job** (or a reservation) needs the GPUs you're using |
| 2 | Slurm sends **SIGTERM**, then **SIGKILL** after a grace period (typically 60–300 s, configurable) |
| 3 | Your **checkpoint handler** saves weights + optimizer state + step number to **shared storage** |
| 4 | Slurm **re-queues** the job. On the next allocation it loads the checkpoint and **continues from step N** |

```python
# Catch SIGTERM; save checkpoint before Slurm kills the job
import signal, torch

def checkpoint_handler(signum, frame):
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict()}, 'ckpt.pt')
    exit(0)

signal.signal(signal.SIGTERM, checkpoint_handler)
# In your training loop: load ckpt.pt if it exists on startup
```

**Key points**

- Save to **shared storage**, not node-local disk. The re-queued job may land on another node.
- On startup: *if a checkpoint exists, resume from it.*

> ⚠️ **Practical caveats for the sample code:**
> - `epoch`, `model` and `optimizer` must be reachable (globals) from the handler.
> - Saving inside a signal handler can interrupt a half-written step. A safer pattern is to set a flag in the handler and checkpoint at the next step boundary.
> - Requeue only happens if the job is submitted with `--requeue` and the partition's preemption mode is `REQUEUE`.
> - The lecture said "continues from step one". It should be **step N** (where it stopped).

---

## 16. Policy: queues, priorities, quotas, backfill

Scheduling is **policy, not just placement**: fairness, queues, priorities, quotas.

- **Queues/partitions** segment hardware by purpose: interactive vs batch vs preemptible.
- **Fair-share:** heavy recent users sink, idle teams float up.
- **Preemption:** low-priority jobs yield to urgent ones; checkpointing makes this survivable.
- **Quotas** cap concurrent GPUs per team or namespace: the brake on noisy neighbours.
- **Backfill:** short, well-sized jobs slot into scheduling gaps. **Declare honest time limits.**

---

## 17. GPU accounting and chargeback

**Why it matters:** without accounting, GPU usage is **invisible**. Teams over-allocate, jobs run overnight at 0% utilization, and the infrastructure team can't justify buying more hardware.

| | Slurm | Kubernetes |
|---|---|---|
| Per-job data | `sacct`: `AllocGRES`, `Elapsed`, `CPUTime`, `MaxRSS` | kube-state-metrics + **DCGM Exporter** + **Prometheus** |
| Aggregation | `sreport`: per-user, per-account GPU-hours | Per-**namespace** GPU-hours |
| Money | GPU-hours x rate = team bill | **Kubecost / OpenCost** add dollar amounts; monthly chargeback to team leads |

```bash
# Slurm: GPU-hours used by each user last week
sreport cluster AccountUtilizationByUser \
  Start=$(date -d '7 days ago' +%Y-%m-%d) End=now \
  --tres=gres/gpu -t Hours

# Kubernetes: GPU-hours per namespace (Prometheus query, from slide)
# sum by (namespace)(rate(DCGM_FI_DEV_GPU_UTIL[1h]) * on(pod) kube_pod_info) * 720
```

> ⚠️ The Prometheus query on the slide is schematic. `DCGM_FI_DEV_GPU_UTIL` is a **gauge** (percent), so `rate()` isn't meaningful on it, and joining on `pod` normally needs `group_left`. Also, **allocated GPU-hours** (what you were granted) and **utilized GPU-hours** (what you actually used) are different numbers. Chargeback usually bills the former; efficiency reports use the latter.

---

## 18. Kubernetes quotas and namespace isolation

```yaml
# ResourceQuota caps GPU use per namespace
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-a-gpu-quota
  namespace: team-a
spec:
  hard:
    requests.nvidia.com/gpu: "4"
    limits.nvidia.com/gpu: "4"
    requests.memory: "256Gi"
---
# LimitRange sets per-pod defaults
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limit-range
  namespace: team-a
spec:
  limits:
    - type: Container
      default:
        nvidia.com/gpu: "1"
      max:
        nvidia.com/gpu: "2"
```

| Object | Role |
|---|---|
| **ResourceQuota** | New pods that would exceed the namespace quota are **rejected at admission**. Running pods are **not evicted**. The quota is a **ceiling, not a guarantee**. |
| **LimitRange** | Per-container defaults and maximums |
| **PriorityClass** | Mark pods critical (won't be preempted) vs low (first to go). Combine with quotas for **SLA tiers**. |
| **Admission webhooks** | Enforce GPU annotations, require team labels, block oversized requests before they reach the scheduler |

> ⚠️ Kubernetes docs say extended resources (like GPUs) can be quota'd **only with the `requests.` prefix**. The `limits.nvidia.com/gpu` line may be rejected, so verify on your cluster version.

---

## 19. Multi-node jobs: gang scheduling

**The multi-node problem:** a 4-node distributed training job needs **all 4 nodes at once**. If 3 start and 1 is unavailable, the 3 sit idle holding resources: the **"stranded GPU"** problem.

**Slurm:** all tasks of a job are launched simultaneously. If any node is unavailable, the whole job waits in the queue. **No partial allocations.**

**Kubernetes:** the default scheduler doesn't do all-or-nothing. A batch scheduler such as **Volcano** adds it.

```bash
# Slurm: 4 nodes x 8 GPUs each = 32 GPUs, all at once
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
```

```yaml
# Kubernetes: Volcano for gang scheduling (fragment)
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
spec:
  minAvailable: 4        # all 4 pods or none
  tasks:
    - replicas: 4
      template:
        spec:
          # containers: ... resources: limits: nvidia.com/gpu: "8"
```

**Lecture advice:** first ask whether you **really need** multi-GPU. If one GPU isn't well utilized, adding more won't help and can make the run slower. Get a single GPU to ~99% utilization first.

> ⚠️ In Slurm terminology, "gang scheduling" actually refers to time-slicing jobs on the same resources. What the slide describes (all-or-nothing allocation of a job's nodes) is Slurm's **default allocation behaviour**. The *concept* is right, but the label is loose.

---

## 20. Topology-aware scheduling

**Why topology matters:** GPU-to-GPU bandwidth depends on the link.

- **NVLink** is NVIDIA's proprietary GPU interconnect, very high bandwidth.
- **PCIe** is much lower, and crossing CPU sockets (`SYS`) is the slowest path.
- Slide's numbers: NVLink 900 GB/s vs PCIe 32 GB/s, so **~28x slower** with wrong placement (900 / 32 ~ 28).
- Collective operations such as **all-reduce** are bottlenecked by the slowest link.

```bash
nvidia-smi topo -m
#        GPU0 GPU1 GPU2 GPU3 ...
# GPU0    X   NV5  NV5  SYS
# GPU1   NV5   X   NV5  SYS
# NV# = NVLink connection; SYS = cross-CPU-socket via PCIe (slow)
```

**How schedulers handle it**

- `nvidia-smi topo -m` shows the **topology matrix**.
- A **topology-aware scheduler** reads it and prefers **co-located GPUs** for multi-GPU jobs.
- `CUDA_VISIBLE_DEVICES` controls which ones the job sees.
- Kubernetes: **Topology Manager** + **CPU Manager** (`--topology-manager-policy=best-effort`, `--cpu-manager-policy=static`), plus a NUMA-aware device plugin placing GPUs on the **same NUMA node as the allocated CPUs**.

> ⚠️ **Slide inaccuracies:**
> - In `nvidia-smi topo -m`, `NV5` means **5 NVLink connections** between the GPUs, **not "NVLink 5"** (the generation).
> - 900 GB/s is Hopper-era (NVLink 4) per-GPU bandwidth. NVLink 5 on Blackwell is about 1.8 TB/s. The "28x" ratio is illustrative.
> - On NVSwitch-based 8-GPU systems (e.g. DGX/HGX), all 8 GPUs are typically in **one** NVLink domain. The "0-3 vs 4-7" split is true for some PCIe/NVLink-bridge servers but isn't universal. **Always check `topo -m` on your hardware.**

---

## 21. Debugging: pods stuck in Pending

```text
Pod Pending?
  -> kubectl describe pod <name>   (read the Events section)

  "Insufficient nvidia.com/gpu"?  -> no node has free GPUs; check quota + running pods
  "No nodes available"?           -> all nodes full or tainted; check node labels vs selector
  GPU allocated but 0% util?      -> check container env vars / that your code actually uses the GPU
```

```bash
# 1. Why is the pod pending?
kubectl describe pod <pod-name> -n <namespace>          # check Events

# 2. Node GPU capacity
kubectl get nodes -o custom-columns="NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# 3. What's consuming the quota?
kubectl describe resourcequota -n <namespace>

# 4. Is the device plugin running on the node?
kubectl get pods -n kube-system | grep nvidia-device-plugin
```

> ⚠️ "GPU allocated but 0% utilization" has many causes beyond `CUDA_VISIBLE_DEVICES`: CPU-only framework wheel, code never moved to `.cuda()`, **input-pipeline starvation** (see the lab), or an idle notebook.

---

## 22. Cost-efficiency: utilization is the number that matters

**Waste patterns**

- Notebooks holding GPUs overnight at **0%**.
- Jobs sized "GPU because habit".
- **Data-loading bottlenecks** leaving SMs starved.
- Whole GPUs used for tiny inference.

**Levers**

- **Idle reaping** for interactive sessions.
- **Right-sizing via MIG.**
- **Profiling input pipelines.**
- **Utilization dashboards per team**: visibility changes behaviour.

> A modern 8-GPU node costs as much as several engineers' salaries per year. **Idle GPUs are the most expensive idle thing you own.**

---

## 23. Lab 2 and lecture demo: watch a GPU work

### 23.1 Slide commands

```bash
# Terminal 1: a small training loop in your Week-2 container
docker run --rm --gpus all gpu-hello python3 busy_train.py

# Terminal 2: live device metrics, 1 s resolution
$ nvidia-smi dmon -s um      # u = utilization, m = memory
# gpu  sm  mem ... fb(MB)
#  0   97  64      14210   <- healthy training
#  0    8   2      14210   <- input-starved: GPU waits on data

# Per-process accounting:
$ nvidia-smi pmon -c 5
```

**Experiment:** shrink DataLoader workers to 0 and watch `sm%` collapse. **That dip is money.**

### 23.2 What was shown in the demo

1. **Identify the GPU:** `nvidia-smi` showed an **RTX 3050 laptop GPU** (not data-center grade).
2. **Check MIG support:** the MIG-mode query returned **[N/A]** (the command is in the course README).
3. **Run `busy_train.py` normally:** a small CNN training loop that saturates the GPU. `nvidia-smi` showed **SM% ~96-100%**.
4. **Run with the starve flag:** the lecturer then ran the same program with a "starve" option. SM% dropped to a low level, which demonstrates that the GPU is **waiting on the input pipeline**, not computing.
5. **Shared memory note:** the container used a **2 GB shared-memory** allocation (`--shm-size`). If it is too small or unavailable, DataLoader workers can fail, and the flag/option changes the behaviour.

**Takeaway:** a GPU can be "allocated" yet nearly idle. Utilization (SM%) is the real signal, and the input pipeline is a common culprit.

> ⚠️ The transcript renders the flag as "star". From context it is a **"starve"** option (likely reducing data-loader workers). Check the README/script for the exact flag name.

---

## 24. Week 2 recap and bridge to Week 3

| Session | One-line takeaway |
|---|---|
| **S1** | Containers freeze **userspace**; the NVIDIA toolkit wires GPUs in at run time |
| **S2** | **Lockfiles + the compatibility chain** make environments deterministic; Jupyter needs port, volume, `shm` |
| **S3** | A **3-level probe** (driver -> toolkit -> framework) localizes GPU failures; **CI** rebuilds trust on every commit |
| **S4** | **Schedulers** turn GPUs into a **metered, fair, measurable utility** |

**Week 3:** Kubernetes for AI in depth: deployments, services, storage, device plugins, requests & limits.

---

## 25. Corrections and gotchas

| Topic | What the slide/lecture said | Correct / clarified |
|---|---|---|
| Fractional requests | "Can't request half, *can* request a quarter" | Neither by default. Fractions need time-slicing/MIG enabled by the admin. |
| GPU "throttling" | Over-limit resources are throttled | **CPU** is throttled, **memory** is OOM-killed. GPUs are not throttled. |
| QoS Guaranteed | "GPU pods should always be Guaranteed" | QoS class depends on CPU/memory request = limit. Setting GPU alone doesn't determine it. |
| `CUDA_VISIBLE_DEVICES` | "Isolation" | Visibility control only. Enforcement comes from cgroups/runtime. |
| Time-slice granularity | "Microsecond" | Typically milliseconds. Not user-controllable either way. |
| MPS isolation | "Buggy process can corrupt another's memory" | Overstated on Volta+ (separate address spaces). Fatal faults can still propagate. |
| Fair-share decay | "A week idle restores priority" | Usage decays by half per half-life, gradually. |
| Preemption restart | "Continues from step one" (lecture) | Resumes from **step N**. Needs `--requeue`/REQUEUE mode. |
| Gang scheduling | Slurm "gang scheduling" = all-or-nothing | That is default *allocation* behaviour. "Gang" in Slurm formally means job time-slicing. |
| NVLink numbers | "NVLink 5 = 900 GB/s", `NV5` = "NVLink 5" | `NV5` = 5 links. 900 GB/s is Hopper-era. NVLink 5 is ~1.8 TB/s. |
| 8-GPU domains | "GPUs 0-3 and 4-7 in different domains" | Depends on the server. NVSwitch systems are usually one domain. |
| Prometheus query | `rate()` of a utilization gauge | Gauges aren't rate-able. Allocated vs utilized hours differ. |
| ResourceQuota | `limits.nvidia.com/gpu` in quota | Extended resources are quota'd with `requests.` only. Verify. |
| Pending diagnosis | 0% util = bad `CUDA_VISIBLE_DEVICES` | Also check CPU-only wheel, code on CPU, input starvation. |
| MIG "data-center only" | Consumer GPUs impossible | Laptop/GeForce cards: no. Some newer workstation/server cards do support it. |
| Transcript | "non-nodes", "star flag", "GPU hours... Slum" | Garbled speech-to-text: MIG slices, `--starve`-style flag, Slurm. |

---

## 26. Cheat sheet and review questions

### Cheat sheet

```text
Scheduler:    declare needs -> queue -> place -> isolate -> preempt -> account
Slurm:        batch, partitions, fair-share, --gres=gpu:N, sets CUDA_VISIBLE_DEVICES
Kubernetes:   services + jobs, nvidia.com/gpu via device plugin, whole GPUs by default
Sharing:      time-slicing (no isolation) < MPS (soft) < MIG (hard, A100+ only)
GPU request:  request == limit; set CPU/memory request == limit for Guaranteed QoS
Preemption:   SIGTERM -> checkpoint to shared storage -> requeue -> resume at step N
Quota:        ceiling not guarantee; excess pods rejected at admission
Multi-node:   all-or-nothing (Slurm default; Volcano on K8s); avoid stranded GPUs
Topology:     nvidia-smi topo -m ; keep multi-GPU jobs inside one NVLink domain
Pending pod:  describe pod -> node GPU capacity -> quota -> device plugin running?
Utilization:  nvidia-smi dmon -s um ; nvidia-smi pmon -c 5
```

### Review questions

1. Why does ad-hoc GPU sharing collapse beyond a handful of users? What four things does a scheduler add?
2. Compare Slurm and Kubernetes: workload model, strengths, typical use. Why do many orgs run both?
3. What does `CUDA_VISIBLE_DEVICES=2,3` do? What does `cuda:0` refer to?
4. Why must a GPU request equal its limit? What happens when CPU vs memory limits are exceeded?
5. Rank time-slicing, MPS and MIG by isolation and by flexibility. Give one use case for each.
6. Why does a laptop RTX GPU show **[N/A]** for MIG mode?
7. Explain fair-share priority and half-life decay. Why should long jobs checkpoint?
8. Walk through the four steps of preemption. What should a checkpoint contain, and where should it be stored?
9. What is the "stranded GPU" problem? How do Slurm and Kubernetes (Volcano) avoid it?
10. Why does GPU placement (NVLink vs PCIe) matter for all-reduce? How do you inspect topology?
11. A pod is stuck in Pending. List the first four diagnostic steps.
12. What is the difference between allocated and utilized GPU-hours? Why do teams care?
13. In the lab, what did the SM% drop show, and what causes it?

### Practice tasks

- Run `nvidia-smi` and the MIG-mode query on your machine. Record whether MIG is supported.
- Run `nvidia-smi topo -m` and identify the link types between your GPUs (a single-GPU laptop will show only `X`).
- Run the lab in two terminals (`busy_train.py` + `nvidia-smi dmon -s um`). Capture SM% with and without the starve option / `num_workers=0`.
- Write a Slurm batch header for a 2-node, 8-GPU-per-node job with a 12-hour limit.
- Write a Guaranteed-QoS pod resource block for 1 GPU, 32Gi memory, 8 CPUs.
- Add a SIGTERM checkpoint-and-resume pattern to a toy training loop (flag-based, saving at a step boundary).
