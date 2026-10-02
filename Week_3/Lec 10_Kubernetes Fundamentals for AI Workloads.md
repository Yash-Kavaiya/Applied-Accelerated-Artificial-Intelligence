# Week 3 · Session 1: Kubernetes Fundamentals for AI Workloads

**Course:** AI Systems Engineering (Containerized AI Systems track)
**Instructor:** Satyadhyan Chickerur, Ph.D., Director, Centre for AI Research, KLE Technological University

---

## Table of Contents

1. [Learning goals and recap](#1-learning-goals-and-recap-of-week-2)
2. [The problem: one node isn't enough](#2-the-problem-one-node-isnt-enough)
3. [What is container orchestration?](#3-what-is-container-orchestration)
4. [What is Kubernetes?](#4-what-is-kubernetes)
5. [Kubernetes vs alternatives](#5-kubernetes-vs-alternatives)
6. [Why Kubernetes for AI infrastructure](#6-why-kubernetes-for-ai-infrastructure)
7. [Architecture: control plane and data plane](#7-architecture-control-plane-and-data-plane)
8. [Anatomy of a `kubectl apply`](#8-anatomy-of-a-kubectl-apply-request)
9. [Core Kubernetes objects](#9-core-kubernetes-objects)
10. [Pod anatomy](#10-pod-anatomy)
11. [Deployments, ReplicaSets and rollouts](#11-deployments-replicasets-and-rollouts)
12. [Services and networking](#12-services-and-networking-types)
13. [GPU resource model](#13-gpu-resource-model-in-kubernetes)
14. [Scheduler decision flow for GPU pods](#14-scheduler-decision-flow-for-gpu-pods)
15. [GPU pod manifest, field by field](#15-gpu-pod-manifest-field-by-field)
16. [Taints, tolerations and affinity](#16-taints-tolerations-and-affinity)
17. [Namespaces, RBAC, quotas](#17-namespaces-rbac-and-resource-quotas)
18. [Multi-tenant cluster end to end](#18-multi-tenant-cluster-end-to-end)
19. [Lab 1: tiny inference service](#19-lab-1-the-tiny-inference-service)
20. [Glossary and session summary](#20-glossary-and-session-summary)
21. [Corrections and gotchas](#21-corrections-and-gotchas)
22. [Cheat sheet and review questions](#22-cheat-sheet-and-review-questions)

---

## 1. Learning goals and recap of Week 2

**Where Week 2 left off**

| Theme | What we did |
|---|---|
| **One machine** | Containerized GPU workloads with Docker: images, layer caching, CUDA base images, reproducible builds |
| **One GPU, many jobs** | Sharing a single GPU: `CUDA_VISIBLE_DEVICES`, time-slicing, MPS, MIG |
| **The ceiling** | Every technique assumed **one host**. Real training/serving needs many hosts, many GPUs, many teams |

**The question for Week 3:** in Week 2 *you* were the scheduler on a single Docker host. Week 3 hands that job to **Kubernetes**: many nodes, many GPUs, many tenants.

**Session 1 goals**

- Understand why orchestration is needed and what Kubernetes does.
- Know the control-plane and data-plane components and the control loop.
- Know the core objects (Pod, Deployment, Service, ConfigMap/Secret, PVC).
- Understand how GPUs are exposed and scheduled (device plugin, DRA, MIG, taints/tolerations).
- Understand multi-tenancy basics (namespaces, RBAC, ResourceQuota, LimitRange).
- Run Lab 1: containerize a model server and deploy it with readiness/liveness-style health checking.

> The lecture was run on a **laptop** (RTX 3050, 4 GB VRAM, Windows + PowerShell + Docker Desktop with WSL2) so students can reproduce everything, then move to data-center GPUs later.

---

## 2. The problem: one node isn't enough

| Concern | Detail |
|---|---|
| **Capacity** | A single training job may need 8, 64 or 512 GPUs across many physical hosts, not one box |
| **Sharing** | Multiple teams compete for the same GPU pool; someone must enforce fairness and prevent starvation |
| **Resilience** | Hosts fail, jobs crash, nodes get drained for maintenance; work must be **rescheduled automatically** |

**Doing it by hand:**

```bash
# manual, brittle, does not scale
ssh gpu-node-07 'docker run --gpus=4 ...'
ssh gpu-node-12 'docker run --gpus=4 ...'
# node-07 just died, now what?
# who is tracking which team used which GPU?
```

If a 20-hour job was on node-07, you lose it and nobody notices until someone checks.

> **Kubernetes closes this gap:** a control system that **places, watches and reschedules** work across a fleet.

---

## 3. What is container orchestration?

> Orchestration is software that decides **where** your containers run, **keeps them running**, and **reacts automatically** when reality changes: a node fails, load spikes, a job finishes.

| Step | Meaning |
|---|---|
| **You declare** | "I want 3 replicas of this container, each with 1 GPU." |
| **It watches** | Continuously compares **desired state** to **actual cluster state** |
| **It reconciles** | Starts, stops or moves containers until reality matches the declaration |

This **declarative** model (describe the end state, not the steps) is the core idea of Kubernetes.

---

## 4. What is Kubernetes?

- **Kubernetes** ("K8s": 8 letters between K and s) is an **open-source system for automating deployment, scaling and management of containerized applications**.
- **K3s** is a lightweight Kubernetes distribution (the lecturer may use it in some labs).
- **Origin:** Google's internal **Borg** system, open-sourced in **2014**.
- **Governance:** donated to the **CNCF** in **2015** (vendor-neutral, community-run).
- **2026 baseline (per slides):** stable line v1.36; the course targets that release.
- **Why it won:** a large ecosystem (Helm, Argo, Kubeflow, Ray), broad cloud support, and an extensible API that can model almost any workload, including GPUs.
- It is **not GPU-specific**. It orchestrates CPU workloads just as well. This course focuses on GPU/AI use.

---

## 5. Kubernetes vs alternatives

| | Docker Compose | Docker Swarm | Nomad | **Kubernetes** |
|---|---|---|---|---|
| **Scope** | Single host | Small clusters | Multi-workload | **Large clusters** |
| **GPU scheduling** | Manual | Basic | Plugin-based | **Native + DRA** |
| **Ecosystem** | Minimal | Small | Moderate | **Largest (CNCF)** |
| **Multi-tenancy** | None | Limited | Namespaces | **RBAC + quotas** |

> Not a knock on the alternatives. Compose and Nomad are great at smaller scope. **K8s wins once you need multi-team, multi-GPU, self-healing infrastructure**, which is what data-center clusters require.

---

## 6. Why Kubernetes for AI infrastructure

| Benefit | Detail |
|---|---|
| **Scale** | Orchestrate hundreds of GPU nodes; autoscale based on queue depth |
| **Isolation** | Namespace + RBAC separation; each team gets a fair-share GPU quota |
| **Reproducibility** | Same container image from laptop to multi-node cluster |

- **ML platforms** commonly run on top of K8s: **MLflow, Kubeflow, Ray, Argo**.
- **K8s in 2026 (slides):** Dynamic Resource Allocation (DRA) for richer GPU/NIC composability; **Karpenter v1.3** supports Blackwell B200 node-shape provisioning on major clouds.

---

## 7. Architecture: control plane and data plane

### 7.1 Overview

| Plane | Component | Role |
|---|---|---|
| **Control plane** | **API Server** | Gateway for all control-plane operations. Every `kubectl`, webhook and controller goes through it. Does **auth, validation, admission**: "the only door in" |
| | **etcd** | Distributed key-value store; **source of truth** for all cluster state; needs HA in production |
| | **Scheduler** | Matches pods to nodes using resource requests, affinity, taint/toleration rules |
| | **Controller Manager** | Runs many reconciliation loops (Deployment, ReplicaSet, Node, ...) |
| **Data plane (worker nodes)** | **kubelet** | Node agent: starts/stops containers (via CRI), reports node and pod status |
| | **kube-proxy** | Service networking: programs iptables/eBPF-style rules so Services route to live pod IPs |
| | **Container runtime** | **containerd 2.0** (CRI-compliant): pulls images, runs containers |

### 7.2 The control loop (one sentence)

> Every controller repeats: **observe current state → compare to desired state → act to close the gap.**

- The **Scheduler** is just one controller: it watches for pods with **no node assigned** and picks one.
- The **Controller Manager** runs many such loops in parallel.

### 7.3 GPU-specific data-plane jobs

- The kubelet works with the **NVIDIA device plugin**, which advertises `nvidia.com/gpu` as a schedulable resource.
- containerd hands off to the **NVIDIA container runtime**, which exposes the physical GPU device inside the container.
- **Most GPU errors appear at this seam** (Docker/WSL passthrough, runtime, device plugin, pod spec). Everything must be in sync before the pod sees the hardware.

> ⚠️ **Slide diagram caveat:** the control-plane slide shows arrows `etcd → Scheduler → Controller Manager` and says "etcd watches the API server". That's misleading. In reality **only the API server talks to etcd**. The scheduler, controllers and kubelets all **watch the API server**. etcd just stores what the API server writes. Likewise the kubelet doesn't poll on a timer, it holds **watch** connections to the API server.

---

## 8. Anatomy of a `kubectl apply` request

1. `kubectl` sends the pod manifest as an **HTTPS request** to the API Server.
2. The API Server **authenticates** you, **validates** the YAML and runs **admission webhooks**.
3. The object is **persisted to etcd** as the new desired state.
4. The **Scheduler** notices an unscheduled pod, picks the best-fit node and **writes the assignment back** (binding).
5. The **kubelet** on that node sees the assignment and **pulls the image** via containerd.
6. The kubelet **reports Running** back through the API Server; you see it in `kubectl get pods`.

---

## 9. Core Kubernetes objects

| Object | One-liner | Details |
|---|---|---|
| **Pod** | Runs your container | Smallest deployable unit; one or more containers sharing network & storage |
| **Deployment** | Keeps N pods running | Manages a ReplicaSet; declarative rollouts and rollbacks |
| **Service** | Networking abstraction | Stable ClusterIP / NodePort / LoadBalancer in front of a pod set |
| **ConfigMap / Secret** | Config & credentials | Decouple configuration from the image; Secrets are **base64-encoded** (use an external-secrets solution for production) |
| **PersistentVolumeClaim (PVC)** | Storage request | Requests durable storage bound to a **StorageClass** (NFS, Ceph, CSI driver) |

---

## 10. Pod anatomy

A pod is a **shared network namespace + shared volumes** around one or more containers.

```
Pod (shared network namespace + volumes)
├── Container: pytorch        (main process; requests nvidia.com/gpu: 1)
└── Container: log-shipper    (optional sidecar; shares localhost + volume)
```

| Aspect | Behaviour |
|---|---|
| **IP address** | **One pod IP**, shared by every container in it |
| **Communication** | Containers talk over `localhost` |
| **Storage** | Volumes can be mounted into any container in the pod |
| **Lifecycle** | Containers in a pod are scheduled, started and stopped together |

> ⚠️ **Nuance:** the whole pod is scheduled to **one node** as a unit, but individual containers can crash and restart independently within the pod (per `restartPolicy`). "Start/stop together" is true at the pod level (create/delete), not for every container restart.

---

## 11. Deployments, ReplicaSets and rollouts

```
Deployment (you edit this)  ->  ReplicaSet (auto-generated per version)  ->  Pods x N
```

**Rolling update, step by step**

1. You change the container **image tag** on the Deployment.
2. A **new ReplicaSet** is created. The old one scales **down** as the new one scales **up**, gradually.
3. **Rollback is instant:** `kubectl rollout undo` points back at the previous ReplicaSet.

```bash
kubectl create deployment inference \
  --image=nvcr.io/nvidia/pytorch:25.04-py3 --replicas=3
kubectl set image deployment/inference \
  pytorch=nvcr.io/nvidia/pytorch:25.05-py3     # triggers rollout
kubectl rollout status deployment/inference
kubectl rollout undo deployment/inference      # instant rollback
```

**Why it matters for inference:** you can run 3 replicas of a model server, roll out a new model/image version gradually, watch it, and roll back if it misbehaves.

> ⚠️ This example is illustrative. A bare NGC PyTorch image has no server command, so a real Deployment would crash-loop. You'd specify a command or a serving image. Also, a Deployment's pod template must use `restartPolicy: Always` (the default). `Never` is for one-off Pods/Jobs.

---

## 12. Services and networking types

| Type | Meaning |
|---|---|
| **ClusterIP** | Internal-only virtual IP; **default**; pod-to-pod traffic inside the cluster |
| **NodePort** | Opens a static port on **every node**; simple, rarely used directly in production |
| **LoadBalancer** | Provisions a **cloud load balancer** that routes external traffic in |

```
Client  ->  Service (stable virtual IP)  ->  Pod (any replica; IP changes on restart)
```

> **A Service is the stable front door** in front of pods whose IPs keep changing as they're rescheduled. A recreated pod gets a **new IP**, so you never address pods directly.

---

## 13. GPU resource model in Kubernetes

### 13.1 Device plugin (legacy, still the default)

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

- The NVIDIA device plugin advertises GPUs as **countable resources**.
- **Integer only:** no fractional GPU unless MIG or time-slicing is configured.

### 13.2 DRA: Dynamic Resource Allocation

```yaml
resourceClaims:
  - name: gpu-claim
    resourceClaimTemplateName: my-gpu-template
```

- **Structured parameters**, **composite devices** (e.g. GPU + NIC) and **partial-device claims** (e.g. MIG slices).
- Instead of "give me N of a counter", you **claim a device matching a template**.

### 13.3 MIG on Blackwell B200 (from slides)

- Up to **7 MIG slices** (profiles from `1g.23gb` up to `7g.160gb` per the slide).
- Each slice can be exposed as a separate resource, e.g. `nvidia.com/mig-1g.23gb`.
- **Time-slicing** (shared access) via the device plugin's time-slicing config (slide cites `nvidia-device-plugin 0.17+`).
- Your **laptop RTX 3050 has no MIG** (see Week 2 Session 4).

---

## 14. Scheduler decision flow for GPU pods

| Step | What happens |
|---|---|
| 1. **Pod requests** | `nvidia.com/gpu: 1` |
| 2. **Filter nodes** | Which nodes advertise **enough free GPU** (plus taints/affinity etc.)? |
| 3. **Score nodes** | Rank by fit (bin-packing vs spreading) |
| 4. **Bind** | Scheduler writes the node assignment to the API server / etcd |
| 5. **kubelet notices** | Pulls the image, **requests the device from the device plugin** |
| 6. **Container starts** | GPU device node is mounted in |

**If no node has a free GPU:**

- The pod stays **`Pending`**. Check `kubectl describe pod <name>` (Events section).
- A **cluster autoscaler / Karpenter** can provision a new GPU node on demand.
- **Without autoscaling, the pod waits indefinitely.**

> ⚠️ In the lecture the **bind** step and the **container start** got blurred together. The order is: filter → score → **bind** (scheduler) → kubelet pulls/starts → container sees the GPU. A pod can be *Pending* (not bound) for lack of a GPU. The container never starts in that case.

---

## 15. GPU pod manifest, field by field

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-gpu-smoke
  labels:
    app: pytorch-test
spec:
  runtimeClassName: nvidia            # use NVIDIA container runtime
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  containers:
    - name: pytorch
      image: nvcr.io/nvidia/pytorch:25.04-py3
      resources:
        limits:
          nvidia.com/gpu: "1"
      command: ["python", "-c",
                "import torch; print(torch.cuda.get_device_name(0))"]
  restartPolicy: Never
```

| Field | Meaning |
|---|---|
| `runtimeClassName: nvidia` | Tells the kubelet to hand off to the **NVIDIA container runtime** instead of the plain OCI runtime |
| `tolerations: [nvidia.com/gpu: Exists]` | **Required** if GPU nodes carry the taint. Without it the pod can't land there |
| `resources.limits.nvidia.com/gpu: "1"` | The actual GPU request. The scheduler **filters and scores** on this |
| `image: nvcr.io/nvidia/pytorch:25.04-py3` | Pinned image (slide says PyTorch 2.11 / CUDA 13.0, see caveat) |
| `restartPolicy: Never` | A smoke test shouldn't restart forever. **Never** surfaces the error immediately |

> *Etymology aside:* "smoke test" comes from hardware testing: power it on and check that nothing literally smokes before deeper testing.

> ⚠️ **Caveats**
> - **Image tag vs. versions:** NGC tag `25.04-py3` is from April 2025. To my knowledge it ships PyTorch ~2.7 / CUDA 12.9, **not** "PyTorch 2.11 / CUDA 13.0". Check the NGC release notes and pick the tag that matches the baseline.
> - **`runtimeClassName: nvidia` doesn't "fall back to CPU"** if the runtime is missing. The lecture suggested it would. In practice the pod fails to start (RuntimeClass not found or runtime error). Remove the field entirely if the default runtime is already NVIDIA-configured, or create the RuntimeClass first.
> - Extended resources like GPUs can be set under `limits` only; the request defaults to the limit.

---

## 16. Taints, tolerations and affinity

> **Taints repel pods. Tolerations allow pods to land on tainted nodes. Affinity attracts pods to specific nodes.**

```bash
# Label & taint a GPU node
kubectl label node gpu-node-01 accelerator=nvidia-rtx3050
kubectl taint node gpu-node-01 nvidia.com/gpu=present:NoSchedule
```

```yaml
# Pod side
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: accelerator
              operator: In
              values: ["nvidia-rtx3050"]
```

**Scheduling decision (slide)**

1. Pod submitted. Does the node have a matching **taint**?
2. **No taint** → any pod may schedule there.
3. **Has taint** → the pod must have a **matching toleration**.
4. Then **affinity rules** apply (optional: prefer/require specific node labels).

**Result:** only GPU-requesting pods with the right toleration land on GPU nodes; everything else is filtered out first.

**Best practice (slide):** taint **all** GPU nodes; every GPU workload must tolerate the taint explicitly.

**Memory aid:** taint = "keep out unless you opt in"; toleration = "I opt in"; affinity = "I *want* this kind of node". A toleration allows a node, it doesn't force it. Affinity does the attracting.

> ⚠️ **Clarifications**
> - The slide says the taint "prevents CPU workloads from accidentally consuming GPU quota". More precisely, the real risk is **CPU-only pods occupying CPU/RAM/disk on expensive GPU nodes** and starving GPU pods. GPU *quota* is only charged to pods that request `nvidia.com/gpu`.
> - **Do not taint your only node** in a single-node lab cluster: system pods and non-GPU pods without tolerations would become unschedulable. Taint only dedicated GPU nodes in multi-node clusters. Many managed clusters and the GPU Operator can apply/handle this for you.
> - The slide's `nodeAffinity: {required: {...}}` is shorthand. The real field is `requiredDuringSchedulingIgnoredDuringExecution` (as above).

---

## 17. Namespaces, RBAC and resource quotas

| Object | Purpose |
|---|---|
| **Namespace** | Logical partition of cluster resources. Convention: **one namespace per team or project** |
| **ResourceQuota** | Limits **total** CPU/memory/GPU across all pods in a namespace; enforces fair-share |
| **LimitRange** | Sets **default and max** resource requests per pod/container; prevents runaway single pods |
| **RBAC Role / ClusterRole** | Controls **who** can create/delete pods, secrets, PVCs within a namespace (bound with RoleBinding) |

Quota details were covered in Week 2 Session 4 (rejected at admission; running pods not evicted; ceiling, not a guarantee).

---

## 18. Multi-tenant cluster end to end

```
One Kubernetes Cluster
├── namespace: team-alpha
│     ResourceQuota: 2 GPUs | RoleBinding: alpha-devs
│     Pod (1 GPU)   Pod (1 GPU)
└── namespace: team-beta
      ResourceQuota: 2 GPUs | RoleBinding: beta-devs
      Pod (1 GPU)   [quota headroom: 1 GPU free]
```

- **team-beta cannot exceed its 2-GPU quota even if team-alpha's GPUs are idle.** Quotas are **per-namespace**, not pooled cluster-wide by default.
- Quotas protect fairness, but can leave capacity idle. Pairing them with priority classes, preemption or elastic-quota tools is the usual remedy.

---

## 19. Lab 1: the tiny inference service

Today's hands-on completed **Lab 1 only**. The Week 3 folder has **five labs**. Each lab folder has a `Dockerfile`, YAML manifest(s) and an `app.py`, plus a README with verified commands (tested on an RTX 3050 4 GB). **Demo 2 (the GPU pod) needs a GPU-enabled cluster**. Demos 1, 3 and 4 run on the Docker Desktop cluster.

### 19.1 What the lab teaches

- **Containerizing a model server.**
- The `/health` endpoint returns **503 until the model finishes loading**. This is exactly why Kubernetes **readiness/liveness probes** exist: *never route traffic to a pod that isn't actually ready.*
- Later labs: GPU scheduling (node labels, taints/tolerations, `nvidia.com/gpu`), PyTorch + storage access modes (ReadWriteOnce vs ReadWriteMany), ResourceQuota/LimitRange for multi-tenancy, what the **NVIDIA GPU Operator** automates.

### 19.2 Environment

- Docker Desktop with **WSL2 backend**, **Kubernetes enabled** (Settings → Kubernetes). Cluster provisioning options include **kind** (multi-node capable) and **kubeadm** (single node). The demo used a **kind** cluster with **one node**, and the node count can be raised if you want multiple nodes.
- Verify the setup:

```bash
docker version
kubectl config current-context     # should be docker-desktop
kubectl get nodes
```

### 19.3 `app.py` walkthrough (FastAPI inference service)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness/liveness probe target. **503 `not ready`** until model is loaded, then **200** `{"status":"ok","device":...}` |
| `POST /predict` | Takes a **16-float feature vector**, returns softmax probabilities + `latency_ms` |
| `GET /metrics` | Prometheus-format text (`inference_requests_total`, `inference_errors_total`, `inference_latency_ms_avg`) |

**Key design points**

- **Model:** `TinyMLP` (16 → 64 → 64 → 3). Tiny so the container starts fast.
- **Device selection:** `device = "cuda" if torch.cuda.is_available() else "cpu"`. In this lab the deployment requests **no GPU**, so it runs on **CPU** even though the laptop has one.
- **Lifespan hook:** loads the model, does a **warm-up forward pass** (forces lazy CUDA init), then sets `metrics.ready = True`. Traffic isn't served until then.
- **Shutdown:** prints "draining".
- **Metrics:** simple in-process counters (use `prometheus_client` for production).
- **Local run:** `uvicorn` on `PORT` (default 8000). In-cluster, uvicorn is the container `CMD`.

```bash
# example request
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6]}'
```

> ⚠️ **Practical notes on `app.py`**
> - `Field(min_length=16, max_length=16)` on a list requires **Pydantic v2** (v1 used `min_items`/`max_items`).
> - Metrics are **per-replica and in-memory**. With 2 replicas, each pod reports its own counters and they reset on restart.
> - `print("[shutdown] draining")` doesn't by itself drain anything. Real graceful shutdown needs the Service to stop sending traffic (readiness flip / `preStop` hook) and uvicorn to finish in-flight requests within `terminationGracePeriodSeconds`.
> - The transcript's "health returns 53" is a speech-to-text error for **503**.

### 19.4 The deploy script (as described in the lecture)

A PowerShell script (works from WSL too) that automates the whole path:

1. Confirms the `kubectl` context is **docker-desktop**.
2. **Builds the image** `tiny-inference:local` from the Dockerfile.
3. **Smoke-tests it locally on CPU:** `docker run --name tiny -p 8000:8000 tiny-inference:local`, polls `/health` (with short sleeps) and errors out if it isn't ready within ~30 s ("container did not become ready within 30 seconds"). Output showed `status ok, device cpu`.
4. **Deploys to Kubernetes** (`kubectl apply`), waits for the rollout to succeed.
5. Lists pods (`kubectl get pods -l app=tiny-inference`) and prints how to test the Service from another terminal.

**Why script it:** repeatable, shows the real workflow people use, and avoids typing each command during teaching.

> ⚠️ The transcript garbles the polling loop as "mini cube" and "2 seconds to 20 seconds". It's a simple wait-until-healthy loop with a timeout, not Minikube. Check the actual script for the interval and timeout.

### 19.5 Typical Deployment shape for the lab

*(Reconstructed from the lecture's description: 2 replicas, container port 8000, health probe, CPU/memory requests, 1 CPU limit, no GPU.)*

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tiny-inference
spec:
  replicas: 2
  selector:
    matchLabels: { app: tiny-inference }
  template:
    metadata:
      labels: { app: tiny-inference }
    spec:
      containers:
        - name: tiny-inference
          image: tiny-inference:local
          imagePullPolicy: IfNotPresent      # local image, don't try a registry
          ports:
            - containerPort: 8000
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 2
            periodSeconds: 5
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 15
            periodSeconds: 10
          resources:
            requests: { cpu: "250m", memory: "256Mi" }
            limits:   { cpu: "1",    memory: "512Mi" }
---
apiVersion: v1
kind: Service
metadata:
  name: tiny-inference
spec:
  selector: { app: tiny-inference }
  ports:
    - port: 80
      targetPort: 8000
```

**Probe semantics**

| Probe | On failure | Use |
|---|---|---|
| **readinessProbe** | Pod is **removed from Service endpoints** (no traffic), not restarted | "Model loaded, safe to serve" |
| **livenessProbe** | Container is **restarted** | "Process is wedged" |
| **startupProbe** | Delays liveness/readiness until startup completes | Slow model loads (important for big models) |

> ⚠️ Reusing a `/health` that returns 503 during load as a **liveness** probe can kill a slow-starting container before it ever finishes loading. For real models, add a **startupProbe** or a generous `initialDelaySeconds`, or give liveness a separate cheaper endpoint.
> If you build the image with Docker Desktop's **kind** option, confirm the image is visible to the cluster nodes (Docker Desktop normally shares its image store, plain kind needs `kind load docker-image`). Use `imagePullPolicy: IfNotPresent`/`Never` for local tags.

### 19.6 Demo observations

- **Docker Desktop UI:** Kubernetes view showed **1 node, 2 pods** of `tiny-inference` (age ~31 h, restarts 0, readiness gate OK).
- **Pod status:** `/health` returned `ok` with `device: cpu`.
- **Builds tab:** shows build history, cache hits, per-step timing and dependencies. Rebuilding showed cached steps (a rebuild a minute earlier completed in about 2 s for the cached steps). Useful for understanding layer caching from Week 2.
- **Scaling idea:** increase nodes in Docker Desktop (kind) to spread the two pods across nodes.

### 19.7 Useful commands to reproduce

```bash
kubectl config current-context
kubectl get nodes
kubectl apply -f tiny-inference-deployment.yaml
kubectl get pods -l app=tiny-inference -o wide
kubectl describe pod <pod>                  # Events: why Pending / ImagePullBackOff
kubectl logs deploy/tiny-inference
kubectl port-forward svc/tiny-inference 8000:80
curl localhost:8000/health
kubectl rollout status deployment/tiny-inference
kubectl rollout undo deployment/tiny-inference
```

---

## 20. Glossary and session summary

| Term | Definition |
|---|---|
| **Node** | Physical/virtual machine running kubelet |
| **Pod** | Smallest deployable unit; one or more co-located containers |
| **Control loop** | Observe → compare → reconcile, repeated forever |
| **Taint / Toleration** | Repel-by-default / opt-in-to-land-here pairing |
| **DRA** | Dynamic Resource Allocation: structured, composable device claims |
| **Namespace** | Logical partition of one cluster's resources |

**Session summary**

- Kubernetes orchestrates AI workloads at scale: **isolation, reproducibility, autoscaling**.
- **Control plane** (API Server, etcd, Scheduler, Controller Manager) + **data plane** (kubelet, kube-proxy, containerd).
- GPUs are exposed via the **device plugin** (`nvidia.com/gpu`) or **DRA**.
- **B200 MIG:** up to 7 independent GPU instances from one card.
- **Taints + tolerations** keep non-GPU workloads off GPU nodes.
- **Namespaces + ResourceQuota + RBAC** = foundation of multi-tenant governance.

---

## 21. Corrections and gotchas

| Topic | What was said | Correct / clarified |
|---|---|---|
| etcd role | "etcd watches API server" | Only the **API server** talks to etcd. Components watch the **API server**. |
| kubelet "polls" | Slide/lecture | It uses **watch** streams (long-lived), not simple polling. |
| DRA versions | "K8s 1.33 ships DRA v1beta2" and "DRA K8s 1.31+"; lecture says "1.32 I suppose" | Slides are inconsistent. To my recollection DRA structured parameters went beta around 1.32 and the API was later promoted (v1beta2 in 1.33; GA in 1.34). **Verify against the Kubernetes release notes for your version.** |
| NGC image tag | `25.04-py3` = "PyTorch 2.11 / CUDA 13.0" | The 25.04 NGC container predates that stack (PyTorch ~2.7, CUDA ~12.9). Check NGC release notes and choose a tag matching your target. |
| `runtimeClassName` | "If NVIDIA runtime isn't available we can go with the simple CPU runtime" | There's no automatic fallback. The pod errors out. Remove the field or define the RuntimeClass. |
| Taint purpose | "Prevents CPU workloads from consuming GPU quota" | It stops non-GPU pods from occupying CPU/RAM on GPU nodes. GPU quota is only charged to GPU-requesting pods. |
| Tainting a lab node | Slide commands tainted `gpu-node-01` | Never taint the **only** node of a single-node cluster. |
| Bind vs container start | Blurred in the lecture | filter → score → **bind** → kubelet pulls → device plugin allocates → container starts. |
| "Containers in a pod start/stop together" | Slide | Scheduled and deleted together; individual containers may restart independently. |
| Secrets | "base64-encoded" | That's **encoding, not encryption**. Enable encryption-at-rest and RBAC, and use external secret stores. |
| kube-proxy | "iptables/eBPF" | kube-proxy uses iptables/IPVS (nftables in newer versions). eBPF dataplanes come from CNIs such as Cilium. |
| MIG profile names | B200 `1g.23gb`…`7g.160gb` (this deck) vs `3g.90gb`/`2g.45gb` (Session 4 deck) | Inconsistent. Actual profiles come from `nvidia-smi mig -lgip` on your hardware. |
| Deployment + `restartPolicy: Never` | Slide uses `Never` for the smoke pod | Fine for a bare Pod/Job. Deployments require `Always`. |
| Health probe | Same `/health` for readiness and liveness | Add a **startupProbe** or separate liveness endpoint for slow loads. |
| Transcript | "53", "kura", "Nvidia SMI", "mini cube", "cube control" | 503, CUDA, `nvidia-smi`, a polling loop, `kubectl`. |
| Slide cross-reference | "see slide 20" | The taint explanation is the *Taints, Tolerations & Affinity* slide. |

---

## 22. Cheat sheet and review questions

### Cheat sheet

```text
Orchestration:  declare desired state -> controllers reconcile actual state
Control plane:  API server (only door) | etcd (state) | scheduler | controller-manager
Data plane:     kubelet | kube-proxy | containerd (+ NVIDIA runtime & device plugin)
apply flow:     kubectl -> API server (authn/validate/admission) -> etcd -> scheduler binds -> kubelet pulls/starts -> status back
Objects:        Pod < ReplicaSet < Deployment ; Service ; ConfigMap/Secret ; PVC
GPU request:    resources.limits."nvidia.com/gpu": "1" (integers; request == limit)
Fractions:      MIG (hardware) or time-slicing (soft) ; DRA for structured claims
Pending pod:    kubectl describe pod -> Events (Insufficient nvidia.com/gpu? taint? quota?)
Taint/tol:      taint repels, toleration permits, affinity attracts
Tenancy:        Namespace + ResourceQuota + LimitRange + RBAC
Probes:         readiness = traffic gate ; liveness = restart ; startup = slow-boot guard
Rollout:        kubectl set image ... ; rollout status ; rollout undo
```

### Review questions

1. Give three reasons a single Docker host stops being enough. What does "you were the scheduler" mean?
2. Define orchestration. What does "declarative desired state" mean?
3. Why was Kubernetes chosen over Compose, Swarm and Nomad for this course?
4. Name the control-plane and data-plane components and what each does.
5. Walk through the six steps of `kubectl apply`.
6. What is the control loop? Why is the scheduler "just another controller"?
7. Pod vs Deployment vs ReplicaSet vs Service: what does each do?
8. Why do pods need a Service in front of them?
9. How does a pod get a GPU? Trace from the pod spec to the container (device plugin, kubelet, runtime).
10. What is DRA and how does it differ from `nvidia.com/gpu: 1`?
11. A GPU pod stays `Pending`. Give the first diagnostic command and three likely causes.
12. Explain taint, toleration and node affinity. Why taint GPU nodes?
13. Why did the lab's `/health` return 503 during startup, and what do readiness and liveness probes do differently?
14. Why can team-beta not use team-alpha's idle GPUs in the example?
15. In Lab 1, why did the service run on CPU even though the laptop has a GPU?

### Practice tasks

- Enable Kubernetes in Docker Desktop. Run `kubectl get nodes` and `kubectl config current-context`.
- Build `tiny-inference:local`, deploy it with 2 replicas, and confirm both pods become Ready.
- `curl` `/health`, `/predict` and `/metrics` through `kubectl port-forward`.
- Delete one pod (`kubectl delete pod ...`) and watch the Deployment recreate it. Note the new pod IP.
- Change the image tag or an env var to trigger a rollout, then `kubectl rollout undo`.
- Add a `startupProbe` and simulate a slow model load (e.g. `time.sleep(20)` in lifespan) to see readiness gating.
- Write a ResourceQuota (2 GPUs) and a LimitRange for a `team-alpha` namespace.
- Write the GPU pod manifest from §15 (to run in the GPU-cluster lab) and predict what `kubectl describe pod` shows if the node is tainted but the pod has no toleration.
