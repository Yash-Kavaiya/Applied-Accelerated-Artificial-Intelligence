# Week 3 · Session 2: GPU Pod Scheduling Hands-On

**Course:** AI Systems Engineering (Containerized AI Systems track)
**Instructor:** Satyadhyan Chickerur, Ph.D., Director, Centre for AI Research, KLE Technological University
**Format:** mostly hands-on (Lab guide: *Kubernetes for ML Inference & GPU Workloads*, Demo 2)

---

## Table of Contents

1. [Goals and recap of Session 1](#1-goals-and-recap-of-session-1)
2. [The environment stack](#2-the-environment-stack-why-docker-desktop-kubernetes-cant-run-gpu-pods)
3. [Why minikube](#3-why-minikube)
4. [Contexts](#4-two-clusters-two-contexts)
5. [What the GPU pod needs (three ingredients)](#5-the-three-ingredients-of-a-gpu-pod)
6. [The lab files](#6-the-lab-files-in-02_gpu_pod)
7. [Running the demo](#7-running-the-demo)
8. [Why the nested-WSL2 workaround exists](#8-why-the-nested-wsl2-workaround-exists)
9. [What happened in the live demo](#9-what-happened-in-the-live-demo-and-what-each-error-means)
10. [Troubleshooting flow](#10-troubleshooting-flow)
11. [Lab questions with answers](#11-lab-questions-with-answers)
12. [Quick recap of Demo 1](#12-quick-recap-of-demo-1-inference-service)
13. [Corrections and gotchas](#13-corrections-and-gotchas)
14. [Cheat sheet and review questions](#14-cheat-sheet-and-review-questions)

---

## 1. Goals and recap of Session 1

**Session 1 introduced the GPU pod pattern conceptually:**

- a **nodeSelector** (pin to GPU-labelled nodes),
- a **toleration** (to land on tainted GPU nodes),
- a **resource limit** `nvidia.com/gpu` (the extended resource),
- plus the **pass-through** that lets the container see the driver/GPU.

**Gap from Session 1:** we wrote a GPU pod but never ran it on a real GPU, and never proved it was using one.

**Session 2 goals**

- Get a pod **scheduled onto a GPU node** and print `nvidia-smi` **from inside the pod**.
- Understand why Docker Desktop's built-in Kubernetes can't do it, and why **minikube** can.
- Understand context switching between two clusters.
- Learn to read the failure modes (Pending, immutable-field errors, missing libraries).

---

## 2. The environment stack: why Docker Desktop Kubernetes can't run GPU pods

```
Windows host   (real GPU + NVIDIA driver)
  └─ WSL2 (Ubuntu)
       └─ Docker Desktop VM / engine
            └─ containers  (docker run --gpus all works here)
                 └─ [Kubernetes node(s)]
                      └─ pod -> container (e.g. hello-world)
```

- The **real GPU and NVIDIA driver live on the Windows host.** WSL2 exposes the GPU to Linux.
- **Plain `docker run --gpus all`** works (Week 2).
- **Docker Desktop's built-in Kubernetes** has **no NVIDIA device plugin**, so it never advertises `nvidia.com/gpu`. The scheduler sees **0 GPUs**, so any pod requesting one stays **`Pending` forever**.

> Key idea: *GPU scheduling in Kubernetes needs the cluster to advertise the GPU as a resource.* A GPU being present on the machine is not enough.

---

## 3. Why minikube

- **minikube** is a tool that runs a **local, single-node Kubernetes cluster** (here, inside a Docker container) for learning and testing.
- In this setup it **supports GPU pass-through**, so the NVIDIA device plugin can advertise `nvidia.com/gpu: 1` and pods can be scheduled onto the GPU.
- Treat it as the "**second, GPU-aware cluster**" mentioned in the lab guide. Setup scripts are in the session materials.

> ⚠️ In the first part of the lecture "mini cube node" is introduced loosely ("I'll tell you what it is later"). It is **not** a Docker Desktop feature. It's a separate cluster you start yourself. Typical GPU start (check your setup scripts for the exact flags):
> ```bash
> minikube start --driver=docker --gpus all
> ```
> This requires the NVIDIA Container Toolkit to be configured for Docker.

---

## 4. Two clusters, two contexts

You now have **two Kubernetes contexts** on one machine:

| Context | Cluster | GPU scheduling? |
|---|---|---|
| `docker-desktop` | Docker Desktop's built-in Kubernetes (kind-style, 1 node) | **No** (no device plugin) |
| `minikube` | minikube cluster | **Yes** (with GPU support) |

```bash
kubectl config get-contexts            # list contexts, * marks the current one
kubectl config current-context
kubectl config use-context minikube    # switch to the GPU cluster
kubectl config use-context docker-desktop
kubectl get pods --context docker-desktop   # one-off, without switching
```

**Important behaviours seen in the demo**

- `kubectl` talks to **whichever context is current**. The *same namespace name* (`default`) exists in **both clusters** and they are completely separate.
- The **Docker Desktop dashboard only shows the `docker-desktop` cluster.** Pods in minikube **never appear there**.
- A `hello-gpu` pod from an earlier try was **Pending in the `docker-desktop` cluster** (no GPU resource there). A different `hello-gpu` ran fine in **minikube**. Deleting one does **not** delete the other.

> ⚠️ The lecture says "namespace" in many places where it means **context/cluster**. Namespaces are partitions *inside* a cluster. The distinction here is **which cluster** you are talking to.

---

## 5. The three ingredients of a GPU pod

| Ingredient | Field | Purpose |
|---|---|---|
| **Node selector** | `nodeSelector: accelerator: nvidia` | Pins the pod to nodes **labelled** as GPU workers |
| **Toleration** | `tolerations: [{key: nvidia.com/gpu, operator: Exists}]` | Lets the pod land on **tainted** GPU nodes, so only GPU-requesting pods go there |
| **Resource limit** | `resources.limits."nvidia.com/gpu": 1` | The **extended resource** advertised by the **NVIDIA device plugin**; the scheduler filters nodes on it |

**Label the node so the selector matches:**

```bash
kubectl label node minikube accelerator=nvidia --overwrite
```

- `--overwrite` lets the command **replace the label's value** if the label already exists (no error on re-run). Handy on a single-node lab cluster.
- Verify: `kubectl get node minikube --show-labels`
- The **label key/value must match** the pod's `nodeSelector` exactly (Session 1 used `accelerator=nvidia-rtx3050`; here it is `accelerator=nvidia`).

**Toleration semantics (as explained in the lecture)**

- `key` is the **taint key** the toleration applies to.
- **Empty key + `operator: Exists`** means "match **all** taints" (all keys and values).
- `Exists` with a key means "tolerate any value for that key".

> ⚠️ The lecture describes the *toleration* as "a very strict rule". It's the **taint** that is strict ("keep out"). A toleration is the pod's **opt-in**. An **empty-key `Exists`** toleration is actually the **most permissive** one: it tolerates *everything*. Fine for a lab, too broad for production.

---

## 6. The lab files in `02_gpu_pod`

| File | Purpose |
|---|---|
| `hello-gpu.yaml` | Tiny pod: prints "hello world from a GPU-scheduled Kubernetes pod" and (in the demo) the GPU name/memory |
| `gpu-smoke.yaml` | First attempt at the `nvidia-smi` pod. **Errors** in the nested-WSL2 setup |
| `gpu-smoke-working.yaml` | The **fixed** version, with driver libraries mounted into the pod |
| `README` / lab guide | Commands and expected output |

### 6.1 What the fixed manifest contains (reconstructed)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-smoke
spec:
  restartPolicy: Never
  nodeSelector:
    accelerator: nvidia
  tolerations:
    - operator: Exists            # empty key + Exists = tolerate all taints
  containers:
    - name: smoke
      image: nvidia/cuda:13.0.0-base-ubuntu24.04   # illustrative; use your lab image
      command: ["nvidia-smi"]
      securityContext:
        privileged: true          # workaround for nested virtualization
      resources:
        limits:
          nvidia.com/gpu: 1
      volumeMounts:
        - name: nvidia-driver
          mountPath: /usr/local/nvidia   # illustrative path
  volumes:
    - name: nvidia-driver
      hostPath:
        path: /path/to/driver/libs/on/minikube/node   # the node's own driver libs
```

> ⚠️ **This is a sketch.** The lecture says the working file uses a **`hostPath` volume + privileged mode** so the pod can see the driver. Exact paths come from your `gpu-smoke-working.yaml`. `privileged: true` and `hostPath` are **security-sensitive** and acceptable only in a throwaway lab.

---

## 7. Running the demo

```bash
# 1. Make sure you're on the GPU cluster
kubectl config use-context minikube
kubectl get nodes                       # should show minikube Ready
kubectl describe node minikube | grep -i nvidia   # expect nvidia.com/gpu: 1

# 2. Label the node so the nodeSelector matches
kubectl label node minikube accelerator=nvidia --overwrite

# 3. Apply the working manifest (name it explicitly)
kubectl apply -f gpu-smoke-working.yaml

# 4. Watch it, then read the logs
kubectl get pod gpu-smoke -w
kubectl logs gpu-smoke

# 5. Clean up
kubectl delete pod gpu-smoke
```

**Expected output** (from the guide): a real `nvidia-smi` table printed **from inside the pod**:

```
+-----------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01   Driver Version: 591.59   CUDA Version: 13.1    |
| 0  NVIDIA GeForce RTX 3050 ...  00000000:01:00.0  Off |        N/A    |
+-----------------------------------------------------------------------+
```

- The **Driver Version** (591.59) is the **Windows host driver**. The **NVIDIA-SMI** version is the WSL-side tool.
- `hello-gpu` showed the GPU model and memory (the lecture said an RTX 3050 **Ti**; the guide's table shows RTX 3050. Whichever your laptop has will appear).

**Fast one-liner flow from the lecture:**

```bash
kubectl apply -f hello-gpu.yaml
kubectl logs hello-gpu
kubectl delete pod hello-gpu
```

> A pod that ran to completion shows **`Completed`**, not `Running`. That is expected for one-shot commands (`restartPolicy: Never`). Logs remain available until you delete the pod.

---

## 8. Why the nested-WSL2 workaround exists

The guide explains the layering on a laptop:

```
Windows -> WSL2 -> Docker Desktop's VM -> minikube node (itself a Docker container) -> pod
```

- **GPU scheduling works correctly**: `kubectl describe node` genuinely shows `nvidia.com/gpu: 1`.
- But at this nesting depth, **containerd's automatic NVIDIA library injection does not fire** for Kubernetes-scheduled pods. The pod starts but `nvidia-smi`/driver libraries are missing.
- **Fix used:** **manually mount the node's own driver libraries** into the pod (plus privileged access).
- **On a real GPU cluster with the NVIDIA GPU Operator** (Demo 5) none of this manual plumbing is needed. The Operator installs the driver/toolkit/device plugin and the runtime injects libraries automatically.

**Two different problems, don't confuse them**

| Symptom | Layer | Meaning |
|---|---|---|
| Pod **Pending**, `Insufficient nvidia.com/gpu` | **Scheduling** | Cluster doesn't advertise a GPU (no device plugin / no free GPU) |
| Pod **Running/Error**, `nvidia-smi: not found` or driver errors | **Runtime / injection** | Scheduled fine, but the container can't see driver libs |

This matches the Week 2 *3-level probe*: driver → toolkit → framework, now with an extra Kubernetes layer above it.

---

## 9. What happened in the live demo (and what each error means)

| What was seen | Explanation |
|---|---|
| `gpu-smoke.yaml` "does not exist" | **Filename mismatch.** The file is named `gpu-smoke-working.yaml` (plus `gpu-smoke.yaml`). Check `ls` and the path/working directory. |
| Apply produced `pod "gpu-smoke" is invalid ... may not change the fields` | A pod named `gpu-smoke` **already existed** (from the broken manifest). **Pod specs are mostly immutable.** You can't edit most fields in place. **Delete and recreate:** `kubectl delete pod gpu-smoke && kubectl apply -f ...` |
| `kubectl logs` -> "pod not found" | Timing, or wrong context/namespace. Run `kubectl get pods` first and wait for the pod to exist. Always confirm the **context** is minikube. |
| Pod listed as `Completed` | Success for a run-once pod. `kubectl logs gpu-smoke` shows the output. |
| `hello-gpu` stuck **Pending** in Docker Desktop's dashboard | That's the **docker-desktop** cluster, which has no GPU device plugin. **Expected.** Delete it there with `kubectl --context docker-desktop delete pod hello-gpu`. |
| Hello-GPU "deleted from default namespace" but Pending one still visible | Deleted in **minikube's** `default`; the Pending one is in **docker-desktop's** `default`. Different clusters. |
| Output glitches on first runs | Logs requested **before** the container finished. Re-run `kubectl logs`, or use `kubectl get pod -w` first. |

**Lesson:** in the demo, much of the confusion came from **which cluster** each command ran against and **leftover pods** with the same name. Before every command: `kubectl config current-context` and `kubectl get pods`.

---

## 10. Troubleshooting flow

```text
Pod stuck Pending?
  kubectl describe pod <name>        -> Events at the bottom
    "Insufficient nvidia.com/gpu"    -> node has no free GPU / no device plugin advertised
    "didn't match Pod's node selector" -> label missing: kubectl label node ...
    "had untolerated taint"          -> add a toleration (or remove the taint)
  kubectl describe node <node> | grep -A8 -i "capacity\|allocatable"   # is nvidia.com/gpu listed?
  kubectl get pods -n kube-system | grep -i nvidia                      # device plugin running?

Pod Running/Completed but no GPU inside?
  kubectl logs <pod>                 -> "nvidia-smi: not found" / "Failed to initialize NVML"
  -> runtime/library injection problem (nested virtualization, missing runtime, missing mounts)
  -> on real clusters check GPU Operator / container toolkit / runtimeClass

Applied a change and got "may not change fields"?
  -> pods are immutable: kubectl delete pod <name>, then re-apply
```

---

## 11. Lab questions with answers

**Q1. What happens if you remove the `nodeSelector`: where does the scheduler try to place the pod?**
It considers **every node** that passes the other filters: it must advertise enough `nvidia.com/gpu` and the pod must tolerate the node's taints. On this single-node minikube cluster, it still lands on the same node. In a multi-node cluster, the selector is what restricts the pod to a **specific GPU type/pool** (for example H100 vs RTX nodes). Without it, the pod can go to any GPU node that fits. *(The GPU resource limit already keeps it off non-GPU nodes.)*

**Q2. What happens if you request `nvidia.com/gpu: 2` on a node that only has 1 GPU?**
The pod stays **`Pending`**. `kubectl describe pod` shows a `FailedScheduling` event such as `0/1 nodes are available: 1 Insufficient nvidia.com/gpu`. The scheduler won't split or fractionalize GPUs. The request is for **whole units** (unless MIG/time-slicing exposes more).

**Q3. Why do GPU nodes typically carry a taint, and what does it prevent?**
It keeps **non-GPU workloads** (web apps, logging agents, batch CPU jobs) from being scheduled onto **expensive GPU nodes**, where they would use CPU/RAM/disk and crowd out GPU jobs. Only pods that **explicitly tolerate** the taint (GPU workloads) can land there.

> Careful: the taint doesn't charge or protect "GPU quota". It protects the node's *general* capacity and keeps GPU nodes reserved for GPU work. Also, **never taint the only node** of a single-node cluster, or system pods can't schedule.

---

## 12. Quick recap of Demo 1: inference service

*(Guide values, which take precedence over my earlier reconstructed YAML.)*

```bash
cd 01_inference_service
docker build -t tiny-inference:local .
docker run -d --name test -p 8000:8000 tiny-inference:local
curl http://localhost:8000/health        # {"status":"ok","device":"cpu"}
docker rm -f test

kubectl apply -f tiny-inference-deployment.yaml     # 2 replicas, probes -> /health
kubectl rollout status deployment/tiny-inference
kubectl get pods -l app=tiny-inference

kubectl port-forward svc/tiny-inference-svc 8001:8000
# second terminal:
curl http://localhost:8001/predict -X POST -H "Content-Type: application/json" \
  -d '{"features":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}'
# {"probs":[0.165,0.302,0.533],"latency_ms":2.26}
```

- The **Service is `tiny-inference-svc`**, exposing port **8000** (so `port-forward svc/tiny-inference-svc 8001:8000`).
- `/health` returns **503 until the model loads**, then 200. This is what **readiness/liveness probes** use.
- It runs on **CPU** because the Deployment requests **no GPU**.
- Demos 1, 3 and 4 run on the Docker Desktop cluster. **Demo 2 needs minikube (GPU)**.

---

## 13. Corrections and gotchas

| Topic | What was said | Correct / clarified |
|---|---|---|
| Namespace vs context | "Docker desktop name space", "mini cube name space" | They're **clusters/contexts**. `default` is a namespace that exists in each cluster separately. |
| Docker Desktop dashboard | Implied it shows all pods | Shows **only the docker-desktop cluster**. minikube pods never appear. |
| Toleration "strict" | "Very strict rule, not everybody can come near the tainted GPU" | The **taint** is the strict part. An empty-key `Exists` toleration matches **all** taints (very permissive). |
| `--overwrite` | "Any other person would have been using it ... so everything becomes free" | It only **overwrites the label's value** if the label already exists. It doesn't free resources. |
| Pending cause | "Pod pending because it can't see the GPUs" | More precisely: **scheduler sees no `nvidia.com/gpu` resource** (no device plugin). Different from a runtime-visibility failure. |
| `may not change fields` | Shown as a mysterious error | Pods are **immutable**. Delete and recreate. |
| Wrong file / name | `gpu-smoke.yaml` vs `gpu-smoke-working.yaml` | Use the **working** file. Don't `apply` the broken one over an existing pod. |
| GPU name | "RTX 3050 Ti" (lecture) vs "RTX 3050" (guide) | Depends on the laptop. Not a conceptual difference. |
| Privileged + hostPath | Presented as the fix | Lab-only workaround for **nested virtualization**. Not for production. On real clusters the **GPU Operator** handles it. |
| `runtimeClassName: nvidia` | Used in Session 1's manifest | May not exist in minikube/nested setups. If the pod fails with a RuntimeClass error, remove it (or create the RuntimeClass). |
| "Mini cube node" | Introduced as something vague | A separate local cluster tool. Start it with GPU support before the demo. |
| Transcript | "cube control", "kura", "mini cube" | `kubectl`, CUDA, minikube. |

---

## 14. Cheat sheet and review questions

### Cheat sheet

```text
Docker Desktop K8s : NO device plugin -> nvidia.com/gpu never advertised -> GPU pods Pending
minikube (GPU)     : advertises nvidia.com/gpu: 1 -> GPU pods schedule
Contexts           : kubectl config get-contexts | use-context minikube | current-context
GPU pod needs      : nodeSelector (label) + toleration (taint) + limits.nvidia.com/gpu
Label node         : kubectl label node minikube accelerator=nvidia --overwrite
Run                : kubectl apply -f gpu-smoke-working.yaml ; kubectl logs gpu-smoke
Pending?           : kubectl describe pod  -> Events (Insufficient nvidia.com/gpu / selector / taint)
Immutable fields   : kubectl delete pod X, then apply again
Nested virt gotcha : scheduling OK but no driver libs in pod -> mount node libs (lab only)
Real clusters      : NVIDIA GPU Operator automates driver/toolkit/plugin/injection
```

### Review questions

1. Why does a GPU pod stay Pending on Docker Desktop's built-in Kubernetes?
2. Where does the real GPU driver live in the Windows → WSL2 → Docker → pod stack?
3. What is minikube, and why is it used for Demo 2?
4. Name the three ingredients of a GPU pod spec and what each does.
5. What does `kubectl label node minikube accelerator=nvidia --overwrite` do, and why must it match the `nodeSelector`?
6. What does an empty-key `Exists` toleration match? Is it appropriate for production?
7. Explain the difference between a *scheduling* failure and a *runtime library injection* failure.
8. Why did the apply say "may not change the fields"? How do you fix it?
9. Why did `hello-gpu` stay Pending in one place and run in another?
10. What happens if you request `nvidia.com/gpu: 2` on a one-GPU node?
11. Why do GPU nodes carry a taint? What shouldn't you do on a single-node cluster?
12. What does the GPU Operator remove the need for?

### Practice tasks

- Run `kubectl config get-contexts` and note the current context before every command.
- Start minikube with GPU support, then confirm `kubectl describe node minikube | grep nvidia` shows `nvidia.com/gpu: 1`.
- Apply `hello-gpu.yaml` on **docker-desktop** and observe `Pending` + the `Insufficient nvidia.com/gpu` event. Then apply it on **minikube** and observe it complete.
- Remove the `nodeSelector`, re-apply, and compare `kubectl describe pod`.
- Change the limit to `nvidia.com/gpu: 2` and record the scheduler event.
- Taint the minikube node (`nvidia.com/gpu=present:NoSchedule`) and test one pod with and without a toleration. Remove the taint afterwards.
- Edit `gpu-smoke-working.yaml` (e.g. command or image), apply again, and handle the immutable-field error by deleting the pod first.
