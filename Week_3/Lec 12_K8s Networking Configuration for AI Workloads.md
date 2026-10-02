# Week 3 · Session 3: K8s Networking & Configuration for AI Workloads

**Course:** AI Systems Engineering (Containerized AI Systems track)
**Instructor:** Satyadhyan Chickerur, Ph.D., Director, Centre for AI Research, KLE Technological University
**Topics:** Services, ConfigMaps, Secrets, Istio, Gateway API, storage, headless DNS
**Format:** concepts and manifest walkthroughs; hands-on execution continues next session

---

## Table of Contents

1. [Goals and context](#1-goals-and-context)
2. [Kubernetes Service types](#2-kubernetes-service-types)
3. [Gateway API (the modern Ingress)](#3-gateway-api-the-modern-ingress)
4. [ConfigMaps and Secrets for training jobs](#4-configmaps-and-secrets-for-training-jobs)
5. [Istio service mesh for inference traffic](#5-istio-service-mesh-for-inference-traffic)
6. [Storage for AI workloads](#6-storage-for-ai-workloads-pvcs-and-storageclasses)
7. [Lab walkthrough: ConfigMap](#7-lab-configmap-train-config)
8. [Lab walkthrough: training pod](#8-lab-training-pod-consuming-configmap--secret)
9. [Lab walkthrough: headless Service](#9-lab-headless-service-training-worker)
10. [Lab walkthrough: StatefulSet](#10-lab-statefulset-worker)
11. [Lab walkthrough: LoadBalancer Service](#11-lab-loadbalancer-service-tiny-inference-lb)
12. [Lab walkthrough: GatewayClass, Gateway, HTTPRoute](#12-lab-gatewayclass-gateway-httproute)
13. [Lab order and command reference](#13-lab-order-and-command-reference)
14. [Corrections and gotchas](#14-corrections-and-gotchas)
15. [Cheat sheet and review questions](#15-cheat-sheet-and-review-questions)

---

## 1. Goals and context

By the end of this session you should be able to:

- Choose the right **Service type** (ClusterIP, NodePort, LoadBalancer, Headless) for a workload.
- Explain why **headless Services + StatefulSets** are the backbone of distributed training.
- **Externalize configuration** (ConfigMap) and **credentials** (Secret) so training images are never rebuilt for a hyperparameter change or token rotation.
- Explain what **Gateway API** is and how it replaces Ingress.
- Describe what a **service mesh (Istio)** gives inference traffic: mTLS, canary splits, observability, rate limiting.
- Pick an access mode and StorageClass for AI data (RWO / RWX / ROX).

**Where this fits:** Session 1 covered core objects. Session 2 ran a GPU pod. This session covers how pods **talk to each other and the outside world**, and how they get their **configuration**.

---

## 2. Kubernetes Service types

A **Service** gives a stable network identity to a changing set of pods (pod IPs change on every restart or reschedule).

| Type | What it is | Typical AI use |
|---|---|---|
| **ClusterIP** | **Default.** Stable **virtual IP inside the cluster only** | Pod-to-pod traffic (e.g. API → model server) |
| **NodePort** | Exposes the Service on **every node's IP at a static port** (range **30000-32767**) | **Dev/debug only** |
| **LoadBalancer** | Provisions an **external load balancer** (cloud ALB/NLB, MetalLB on bare metal) | **Production inference endpoints** |
| **Headless** (`clusterIP: None`) | **No virtual IP.** DNS returns **each pod's IP directly** | **StatefulSets and distributed training** (NCCL rendezvous needs per-pod DNS) |

```
ClusterIP:     client(in cluster) -> virtual IP -> one of N pods (kube-proxy picks)
NodePort:      client -> <any node IP>:3xxxx -> Service -> pod
LoadBalancer:  client -> external IP -> NodePort/Service -> pod
Headless:      DNS lookup of the service name -> [pod-0 IP, pod-1 IP, pod-2 IP]
```

> **Headless Services are the backbone of distributed training:** each training pod gets a stable DNS entry (e.g. `worker-0.training-worker`) that NCCL initialization depends on.

> ⚠️ The slide's example names `pod-0.svc`, `pod-1.svc` are generic. The real DNS format is `<pod-name>.<service-name>.<namespace>.svc.cluster.local` (see §10). LoadBalancer and NodePort are **layered**: a LoadBalancer Service also allocates a NodePort and ClusterIP underneath.

---

## 3. Gateway API (the modern Ingress)

**Gateway API** is the standard HTTP/gRPC routing layer that **supersedes Ingress**. Its main idea is **separation of roles**:

| Resource | Owned by | Purpose |
|---|---|---|
| **GatewayClass** | Platform/infra team | Picks the **controller implementation** (here, Istio). One per controller, created once |
| **Gateway** | Platform/cluster operator | A concrete **listener** (port/protocol) backed by a proxy |
| **HTTPRoute** | **ML/app team**, in their own namespace | **Path/host-based routing rules** attached to a Gateway |

```
GatewayClass ──► Gateway (team-alpha, :8082) ──► HTTPRoute (/v1/completions) ──► Service (llm-service)
(controller)        (listener + Envoy)             (owned by the ML team)
```

**Why it beats classic Ingress**

- **Decouples operators from app teams:** the ML team creates an HTTPRoute in **its own namespace** with **no cluster-admin access**.
- Multiple teams' HTTPRoutes can attach to **one Gateway**, each matching its own paths.
- Richer, standardized features: traffic splitting, header matching, gRPC, cross-namespace controls.

**Slide snippet (GatewayClass + HTTPRoute)**

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: inference-gateway
spec:
  controllerName: istio.io/gateway-controller
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-inference-route
  namespace: team-alpha
spec:
  parentRefs:
  - name: inference-gateway
  rules:
  - matches:
    # ... (path match + backendRefs, see §12)
```

> ⚠️ **Lecture wording:** the instructor described Ingress as "a standard protocol for connecting two points". More precisely, **Ingress is a Kubernetes API object** for routing external HTTP(S) traffic to Services. It's older, limited and largely **frozen** in favour of Gateway API. It still works in many clusters, so "superseded" is accurate while "removed" is not.

---

## 4. ConfigMaps and Secrets for training jobs

### 4.1 The idea

```
 training image (code + PyTorch)   +   ConfigMap (batch_size=128, ...)
                         \              /
                       Pod starts a container
                                ↓
                  python train.py --batch-size 128
```

> **Same training image + different ConfigMap values = different training run.**

- Different people/runs need different **batch size, learning rate, max steps, model architecture**.
- If those are baked into the image, every hyperparameter change means a **rebuild**. Don't do that.
- **Separate the image (code + dependencies) from configuration.**

### 4.2 ConfigMap (non-sensitive config)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: train-config
data:
  batch_size: "128"
  learning_rate: "3e-4"
  max_steps: "50000"
  warmup_steps: "2000"
  model_arch: llama3-8b
```

- **All values are strings** (note the quotes).
- Stored as **plain key/value pairs, visible in cleartext**. **Never put secrets in a ConfigMap.**
- Any pod **in the same namespace** can reference it by name.
- Editing a ConfigMap **updates the object immediately**, but a **running pod does not pick up env-var changes**. **Restart the pod.**

### 4.3 Secret (sensitive config)

```bash
# Create a secret from literals
kubectl create secret generic ml-tokens \
  --from-literal=WANDB_API_KEY=... \
  --from-literal=HF_TOKEN=...
```

- Holds things like the **Weights & Biases API key** and the **Hugging Face token**.
- Keeps credentials **out of the image and out of Git**, so you never hand tokens to others by shipping a container.
- **Production tip (slide):** use the **External Secrets Operator (ESO)** to sync **AWS Secrets Manager / Vault → K8s Secret** automatically. **Never commit secrets to Git.**

### 4.4 Consuming both in a pod

```yaml
envFrom:
  - configMapRef: { name: train-config }
  - secretRef:    { name: ml-tokens }
```

- `envFrom` **merges every key** from both sources into the container's **environment variables**, with no per-key wiring.
- Tuning a run or rotating a token **never requires a rebuild**.

> ⚠️ **Important correction (lecture):** the instructor explained the Secret as holding "the weights and biases" and linked it to model-weight privacy (federated learning, data reconstruction). On the slide, **"W&B" means Weights & Biases, the experiment-tracking service**, so the Secret holds its **API key**, not neural-network weights/biases. Model weights are **checkpoint/artifact data** that belong in storage (PVC/object store), not in Secrets.
>
> ⚠️ **Secrets are base64-encoded, not encrypted** by default. Protect them with RBAC, **encryption at rest**, and external secret managers.
>
> ⚠️ **Update behaviour:** env vars from `envFrom` are set **at container start** and never change. If you instead **mount** a ConfigMap/Secret as a **volume**, files update eventually, but your app must re-read them.

---

## 5. Istio service mesh for inference traffic

A **service mesh** adds a proxy layer (Istio uses **Envoy** sidecars or ambient mode) handling traffic features **without changing app code**.

| Feature | What it does for inference |
|---|---|
| **mTLS** | Automatic **mutual TLS** between all pods: a **zero-trust** network inside the cluster |
| **Traffic splitting** | **Canary rollouts:** route e.g. **10% to the new model, 90% to stable**, then ramp up gradually |
| **Observability** | Auto-injected **metrics (Prometheus), traces (Jaeger), access logs** |
| **Rate limiting** | Envoy-level limits to protect inference endpoints from request bursts |

**Canary rollout idea (from the lecture)**

- You have a stable model serving traffic and an improved candidate.
- Don't switch all traffic at once. Send **10%** to the new one, watch quality/latency/errors, then raise to 25%, 50%, **100%**.
- Roll back by setting the weight back to 0.

**Slide snippet: `VirtualService` (10% canary)**

```yaml
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: llm-inference-vs
spec:
  hosts: [llm-service]            # reconstructed
  http:
  - route:
    - destination: { host: llm-service, subset: stable }   # reconstructed
      weight: 90
    - destination: { host: llm-service, subset: canary }
      weight: 10
```

*(The slide's snippet was cut off after `route:`. The weights and subsets above are reconstructed. Subsets also need a `DestinationRule` defining `stable` and `canary` by pod labels.)*

> ⚠️ Istio is **optional infrastructure**: it adds latency, memory per pod, and operational complexity. Gateway API's `HTTPRoute` can also do **weighted backendRefs** without Istio-specific objects. The lecture said a demo "LLM inference" setup would be shown, but this session only previewed it.

---

## 6. Storage for AI workloads: PVCs and StorageClasses

| Access mode | Meaning | Use |
|---|---|---|
| **ReadWriteOnce (RWO)** | One **node** read-write | Local NVMe/SSD scratch; fast, **not shareable across nodes** |
| **ReadWriteMany (RWX)** | Many nodes read-write at once | **Shared datasets** during distributed training |
| **ReadOnlyMany (ROX)** | Many nodes **read-only** | **Shared model weights** served to many inference pods |

**StorageClass examples for AI clusters (slide)**

| StorageClass | Backing | Use |
|---|---|---|
| `local-nvme` | local provisioner | Ultra-fast **scratch space for dataset staging** |
| `efs-sc` / `azurefile-csi` | Managed NFS | **Shared dataset volumes** across training pods |
| `weka-sc` / `vast-csi` | High-performance parallel FS | **1M+ IOPS** for large-scale training |

> ⚠️ RWO is an **access mode**, not "the default for NVMe". NVMe/local volumes are *typically* RWO because the disk is attached to one node. Note that RWO limits access per **node**, so multiple pods on the same node can still share it (use **ReadWriteOncePod** for strictly one pod). Also, RWX/ROX support depends on the **storage backend**: block volumes (EBS, local NVMe) generally don't support RWX.

---

## 7. Lab: ConfigMap `train-config`

*(File: `04_configmap.yaml`)*

- Stores non-sensitive hyperparameters **outside** the image.
- Tuning = **edit the ConfigMap**, not rebuild the image.
- Consumed via `envFrom → configMapRef`.

```bash
kubectl apply -f 04_configmap.yaml
kubectl get configmap train-config -o yaml
```

**Note:** pair with a Secret (`05_training_pod/create_secret.sh`) for anything sensitive. WANDB/HF tokens **never** belong in a ConfigMap.

---

## 8. Lab: training pod consuming ConfigMap + Secret

*(File: `05_training_pod.yaml`; script: `create_secret.sh`)*

- A **one-shot training container** pulling its whole environment from **two sources**: ConfigMap and Secret.
- `restartPolicy: Never` models a **finite training job**, not a long-lived service.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
spec:
  restartPolicy: Never
  containers:
    - name: trainer
      image: busybox:1.36
      command: ["sh","-c","sleep 3600"]
      envFrom:
        - configMapRef:
            name: train-config
        - secretRef:               # not visible in the slide extract; required per the slide text
            name: ml-tokens
```

*(`busybox` + `sleep 3600` is a **stand-in** for a real trainer, just so you can inspect the environment.)*

```bash
# 1. Secret FIRST (the pod references it)
./create_secret.sh                       # WSL/bash; use the PowerShell variant on Windows
# 2. Apply the pod
kubectl apply -f 05_training_pod.yaml
# 3. Verify the environment inside the container
kubectl exec -it training-pod -- env | grep -E "batch_size|WANDB|HF_TOKEN"
# 4. Clean up
kubectl delete -f 05_training_pod.yaml
```

> ⚠️ **Order matters:** if `ml-tokens` doesn't exist, the pod gets stuck in **`CreateContainerConfigError`**. The lecture described it as "the pod keeps checking", but it's an actual error state. Run `kubectl describe pod training-pod` to see `secret "ml-tokens" not found`.
> ⚠️ The slide's visible manifest only shows `configMapRef`. The `secretRef` is described in the text, and the slide extract was truncated.
> ⚠️ `grep -E "batch_size|WANDB|HF_TOKEN"` is **case-sensitive**: ConfigMap keys here are lowercase (`batch_size`), Secret keys uppercase. **Don't paste real tokens into screen-shares or chat.**

---

## 9. Lab: headless Service `training-worker`

*(File: `06_headless_svc.yaml`)*

```yaml
apiVersion: v1
kind: Service
metadata:
  name: training-worker
spec:
  clusterIP: None            # headless: no virtual IP at all
  selector:
    app: training-worker
  ports:
    - name: nccl
      port: 29500
      targetPort: 29500
```

**How it works**

- `clusterIP: None` → **no virtual IP** is allocated.
- A DNS lookup of the Service name returns **every pod's IP directly**, not one load-balanced address.
- Each pod additionally gets its own **stable hostname record**: `<pod-name>.training-worker`.
- **Why training needs it:** NCCL rendezvous requires every worker to address every other worker by a **stable hostname**. A normal ClusterIP would resolve all workers to **one virtual IP**, so peers can't target a specific rank.
- Port **29500** is the conventional **rendezvous/master port** (PyTorch `torchrun` default).

```bash
kubectl apply -f 06_headless_svc.yaml
```

> ⚠️ The slide says a ClusterIP Service would make "NCCL hang at initialization". That's a plausible failure mode but stated too strongly. The core issue is that rank discovery needs **individually addressable peers**. Also: **per-pod DNS records are created only for pods that are Ready**, unless the Service sets `publishNotReadyAddresses: true`, which distributed-training setups often enable so workers can find each other *before* they pass readiness.
> The lecture noted a typo in the diagram's arrow placement. The Service points at all three workers.

---

## 10. Lab: StatefulSet `worker`

*(File: `07_statefulset.yaml`)*

**Why a StatefulSet (vs Deployment)**

| | Deployment | StatefulSet |
|---|---|---|
| Pod names | random hash (`web-7d9f...`) | **ordinal** (`worker-0`, `worker-1`, `worker-2`) |
| Identity after reschedule | new name, new IP | **same name, same ordinal, same DNS name** |
| Startup | all at once | **ordered** (0, then 1, then 2) by default |
| Storage | shared | can have **per-pod PVCs** (`volumeClaimTemplates`) |

**Reconstructed manifest** (the slide's code was cut off):

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: worker
spec:
  serviceName: training-worker        # binds to the headless Service
  replicas: 3
  selector:
    matchLabels:
      app: training-worker
  template:
    metadata:
      labels:
        app: training-worker
    spec:
      containers:
        - name: worker
          image: busybox:1.36
          command: ["sh","-c","sleep 3600"]   # slide shows sleep; exact value not visible
```

**DNS of each pod**

```
worker-<N>.training-worker.<namespace>.svc.cluster.local
```

```bash
kubectl apply -f 07_statefulset.yaml
kubectl get pods -l app=training-worker
kubectl exec -it worker-0 -- nslookup worker-1.training-worker      # verify peer discovery
kubectl delete -f 07_statefulset.yaml
```

**Key points**

- **Pod identity survives rescheduling**: a replacement `worker-1` keeps its name, ordinal and DNS entry. (Its **IP address may change**, which is exactly why you use the name.)
- **Ordinal startup order** (0, 1, 2) matters for rendezvous protocols expecting a **rank-0 coordinator**. Rank can be derived from the ordinal suffix.
- The Service's `selector` label (`app: training-worker`) must match the pod template labels.

> ⚠️ For real GPU training you'd add `resources.limits.nvidia.com/gpu`, a toleration/nodeSelector (Session 2), a PVC for checkpoints, and consider `podManagementPolicy: Parallel` (so workers don't wait on each other to start). **Gang scheduling** (Volcano/Kueue) is still needed to avoid stranded GPUs (Week 2 Session 4). A StatefulSet alone doesn't guarantee all-or-nothing placement.

---

## 11. Lab: LoadBalancer Service `tiny-inference-lb`

*(File: `08_loadbalancer_svc.yaml`)*

```yaml
apiVersion: v1
kind: Service
metadata:
  name: tiny-inference-lb
spec:
  type: LoadBalancer
  selector:
    app: tiny-inference
  ports:
    - port: 8080        # external/service port
      targetPort: 8000  # container port
```

```
Client --> :8080 Service (tiny-inference-lb) --> pods tiny-inference :8000
```

**Points**

- `type: LoadBalancer` requests an **externally reachable IP** from the cluster's cloud or local LB provider.
- This is the **production pattern for a real inference endpoint**, as opposed to ClusterIP (internal) or NodePort (dev).
- **Prerequisite:** the **`tiny-inference` Deployment** from Lab 1 must already be running. Otherwise the selector has **nothing to route to**.
- **Docker Desktop's local Kubernetes** publishes LoadBalancer Services on **`localhost`** automatically, so no cloud LB controller is needed.
- On a **real cluster** it provisions an actual external IP via the cloud provider, or **MetalLB** on bare metal.

```bash
kubectl apply -f ../01_inference_service/tiny-inference-deployment.yaml   # prerequisite
kubectl apply -f 08_loadbalancer_svc.yaml
kubectl get svc tiny-inference-lb
curl http://localhost:8080/health
kubectl delete -f 08_loadbalancer_svc.yaml
```

> ⚠️ This lab Service is named `tiny-inference-lb` (port 8080). Lab 1's ClusterIP Service was `tiny-inference-svc` (port 8000, used with `port-forward`). On `minikube`, a LoadBalancer stays `<pending>` until you run `minikube tunnel`. This is **one more reason to confirm your kubectl context** before testing.

---

## 12. Lab: GatewayClass, Gateway, HTTPRoute

Apply in this order: **GatewayClass → Gateway → HTTPRoute** (each depends on the previous).

### 12.1 GatewayClass `inference-gateway` (`gatewayclass.yaml`)

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: inference-gateway
spec:
  controllerName: istio.io/gateway-controller
```

- **One GatewayClass per cluster controller**, created once by the **platform team**. App teams never touch it.
- `controllerName` **must match** what the mesh (Istio) actually registers.
- Many Gateways (team-alpha, team-beta, future teams) reference it by `gatewayClassName`.

```bash
kubectl apply -f gatewayclass.yaml
kubectl get gatewayclass inference-gateway
kubectl get gatewayclasses -o wide       # confirm the controller Istio registered
```

### 12.2 Gateway `inference-gateway` (`gateway.yaml`)

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
  namespace: team-alpha
spec:
  gatewayClassName: inference-gateway
  listeners:
    - name: http
      port: 8082
      protocol: HTTP
      allowedRoutes:
        namespaces:
          from: Same          # only HTTPRoutes in team-alpha may attach
```

- Once created, **Istio provisions a real Envoy proxy plus a LoadBalancer Service**.
- Listens on **8082**, **not 80**, because **Docker Desktop's own engine already holds port 80** on localhost. 8082 was chosen over 8080/8081, which `tiny-inference-lb` and other local tools use.
- `allowedRoutes.namespaces.from: Same` restricts attachment to HTTPRoutes in the Gateway's own namespace.

```bash
# (PowerShell) confirm port 8082 is free first:
Get-NetTCPConnection -LocalPort 8082 -State Listen
kubectl apply -f gateway.yaml
kubectl get gateway -n team-alpha inference-gateway
```

> **Note:** on a real cluster (cloud LB or MetalLB) every LoadBalancer Service gets its **own IP**, so the port-8082 workaround is a **Docker-Desktop-only quirk**. Don't present it as universal.
> The `team-alpha` namespace must exist first (`kubectl create namespace team-alpha`).

### 12.3 HTTPRoute `llm-inference-route` (`httproute.yaml`)

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-inference-route
  namespace: team-alpha
spec:
  parentRefs:
    - name: inference-gateway
  rules:
    - matches:
        - path:
            value: /v1/completions        # slide shows value only; type defaults to PathPrefix
      backendRefs:
        - name: llm-service               # from slide text
          port: 8080                      # from slide text
```

- Defines a **path-based rule** (`/v1/completions`) and attaches to a Gateway via **`parentRefs`**.
- `backendRefs` sends matching traffic to the **`llm-service`** Service on **port 8080**.
- Created by the **ML team in its own namespace**: **no cluster-admin** needed (unlike classic Ingress).
- **Multiple HTTPRoutes from different teams** can attach to one Gateway, each matching its own path.
- **Canary/traffic-splitting rules** build on this same HTTPRoute resource.

```bash
kubectl apply -f httproute.yaml
kubectl get httproute -n team-alpha llm-inference-route
```

> ⚠️ **Ownership:** the lecture says "the Gateway Class and Gateway are platform-owned; only this HTTPRoute lives in the application team's namespace". The slide's Gateway lives in `team-alpha` for the demo, so in this lab one person plays both roles.
> ⚠️ A `llm-service` Service with a backing deployment must exist, or the route is accepted but traffic returns 5xx/503. Check the route's `status` conditions (`kubectl describe httproute ...`).
> ⚠️ **Naming:** a GatewayClass, a Gateway and an Istio VirtualService all named "inference-gateway/llm-..." can be confusing. The names are arbitrary labels. They aren't related to the separate **Gateway API Inference Extension** project (which adds `InferencePool` for LLM-aware routing). The lecture's "inference gateway class" refers only to this lab's naming.

---

## 13. Lab order and command reference

| # | File | Purpose | Depends on |
|---|---|---|---|
| 04 | `04_configmap.yaml` | `train-config` ConfigMap | none |
| 05 | `create_secret.sh` then `05_training_pod.yaml` | `ml-tokens` Secret, then `training-pod` | 04 + secret |
| 06 | `06_headless_svc.yaml` | `training-worker` headless Service | none |
| 07 | `07_statefulset.yaml` | `worker-0..2` | 06 |
| 08 | `08_loadbalancer_svc.yaml` | `tiny-inference-lb` | Lab 1 Deployment |
| - | `gatewayclass.yaml` | GatewayClass | Istio + Gateway API CRDs installed |
| - | `gateway.yaml` | Gateway in `team-alpha` (:8082) | GatewayClass |
| - | `httproute.yaml` | `/v1/completions` → `llm-service:8080` | Gateway, `llm-service` |

```bash
# Quick checks
kubectl get svc,pods,statefulset,configmap,secret
kubectl describe pod <name>                      # CreateContainerConfigError, Pending, etc.
kubectl get gatewayclass,gateway,httproute -A
kubectl exec -it worker-0 -- nslookup worker-1.training-worker
kubectl exec -it training-pod -- env | sort
```

**Prerequisite:** the Gateway API CRDs and Istio must be installed. Without them `kubectl apply` fails with `no matches for kind "GatewayClass"`.

---

## 14. Corrections and gotchas

| Topic | What was said | Correct / clarified |
|---|---|---|
| W&B Secret | "Secret = weights and biases ... backtrack the data (federated learning)" | **W&B = Weights & Biases (tracking service) API key.** Model weights aren't stored in Secrets. |
| Ingress | "A standard protocol connecting two points" | An **API object** for HTTP routing; frozen/superseded by Gateway API. |
| Secrets security | "Isolate secrets from the container" | They're **base64 only**. Enable encryption at rest + RBAC; use ESO/Vault. |
| ConfigMap updates | "Pod still needs restart" | True for **env vars**. Volume-mounted ConfigMaps do update (eventually). |
| `CreateContainerConfigError` | "Pod keeps checking for the secret" | An actual error state. Use `kubectl describe pod`. |
| Slide manifest for training pod | Shows only `configMapRef` | Needs the `secretRef` too (per slide text). |
| Headless DNS names | `pod-0.svc` (slide 1) | `<pod>.<svc>.<ns>.svc.cluster.local`. |
| Headless + readiness | Not mentioned | Per-pod DNS only exists for **Ready** pods unless `publishNotReadyAddresses: true`. |
| "NCCL would hang" | Slide | Overstated; the real problem is no per-peer addressing. |
| StatefulSet identity | "Keeps its name, ordinal, DNS" | Yes, but the **IP can change**. Always use the DNS name. |
| StatefulSet & GPUs | Implied sufficient for training | Still needs GPU requests, tolerations, PVCs, and **gang scheduling**. |
| RWO | "Default for NVMe local SSDs" | RWO is an **access mode**. Local NVMe is usually RWO by nature. |
| RWX/ROX | Listed as options | Depend on the **backend** (NFS/parallel FS yes, block disks usually no). |
| Port 8082 | Gateway listener | A **Docker Desktop quirk** (port 80 taken). Not universal. |
| LoadBalancer locally | "Publishes on localhost automatically" | True on Docker Desktop. On minikube needs `minikube tunnel`; on bare metal needs MetalLB. |
| GatewayClass name | `inference-gateway` for class *and* Gateway | Allowed (different kinds), but confusing. Istio's default class is usually named `istio`. |
| Istio vs HTTPRoute | "Canary built on HTTPRoute" and "VirtualService config" | Two ways: **Gateway API weighted `backendRefs`** or **Istio VirtualService + DestinationRule**. Pick one per route. |
| Slide cross-reference | "Reference: Session 2, slide 6" / PDF titled "Session 2" | The deck metadata says Session 2; this is Week 3 **Session 3**. |
| Transcript | "Kubernetes ... STO", "cube control", "mini cube" | Istio, `kubectl`, local Docker Desktop cluster. |

---

## 15. Cheat sheet and review questions

### Cheat sheet

```text
ClusterIP    internal virtual IP (default)           NodePort   node:30000-32767 (dev only)
LoadBalancer external IP (prod; MetalLB on bare metal)   Headless   clusterIP: None -> DNS returns pod IPs
StatefulSet  worker-0..N, stable name+DNS, ordered start ; serviceName -> headless Service
Pod DNS      <pod>.<svc>.<ns>.svc.cluster.local
ConfigMap    non-sensitive config, plain text, strings ; env vars need pod restart
Secret       tokens/keys, base64 (NOT encrypted) ; create BEFORE pod or CreateContainerConfigError
envFrom      configMapRef + secretRef -> all keys become env vars
Gateway API  GatewayClass (platform) -> Gateway (listener) -> HTTPRoute (app team, own namespace)
Istio        mTLS | canary weights (90/10) | metrics+traces | rate limits
Storage      RWO (one node) | RWX (shared datasets) | ROX (shared model weights)
```

### Review questions

1. Compare ClusterIP, NodePort, LoadBalancer and Headless Services. Which is for production inference? Which for distributed training?
2. What does `clusterIP: None` do, and how does DNS behave differently?
3. Why does NCCL rendezvous need stable per-pod hostnames?
4. How does a StatefulSet differ from a Deployment? What survives a reschedule and what doesn't?
5. Why separate the training image from its hyperparameters? What changes when you edit a ConfigMap, and what does a running pod see?
6. Why must tokens go in a Secret, not a ConfigMap? Is a Secret encrypted by default?
7. What does `envFrom` do, and what error appears if the referenced Secret is missing?
8. What does "W&B token" mean on the slide? What was misunderstood in the lecture?
9. Describe the three Gateway API resources and who owns each. How is this better than Ingress?
10. What does `allowedRoutes.namespaces.from: Same` do?
11. Why did the lab use port 8082? Is that true on a real cluster?
12. Explain a 10/90 canary rollout and what you would watch before ramping up.
13. Match each access mode (RWO / RWX / ROX) to an AI use case.

### Practice tasks

- Apply `04_configmap.yaml`, `create_secret.sh` and `05_training_pod.yaml`. Inspect `env` inside the pod. Then **edit the ConfigMap**, confirm the running pod still shows the old value, delete and recreate it, and confirm the new value appears.
- Apply the pod **before** creating the Secret and read the `describe pod` event. Then fix it.
- Apply the headless Service and StatefulSet. Run `nslookup worker-1.training-worker` from `worker-0`. Delete `worker-1` and watch it come back with the same name (note whether the IP changed).
- Compare `nslookup` output for a headless Service vs a normal ClusterIP Service.
- Deploy Lab 1, apply `08_loadbalancer_svc.yaml`, and `curl localhost:8080/health`.
- Install Gateway API CRDs + Istio, then apply GatewayClass → Gateway → HTTPRoute, and check each resource's status conditions.
- Write a weighted `backendRefs` HTTPRoute that sends 10% to a `canary` Service and 90% to `stable`.
