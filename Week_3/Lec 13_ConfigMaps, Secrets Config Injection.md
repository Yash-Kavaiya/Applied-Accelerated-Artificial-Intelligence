# Week 3 · Session 4: Hands-On Lab, Services, ConfigMaps, Headless DNS, Gateway API and Istio Canary

**Course:** AI Systems Engineering (Containerized AI Systems track)
**Instructor:** Satyadhyan Chickerur, Ph.D., Director, Centre for AI Research, KLE Technological University
**Format:** hands-on lab (Lab Guide: *K8s Networking & Configuration for AI Workloads*)
**Prerequisite:** Week 3 Session 3 concepts (Services, ConfigMaps/Secrets, Gateway API, Istio)

---

## Table of Contents

1. [Goals and environment](#1-goals-and-environment)
2. [Part 1a: ConfigMap, Secret and envFrom](#2-part-1a-configmap-secret-and-envfrom)
3. [Part 1b: headless Service and StatefulSet DNS](#3-part-1b-headless-service-and-statefulset-dns)
4. [Part 1c: LoadBalancer for production endpoints](#4-part-1c-loadbalancer-for-production-endpoints)
5. [Part 1d: Gateway API](#5-part-1d-gateway-api-the-modern-ingress)
6. [Part 2: Istio canary rollout](#6-part-2-istio-canary-rollout)
7. [What was seen in the live demo](#7-what-was-seen-in-the-live-demo)
8. [Command reference and cleanup](#8-command-reference-and-cleanup)
9. [Corrections and gotchas](#9-corrections-and-gotchas)
10. [Cheat sheet and review questions](#10-cheat-sheet-and-review-questions)

---

## 1. Goals and environment

**You will practise**

- **ConfigMaps + Secrets** injected with `envFrom`, so the training image is never rebuilt to tune a run or rotate a token.
- **Headless Service + StatefulSet** DNS: the backbone of distributed training (NCCL rendezvous).
- **LoadBalancer Service** vs ClusterIP / NodePort for an inference endpoint.
- **Gateway API** (GatewayClass + Gateway + HTTPRoute) as the replacement for Ingress.
- **Istio** DestinationRule + VirtualService for a **weighted canary rollout**, plus automatic mTLS and observability (Kiali/Prometheus).

**Environment**

- Docker Desktop + **WSL2 backend** + **Kubernetes enabled**.
- Everything runs on the plain **`docker-desktop`** context. **No GPU-aware (minikube) cluster is needed** for this lab.
- The lecture ran commands in **PowerShell** (the guide notes PowerShell variants).

```bash
kubectl config use-context docker-desktop
kubectl get nodes        # should show a Ready control-plane node
```

**Folders**

| Folder | Contents |
|---|---|
| `03_networking_config` | `04_configmap.yaml`, `create_secret.sh`, `05_training_pod.yaml`, `06_headless_svc.yaml`, `07_statefulset.yaml`, `08_loadbalancer_svc.yaml`, `gatewayclass.yaml`, `gateway.yaml`, `httproute.yaml` |
| `07_istio` | `llm_service_base.yaml`, `09_istio_destinationrule.yaml`, `10_virtualservice.yaml`, `11_inference_v2.yaml`, `patch.json` (revert to 90/10), `patch-promote.json` (promote to 0/100) |
| `01_inference_service` | The Lab 1 `tiny-inference` Deployment (reused in Part 1c) |

> ⚠️ The guide says "unlike Session 1's GPU pod demo", but the GPU pod demo was **Week 3 Session 2**. Only the numbering is off. The point stands: this lab needs no GPU cluster.

---

## 2. Part 1a: ConfigMap, Secret and envFrom

**Files:** `04_configmap.yaml`, `create_secret.sh`, `05_training_pod.yaml`

**Idea:** a ConfigMap holds hyperparameters, and a Secret holds tokens (W&B, Hugging Face). A pod consumes **both identically via `envFrom`**. The Secret is created **imperatively** so real values never get committed to Git as plain YAML.

### Steps

```bash
cd 03_networking_config
kubectl apply -f 04_configmap.yaml
bash create_secret.sh                       # creates Secret ml-tokens (type Opaque)
kubectl apply -f 05_training_pod.yaml
```

```powershell
# PowerShell check (inside the pod)
kubectl exec -it training-pod -- env | Select-String -Pattern "batch_size|learning_rate|max_steps|model_arch|WANDB|HF_TOKEN"
```

```bash
# WSL / bash equivalent
kubectl exec -it training-pod -- env | grep -E "batch_size|learning_rate|max_steps|model_arch|WANDB|HF_TOKEN"
```

### Expected output

```
batch_size=128
learning_rate=3e-4
max_steps=50000
model_arch=llama3-8b
HF_TOKEN=demo-hf-token-not-real
WANDB_API_KEY=demo-wandb-key-not-real
```

- ConfigMap keys appear **lowercase**, and the Secret keys appear **uppercase**, exactly as defined.
- Values are **demo placeholders** ("not real"). Never paste real tokens into a lab, chat or screen share.
- You can inspect the objects: `kubectl get configmap train-config -o yaml`, `kubectl get secret ml-tokens -o yaml` (values shown **base64-encoded**, which is encoding, not encryption).

### Demo observation: "cannot exec into a completed pod"

When re-running, the lecture hit: *cannot exec into a container in a completed pod; current phase is Succeeded*.

- The training pod's stand-in command is a **`sleep 3600`**, so the container **exits after an hour**.
- With `restartPolicy: Never`, the pod then stays in phase **`Succeeded`** and you can't `exec` into it.
- **Fix:** delete and recreate it (`kubectl delete -f 05_training_pod.yaml` then apply again), or just re-apply after cleanup.

**Quick questions**

- What if you edit `train-config` now? The running pod's env **does not change**. Recreate the pod.
- What if `ml-tokens` doesn't exist? The pod gets stuck in **`CreateContainerConfigError`**.

---

## 3. Part 1b: headless Service and StatefulSet DNS

**Files:** `06_headless_svc.yaml`, `07_statefulset.yaml`

**Idea:** a headless Service (`clusterIP: None`) allocates **no virtual IP**. DNS for the Service name returns **pod IPs directly**, and each StatefulSet pod gets its own stable hostname (`worker-0.training-worker`, `worker-1.training-worker`, ...). **NCCL rendezvous** needs every worker to dial every other worker by a stable hostname.

### Steps

```bash
kubectl apply -f 06_headless_svc.yaml
kubectl apply -f 07_statefulset.yaml
kubectl get pods -l app=training-worker           # worker-0, worker-1, worker-2

# ONE specific pod
kubectl exec worker-0 -- nslookup worker-1.training-worker.default.svc.cluster.local
# ALL pod IPs (no VIP)
kubectl exec worker-0 -- nslookup training-worker.default.svc.cluster.local
```

### Expected output (IPs will differ)

```
Name:    worker-1.training-worker.default.svc.cluster.local
Address: 10.244.0.9          <- ONE specific pod

Name:    training-worker.default.svc.cluster.local
Address: 10.244.0.10
Address: 10.244.0.8
Address: 10.244.0.9          <- ALL pod IPs, no VIP
```

### Contrast: a regular ClusterIP Service on the same pods

```bash
kubectl create service clusterip training-worker-demo-vip --tcp=29500:29500
kubectl patch service training-worker-demo-vip -p '{"spec":{"selector":{"app":"training-worker"}}}'
kubectl exec worker-0 -- nslookup training-worker-demo-vip.default.svc.cluster.local
kubectl delete service training-worker-demo-vip      # clean up
```

```
Name:    training-worker-demo-vip.default.svc.cluster.local
Address: 10.96.168.200       <- ONE virtual IP, always
```

**What to notice**

| Service | `CLUSTER-IP` | DNS answer |
|---|---|---|
| `training-worker` (headless) | `None` | **3 pod IPs** (+ per-pod names) |
| `training-worker-demo-vip` (ClusterIP) | `10.96.x.x` | **one virtual IP** |

- Run `kubectl get svc` to see `None` vs a real ClusterIP side by side.
- The extra Service is **deleted** at the end so it doesn't linger. The lecturer ran only the headless Service first, then the VIP Service for comparison, and deleted it "to keep things sane".
- With **one VIP**, every worker resolves to the same address, so peers cannot target a specific rank.

---

## 4. Part 1c: LoadBalancer for production endpoints

**File:** `08_loadbalancer_svc.yaml` (reuses the `tiny-inference` Deployment from Lab 1)

| Type | Scope |
|---|---|
| ClusterIP | Internal only |
| NodePort | Dev/debug only (static port 30000-32767 on **every** node) |
| **LoadBalancer** | **Production pattern**: provisions an external IP via a cloud LB (ALB/NLB on AWS, equivalents elsewhere) |

```bash
kubectl apply -f ../01_inference_service/tiny-inference-deployment.yaml   # prerequisite
kubectl apply -f 08_loadbalancer_svc.yaml
kubectl get svc tiny-inference-lb
curl http://localhost:8080/health
```

**Expected output**

```
NAME                TYPE           EXTERNAL-IP   PORT(S)
tiny-inference-lb   LoadBalancer   172.18.0.5    8080:32023/TCP
{"status":"ok","device":"cpu"}
```

- `8080:32023/TCP` = **service port 8080** and an auto-allocated **NodePort 32023**. A LoadBalancer sits on top of NodePort and ClusterIP.
- The response `device: cpu` confirms the service from Lab 1 is the one answering (it requests no GPU).
- In the live demo, `curl` against `localhost:8080` returned `status ok, device cpu`.

**Discussion questions (from the guide)**

1. **On a real cloud cluster, what happens when you create a LoadBalancer Service?** The **cloud controller manager** watches the Service, calls the provider API to create a load balancer (e.g. an NLB/ALB on AWS, with target groups for node/pod endpoints and health checks), then writes the external address into the Service's status.
2. **Why is NodePort dev/debug only?** Clients must know node IPs and high ports, nodes can come and go, there's no health-aware external load balancing or TLS termination, ports are limited, and every node's port is exposed (bigger attack surface).

---

## 5. Part 1d: Gateway API (the modern Ingress)

**Files:** `gatewayclass.yaml`, `gateway.yaml`, `httproute.yaml`

**Idea:** the platform team owns **GatewayClass + Gateway**. App teams own **HTTPRoute** in their own namespace, with no cluster-admin needed.

### 5.1 One-time setup (once per cluster)

Istio doesn't bundle the Gateway API CRDs, and here Istio is also the controller that implements them (`controllerName: istio.io/gateway-controller`).

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml
istioctl install --set profile=demo -y
kubectl create namespace team-alpha
kubectl label namespace team-alpha istio-injection=enabled
```

> ⚠️ The guide's PDF text shows `standardinstall.yaml`. That's a line-break artifact. The release asset is **`standard-install.yaml`**.
> `istio-injection=enabled` makes new pods in `team-alpha` get an **Envoy sidecar** (existing pods need a restart to pick it up). The lecture noted the `team-alpha` namespace "already exists and isn't labelled yet". The label is what enables injection.
> **Version compatibility:** the guide pins Gateway API **v1.2.0** while Part 2 uses **Istio release-1.30** addons. Check that your Istio version supports the Gateway API CRD version you install.

### 5.2 Apply the resources

```bash
cd 03_networking_config
kubectl apply -f gatewayclass.yaml
kubectl apply -f gateway.yaml
kubectl apply -f httproute.yaml
kubectl get gatewayclass inference-gateway -o wide
```

**Expected output**

```
NAME                CONTROLLER                    ACCEPTED   AGE
inference-gateway   istio.io/gateway-controller   True       3s
```

`ACCEPTED: True` means Istio has registered itself as the controller for this class. If it's `Unknown`/`False`, the controller name doesn't match what's installed.

### 5.3 Docker Desktop quirks (important)

1. **Host port 80 is often taken.** On Docker Desktop, each LoadBalancer Service is published on the host by a small per-Service proxy container, and Docker's own engine backend already listens on **port 80**. A second Service asking for 80 fails (`SyncLoadBalancerFailed ... exit status 125`) and stays `<pending>` forever.
   - **Fix:** check the port first (`Get-NetTCPConnection -LocalPort 80 -State Listen`) and move the Gateway listener to a free port. **This lab uses 8082.**
   - Also remove the demo profile's default **`istio-ingressgateway`** Deployment/Service in `istio-system` if it is squatting on the same port. Gateway API makes it redundant.
2. **EXTERNAL-IP is unreliable for a second LoadBalancer.** Docker Desktop's local LB publishing reliably covers only the **first** LoadBalancer Service. A second one (the Gateway) can report `Programmed: True` with an IP, yet **nothing is listening**. Verify with **port-forward** instead.

```bash
kubectl port-forward -n team-alpha svc/inference-gateway-inference-gateway 18082:8082
curl http://localhost:18082/v1/completions
# -> llm-service stable response
```

- Service name pattern: Istio names the auto-created Service `<gateway-name>-<gatewayclass-name>` (`inference-gateway-inference-gateway`).
- `18082 -> 8082` maps your local port to the Gateway listener port.
- In the demo, the response text was **"llm-service stable response"**. It's a **stub backend** that demonstrates routing/port-forwarding, not a real LLM.

> **On a real cloud/MetalLB cluster** every LoadBalancer gets its own IP, so none of the port-8082 workaround applies. It is a Docker-Desktop-only quirk.

### 5.4 Role split recap

| Before (Ingress) | After (Gateway API) |
|---|---|
| Teams often needed elevated rights to edit shared Ingress | Platform team creates **one Gateway**; app teams attach **HTTPRoutes** in their namespace |

---

## 6. Part 2: Istio canary rollout

**Folder:** `07_istio`
**Files:** `llm_service_base.yaml`, `09_istio_destinationrule.yaml`, `10_virtualservice.yaml`, `11_inference_v2.yaml`, `patch.json`, `patch-promote.json`

### 6.1 What Istio gives (zero app code changes)

1. **Automatic mutual TLS** between pods (zero-trust inside the cluster).
2. **Weighted traffic splitting** for canary rollouts.
3. **Auto-injected observability** (metrics, traces, access logs).
4. **Envoy-level rate limiting.**

This demo focuses on the **canary**: a **DestinationRule** defines named **subsets** (`stable`, `canary`) from a pod label, and a **VirtualService** splits traffic across subsets **by weight**.

### 6.2 One-time setup: Prometheus + Kiali

```bash
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.30/samples/addons/prometheus.yaml
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.30/samples/addons/kiali.yaml
```

- **Prometheus** collects the mesh metrics. **Kiali** is the browser UI that draws the traffic graph.
- Istio itself must already be installed (Part 1d). The lecture noted the add-on configs came back "unchanged" because they were already applied.

> ⚠️ The guide's text shows `release1.30`. The real branch is **`release-1.30`** (hyphen lost at a line break).

### 6.3 Deploy: stable, rules, then canary

```bash
cd 07_istio
kubectl apply -f llm_service_base.yaml          # stable (v1), 2 replicas, plus Service + curl-client
kubectl apply -f 09_istio_destinationrule.yaml  # subsets: stable / canary
kubectl apply -f 10_virtualservice.yaml         # 90/10 split
kubectl apply -f 11_inference_v2.yaml           # canary (v2), 1 replica
```

**Reconstructed shape of the routing objects** (verify against your files):

```yaml
apiVersion: networking.istio.io/v1
kind: DestinationRule
metadata:
  name: llm-inference-dr
  namespace: team-alpha
spec:
  host: llmsvc.team-alpha.svc.cluster.local
  subsets:
    - name: stable
      labels: { version: v1 }
    - name: canary
      labels: { version: v2 }
---
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: llm-inference-vs
  namespace: team-alpha
spec:
  hosts: [llmsvc.team-alpha.svc.cluster.local]
  http:
    - route:
        - destination: { host: llmsvc.team-alpha.svc.cluster.local, subset: stable }
          weight: 90
        - destination: { host: llmsvc.team-alpha.svc.cluster.local, subset: canary }
          weight: 10
```

### 6.4 Generate traffic and count the split

Run from **inside the mesh** (the `curl-client` pod has a sidecar). The counting happens in the pod's shell, so it works unchanged in PowerShell, WSL or Git Bash:

```bash
kubectl exec curl-client -n team-alpha -- sh -c \
 'for i in $(seq 1 40); do curl -s http://llmsvc.team-alpha.svc.cluster.local:8080/; echo; done | sort | uniq -c'
```

**Expected output (roughly 90/10)**

```
 34 response from STABLE (v1)
  6 response from CANARY (v2)
```

- The lecture run showed **35 stable / 5 canary** (40 requests). Any result near 36/4 is normal. The split is **probabilistic**, so with 40 samples expect noise.

### 6.5 Promote the canary to 100%

```bash
kubectl patch virtualservice llm-inference-vs -n team-alpha --type=json --patch-file=patch-promote.json
# re-run the same 40-request loop:
#  40 response from CANARY (v2)
```

**Reconstructed `patch-promote.json`** (JSON Patch; your file may differ):

```json
[
  {"op": "replace", "path": "/spec/http/0/route/0/weight", "value": 0},
  {"op": "replace", "path": "/spec/http/0/route/1/weight", "value": 100}
]
```

- `patch.json` reverts to **90/10**.
- Real rollouts ramp in steps (10 -> 25 -> 50 -> 100) while watching error rate and latency, and roll back by restoring the weights.

### 6.6 mTLS and observability

```bash
istioctl x describe pod <llm-svc-stable-pod> -n team-alpha
# Workload mTLS mode: PERMISSIVE

kubectl port-forward -n istio-system svc/kiali 20001:20001
# open: http://localhost:20001/kiali/console/graph/namespaces?namespaces=team-alpha
```

- **PERMISSIVE** = the sidecar accepts **both mTLS and plaintext**. Sidecar-to-sidecar traffic is upgraded to mTLS automatically, but plaintext clients aren't rejected.
- In **Kiali's graph** (namespace `team-alpha`) you see: client -> service -> **two app versions** (stable v1 and canary v2), with **HTTP requests per second in/out and error rates** per version. This is where you watch whether the canary is healthy.

---

## 7. What was seen in the live demo

| Step | Observation |
|---|---|
| Context | `docker-desktop`, one Ready control-plane node |
| ConfigMap/Secret | Applied (unchanged on re-run). Secret `ml-tokens` is type `Opaque` with demo values |
| Training pod | First re-run failed with the "completed pod / phase Succeeded" message. After recreate, `env` showed the ConfigMap + Secret values together |
| StatefulSet | `worker-0/1/2` Running. The headless Service showed **no ClusterIP** but three pod IPs |
| VIP experiment | A regular ClusterIP Service collapsed all workers to **one IP**. It was deleted afterwards |
| LoadBalancer | `tiny-inference-lb` -> `/health` returned `ok`, device `cpu` |
| Istio install | Core + ingress gateway components installed. `team-alpha` existed |
| Gateway | GatewayClass accepted. The stub **LLM service** answered through the Gateway path via port-forward |
| Canary | 40 requests -> 35 stable / 5 canary, then a patch moved 100% to canary |
| Kiali | One service, two apps, three versions in the graph. Docker Desktop's namespace view showed the `inference-gateway`, `svc-stable` (2 pods), `svc-canary` (1 pod), `curl-client` and the emulated LLM backend |
| Wrap-up | Cleanup commands free memory and keep Docker Desktop stable |

---

## 8. Command reference and cleanup

| Demo | Run | Verify |
|---|---|---|
| 1a ConfigMap/Secret | `kubectl apply -f 04_configmap.yaml && bash create_secret.sh` | `kubectl exec training-pod -- env` |
| 1b Headless + StatefulSet | `kubectl apply -f 06_headless_svc.yaml -f 07_statefulset.yaml` | `kubectl exec worker-0 -- nslookup ...` |
| 1c LoadBalancer | `kubectl apply -f 08_loadbalancer_svc.yaml` | `curl http://localhost:8080/health` |
| 1d Gateway API | `kubectl apply -f gatewayclass.yaml -f gateway.yaml -f httproute.yaml` | `kubectl get gatewayclass inference-gateway -o wide` |
| 2 Istio canary | `kubectl apply -f 09_istio_destinationrule.yaml -f 10_virtualservice.yaml -f 11_inference_v2.yaml` | curl loop + `uniq -c` |

### Cleanup

```bash
kubectl delete pod training-pod --ignore-not-found
kubectl delete configmap train-config --ignore-not-found
kubectl delete secret ml-tokens --ignore-not-found
kubectl delete statefulset worker --ignore-not-found
kubectl delete svc training-worker tiny-inference-lb --ignore-not-found
kubectl delete namespace team-alpha --ignore-not-found
kubectl delete gatewayclass inference-gateway --ignore-not-found
kubectl delete -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml --ignore-not-found
istioctl uninstall --purge -y
kubectl delete namespace istio-system --ignore-not-found
```

> ⚠️ Cleanup order matters: delete workloads/Gateways **before** removing the Gateway API CRDs and Istio. Deleting the CRDs first can leave resources stuck. Deleting the `team-alpha` namespace also removes the Gateway, HTTPRoute, VirtualService, DestinationRule and the canary/stable Deployments.
> The Prometheus and Kiali add-ons live in `istio-system`, so deleting that namespace removes them.

---

## 9. Corrections and gotchas

| Topic | What was said | Correct / clarified |
|---|---|---|
| Replica counts vs split | "Stable has 2 replicas and canary 1, there's always one backup replica" (lecture) | The **90/10 comes from VirtualService weights**, not replica counts. The canary's single replica is just its capacity. With weights set, 10% goes to canary regardless of pod counts. |
| Promotion | "Shift from stable replicas to canary again and again" | Promotion is **patching the weights** (e.g. 0/100), not moving replicas. |
| mTLS | "mTLS is on by default" (guide) / "PERMISSIVE" (output) | **PERMISSIVE accepts plaintext too.** For zero-trust, apply a `PeerAuthentication` with mode **STRICT**. |
| Sidecar injection | Label on `team-alpha` | Only affects **new** pods. Restart older pods to inject the sidecar. |
| `curl-client` | Run from the mesh | Needs a sidecar to be a meaningful mesh client. Traffic from outside the mesh bypasses the VirtualService (use the Gateway for that). |
| Split counts | 34/6 (guide), 35/5 (live) | Both are normal sampling noise around 90/10 (expected about 36/4). |
| Docker Desktop LB | Gateway reports `Programmed: True` with an IP | Can be **false comfort** for a second LoadBalancer. Verify with `port-forward`. |
| Port 8082 | Gateway listener | A **local quirk**. On real clusters port 80/443 is normal. |
| `standardinstall.yaml`, `release1.30`, `patchpromote.json` | Guide text | Line-break artifacts. Real names: `standard-install.yaml`, `release-1.30`, `patch-promote.json`. |
| Gateway API vs Istio versions | v1.2.0 CRDs, Istio 1.30 addons | Check Istio's supported Gateway API version before installing. Mismatch can leave the class not Accepted. |
| Completed pod exec | "Cannot exec into completed pod" | Expected after `sleep 3600` ends under `restartPolicy: Never`. Recreate. |
| Secret "demo-*-not-real" | Printed in plain text | Fine for a lab. Real tokens would leak via `env`, `kubectl describe`, logs and screen-shares. |
| `kubectl exec -it ... | grep` | Works in WSL/bash | In PowerShell use `Select-String`. `-it` with a pipe can warn about a TTY, so `kubectl exec <pod> -- env` also works. |
| HTTPRoute backend name | `llm-service` (Session 3) vs `llmsvc` (mesh host) | Different objects in the lab (HTTPRoute backend vs mesh Service). Check actual Service names with `kubectl get svc -n team-alpha`. |
| NCCL hang | "Would hang at initialization" | A likely failure mode, not a guaranteed one. The real point: peers need per-pod addresses. |
| Transcript | "STO", "agris gateways", "kali", "LV LL MSVC", "cube control" | Istio, ingress gateways, Kiali, `llm-svc`, `kubectl`. |

---

## 10. Cheat sheet and review questions

### Cheat sheet

```text
Env inject   : envFrom: configMapRef + secretRef ; Secret first or CreateContainerConfigError
Config change: env vars frozen at container start -> recreate pod
Headless     : clusterIP: None -> nslookup <svc> returns ALL pod IPs ; <pod>.<svc> returns ONE
ClusterIP    : one VIP for every pod (bad for NCCL rank addressing)
LoadBalancer : cloud LB (prod) ; Docker Desktop publishes first LB on localhost ; 2nd LB -> use port-forward
Gateway API  : CRDs + Istio -> GatewayClass (Accepted: True) -> Gateway (:8082) -> HTTPRoute
Canary       : DestinationRule subsets (labels) + VirtualService weights ; promote with patch
Check split  : kubectl exec curl-client -- sh -c 'for ...; done | sort | uniq -c'
mTLS         : PERMISSIVE (mixed) vs STRICT (enforced) ; observe in Kiali (port-forward 20001)
Cleanup      : workloads first, then CRDs, then istioctl uninstall --purge
```

### Review questions

1. Why does `envFrom` mean you never rebuild the training image to tune a run or rotate a token? What does a running pod see if you edit the ConfigMap?
2. Why was the training pod "completed", and why couldn't you `exec` into it?
3. What do `nslookup training-worker...` and `nslookup worker-1.training-worker...` return, and how does a ClusterIP Service change that?
4. Why does NCCL rendezvous need per-pod hostnames?
5. What does the `8080:32023/TCP` mapping mean? Why is NodePort dev/debug only?
6. What does a cloud controller manager do when you create a LoadBalancer Service?
7. Why did the Gateway listener use port 8082, and why use `port-forward` to test it?
8. What does `ACCEPTED: True` on a GatewayClass tell you?
9. How do DestinationRule subsets and VirtualService weights combine to make a canary?
10. How would you promote the canary, and how would you roll back?
11. Why is "PERMISSIVE" not full zero-trust? What would you change for STRICT mTLS?
12. What can you learn from the Kiali graph during a canary?

### Practice tasks

- Run Part 1a. Edit `train-config`, confirm the **running pod's env is unchanged**, recreate the pod, and confirm the new value.
- Delete `ml-tokens`, re-apply the pod, and read the `CreateContainerConfigError` event.
- Compare `kubectl get svc` and `nslookup` results for the headless Service vs the temporary VIP Service.
- Delete `worker-1` and confirm it returns with the same name (compare IPs before and after).
- Expose `tiny-inference` with the LoadBalancer Service and `curl localhost:8080/health`. Then delete it and note what happens to the host port.
- Follow Part 1d. Verify the Gateway with `port-forward` and explain why `EXTERNAL-IP` alone isn't proof.
- Run the 40-request loop three times and record the stable/canary counts. Then run it with weights 50/50 and 0/100.
- Apply a `PeerAuthentication` with STRICT mode in `team-alpha` and test a plaintext request from a pod **without** a sidecar.
- Run the full cleanup and confirm that `kubectl get ns` and `kubectl get crd | grep -i gateway` are clean.
