# Week 2 · Session 2 — Deterministic Environments & Containerized Jupyter

## 1. Learning Objectives

By the end of this session you should be able to:

- Explain the three levels of pinning: **ranges**, **exact versions**, **hashes**.
- Generate a lockfile with `pip-compile` and consume it in a Dockerfile (hands-on).
- Pick compatible **CUDA / driver / framework** versions using the compatibility chain.
- Slim production images with **multi-stage builds**.
- Launch a **GPU-backed Jupyter container** with correct volumes and shared memory (hands-on).

### Context from Session 1
Containers are **isolated by design**. A containerized Jupyter notebook needs four things wired in from the outside world:
1. A browser connection
2. Physical GPU access
3. A place to persist files
4. Enough (shared) memory for data loading

None of these are connected automatically — this session covers how to wire them correctly, and how to make the environment itself **reproducible**.

---

## 2. The Drift Problem — Why `pip install torch` Isn't Reproducible

**Unpinned install:**
```bash
pip install torch transformers
```
- Resolves to whatever is **newest today**.
- Transitive dependencies float too.
- Two builds a week apart can differ in **dozens of packages**.
- Classic symptom: a bug report that starts with *"nothing changed in our code"* — but something changed in the resolver's world.

**Pinned + locked:**
```
torch==2.11.2
```
- Every transitive dependency is also pinned (optionally with `sha256` hashes).
- The same input always yields the same environment.

> **Core idea:** Solve drift by **pinning**, then **locking**.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/8094790a-548b-43fe-a34a-3f316547c4dc" />

## 3. Three Levels of Strictness — The Pinning Ladder

| Level | Example | Behavior | Use case |
|---|---|---|---|
| **1. Ranges** | `torch>=2.10,<3` | Flexible, drifts | OK for libraries; **risky for training** |
| **2. Exact pins** | `torch==2.11.2` | Repeatable install while wheels stay published | Good general default |
| **3. Hash-locked** | `--require-hashes` | Tamper-evident, supply-chain safe, byte-identical installs | CI-grade / production |

- **Ranges** allow any version from the floor up to (not including) the ceiling — flexible but drift-prone.
- **Exact pins** fix the version number, but rely on the package index/wheel still being published as-is.
- **Hash-locked** installs verify the exact bytes of every package — if anyone tampers with a package, the install fails. This is the strictest, most reproducible tier.


<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/c892c6c3-77d4-4f38-a4cf-9071deeb8b1d" />

## 4. Lab 1 — Lockfile Workflow with `pip-compile`

**Workflow:**
1. Author intent in `requirements.in` (human-edited, high-level).
2. Compile it into `requirements.txt` (generated, committed, hash-locked).
3. Install strictly from the generated file in your Dockerfile.

```bash
# requirements.in (what you actually care about)
torch==2.11.*
transformers>=4.50
datasets
```

```bash
$ pip install pip-tools
$ pip-compile --generate-hashes requirements.in   # -> requirements.txt
$ head -4 requirements.txt
torch==2.11.2 \
    --hash=sha256:9f2c1a... \
    --hash=sha256:41d77b...
```

```dockerfile
# In the Dockerfile:
RUN pip install --no-cache-dir --require-hashes -r requirements.txt
```

> **Key principle:** *Intent lives in `.in` (human-edited); truth lives in `.txt` (generated, committed).*

Everyone who runs `pip-compile --generate-hashes` against the same `requirements.in` gets an identical `requirements.txt` — same packages, same hashes, **byte-equivalent** installs. No person-to-person variation.

<img width="2170" height="725" alt="image" src="https://github.com/user-attachments/assets/fd0810a3-1665-4f75-9377-9b5e41bb513b" />

## 5. The Compatibility Chain — Driver → CUDA → Framework → Python

Four things must be aligned simultaneously:

```
Host driver >= CUDA toolkit in image == CUDA of framework wheel
      |                                        |
  nvidia-smi                      nvidia/cuda:13.0.0-*   torch==2.11.2+cu130
                                          v                        v
                          Python in image == Python the wheel was built for (3.12)
```

**Rules to pin in writing:**
- **Driver floor** — the minimum host GPU driver version required.
- **CUDA tag** — the CUDA toolkit version baked into the base image.
- **Framework wheel** — PyTorch/TensorFlow wheels encode their CUDA version in the suffix (e.g., `+cu130`); the index URL you install from is part of the pin.
- **Python minor version** — the wheel must match the Python version in the image (e.g., 3.12).

> **Important:** Document the **driver floor** in your README. It's the one thing the container image *cannot* carry — the host driver lives outside the container and must be verified separately (e.g., via `nvidia-smi`).

<img width="1942" height="809" alt="image" src="https://github.com/user-attachments/assets/9b9f86ba-f1b9-4925-acfa-e39989bb0952" />


## 6. Image Diet — Multi-Stage Builds

Goal: compile fat, ship thin.

```dockerfile
# Stage 1 - builder: has nvcc, compiles custom kernels / wheels
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# Stage 2 - runtime: no compilers, just the artifacts
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt
```

- The `devel` image (with compilers, ~7 GB) never ships to production.
- The `runtime` image shrinks by **gigabytes**.
- Smaller image → faster node pull → faster autoscaling → faster job starts.

<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/9dfd9944-3f53-486c-89a8-0acd9bf69ef2" />

## 7. Dev Workflow — What Actually Needs Wiring for Containerized Jupyter

A container is isolated by design; three "wires" connect it to the outside world:

| Wire | How | Why |
|---|---|---|
| **Network** | `-p 8888:8888`, bind to `0.0.0.0` inside, use the printed token URL | Publishes the Jupyter port so a browser can reach it |
| **Data** | `-v $PWD:/workspace` | Notebooks/files survive container restarts; container images stay stateless |
| **Memory** | `--shm-size=8g` | DataLoader workers exchange tensors via `/dev/shm`; Docker's 64 MB default causes crashes |

> **Mental model:** *The container is disposable. The volume is not. Treat every notebook server as cattle, not a pet.*

---

## 8. Lab 2 — Launch a GPU-Backed Jupyter Container

```bash
docker run --rm -it \
  --gpus all \
  --shm-size=8g \
  -p 8888:8888 \
  -v "$PWD":/workspace -w /workspace \
  quay.io/jupyter/pytorch-notebook:cuda13-latest \
  start-notebook.py --IdentityProvider.token='lab2'
```

Then open `http://localhost:8888/?token=lab2` and in a cell:
```python
import torch
torch.cuda.get_device_name(0)
```

**Verification checklist:**
- GPU name prints correctly.
- A file saved in `/workspace` appears on the host filesystem.
- Stop and relaunch the container — the notebook content survives (proof the **volume**, not the container, holds state).

### Demo output observed in class (`GPUTest.ipynb`)
```python
import torch
print(torch.cuda.is_available())  # False with cpu wheel - expected
print(torch.__version__)          # confirms deterministic install
print("Jupyter is alive inside the container")
```
```
True
2.11.0+cu130
Jupyter is alive inside the container
```

### Course demo file set (Week 2 folder)
- `Dockerfile`
- `docker-compose.yml`
- `requirements.in`
- `requirements.txt`
- `GPUTest.ipynb`
- `.dockerignore`

**Demo `docker-compose.yml` commands:**
```bash
# First run (builds the image):
docker compose up --build

# Subsequent runs:
docker compose up

# Stop:
docker compose down
```

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/165118ec-4108-4f4b-91a8-1b59ef4034aa" />


## 9. Dev Loops — Where Should the Code Live?

| Mode | Description |
|---|---|
| **Bind-mounted (dev)** | Edit on host, run in container → instant feedback. Pair with an editor that attaches to containers (e.g., **dev containers**). |
| **Baked in (ship)** | `COPY` the code at build time. The image alone is the deployable, auditable artifact — nothing depends on a host checkout. |

> Same image, two modes: mount over `/workspace` during the day for active development, then **rebuild to freeze a version** before shipping.

---

## 10. Hygiene — `.dockerignore` and Secrets

```dockerfile
# .dockerignore
.git
data/                  # datasets do not belong in images
checkpoints/
*.ipynb_checkpoints
.env                   # secrets NEVER enter the build context
```

- **Everything in the build context is sent to the Docker daemon** — a stray 200 GB `data/` folder makes builds crawl.
- **Secrets in `ENV` or `COPY` persist in image layers forever**, even if "deleted" in a later layer.
- Use **build secrets** instead, which are never written to a layer:
```dockerfile
RUN --mount=type=secret,id=hf_token ...
```

---

## 11. Exercise — Before Next Session

**Task:** Convert Session 1's image to a locked build:
1. `requirements.in` → `pip-compile --generate-hashes` → produces `requirements.txt`.
2. Install with `--require-hashes`.
3. Launch Jupyter from the resulting image with GPU access, a workspace volume, and `--shm-size=8g`.

**Check-for-understanding questions:**
1. What can change between two unpinned builds run a month apart?
2. Why must the wheel's CUDA suffix match the image's toolkit version?
3. Why does the DataLoader care about `--shm-size`?

---

## 12. Session Recap

- Intent lives in `.in`, truth lives in a **hash-locked `.txt`** — generated, committed, reviewed.
- Pin the **whole chain**: driver floor, CUDA tag, framework wheel, Python minor.
- Use **multi-stage builds**: compile in `devel`, ship on `runtime`.
- Jupyter needs **three wires**: port, volume, shared memory.
- **Next session (Session 3):** A CUDA primer inside containers, and CI that validates every image build.


## Appendix — Reference Files from the Demo

### `requirements.in` → `requirements.txt` (illustrative)
```
# Normally this file is produced by `pip-compile` from requirements.in.
# A real generated file would include the full transitive closure with
# hashes. For demo purposes this lists the top-level pins only.
#
# To regenerate:
#   pip-compile --generate-hashes --output-file=requirements.txt requirements.in

torch==2.11.0
torchvision==0.26.0
jupyterlab==4.2.0
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.0
scikit-learn==1.4.2
ipywidgets==8.1.2
```

### `docker-compose.yml`
```yaml
# Brings up a GPU-accelerated Jupyter Lab with your local working
# directory bind-mounted into /workspace inside the container.
#
# Start:   docker compose up --build
# Stop:    docker compose down
#
# After start, open: http://localhost:8888/

services:
  jupyter:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ai-dev-jupyter

    ports:
      - "8888:8888"

    volumes:
      # Mount the project directory so edits on the host show up live
      # inside the container.
      - .:/workspace
      # Optional: a named volume for large datasets / caches that
      # survives container rebuilds.
      - dataset-cache:/data

    # PyTorch DataLoader workers share memory via /dev/shm. Docker's
    # default 64 MB will cause cryptic crashes; bump it.
    shm_size: "8gb"

    # GPU allocation via the compose v3.8+ syntax. Requires
    # nvidia-container-toolkit on the host.
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    # Friendly environment variables for notebook use.
    environment:
      JUPYTER_ENABLE_LAB: "yes"
      PYTHONPATH: "/workspace"

volumes:
  dataset-cache:
```

### `Dockerfile`
```dockerfile
# ------------------------------------------------------------------
# Jupyter development container with GPU access and pinned deps.
#
# Key techniques demonstrated:
#   - Multi-stage build (builder -> runtime) to keep final image small
#   - requirements.txt is compiled from requirements.in via pip-compile
#     for a fully-pinned, reproducible dependency set
#   - Layer ordering puts slowly-changing content on top so rebuilds
#     after code edits are fast
# ------------------------------------------------------------------

# -------- Stage 1: builder - has compilers if we need to build any
# native extensions; the artifacts are copied into the runtime stage.
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python — CUDA devel image has compilers but not Python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment that we will copy wholesale
# into the runtime image. This keeps the final image free of build
# caches and apt metadata.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install dependencies from the fully-pinned requirements.txt.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
                --extra-index-url https://pypi.org/simple \
                -r /tmp/requirements.txt

# -------- Stage 2: runtime - smaller base, no compilers.
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Install Python in runtime stage — runtime image also lacks Python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy only the installed virtual environment from the builder.
# No pip, no compilers, no build cache in the final image.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace
EXPOSE 8888

# Run Jupyter Lab without a token for local dev convenience.
# DO NOT deploy this command line to production; it exposes an open
# Jupyter server.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=", \
     "--ServerApp.password="]
```

> **Security note:** The demo `Dockerfile`'s `CMD` disables the Jupyter token/password for local dev convenience — this is explicitly flagged in the file as **not safe for production/deployment**, since it exposes an open Jupyter server.
