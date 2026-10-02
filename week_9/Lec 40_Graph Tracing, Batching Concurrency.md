# Segment 2: Graph Tracing, Batching, and Concurrency

*Optimising the compute graph for inference: tracing, operator fusion, dynamic shapes, and concurrent request handling.*

---

## Learning Objectives

1. Explain the difference between **eager execution** and **traced/compiled graphs** for inference, and the overhead each eliminates.
2. Use `torch.jit.trace`, `torch.jit.script`, and `torch.export` to capture an inference graph and apply optimisations.
3. Implement **dynamic batching** and explain how it improves GPU utilisation without pushing latency beyond the SLO.
4. Describe **CUDA Graphs** and explain why they are mandatory for low-latency small-batch inference.
5. Configure **concurrent request handling** in a serving framework using async processing and request queuing.

**Context from the lecture:** The graph idea is the same as in training (see Segment 1 for the training vs. inference pipeline difference). The key difference is that for inference the captured graph can be **reused**, and it must support **dynamic shapes** so we don't rebuild a graph for every new input shape.

---

## 1. Graph Tracing: Capturing the Computation Graph

### 1.1 Why trace a graph for inference?

- **Eager execution** (line-by-line, "Pythonic" execution) pays Python overhead on every forward pass: **~3-10 µs per operator call**.
- Worked example: a 32-layer Transformer with 200 ops/layer:
  `32 × 200 × 5 µs = 32 ms` of pure Python dispatch overhead **per step** (and the same again on the next step).
- **Graph capture** removes this: Python runs **once** to trace, then the compiled graph executes **without Python**.
- Additional benefits from graph-level optimisation:
  - **Operator fusion**: several ops become one (e.g., Conv + MaxPool in a single kernel call).
  - **Layout optimisation**.
  - **Dead-code elimination**: unneeded layers/ops are removed.

> **Intuition:** a model is a long chain of calls (matmul, matmul, flash attention, MLP, softmax, ...). In eager mode each is dispatched individually. On CUDA each one launches kernels; on CPU, other kernels. Latency here means the time taken to produce the first output.

### 1.2 `torch.jit.trace`: tensor-shape-specific tracing

```python
import torch
model = MyModel().eval()
example_input = torch.randn(1, 512)

# trace records ops for this specific input shape
traced = torch.jit.trace(model, example_input)
torch.jit.save(traced, 'model_traced.pt')

# Load and run: no Python, pure TorchScript
loaded = torch.jit.load('model_traced.pt')
output = loaded(example_input)
```

- Records the ops executed for **one specific input shape** (here `1 × 512`).
- The saved model is loaded and run as pure TorchScript, with no Python dispatch.
- **Limitation:** does **not** handle Python control flow (`if/else` on tensor values). Control flow depends on data and would change the graph dynamically (e.g., choosing flash attention vs. self-attention based on some metric). A trace bakes in only the path taken during tracing.
- **Use for:** CNNs, encoders, and models with **fixed-shape forward passes** (e.g., vision models with constant image size). In such cases `jit.trace` is perfectly fine and can sometimes even perform better.

### 1.3 `torch.jit.script`: control-flow-aware compilation

```python
@torch.jit.script
def beam_search_step(logits: torch.Tensor,
                     beam_scores: torch.Tensor,
                     k: int) -> torch.Tensor:
    # if/for loops compiled to TorchScript IR
    scores = logits + beam_scores.unsqueeze(-1)
    top_scores, top_ids = scores.topk(k)
    return top_ids
```

- Compiles Python to **TorchScript IR**, supporting `if / for / while`.
- **Restrictions:** type annotations are required; limited Python standard-library support.

### 1.4 `torch.export` (PyTorch 2.x): the modern approach

```python
from torch.export import export

# Capture with dynamic shape support
exported = export(
    model,
    args=(example_input,),
    dynamic_shapes={'x': {0: torch.export.Dim('batch', min=1, max=256)}},
)
# Export to TensorRT, ONNX, or run directly
```

- **Recommended API for PyTorch 2.x inference.**
- Supports **dynamic dimensions** (batch size, sequence length), unlike `jit.trace`. Here the batch dimension is allowed to range from 1 to 256.
- Integrates with `torch.compile` for end-to-end optimisation.

### 1.5 Quick comparison

| Method | Control flow | Dynamic shapes | Best for |
|---|---|---|---|
| `torch.jit.trace` | No (bakes in one path) | No (shape-specific) | CNNs, encoders, fixed-shape forward passes |
| `torch.jit.script` | Yes (`if/for/while`) | Limited | Models/functions with data-dependent logic (e.g., beam search step) |
| `torch.export` | Handled via export semantics | Yes (`Dim` with min/max) | Recommended default for PyTorch 2.x |

---

## 2. CUDA Graphs: Eliminating CPU Overhead in Low-Latency Inference

### 2.1 The CPU overhead problem at small batch sizes

- Each CUDA kernel launch needs a CPU-GPU round trip: **~5-10 µs** of CPU time. (Kernels here means, e.g., a sparse matmul launched on the GPU; the CPU initiates every call.)
- A Transformer forward pass launches **500-2000 kernels** (softmax, FFN matmuls, attention, ...), giving **up to ~20 ms** of CPU overhead.
- At **batch = 1** with ~5 ms of real GPU compute, the 20 ms CPU overhead dominates.
- The GPU sits **idle** waiting for the CPU to issue the next kernel, so the workload is **CPU-bound, not GPU-bound**.

### 2.2 CUDA Graphs: record once, replay many times

- **Record phase:** the GPU records all kernel launches and their arguments into a graph.
- **Replay phase:** a **single** graph launch replays all kernels with **zero CPU involvement**.
- CPU overhead goes from **O(n_kernels)** to **O(1)** per step: **~3 µs vs ~20 ms**.
- Conceptually similar to operator fusion, but at the CUDA kernel-launch level rather than the Python operator level.

### 2.3 Capture with `torch.cuda.CUDAGraph`

```python
# Warmup: required to initialise cuDNN and NCCL
for _ in range(3):
    output = model(static_input)
torch.cuda.synchronize()

# Capture the graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay: just copy new input, replay graph
static_input.copy_(new_input)
g.replay()
result = static_output.clone()
```

Notes from the lecture:

- `torch.cuda.synchronize()` is there to make **timing measurements** accurate; it is not part of CUDA Graphs and can be removed in production.
- Inside the `with torch.cuda.graph(g)` block, the kernel calls issued by the model are **recorded** into `g`.
- **CRITICAL:** input/output tensors must live at the **same memory addresses**. That's why you use `copy_()` into the static input rather than rebinding a new tensor. The graph has the addresses baked in.

### 2.3.1 `torch.compile` + CUDA Graphs (recommended pattern)

```python
# torch.compile uses CUDA Graphs automatically when mode='reduce-overhead'
model = torch.compile(
    model,
    mode='reduce-overhead',  # enables CUDA Graphs
)

# First call triggers compilation + graph capture
output = model(x)   # slow
# Subsequent calls replay the graph
output = model(x)   # fast: O(1) CPU overhead
```

- `mode='reduce-overhead'` handles capture, copy, and replay for you behind the scenes.
- The first call is slow (compilation + capture); subsequent calls are fast.
- The abstraction hides the CUDA code, so you need to know *which arguments/patterns* trigger this behaviour.

### 2.4 Limitations of CUDA Graphs

- **Static shapes required:** the graph is captured for specific `(batch, seq_len)` because kernel arguments encode matrix dimensions. Dynamic shapes need **separate captures per shape**.
- **Static memory:** no dynamic allocation inside the graph.
- **No CPU-GPU synchronisation points** inside the graph (e.g., `.item()`, `print`).
- **Workaround for dynamic batch sizes:** capture one graph per batch size (1, 2, 4, 8, 16, 32, 64) and **pad** smaller batches up to the nearest captured size.

### 2.5 Speedup in practice

| Setting (bs=1) | Per-step time |
|---|---|
| Without CUDA Graphs | ~25 ms (20 ms CPU + 5 ms GPU) |
| With CUDA Graphs | ~5.3 ms (0.3 ms CPU + 5 ms GPU) |

- **Speedup: ~5x at batch=1, ~1.2x at batch=64.** The gain shrinks as batches grow because GPU compute starts to dominate.
- Even a modest speedup matters for LLM serving: milliseconds saved directly improve **TTFT** (time to first token) and **TPOT** (time per output token), which users can feel.
- CUDA Graphs are **mandatory for competitive low-latency LLM inference**; major serving frameworks (vLLM, TGI) use them by default.

---

## 3. Dynamic Batching and Concurrency for Maximum Throughput

### 3.1 Static vs. dynamic vs. continuous batching

- **Static batching:** collect exactly *B* requests and run them as one batch. Simple, but every request waits for the slowest.
- **Dynamic batching:** accumulate requests up to `max_batch` **or** until a timeout, whichever first. Better latency-throughput trade-off.
- **Continuous batching** (Segment 1): batching at the **token level**; current best practice for LLM serving.

### 3.2 Triton Inference Server: production batching configuration

```protobuf
# config.pbtxt: model serving configuration
name: "llama3_8b"
backend: "vllm"
max_batch_size: 128

dynamic_batching {
  preferred_batch_size: [1, 4, 8, 16, 32, 64, 128]
  max_queue_delay_microseconds: 5000   # wait up to 5ms to fill batch
}

instance_group [
  { count: 1, kind: KIND_GPU, gpus: [0] }
]
```

- `max_batch_size` caps the batch.
- `preferred_batch_size` lists sizes the server tries to form.
- `max_queue_delay_microseconds` is the **latency budget you trade for utilisation** (here up to 5 ms to fill a batch); it must stay within your **SLO**.
- `instance_group` controls the number of model instances per GPU.

### 3.3 Concurrency: handling multiple requests simultaneously

- **Single-threaded serving:** requests queue behind each other, giving terrible utilisation. (Analogy: you ask ChatGPT on phone, laptop, and desktop at once; you don't want each response to wait for the previous one.)
- **Async serving:** use `asyncio` to overlap tokenisation, preprocessing, and inference scheduling.
- **Multi-instance:** run N model replicas on N GPUs for roughly linear throughput scaling.

```python
# FastAPI + vLLM async serving pattern
from fastapi import FastAPI
from vllm import AsyncLLMEngine, AsyncEngineArgs

app = FastAPI()
engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model='meta-llama/Llama-3.1-8B-Instruct',
    max_num_seqs=256,         # max concurrent sequences
    max_num_batched_tokens=8192,  # max tokens per iteration
))

@app.post('/generate')
async def generate(request: GenerateRequest):
    async for output in engine.generate(request.prompt, ...):
        pass
    return {'text': output.outputs[0].text}
```

Notes:

- FastAPI is the HTTP front door to vLLM. You can also use **gRPC** or another API layer. What matters is using **async I/O** in whichever framework you pick.
- `max_num_seqs=256`: up to 256 concurrent sequences.
- `max_num_batched_tokens=8192`: up to 8192 tokens per iteration, so tokens from different requests can be overlapped in one iteration.
- The `async for ... in engine.generate(...)` loop is the key piece: it tells the engine output is streamed asynchronously, which is how concurrency is exploited. This improves TTFT/TPOT.

### 3.4 Request scheduling strategies

- **FCFS (First-Come First-Served):** simple, fair; default in most frameworks.
- **Priority scheduling:** SLA-based priority; short requests jump the queue.
- **Chunked prefill (vLLM 0.4+):** break long prefills into chunks to reduce TTFT spikes. It reportedly **halves P99 TTFT** by stopping long prompts from blocking the decode loop.

---

## 4. Operator Fusion and Kernel Optimisation for Inference

### 4.1 Fused kernels reduce memory traffic

- **Unfused:** each op reads from and writes to **HBM**, causing multiple round trips even for simple sequences.
- **Fused:** sequential ops computed in **registers**, giving one HBM read + one HBM write total.
- Fused LayerNorm + Dropout + Add: **3x HBM reads → 1x; 3x HBM writes → 1x**.
- Fused GELU FFN: column-parallel matmul + GELU applied in-register; **no HBM** for the intermediate.

### 4.2 Flash Attention: the inference kernel standard (2024)

- **Flash Attention 3** (Dao et al., 2024): H100-specific, uses **WGMMA** and **TMA** instructions.
- Never materialises the full **T×T** attention score matrix; computation is **tiled in SRAM** (fusing tiles across heads).
- **FA3 FP16 throughput on H100:** up to **740 TFLOPS (~74% of H100 FP16 peak)**.
- **Memory:** **O(T)** instead of **O(T²)**, which enables **128K+ context** serving.
- **Flash Decoding** (Tri Dao, 2023): parallelises attention over the sequence dimension for long KV caches; **~8x speedup vs FA2 at seq=8192**.
- Natively supported by serving stacks to improve KV-cache performance and attention speed.

### 4.3 xFormers and custom kernels

- **Meta's xFormers:** memory-efficient attention and fused ops for Transformers.
- **Triton:** write custom CUDA-like kernels in Python, auto-tuned for the hardware.

```python
# Using Flash Attention in PyTorch (built-in since 2.0)
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False):
    output = F.scaled_dot_product_attention(q, k, v)
```

### 4.4 `torch.compile` for inference: modes

```python
import torch
model = MyModel().eval()

# Default: balanced compile time vs speedup
opt = torch.compile(model, mode='default')

# reduce-overhead: CUDA Graphs, best for fixed-shape inference
opt = torch.compile(model, mode='reduce-overhead')

# max-autotune: exhaustive search (slow compile, fastest runtime)
opt = torch.compile(model, mode='max-autotune')

# For inference: disable gradient computation
with torch.no_grad():
    output = opt(input)
```

| Mode | Trade-off |
|---|---|
| `default` | Balanced compile time vs. speedup |
| `reduce-overhead` | Uses CUDA Graphs; best for fixed-shape inference |
| `max-autotune` | Exhaustive search; slowest compile, fastest runtime |

### 4.5 Speedup from `torch.compile` at inference

| Model | Speedup vs eager |
|---|---|
| Small model (BERT-base) | 1.5-2.0x |
| LLaMA-3-8B (prefill) | 1.3-1.5x |
| LLaMA-3-8B (decode, bs=1) | 2-5x (with CUDA Graphs) |
| Stable Diffusion UNet | 1.8-2.5x |

### 4.6 Weight tying and quantisation fusion

- **Weight tying:** share embedding and LM-head weights; slide claims it saves **512 MB for LLaMA-3-8B**.
- **FP8 matmul fusion:** H100 Transformer Engine computes FP8 matmul and accumulates in FP32 in a **single kernel**.
- **AWQ fused:** quantised-weight dequantisation + matmul fused, giving minimal dequant overhead.

---

## 5. Segment 2 Summary

- Traced/compiled graphs eliminate Python dispatch overhead (~5 µs/op × 1000 ops/step ≈ ~5 ms); critical for low-latency serving. **`torch.export`** is the recommended API for PyTorch 2.x with dynamic shapes.
- **CUDA Graphs** cut per-step CPU overhead from O(n_kernels) to O(1): **~5x speedup at batch=1** where CPU overhead dominates. `torch.compile(mode='reduce-overhead')` enables them automatically.
- **Dynamic batching** with a small queue delay (5-10 ms) improves GPU utilisation by grouping concurrent requests. **Continuous batching (token-level) with chunked prefill** is the 2025 standard for LLM serving.
- **Flash Attention 3** reaches ~740 TFLOPS on H100 FP16 (~74% of peak) and removes the O(T²) attention memory constraint. **Flash Decoding** gives ~8x speedup for long-context KV-cache attention during decode.
- **Operator fusion** reduces HBM traffic from multiple round trips to one per fused block. `torch.compile` applies fusion automatically; Flash Attention, fused LayerNorm, and FP8 matmul are the highest-impact fusions for Transformer inference.

---

## Lecture Clarifications and Cautions

Points where the spoken lecture and slides are loose or may mislead; worth double-checking when studying:

1. **"Default" CUDA Graphs in `torch.compile`:** CUDA Graphs are enabled by `mode='reduce-overhead'`, **not** by the `default` mode.
2. **Trace vs. fusion:** when discussing `jit.trace`, the lecture says it "eliminates the fusion"; what it eliminates is the **Python dispatch overhead** (fusion is an additional graph-level optimisation).
3. **CUDA Graphs speedup numbers:** the slide says ~5x at bs=1 and ~1.2x at bs=64, while the lecture mentions "1.2-1.5x with larger batch sizes"; consistent in spirit: the gain shrinks as batch size grows.
4. **Dynamic batching delay:** the Triton config uses 5 ms (`5000 µs`); the summary says 5-10 ms. Choose the delay based on your SLO.
5. **`torch.jit` status:** `torch.jit.trace`/`script` are in maintenance mode in recent PyTorch; `torch.export` + `torch.compile` are the forward-looking path. Check current PyTorch docs for API details.
6. **Weight tying figure (512 MB for LLaMA-3-8B):** verify against the model config; the saving depends on vocab size, hidden size, and dtype, and LLaMA-3-8B does not tie embeddings by default.
7. **Code run-through:** a demo with an A100 target will be shown in the code section after all segments are covered.

---

## Key Terms

- **Eager execution:** op-by-op Python execution.
- **Tracing:** recording the ops executed for a given example input.
- **TorchScript IR:** the intermediate representation produced by `jit.script`.
- **CUDA Graph:** recorded set of kernel launches replayed with a single launch.
- **SLO:** latency target a service guarantees.
- **TTFT / TPOT:** time to first token / time per output token.
- **HBM / SRAM:** GPU main memory / fast on-chip memory.
- **Prefill / decode:** prompt processing phase / token-by-token generation phase.
- **Chunked prefill:** splitting long prompt processing into chunks so decode isn't blocked.

**Next:** Segment 3.
