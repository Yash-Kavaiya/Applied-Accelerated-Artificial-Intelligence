# Week 9, Segment 1: Inference Acceleration Fundamentals

*Why inference differs from training, where latency comes from, and how hardware is matched to AI serving workloads*

## Learning Objectives

1. Distinguish the inference compute pattern from training: batch size 1 vs large, memory-bandwidth-bound vs compute-bound.
2. Define and calculate latency, throughput, and TTFT/TPOT metrics for a production LLM serving system.
3. Identify the five main sources of inference latency: model loading, tokenisation, prefill, decode, and post-processing.
4. Explain why the KV cache is the dominant memory consumer during LLM decode, and calculate its size.
5. Match inference workload characteristics to GPU/CPU/NPU hardware capabilities, and explain why H100 NVL is preferred for LLM serving.

## 1. Scope of This Segment

This week (Week 9) is dedicated to **inference** for large language models and the surrounding serving/tool-chain ecosystem, as a counterpart to the earlier dedicated week on **accelerated training**.

- Training-specific code (training loop, evaluation loop, validation loop) is **not** covered here — refer back to the training sessions for that.
- This segment focuses on what actually happens when an LLM is **deployed** for a specific purpose: chatbots, summarization, prediction tasks, or as a component in agentic AI systems where the LLM is connected to other tools.
- Segment 1 covers **inference acceleration fundamentals**. A later segment covers **graph tracing**, other tool chains, and further optimization techniques.

Two central themes for this segment:
- **Where does inference latency come from?** (unlike training, we don't care about loss computation, backward pass, gradient updates, or optimizer memory during inference)
- **Hardware and metrics**: what metrics define good serving performance, and the latency/throughput tradeoff in production.

## 2. Training vs Inference: Two Fundamentally Different Compute Patterns

### 2.1 Training Characteristics

| Aspect | Training |
|---|---|
| Batch size | Large: 64–4096 samples — maximizes GPU FLOP utilization |
| Computation | Deterministic — same compute graph every step (same data types/dimensions each iteration) |
| Compute regime | Compute-bound: weights accessed once, large matmuls fill Tensor Cores |
| Throughput metric | Samples/sec or tokens/sec (overall) |
| Latency tolerance | High — users wait seconds to minutes |
| Parallelism | Data, tensor, and pipeline parallelism across many GPUs (distributed computing) |

**Key training idea:** because the same compute graph is reused every iteration with fixed shapes, and batches are large, you can extract high GPU utilization by getting all the weights into large matrix multiplications that fill the Tensor Cores.

### 2.2 Inference Characteristics

| Aspect | Inference |
|---|---|
| Batch size | 1 for **online serving** (e.g., ChatGPT-style single prompt/response); up to ~256 for **offline batch serving** |
| Computation | Dynamic — sequence length and routing vary per request |
| Compute regime | Often memory-bandwidth-bound (especially decode) |
| Decoding | Autoregressive: one token per forward pass — serial dependency |
| Latency metrics | TTFT, TPOT, P50/P99 latency — users feel every millisecond |
| Throughput metric | Tokens/sec **at a target latency SLO** |
| Memory | Weights + KV cache must fit together in GPU HBM |

**Online vs. offline serving:**
- *Online serving* — e.g. typing a prompt into ChatGPT and expecting an immediate response. You send **one prompt at a time**, not batches. This is fundamentally different from training's ability to exploit large-batch parallelism.
- *Offline batching* — e.g. summarizing a large number of documents overnight, where all documents are ready ahead of time and there's a looser time bound. Still typically a smaller batch size than training.

Because online serving uses batch size 1, you **cannot** extract the same parallelism you had during training. Additionally:
- Prompts vary in length (sequence length varies per request) — sometimes a full document, sometimes just "yes" or "no."
- **Autoregressive decoding**: generating the next token happens sequentially, one token per forward pass. Each new token is appended to the sequence and used to predict the next token — this creates a serial dependency that cannot be parallelized away like training batches can.

*Reference cited on slides: Pope et al., "Efficiently Scaling Transformer Inference," MLSys 2023.*

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/90815530-5fde-4f66-b86d-e487fb5e5477" />

## 3. Key Inference Performance Metrics

Four/five main metrics matter for characterizing how well an inference system is performing:

- **TTFT (Time To First Token)** — latency of the **prefill** phase; this is the user-facing responsiveness metric (how long before anything starts appearing).
- **TPOT (Time Per Output Token)** — the **decode** step latency; determines the perceived "streaming speed" of the response.
- **P50 / P99 latency** — median and tail latency. SLO (Service Level Objective) contracts are usually defined on **P99**, not the average, because tail latency is what determines whether the system reliably meets its guarantees.
- **Throughput** — tokens/sec computed across all concurrent requests, measured **at the target latency SLO** (not throughput alone — it only counts if the latency contract is also met).
- **QPS (Queries Per Second)** — a system-level capacity metric.
- **MFU (Model FLOP Utilization)** — efficiency relative to the hardware's theoretical peak FLOP rate. Since the hardware peak is fixed by the batch size regime you're in, MFU tells you how much of that ceiling you're actually capturing.

TTFT is generally slower than the per-token latency during decode, because prefill requires computing key/value pairs for the *entire* prompt at once. Once the KV cache exists, subsequent (decode-phase) tokens don't need to recompute the KV pairs of earlier tokens — this is what keeps per-token decode latency lower than the initial prefill latency.

---

## 4. Prefill vs Decode: The Two Phases of LLM Inference

LLM inference happens in two distinct phases with very different compute characteristics:

### 4.1 Prefill
- Processes the **entire input prompt in parallel** in a single forward pass.
- Operates on a large batch of tokens at once → **compute-bound**.
- Fast, because parallelism can be fully exploited (similar in spirit to a training-style large-batch forward pass).
- Example: for a 1000-token prompt, prefill takes roughly **~20 ms** on an H100.

### 4.2 Decode
- Generates output tokens **one at a time**, autoregressively.
- Strictly **serial** — each token depends on all previous tokens.
- **Memory-bandwidth-bound**, not compute-bound (explained in detail below).
- Slower per-token relative to prefill's per-token cost, because parallelism cannot be exploited.
- Example: for that same request, generating 200 output tokens takes roughly **~200 ms** on an H100.

### 4.3 Causal Attention Refresher (why KV cache matters)
For a sequence of tokens (e.g., A, B, C, D):
- Every token is projected into **key** and **value** vectors, and a **query** is generated for each token.
- **Causal masking**: token *D* (4th token) attends to *all* preceding tokens (A, B, C, D). Token *A* (1st token) attends **only to itself**.
- This attention pattern is what multi-head attention is exploiting — capturing many different relational patterns between tokens.

When a new token (say, token *E*) is generated during decode:
- Its key/value pair is computed.
- Its query must attend to **all previous** key/value pairs (A through D), plus itself.
- If you recomputed the K/V pairs for *every* previous token at *every* decode step, this would take longer and longer, and require more memory bandwidth, as the sequence grows.

**This is exactly why the KV cache exists**: previously computed key/value pairs are stored in memory so they don't need to be recomputed at every decode step. The new query only needs to attend to the cached K/V pairs plus the new token's own K/V.

This is one of the biggest structural differences from training: during training, there's no need to maintain a KV cache, because the backward pass recomputes/uses activation gradients, not a persistent decode-time cache.

---

## 5. The Memory Bandwidth Bottleneck in Decode

### 5.1 Arithmetic Intensity

**Arithmetic intensity** = FLOPs performed ÷ bytes of data loaded. It measures how much compute you're doing per byte moved from memory.

- High arithmetic intensity (ratio → close to hardware's "ridge point") → **compute-bound**.
- Low arithmetic intensity → **memory-bound** (you spend more time loading data than computing on it).

### 5.2 Why Decode at Batch=1 Is Memory-Bound

At batch size 1, a single decode step:
- Requires loading **all** model weights from High Bandwidth Memory (HBM), even though the actual compute (one token forward pass) is tiny.
- Example, using a 7B-parameter model at 2 bytes/parameter (BF16):
  - **Weight read**: 7B × 2 bytes = **14 GB per decode step**
  - **Compute**: 7B × 2 FLOPs = **14 GFLOP at batch=1**
  - **Arithmetic intensity** = 14 GFLOP / 14 GB = **1 FLOP/byte**

### 5.3 Hardware Ridge Point Comparison

- **H100 peak**: 3.35 TB/s HBM bandwidth, 3958 TFLOPS (FP8) → ridge point = **1181 FLOP/byte**
- At an arithmetic intensity of 1 FLOP/byte (far below the ridge point of 1181), the workload is **completely memory-bandwidth-bound** — the GPU sits at only about **0.08% of its peak FLOP capability**.

### 5.4 Batching Is the Fix
- **Batching is essential**: increasing batch size raises arithmetic intensity, because weights are loaded once but reused across more tokens' worth of compute.
- Increasing batch from 1 → 64 raises arithmetic intensity **64-fold**, which correspondingly raises achievable throughput (up to the point where the workload becomes compute-bound instead).
- Concretely, going from batch 1 to 64 was cited as raising the H100's utilization roughly **16-fold** in this scenario (moving it much closer to being compute-bound).

**Bottom line:** for batch=1 decode, you must load *all* the weights just to generate a single token — this is why decode is fundamentally different (and harder to make efficient) than training or prefill.

---

## 6. KV Cache Sizing

### 6.1 Formula

```
KV cache size = 2 × n_layers × n_heads × d_head × seq_len × bytes_per_value
```

The factor of 2 accounts for storing **both** keys and values.

### 6.2 Worked Example: LLaMA-3-70B

- Model: LLaMA-3, ~70B parameters
- Sequence length: 8,192
- Data type: BF16 (2 bytes)
- Calculation: 2 × 80 (layers) × 64 (heads) × 128 (head dim) × 8192 (seq len) × 2 (bytes) = **~21 GB per request**

### 6.3 Why This Matters for Capacity Planning

- On an **H100 80GB**, if model weights take ~140 GB spread across 2 GPUs (70 GB per GPU), that leaves only about **~20 GB per GPU** for KV cache.
- Since KV cache is required **per request**, and each request at this configuration needs ~21 GB, capacity is extremely tight — you can barely fit *one* request's KV cache per GPU in this example.
- **KV cache is the primary constraint on maximum batch size and context length**, not just weight storage.
- KV cache size scales **multiplicatively** with context length: doubling the sequence length doubles the KV cache size for every request. As sequence length grows, KV cache pressure grows right along with it — this creates a hard resource-matching problem between workload demands (context length, batch size, number of concurrent users/requests) and available hardware memory.
- Note: requests can come from **different users** in a shared serving environment, further multiplying total KV cache demand.

*Reference cited on slides: Pope et al., "Efficiently Scaling Transformer Inference," MLSys 2023.*

---

## 7. Inference Hardware: Matching Workload to Accelerator

### 7.1 GPU: The Dominant Inference Platform

| GPU | Bandwidth | Memory | Notes |
|---|---|---|---|
| NVIDIA H100 SXM5 | 3.35 TB/s | 80 GB (HBM3) | Primary LLM serving GPU |
| NVIDIA H100 NVL | — | 188 GB total (two H100 dies) | Fits a 70B model **without** tensor parallelism |
| NVIDIA L40S | 864 GB/s | 48 GB (GDDR6) | Cost-efficient for smaller models |
| NVIDIA A10G (cloud) | 600 GB/s | 24 GB | Common in AWS/GCP instances |
| NVIDIA RTX 4090 | 1 TB/s | 24 GB (GDDR6X) | Consumer-grade; viable for 7B models locally |

### 7.2 Why HBM Bandwidth Matters More Than FLOPS for Inference

- Decode throughput ≈ **min**(HBM_bandwidth / model_size, peak_FLOPS / arithmetic_intensity)
- Example: H100 → 3.35 TB/s ÷ 140 GB (70B model, BF16) = **24 tokens/s theoretical max per request at batch size 1**.
- Increasing batch size to 32 raises throughput roughly proportionally, **until** the workload transitions to being compute-bound.
- **Conclusion**: for LLM serving, memory bandwidth — not FLOPS — is the primary hardware metric to optimize for.

### 7.3 Specialised AI Inference Accelerators (2024–2025)

| Accelerator | Bandwidth | Memory | Notes |
|---|---|---|---|
| Google TPU v5p | 459 TB/s HBM2e | 96 GB | Dominant for Google's internal serving (e.g., Gemini) |
| AWS Inferentia2 | 2.3 TB/s | 32 GB | Cost-efficient for Neuron-compiled models |
| AMD MI300X | 5.3 TB/s HBM3 | 192 GB | Growing LLM inference adoption |
| Groq LPU | SRAM-based, no HBM | — | Extremely low latency; limited model size |

These specialized accelerators are widely used in the servers hosting large deployed models (e.g., Gemini, Grok, and other customized LLMs).

*Reference cited on slides: Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention," SOSP 2023.*

### 7.4 CPU Inference: When and Why

- CPU inference is viable for: **models under 2B parameters**, batch size = 1, latency-insensitive workloads.
- **llama.cpp**: highly optimized CPU inference using AVX-512 / AVX2 SIMD instructions.
- **Apple M-series**: unified memory up to 192 GB — can fit 70B models at 4-bit quantization in RAM.
- **Intel AMX** (Advanced Matrix Extensions): dedicated matrix multiplication hardware on Sapphire Rapids CPUs.
- **Intel Gaudi 2**: 300 GB/s HBM2e, 96 GB — competitive with A100 for inference.

### 7.5 The Serving Latency Stack (End-to-End Breakdown)

A concrete latency budget for a 200-token response at batch size 1 on an H100:

| Stage | Approx. Latency |
|---|---|
| 1. Network: client → load balancer | ~1 ms |
| 2. Request queue: waiting for a GPU slot | ~0–100 ms (load-dependent) |
| 3. Tokenisation: BPE/SentencePiece encode | ~0.1 ms for 1K tokens |
| 4. Prefill: process prompt on GPU | ~20 ms for 1K tokens on H100 |
| 5. Decode: autoregressive generation | ~5 ms/token on H100, batch=1 |
| 6. Detokenisation + postprocessing | ~0.1 ms |
| 7. Network: response → client | ~1 ms |
| **Total P99 (200-token response, bs=1)** | **~1.1 seconds on H100** |

### 7.6 Key Optimisation Levers (Preview)

These four techniques are the primary levers modern serving stacks use to close the gap between naive inference and production-grade throughput/latency:

- **Continuous batching**: removes idle GPU gaps between requests — up to **10–23× throughput gain**.
- **PagedAttention**: eliminates KV cache fragmentation — enables **3–4× more concurrent requests**.
- **Speculative decoding**: up to **3× TTFT/TPOT improvement** at the same hardware cost.
- **Quantization**: 4-bit quantization enables **4× more concurrent requests** in the same VRAM footprint.

---

## 8. PagedAttention and Continuous Batching: The Foundation of Modern LLM Serving

### 8.1 The Problem: KV Cache Fragmentation

- Traditional serving **pre-allocates** KV cache for `max_seq_len` tokens per request — this wastes **60–80% of GPU memory** on average, because most requests don't actually use the full maximum length.
- Example: if `max_seq_len = 4096` and the average output is 200 tokens, **95% of allocated KV memory is never used**.
- This approach also cannot handle dynamic output lengths gracefully — it must reserve the worst case up front, for every request.

### 8.2 PagedAttention (vLLM, Kwon et al., SOSP 2023)

Directly analogous to virtual memory paging in operating systems:
- Divides the KV cache into **fixed-size pages (blocks) of 16 tokens each**.
- Each request's KV cache becomes a **logical sequence of non-contiguous physical pages**.
- Physical pages are **allocated on demand** as tokens are generated — no pre-allocation waste.
- **Sharing**: multiple requests sampling from the same prefix can **share physical pages** (this is the basis of prefix caching), exploiting temporal and spatial locality of KV cache usage.
- **Result**: 3–4× more concurrent requests on the same GPU, at the same latency, by eliminating the repeated loading/storing/re-storing of fragmented KV memory.

Example vLLM usage (PagedAttention is on by default):

```python
# vLLM uses PagedAttention by default
from vllm import LLM, SamplingParams

llm = LLM(model='meta-llama/Llama-3-8B', gpu_memory_utilization=0.90)
outputs = llm.generate(['What is ML?'], SamplingParams(max_tokens=200))
```

### 8.3 Continuous Batching (Orca, Yu et al., OSDI 2022)

- **Traditional static batching**: the system waits for **all** requests in a batch to finish before starting new ones. (Analogy from the lecture: a bus that doesn't leave the station until every seat is filled — the GPU sits idle waiting for the slowest request in the batch to complete.)
- **Continuous batching**: new requests are inserted into the running batch **at the token level**. As soon as a "slot" frees up (one sequence finishes), a new request begins decoding immediately — no waiting for the whole batch to complete.
- Each scheduling iteration, the system checks for completed sequences and adds new ones in their place.
- **Result**: GPU utilization rises from roughly **~30% to 70–90%**; vLLM reports **10–23× throughput improvement** from this technique alone.

### 8.4 Speculative Decoding: Hiding Decode Latency

- A small, fast **draft model** generates **K candidate tokens** ahead of time.
- The large **target model** verifies all K candidates in a **single pass**.
- If tokens are accepted: you effectively get K tokens generated for the cost of **1 target-model step + K draft-model steps** — much cheaper than K full target-model steps.
- Typical acceptance rate ~80% translates to roughly a **2–3× effective speedup** on H100-class hardware.
- **Works best** when the draft and target models **share vocabulary and embeddings** (e.g., using LLaMA-3-8B as the draft model for LLaMA-3-70B as the target) — because the smaller model's token predictions are more likely to genuinely overlap with what the larger model would have produced.

*Note: vLLM, TensorRT-LLM, and SGLang all implement PagedAttention + continuous batching as their baseline.*

---

## 9. Inference Serving Frameworks: vLLM, TGI, and SGLang

Because implementing all of the above optimizations (paged attention, continuous batching, speculative decoding, quantization support) from scratch for every model would be highly inefficient, production LLM deployments rely on dedicated **serving stacks** — software frameworks purpose-built to serve LLMs in production.

### 9.1 vLLM (UC Berkeley, 2023–2025)

- Core innovations: **PagedAttention + continuous batching**.
- Supports LLaMA, Mistral, Qwen, Gemma, DeepSeek, and 100+ HuggingFace models.
- **Tensor parallelism** via a Megatron-LM-style approach, for multi-GPU serving.
- Provides an **OpenAI-compatible API server** — a drop-in replacement for the GPT-4 API.

Launching a vLLM OpenAI-compatible server:

```bash
# Launch vLLM OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

### 9.2 HuggingFace TGI (Text Generation Inference)

- Battle-tested at HuggingFace scale — powers the HuggingFace Inference API.
- Combines **Flash Attention 2 + continuous batching + tensor parallelism**.
- Docker-native deployment, with **Prometheus metrics out of the box**.
- **Best for**: HuggingFace Hub model deployment, production reliability.

```bash
docker run --gpus all ghcr.io/huggingface/text-generation-inference \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --num-shard 1 --max-input-length 4096
```

### 9.3 SGLang (Stanford, 2024)

- **RadixAttention**: automatic KV cache reuse across requests that share a common prefix.
- Supports **structured generation** (JSON/regex-constrained decoding) without overhead.
- **Runtime parallelism**: can pipeline multiple generation calls within a single program.
- Reports up to **5× faster than vLLM** specifically on **prefix-heavy workloads** (RAG, few-shot prompting).

```python
import sglang as sgl

@sgl.function
def multi_turn_chat(s, question):
    s += sgl.system('You are a helpful assistant')
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen('answer', max_tokens=200))
```

### 9.4 Framework Selection Guide (2025)

| Framework | Best For |
|---|---|
| **vLLM** | General-purpose LLM serving; good default choice; widest model support |
| **TGI** | HuggingFace ecosystem; production reliability; Docker/k8s native |
| **SGLang** | Prefix-heavy workloads (RAG, agents); structured output; research |
| **TensorRT-LLM** | Maximum throughput on NVIDIA hardware; requires a compilation step |
| **Ollama** | Local development; consumer hardware; ease of use over raw performance |

*Reference cited on slides: Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS 2024.*

---

## 10. Production Deployment Stack

A full production LLM serving stack typically layers the following around the core inference engine:

- **Load balancer** (NGINX / HAProxy) → routes requests to → **serving framework** → **GPU pool**.
- **Autoscaling**: Kubernetes HPA (Horizontal Pod Autoscaler) + KEDA, scaling based on queue depth or GPU utilization.
- **Observability**: Prometheus (for latency/throughput metrics) + Grafana dashboards, for monitoring model performance, autoscaling behavior, and load balancing effectiveness.

The general point: you do not want to hand-code every one of these concerns (paging, batching, load balancing, autoscaling, observability) for every model you serve — mature serving stacks like vLLM, TGI, and SGLang, combined with standard MLOps tooling (Prometheus/Grafana, Kubernetes autoscalers), handle this so you don't have to reimplement it per deployment.

---

## 11. Segment 1 Summary

**Compute pattern & bottleneck:**
Inference is memory-bandwidth-bound at decode time (batch=1, arithmetic intensity = 1 FLOP/byte vs. H100's ridge point of 1181 FLOP/byte). Batching is the primary throughput lever — raising batch size 64× raises compute utilization roughly proportionally.

**Key metrics:**
TTFT (prefill latency), TPOT (per-token decode latency), P99 latency, and throughput (tokens/sec at SLO) are the metrics that define serving quality. KV cache size = 2 × n_layers × n_heads × d_head × seq_len × bytes — this comes out to **21 GB per request** for LLaMA-3-70B at seq_len=8192, BF16.

**Optimization techniques:**
PagedAttention (as implemented in vLLM) eliminates KV cache fragmentation via OS-style paging, enabling 3–4× more concurrent requests on the same GPU. Continuous batching removes idle GPU time between requests, giving a 10–23× throughput improvement over static batching.

**Hardware:**
H100 NVL (188 GB combined) is the preferred serving GPU for 70B-class models because it fits the full model **without** needing tensor parallelism. AMD MI300X (192 GB, 5.3 TB/s) is the leading alternative. Across all hardware choices, **memory bandwidth — not FLOPS — is what determines LLM serving capacity**.

**Software stack:**
vLLM is the dominant open-source serving framework overall. SGLang leads specifically for prefix-heavy workloads (RAG, agents) via RadixAttention. TGI is preferred for HuggingFace-ecosystem production deployments. TensorRT-LLM maximizes raw throughput on NVIDIA hardware at the cost of a required compilation step.

---

*Sources referenced across slides: Pope et al., "Efficiently Scaling Transformer Inference" (MLSys 2023); Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" (SOSP 2023); Yu et al., "Orca" (OSDI 2022); Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (NeurIPS 2024).*
