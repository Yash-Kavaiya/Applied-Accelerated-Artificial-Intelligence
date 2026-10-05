# Week 06 · Segment 2: Mixed Precision Training and Numerical Stability

**Course:** Applied Accelerated Artificial Intelligence (NPTEL)
**Instructor:** Dr. Satyajit Das, Dept. of CSE, IIT Guwahati

---

## 1. Context and Scope

### Week 06 roadmap (Training Optimization and LLM Fine-tuning)
| Segment | Topic |
|---|---|
| 1 | LLM foundations and terminology |
| **2** | **Mixed precision and numerical stability (this segment)** |
| 3 | Memory optimization and throughput |
| 4 | Transfer learning for large models |
| 5 | Efficient LLM fine-tuning using PEFT and LoRA |

- Segment 1 covered next-token prediction through transformer layers, normalization layers, and probability distribution computation.
- Mixed precision and data types were also covered in the earlier PyTorch week. This segment only revisits them briefly.
- This segment takes a pre-trained model and shows how to retrain it. Adaptation (adapting a pre-trained LLM to a specific domain) comes in later segments.
- **Today's focus:** precision formats, autocast, loss scaling, and the stability tricks that keep LLM training from exploding or silently degrading.

### Framing (slide 2)
- **Why now?** Lower precision is the main reason modern GPUs can train frontier-scale models fast enough and cheaply enough.
- **Key idea:** Use low precision where it is safe. Keep sensitive computations in higher precision.
- **Outcome:** Better throughput and lower memory without sacrificing convergence.

### Why stability matters
- Computing logits via softmax or other activations involves exponentials.
- In reduced precision, results can go to **zero** (underflow) or **infinity** (overflow), depending on which end of the range the value sits at.
- The precise behavior depends on the data type: FP16, BF16, or FP8.

---

## 2. Learning Objectives
1. Explain why mixed precision improves throughput, memory footprint, and memory bandwidth utilization.
2. Distinguish FP32, TF32, BF16, FP16, and FP8 by dynamic range, precision, and typical use.
3. Describe the mixed-precision training loop: autocast, higher-precision accumulation, optimizer update, optional loss scaling.
4. Identify which operations are usually safe in lower precision and which should stay in FP32 or higher.
5. Diagnose failure modes: overflow, underflow, catastrophic cancellation, NaNs, silent accuracy drift.
6. Apply stable formulations: max-subtracted softmax, `log_softmax`, `logsumexp`.
7. Choose between BF16, FP16, and FP8 based on hardware support and model risk profile.
8. Connect mixed precision to the next segment (memory optimization and throughput).

---

## 3. Why Mixed Precision Is No Longer Optional for LLM Training

| Benefit | Detail |
|---|---|
| **Higher throughput** | Tensor Cores deliver much more matrix-multiply throughput with reduced-precision inputs than with plain FP32. |
| **Lower activation memory** | Activations and temporary tensors drop from 4 bytes/value to 2 bytes/value. This directly improves batch size and sequence length headroom. |
| **Less memory traffic** | Lower precision cuts bandwidth pressure. In transformer workloads, moving data is often as expensive as computing on it. |
| **Better energy efficiency** | More work per joule matters for long runs (cost, cooling, cluster utilization). |

**Batch size example from the lecture:** if you fit a batch size of 4 in full precision, halving the bytes per value lets you fit about 8 per step. This improves gradient computation per step and affects the number of iterations needed.

### Weights-only example (7B parameters)
| Precision | Weight storage |
|---|---|
| FP32 | 28 GB |
| BF16 / FP16 | 14 GB |

- Mixed precision halves storage **only for tensors that really live in 16-bit form**.
- Training still needs gradients, optimizer state, and activations.

### Key principles
- The goal is **not** "everything in low precision." The goal is "**the right precision for each operation**."
- **GEMMs and convolutions** are usually good candidates for lower precision (16-bit float gives very good results).
- **Reductions, normalization statistics, and some loss computations** are not good candidates. Reduced dynamic range hurts the loss computation, which feeds the gradients for weights and biases.
- Mixed precision is both a **systems optimization** and a **numerical-analysis problem**.
- Training is not done with one static width of 16 or 32. Some steps use full precision and some use half precision.

> **Rule of thumb:** If a tensor is throughput-critical and numerically well behaved, lower precision is attractive. If it sets dynamic range or accumulates many values, keep more precision.

---

## 4. Floating-Point Formats

| Format | Bits | Exponent / Mantissa | Typical role | Practical note |
|---|---|---|---|---|
| **FP32** | 32 | 8 / 23 | Reference training precision; optimizer states; sensitive accumulations | Stable but slower and heavier on memory/bandwidth |
| **TF32** | 19 (compute mode) | 8 / 10 | Fast matmuls on Ampere+ while keeping FP32 API inputs | Easy speed-up mode; **not a storage dtype** |
| **BF16** | 16 | 8 / 7 | Default mixed-precision training on modern GPUs/TPUs | FP32-like dynamic range makes training much safer than FP16 |
| **FP16** | 16 | 5 / 10 | Legacy/common mixed precision where BF16 is absent or memory pressure is severe | Higher local precision than BF16, but much smaller dynamic range |
| **FP8** (E4M3 / E5M2) | 8 | 4/3 or 5/2 | Emerging transformer training/inference with framework/library support | Needs scaling recipes and mature kernels; not yet the universal default |

### Notes
- **TF32:** a fast Tensor Core execution mode for FP32-style workloads. It keeps FP32's 8-bit exponent but uses a 10-bit mantissa in matmul/convolution paths. It works well on Ampere and Hopper and newer GPUs.
- **Double precision (FP64)** is not usually needed in AI workloads, except for some scientific computations. "Full precision" in this course means **FP32** (single precision).
- **BF16 vs FP16:** both are 16 bits, so both halve the memory. BF16 has an 8-bit exponent (like FP32), so it has a wider dynamic range. FP16 has only a 5-bit exponent.
- BF16 was curated specifically for AI workloads.
- **FP8:** with 8 bits, the exponent has 4 or 5 bits depending on the variant. It usually needs scaling recipes because cutting precision makes gradients vanish quickly. Whenever bits are reduced, scaling has to happen somewhere.

### Where each format is used
- **FP32:** optimizer state, sensitive computations, and accumulations.
- **TF32:** fast matmuls on Ampere/Hopper and newer GPUs.
- **BF16:** default mixed-precision training on modern GPUs and TPUs. **If your stack supports BF16, try it first for large transformers.**
- **FP16:** use it on GPUs older than Ampere, which have no BF16 support. It is also still useful for loss scaling, overflow checks, and architecture-specific validation.
- **FP8:** increasingly important on Hopper/Blackwell-class systems, especially via **Transformer Engine**.

---

## 5. What "Mixed" Means Inside One Training Step

Flow: **Inputs + weights → Autocast region → Sensitive ops → Backward pass → Optimizer step**

| Stage | What happens |
|---|---|
| **Inputs + weights** | Model parameters stay in default precision (FP32) outside the autocast context. |
| **Autocast region** | Matmuls and convolutions run in lower precision (BF16 or FP16), chosen automatically based on what the target GPU supports. |
| **Sensitive ops** | Reductions, normalization statistics, softmax, `log_softmax`, and selected losses stay in FP32. |
| **Backward pass** | Gradients follow the dtypes induced by forward ops. FP16 training may need scaling. |
| **Optimizer step** | The optimizer (Adam, AdamW with weight decay, etc.) typically applies updates in FP32 or another safer accumulator precision. |

**Where the transformer's sensitive operations sit:** layer normalization, attention (multi-head, scaling, dot product with values), logits, MLP, softmax.

### Autocast rules
- Autocast is an **op-level policy, not a blanket cast** of the whole model.
- Do **not** manually call `model.half()` or `model.bfloat16()` when relying on autocast, unless you intentionally want a pure low-precision model. Autocast analyzes the capability of the hardware (streaming multiprocessors) and the kernels being launched, then applies the precision automatically.
- Autocast needs a **region specification** around the forward pass and loss, not around everything.
- **Backward should generally run outside the autocast context.** PyTorch chooses backward dtypes from the forward path.
- **Core idea:** low precision where the hardware is fast (matmuls, convolutions), higher precision where the math is fragile (loss, activations, reductions).

---

## 6. BF16 vs FP16: Same Storage, Different Behavior

| Format | Sign | Exponent | Mantissa | What it buys you |
|---|---|---|---|---|
| BF16 | 1 | 8 | 7 | FP32-like range |
| FP16 | 1 | 5 | 10 | Finer local precision |

- **Range example:** a value like `1e5` overflows in FP16 but fits naturally in BF16, which keeps FP32's 8-bit exponent.
- **Precision example:** around numbers near 1.0, FP16 has a finer mantissa, so tiny relative differences are represented more accurately.
- **Takeaway:** BF16 trades mantissa precision for dynamic range. For large-model training, that is usually the safer trade.

### Concrete gradient example
- Suppose a backward gradient is about **2^-25 ≈ 2.98×10^-8**.
- In FP16 this is below the smallest subnormal scale and **flushes to zero** (underflow).
- In BF16 the dynamic range is far wider, so the value is representable.
- *(Note: the auto-generated transcript says "2^-255", which is a transcription error. The slide says 2^-25.)*

### Why BF16 is often the default
- Same storage as FP16.
- Much wider exponent range.
- Far fewer overflow surprises in LLM training.
- Often no GradScaler is needed.

### When FP16 is still attractive
- Hardware/library support is mature.
- Some workloads value extra mantissa bits.
- It is still common in legacy codebases and inference paths.
- You are on a pre-Ampere GPU (no BF16 support), or your memory budget prefers it.

### Operational recommendation
- On A100/H100/Blackwell-class hardware, **try BF16 first**. Move to FP16 only when your stack or memory budget strongly prefers it.
- **Always check first whether your target GPU/TPU supports BF16.**
- If your target does not support the full BF16 range, you must apply a scaling mechanism. Otherwise you get underflow/overflow, with values becoming 0, inf, or NaN.

---

## 7. Numerical Failure Modes

### 7.1 Overflow
- **Symptoms:** `inf` / `NaN` in loss or activations; logits blow up before softmax.
- **Example:** `exp(1000)` is not representable in FP16 or FP32, so a naive softmax overflows immediately and gives NaN.
- **Fixes:**
  - Subtract the max before softmax.
  - Prefer fused attention / `log_softmax` kernels.
  - Choose BF16 over FP16 where possible.

### 7.2 Underflow
- **Symptoms:** tiny gradients become exactly zero; learning stalls silently in some layers. No weights are updated because the gradients vanish.
- **Example:** very small FP16 gradients in the backward pass flush to zero. This is why loss scaling exists.
- **Fixes:**
  - GradScaler for FP16.
  - BF16 when available.
  - Monitor gradient norms and skipped steps.

### 7.3 Cancellation / bad reductions
- **Symptoms:** unstable norms, variances, or probabilities; tiny accuracy drift that compounds over long training.
- **Example:** subtracting two close values, or summing many small values, in low precision loses meaningful digits.
- **Fixes:**
  - Accumulate in FP32.
  - Keep layer-norm stats and reductions in higher precision.
  - Use stable library ops instead of hand-written formulas.

> **Remember:** instability is not always dramatic. Sometimes the model does not explode; it just converges worse.

### How to detect problems while training
- `inf` / `NaN` in loss, activations, or logits → overflow.
- Gradients become exactly zero and learning saturates → underflow.
- Fluctuating variances/probabilities and slow drift → bad reductions.
- These issues are especially visible in **LLMs/transformers** because of their size. They are less of a concern in smaller networks such as CNNs and RNNs.

---

## 8. Stable Formulas and Safe Transformer Patterns

### Max-subtracted softmax
- Naive logits: `[1000, 1001, 999]`
- Subtract max → `[-1, 0, -2]`
- Gives **mathematically identical** probabilities but avoids huge exponentials.
- Formula: `softmax(x_i) = exp(x_i − max(x)) / Σ_j exp(x_j − max(x))`

### Use `log_softmax`
- Prefer `log_softmax(x)` over `log(softmax(x))`. PyTorch documents the separate form as slower and numerically unstable.

### Use `logsumexp`
- When a loss or normalization contains `log(sum(exp(·)))`, call the stable library primitive instead of composing exp + sum + log by hand.

### Transformer-specific safe patterns
- Scale attention scores by **1/√d_k** before softmax.
- Use **fused attention kernels** when available.
- Keep normalization statistics and important reductions in FP32.

### Autocast already helps (if you let it)
- Under CUDA autocast, PyTorch sends many numerically sensitive ops to **float32**: `softmax`, `log_softmax`, `cross_entropy`, `layer_norm`, `sum`.
- If you write code outside autocast's scope, you must apply the stabilization yourself.

### Engineering habit
- When writing custom kernels or custom autograd functions, identify which intermediate values need more precision (for example, matmul vs convolution vs other ops).
- **The biggest bugs usually happen when a fused path drops a carefully upcast reduction back to low precision.**

---

## 9. Loss Scaling: the Classic FP16 Stabilization Technique

Used when training in FP16, for example on GPUs older than Ampere.

### Steps
1. **Compute loss L.** The forward pass runs under autocast.
2. **Multiply by scale `s`.** Tiny gradients become larger before backward.
3. **Backward on `s·L`.** Gradients are computed in the scaled domain.
4. **Unscale gradients.** Undo the scaling before clipping or the optimizer update.
5. **Step or skip.** If inf/NaN appears, skip the optimizer step and **reduce `s`**.

### Concrete intuition
- A gradient of 2^-25 is too small for FP16 and may flush to zero.
- Scale the loss by 2^10, and the gradient becomes **2^-15** during backward, which is much easier to represent.
- After unscaling, the intended update magnitude is recovered.

> **Important:** this is mainly an FP16 story. BF16 usually does not need dynamic loss scaling because its exponent range is much wider.

---

## 10. A Practical PyTorch 2.x Mixed-Precision Recipe

```python
use_fp16 = False                 # set True only if you really want FP16
mp_dtype = torch.bfloat16        # torch.float16 for FP16 path
scaler   = torch.amp.GradScaler("cuda", enabled=use_fp16)

for x, y in loader:
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", dtype=mp_dtype):
        logits = model(x)
        loss   = loss_fn(logits, y)

    if use_fp16:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)            # before clip_grad_norm_
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

### Do / Avoid
- **Do:** keep the model in default precision and let autocast choose dtypes op by op.
- **Do:** use GradScaler only for FP16-style training. BF16 usually just needs autocast.
- **Do:** unscale before gradient clipping, anomaly checks, or any logic that inspects gradient magnitudes.
- **Avoid:** blindly calling `half()` on the full model when you really want automatic mixed precision.

---

## 11. LLM-Specific Nuances

- **Attention scores can explode.** QKᵀ combines many products (for example 1024×1024 matrices). With long contexts or sharp activations, logits grow large before softmax. This is why scaling and fused kernels matter. It is less relevant for CNN-type networks.
- **LayerNorm / RMSNorm need careful statistics.** Means, variances, and RMS lose accuracy if all intermediate reductions stay in low precision.
- **Long sequences amplify error.** More timesteps mean more reductions, more attention work, and more chances for small rounding errors to accumulate.

### Common transformer block precision policy
| Component | Precision |
|---|---|
| Linear / QKV / MLP GEMMs | BF16 / FP16 / FP8 |
| Attention softmax and log-probabilities | often FP32 |
| Norm statistics | often FP32 |
| Optimizer moments | usually FP32 |
| Final output tensor | downcast if safe |

### Debugging tips
- Use proven kernels for scaled dot-product attention. They already encode many stability choices.
- If NaNs appear **only at long sequence lengths**, inspect attention score magnitudes and reduction precision first.
- Compare a failing mixed-precision run against a short **FP32 or BF16 baseline** rather than guessing layer by layer.

---

## 12. Choosing a Precision Mode

| Situation | Recommended start | Why |
|---|---|---|
| Modern GPU with BF16 (A100/H100/Blackwell) | **BF16 mixed precision** | Best balance of speed, memory, dynamic range |
| Legacy CUDA path or FP16-tuned stack | **FP16 + GradScaler** | Still fast, needs more numerical care |
| Easy speed-up with minimal code change on Ampere+ | **Enable TF32 matmul** | Faster Tensor Core matmuls while staying in FP32 APIs |
| Validated Hopper/Ada/Blackwell stack with Transformer Engine | **FP8 for selected blocks** | Extra throughput/memory gains, needs scaling recipes and robust kernels |
| Debugging unexplained NaNs or convergence loss | **Temporarily increase precision** | Fastest way to separate model bugs from precision bugs |

- **Simple decision rule:** default to BF16 if supported. Use FP16 when compatibility demands it. Use FP8 only with a validated library stack.
- **Debugging rule:** if a run fails only in one precision mode, compare gradients, skipped steps, and loss curves against a safer baseline.
- **Systems rule:** precision choice is tied to hardware, kernels, communication strategy, optimizer, and checkpointing, not just the math.

---

## 13. Code Demo Walkthrough (Colab)

### Setup
- Install `transformers`. Import `AutoTokenizer` and `AutoModelForCausalLM`.
- Runtime options: **T4 GPU** (free, used in the demo), H100/other GPUs, or TPUs (a high-end TPU is not free).
- Use "Change runtime type" to select the device.

### Hardware capability check
- The code queries the GPU name and device capability (here: **Tesla T4**).
- It checks whether the GPU supports BF16 (compute capability major version ≥ 8, i.e., Ampere+). If yes, it uses BF16. Otherwise it falls back to FP16.
- **Result on T4:** FP16 supported, **BF16 not supported**. On an H100 (Hopper), BF16 would be supported.

### Naive vs stable softmax demo
- Naive softmax: `exp(x) / sum(exp(x))` with large values gives **NaN everywhere**.
- Stable softmax: compute `x − x.max()` first, then exponentiate. This gives valid values.
- `torch.softmax` already performs the stable computation internally.

### Dataset
- A tiny hand-made **instruction dataset** of question/answer pairs.
- Format given to the LLM: an **instruction** followed by the desired **response**.
- Example instruction: *"Explain mixed precision training in one paragraph."*

### Model and hyperparameters
| Setting | Value |
|---|---|
| Model | `distilgpt2` (a very tiny causal LM) |
| Max context length | 128 |
| Batch size | 4 |
| Gradient accumulation steps | 4 (do not update after every batch; accumulate over 4 micro-batches, which reduces memory requirements) |
| Max updates | 30 |

- **Causal LM:** GPT-like model trained to predict the next token from the previous tokens.
- The tokenizer's vocabulary and auxiliary files are downloaded automatically.

### Data pipeline
- **Dataset class** ("the warehouse") holds the samples.
- **Collator** ("the packaging") tokenizes the text, copies input IDs as labels, and pads shorter samples to a fixed length (`padding=True`). Padded positions are masked/skipped in the loss.
- **DataLoader** ("the conveyor belt") feeds batches at the defined batch size.

### Baseline check
- Before training, `generate_text` is run with the instruction prompt.
- The untrained model produces meaningless output, as expected.

### Training loop (where mixed precision is applied)
- Model moved to device; optimizer defined.
- Warm-up steps followed by a learning-rate scheduler.
- **GradScaler** created (enabled only for the FP16 path).
- History lists track loss, grad norm, scale value, etc. Peak GPU memory (GB) is tracked.
- Loop logic:
  1. For each batch, enter an `autocast` context (a null context if CUDA is unavailable).
  2. Forward pass and loss, divided by the gradient accumulation steps.
  3. If the scaler is enabled: scaled backward. Otherwise: normal `loss.backward()`.
  4. Every 4 micro-steps: **unscale**, clip gradients, run the **optimizer step**, then the **scheduler step**, then `zero_grad`.
  5. Record update, loss, gradient norm, and scale value.
- Logged at updates 1, 5, 10, 20, 25, 30.

### Results
- After 30 updates, the model generates the learned definition of mixed precision training.
- The loss curve drops to **almost zero** over the update steps.

---

## 14. Summary
- Mixed precision works because **different operations tolerate different numerical error**. The win comes from matching precision to the operation, not from blindly casting everything down.
- **BF16** is usually the safest 16-bit training default on modern accelerators (FP32-like range). **FP16** remains useful but often needs loss scaling and closer monitoring.
- Stability depends on stable formulations: subtract-max softmax, `log_softmax`, `logsumexp`, higher-precision reductions, and sensible attention/norm kernels.
- Autocast, GradScaler, and library kernels encode many best practices, but **custom code can easily bypass them**.
- **Ladders:**
  - **Precision:** FP32 → BF16/FP16 → FP8. Move downward only when the stack is ready.
  - **Stability:** NaNs / infs / silent drift. Start debugging from the most sensitive reductions.
  - **Engineering:** measure speed, memory, and convergence together. Optimizing only one is not enough.

### Coming up
- **Segment 3:** memory optimization and throughput (fitting larger batches and longer contexts on the same hardware, scaling to any GPU, distributed computing for LLMs).
- **Later segments:** transfer learning, fine-tuning, adaptation (PEFT/LoRA).
