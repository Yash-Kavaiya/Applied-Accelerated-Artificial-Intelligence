# LLM Foundations & Terminology — Week 6, Segment 1

This lecture is **Segment 1 of 5** for the week:

| Segment | Topic |
|---|---|
| **1 (this lecture)** | LLM foundations and terminology |
| 2 | Mixed precision and numerical stability |
| 3 | Memory optimization and throughput |
| 4 | Transfer learning for large models |
| 5 | Efficient LLM fine-tuning using PEFT and LoRA |

## Learning Objectives

1. Define an LLM and distinguish pretraining, adaptation, and inference.
2. Explain the tokenizer → embedding → transformer → logits pipeline.
3. Describe the role of masked self-attention in decoder-only models.
4. Relate next-token prediction to large-scale pretraining.
5. Explain why pretrained LLMs still need task or domain adaptation.
6. Identify the main compute and memory bottlenecks in LLM training.
7. Compare prompting, RAG, full fine-tuning, and PEFT at a high level.
8. Connect these foundations to the remaining Week 6 segments.

## 1. What Makes a Model an "LLM"?

An LLM is **not** just "a very big neural network." It's a **pretrained transformer-based language model** whose scale gives it broad, reusable linguistic and reasoning capability. Four properties define it together:

- **Language model** — learns a probability distribution over the next token given prior context.
- **Large** — very large parameter counts, datasets, and training compute.
- **Foundation model** — after pretraining, the same base model can be adapted for many downstream tasks.
- **Useful system** — prompting, RAG, or fine-tuning turns the base model into a task-oriented assistant.

Unlike a CNN or a typical smaller network — where you train once and deploy that same model — LLMs are so large in parameters *and* required training data that "training" almost never means training from scratch for each use case. Instead, you take an existing **pretrained** model and adapt it.

## 2. From Raw Text to Next-Token Prediction

The core task of an LLM: given prior tokens, predict a probability distribution over the next one.

**Raw text → Tokenizer → Token IDs → Embeddings → Transformer → Logits / softmax**

- **Tokenizer** — splits text into subword units ("tokens"). The scheme used (e.g., unigram, BPE) differs by model and determines how finely text is chopped up.
- **Token IDs** — each token gets an integer index that also encodes its **position** in the sequence. Position matters — it can change the meaning of the whole sentence.
- **Embeddings** — token IDs become dense vectors (token embedding), combined with a position embedding to form the **input embedding**. The model never processes raw text directly — only these vectors.
- **Transformer** — stacked transformer blocks perform "context mixing," letting each token's representation be shaped by the others around it, across many layers.
- **Logits / softmax** — the final layer outputs logits; softmax turns these into a probability distribution over possible next tokens.

**Key ideas:**
- Tokenization defines the atomic units the model can ever "see."
- The **context window** limits how many prior tokens are visible when predicting the next one.
- Training repeats this pipeline over a massive corpus, across many token positions, over and over.

## 3. Transformer Foundations: The Decoder Stack

Transformers originate from *"Attention Is All You Need"* — worth reading directly for the original motivation behind self-attention.

A full transformer has **encoder** and **decoder** halves:
- **Encoders** are typically used for translation-style, sequence-to-sequence tasks.
- **Decoders** are what most modern LLMs (GPT-style models) are built from — a **decoder-only** stack.

**One decoder block:**

```
Input tokens + positions
        ↓
Masked multi-head self-attention
        ↓
Add & norm
        ↓
Feed-forward network (MLP)
        ↓
Add & norm → next layer
```

**Why it works:**
- **Self-attention** lets each token mix in information from earlier tokens in the same sequence.
- **Multi-head attention** learns several relations in parallel — syntax, semantics, co-reference, code structure, and more — each head captures a different way tokens relate to one another.
- Stacking many blocks builds increasingly abstract contextual representations.

Decoder-only LLMs use **causal masking**: token *t* can only attend to positions **before** it, both in training and at inference. This is the objective being optimized:

**P(xₜ | x_{<t})** — probability of the current token given all previous tokens.

## 4. Why Self-Attention Changed Large-Scale Language Modeling

Classic long-range dependency example: *"The trophy does not fit in the suitcase because **it** is too large."* Resolving "it" requires connecting back to "the trophy" — self-attention lets the model make that link **directly**, without stepping through every intermediate word.

Three big advantages follow:
- **Direct long-range access** — relevant earlier tokens can be consulted immediately, not step-by-step.
- **Parallel training** — all token positions in a sequence are processed together (unlike RNNs, which are inherently sequential).
- **Scales well with compute** — a major reason transformers dominate modern LLM design; more GPUs means more heads/positions processed in parallel.

## 5. Inside Attention: A Worked Walkthrough

The lecture traced computation step-by-step using a live visualization of a tiny model (see [Resource](#resource-the-llm-visualizer) below):

1. **Tokenization** — the input sentence is broken into tokens.
2. **Embeddings** — token embedding + position embedding combine into the **input embedding**.
3. **Inside one attention head** (the demo used 3 heads), for each head:
   - The layer-normalized input is projected through **learned weight matrices** into a **Query (Q)**, **Key (K)**, and **Value (V)** vector.
   - **Q · K** (dot product) produces **attention weights** — how much each token should attend to every other token.
   - Those weights combine with **V** to produce that head's **attention output**.
4. Attention outputs pass through a **residual connection**, then **layer normalization**.
5. This repeats across every stacked transformer block (the demo showed 3 layers).
6. In the final layer: residual → layer norm → **LM head weights** project the result into **logits** → **softmax** → probability distribution over the vocabulary → the highest-probability token is selected as output.

Bigger, more capable LLMs scale this exact same pattern up — **more heads, more stacked layers** — which is what drives up both parameter count and the cost of the underlying matrix multiplications (attention) and matrix-vector operations (normalization, linear layers).

## 6. How LLMs Learn Before They Become Assistants

**Corpora (web, books, code, docs) → Pretraining (next-token prediction) → Base model (strong continuation model) → Instruction / preference tuning (more helpful behavior)**

- **Pretraining** gives broad language competence — the model learns to continue text well.
- **Tuning** changes the *behavior* of that same model so it follows instructions, matches a target domain, or fits a deployment constraint.

**This is why a chat model is usually not "just pretrained" — it's pretrained *and then* adapted.**

Pretraining at this scale is a huge undertaking — the lecture notes spend that can run into the **millions of dollars** and require **thousands of GPUs**. That's precisely why organizations essentially never pretrain from scratch for each new use case; they adapt an existing pretrained model instead.

## 7. Why Fine-Tuning Is Still Needed After Pretraining

Even a well-pretrained LLM usually needs adaptation because of:

| Driver | What it means |
|---|---|
| **Domain shift** | Medical, legal, scientific, or enterprise language differs from the generic web text used in pretraining. |
| **Task format** | Classification, extraction, summarization, code completion, dialogue — each needs different behavior. |
| **Policy and style** | Tone, refusal behavior, safety rules, and output format often need explicit adaptation (e.g., differs by deployment region). |
| **Efficiency** | Updating only the right subset of parameters can be far cheaper than retraining or full fine-tuning. |

**Typical adaptation choices:** better prompts / in-context learning, retrieval-augmented generation (RAG), supervised fine-tuning, parameter-efficient fine-tuning (PEFT), or full fine-tuning of all parameters.

## 8. Where the Cost Comes From in LLM Training

Training memory is **not just the parameters** — it's a stack of:

**Parameters + Gradients + Optimizer state + Activations**

**Why it grows quickly:**
- **Full fine-tuning** updates every parameter, so gradients and optimizer states add major overhead on top of the parameters themselves.
- Longer sequence lengths increase **attention cost roughly quadratically** with context length.
- Larger batches improve hardware utilization but also raise memory pressure.
- **Activation tensors dominate memory** during backpropagation for deep models.

*This is exactly why the rest of Week 6 covers mixed precision, gradient accumulation, checkpointing, efficient batching, and PEFT — they all exist to manage this cost.*

## 9. Adaptation Strategies at a Glance

| Method | What changes? | Cost | Best fit |
|---|---|---|---|
| **Prompting / ICL** | No weights; only the input context changes | Low | Quick iteration when the base model already knows enough |
| **RAG** | Model weights stay fixed; external knowledge is retrieved at run time | Low–Med | Fresh knowledge or private documents without retraining |
| **Full FT / SFT** | All or most model weights are updated | High | Maximum specialization when enough data and compute exist |
| **PEFT / LoRA** | Only small trainable adapters are updated | Medium | Task adaptation under tight GPU memory and storage budgets |

**Takeaway:** the real question isn't "fine-tune or not" — it's *which part of the system should change*: the prompt, the retrieved context, a few trainable adapters, or the whole model.

## 10. How Segment 1 Sets Up the Rest of Week 6

| Segment | Question it answers |
|---|---|
| **2 — Mixed precision** | How can we reduce precision while preserving stable training? |
| **3 — Memory + throughput** | How do batching, accumulation, and checkpointing fit large jobs into memory? |
| **4 — Transfer learning** | Why can a pretrained model adapt with far less data than training from scratch? |
| **5 — PEFT / LoRA** | How can we modify only a tiny number of parameters and still adapt effectively? |

---

## Summary

- LLMs are large transformer-based next-token models pretrained at scale.
- Pretraining gives general capability; prompting, retrieval, and fine-tuning make the model useful for specific settings.
- Week 6's efficiency methods matter because memory, attention cost, and optimizer overhead make naïve full fine-tuning expensive.

## Quick Glossary

| Term | Meaning |
|---|---|
| Token | A subword unit the model reads/writes — its atomic unit of processing. |
| Embedding | A dense vector representation of a token plus its position. |
| Context window | The maximum number of prior tokens the model can attend to. |
| Self-attention | Mechanism letting each token weigh and combine information from other tokens. |
| Multi-head attention | Several attention "views" run in parallel, each learning different relationships. |
| Causal / masked attention | Restricts each token to attend only to earlier positions (used in decoder-only LLMs). |
| Logits | Raw, unnormalized scores produced before softmax. |
| Softmax | Converts logits into a probability distribution. |
| Pretraining | Initial large-scale next-token training on broad corpora. |
| Fine-tuning | Further training a pretrained model on a narrower objective or dataset. |
| SFT | Supervised fine-tuning — updating most/all parameters on labeled data. |
| PEFT | Parameter-efficient fine-tuning — updating only a small set of added/trainable parameters (e.g., LoRA). |
| RAG | Retrieval-augmented generation — pulling in external documents at inference time instead of retraining. |
| Domain shift | Mismatch between the pretraining data distribution and the deployment domain. |

## Resource: The LLM Visualizer

The link mentioned in the lecture, **bbycroft.net/llm**, is an interactive 3D visualization built by software engineer Brendan Bycroft that walks through a GPT-style transformer's inference process step by step — embeddings, layer norm, self-attention (Q/K/V), the MLP, and the final softmax — all rendered as an animated 3D model you can rotate and inspect.

Its default walkthrough uses **nano-GPT**, a tiny model (~85,000 parameters, based on Andrej Karpathy's minGPT) trained to sort short sequences of the letters A/B/C — small enough to trace every computation by hand, which is exactly why the lecture used it to illustrate the Q/K/V mechanics. The same site also lets you explore the same architecture at GPT-2 and GPT-3 scale, to get a feel for how the identical pattern of heads and layers simply multiplies as models grow.
