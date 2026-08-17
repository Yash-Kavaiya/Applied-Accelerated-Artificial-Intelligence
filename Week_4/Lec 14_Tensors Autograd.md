# Detailed Notes: Segment 1 – PyTorch Tensors, Autograd & the Computational Graph

**Segment Focus:** The mathematical building block of every neural network — and how PyTorch tracks every operation to compute gradients automatically.

## Course Context – Week 04 Overview

Week 04 focuses on **PyTorch for Accelerated Training**. The goal is to maximise training speed for AI workloads while also covering relevant inference aspects.

| Segment | Topic |
|---------|-------|
| Seg 1 | Tensors & Autograd |
| Seg 2 | Building Models with `nn.Module` |
| Seg 3 | The Training Loop |
| Seg 4 | DataLoader & Data Pipeline |
| Seg 5 | First Steps in Performance |

This segment revisits and deepens the fundamentals of tensors and automatic differentiation — the core machinery that makes training possible.

## Learning Objectives — Segment 1

| # | Objective |
|---|-----------|
| 01 | Create tensors of any shape, dtype, and device using `torch.tensor`, `torch.zeros`, `torch.randn`, and `torch.arange` |
| 02 | Perform arithmetic, broadcasting, slicing, reshaping, and reduction operations on tensors |
| 03 | Explain what `requires_grad=True` does and how PyTorch builds a dynamic computational graph during the forward pass |
| 04 | Call `.backward()` on a scalar loss and read per-parameter gradients from `.grad` attributes |
| 05 | Distinguish between leaf tensors, non-leaf tensors, and detached tensors; understand when to use `torch.no_grad()` |

## 1. PyTorch Tensors: The N-Dimensional Array at the Core of AI

A **tensor** is the fundamental data structure in PyTorch. It is the multi-dimensional generalisation of a matrix and is the only data type that GPUs (and the entire autograd system) understand.

### 1.1 Creating Tensors

| Method | Description | Example |
|--------|-------------|---------|
| `torch.tensor(data)` | Creates a tensor from a Python list / NumPy array (copies data) | `torch.tensor([1, 2, 3])` |
| `torch.zeros(*size)` | Tensor filled with zeros (default `float32`) | `torch.zeros(3, 4)` → 3×4 matrix |
| `torch.ones(*size)` | Tensor filled with ones | `torch.ones(2, 3, 4)` |
| `torch.randn(*size)` | Samples from standard normal \(\mathcal{N}(0,1)\) — commonly used for weight initialisation | `torch.randn(B, C, H, W)` |
| `torch.arange(start, end, step)` | Evenly spaced values (like Python `range`) | `torch.arange(0, 10, 2)` → `[0, 2, 4, 6, 8]` |
| `torch.linspace(start, end, steps)` | `steps` evenly spaced values between start and end | `torch.linspace(0, 1, 100)` |

**Key point:** `torch.tensor` always copies the data. For zero-copy conversion from NumPy use `torch.from_numpy`.

### 1.2 Data Types (`dtype`)

| dtype | Bytes | Typical Use |
|-------|-------|-------------|
| `torch.float32` | 4 | Default for model weights |
| `torch.float16` / `torch.bfloat16` | 2 | Mixed-precision / GPU training |
| `torch.int64` | 8 | Class labels, indices (default integer type) |
| `torch.bool` | 1 | Attention masks, boolean indexing |

Conversion methods:
```python
x = x.to(torch.float16)   # or x.half()
x = x.float()             # → float32
x = x.long()              # → int64
x = x.bool()
```

**Rule:** Only floating-point tensors can participate in autograd (`requires_grad=True`). Integer and boolean tensors cannot.

### 1.3 Device Placement

Tensors live either in **CPU RAM** or **GPU HBM**.

```python
x = torch.randn(3, 3)              # lives on CPU by default
x = x.to('cuda')                   # move to GPU 0 (returns a NEW tensor)
x = x.cuda()                       # shorthand
x = x.to('cuda:1')                 # specific GPU index
print(x.device)                    # e.g. cuda:0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

**Important:** `.to()` and `.cuda()` return a **new** tensor. The original is unchanged unless you re-assign (`x = x.to(...)`).

### 1.4 Shape, Stride and Contiguity

| Attribute / Method | Meaning |
|--------------------|---------|
| `x.shape` or `x.size()` | `torch.Size([...])` |
| `x.ndim` | Number of dimensions (`len(x.shape)`) |
| `x.numel()` | Total number of elements |
| `x.dtype`, `x.device` | Data type and device |
| `x.stride()` | How many elements to skip to move one step along each dimension |
| `x.is_contiguous()` | Whether the underlying memory is laid out in C-order |

### 1.5 Essential Tensor Operations

**Element-wise arithmetic (with broadcasting):**
```python
x + y, x - y, x * y, x / y
```

**Matrix multiplication:**
```python
x @ y                 # preferred
torch.matmul(x, y)
```

**Reductions:**
```python
torch.sum(x)          # scalar
x.mean()
x.max(), x.min()
x.sum(dim=0)          # reduce along dimension 0
x.mean(dim=-1)        # reduce along last dimension
x.max(dim=0)          # returns (values, indices)
```

**Transpose:**
```python
x.T                   # for 2-D
x.transpose(0, 1)     # general
```

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/7837f144-c79f-4434-92b8-3d2dc6f5a0db" />


### 1.6 Reshaping – Same Elements, New Interpretation

Reshaping does **not** change the underlying data (when possible); it only changes how we interpret the shape.

| Operation | Behaviour | Memory |
|-----------|-----------|--------|
| `x.view(*shape)` | Reinterprets storage. `-1` infers the missing size. Fails if tensor is non-contiguous. | Shares memory |
| `x.reshape(*shape)` | Same goal as `view`. Makes a copy only when necessary. | View if possible, else copy |
| `x.unsqueeze(dim)` | Inserts a dimension of size 1 at the given position | Shares memory |
| `x.squeeze()` / `x.squeeze(dim)` | Removes dimensions of size 1 | Shares memory |
| `x.permute(*dims)` | Reorders existing dimensions (no data movement of values, only axis meaning) | Shares memory (may become non-contiguous) |

**Mental model:**
- `view` = reinterpret the same storage
- `reshape` = try to view; if impossible, copy
- `unsqueeze` / `squeeze` = add / remove size-1 axes
- `permute` = change the **order** of axes (critical for image layout conversion)

#### Visual Example – `view` / `reshape`

Starting tensor of shape `(2, 3, 4)` containing values 1 … 24:

```
view(2, -1) or reshape(2, -1)  →  shape (2, 12)
```

`-1` tells PyTorch: “compute the missing dimension so that the total number of elements stays the same”.

#### `unsqueeze(0)` example

```
(3, 4)  →  unsqueeze(0)  →  (1, 3, 4)
```

#### `squeeze()` example

```
(3, 1, 4)  →  squeeze()  →  (3, 4)
```
Removes **all** dimensions whose size is 1.

### 1.7 Permute – Reordering the Meaning of Dimensions

`permute` is especially important for images.

**Common conversion:** channel-last → channel-first

```
(H, W, C)  →  x.permute(2, 0, 1)  →  (C, H, W)
```

```
Original axes:  0 = H,  1 = W,  2 = C
New order:      2, 0, 1   →   C, H, W
```

This is required because almost all PyTorch vision models expect **NCHW** (batch, channel, height, width) layout.

```mermaid
flowchart LR
    A["(H, W, C)<br/>Channel-last"] -->|permute(2,0,1)| B["(C, H, W)<br/>Channel-first"]
    style A fill:#e1f5fe
    style B fill:#c8e6c9
```
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/2078c9a5-3aac-441a-989a-925a6d2152b0" />


**Key insight:** The **values** stay the same; only the semantic meaning of each axis changes.

### 1.8 Broadcasting – How PyTorch Handles Differently-Shaped Tensors

**Definition:** When two tensors of different shapes are combined, PyTorch **virtually expands** the smaller tensor to match the larger one **without copying data**.

**Broadcasting rules (align from the right):**
1. Dimensions are compared from the trailing (right-most) dimension.
2. Two dimensions are compatible if they are equal **or** one of them is 1.
3. Missing dimensions are treated as size 1.

**Classic example:**
```python
a = torch.ones(3, 4)   # shape (3, 4)
b = torch.ones(4)      # shape (4,)  → treated as (1, 4)
c = a + b              # shape (3, 4) — b is broadcast over the rows
```
<img width="1402" height="1122" alt="image" src="https://github.com/user-attachments/assets/718aecc9-414d-4942-be09-68e2074acd7f" />

**Image normalisation example (very common):**
```python
img  = torch.randn(8, 3, 224, 224)          # NCHW
mean = torch.tensor([0.485, 0.456, 0.406])  # shape (3,)
mean = mean.view(1, 3, 1, 1)                # reshape for broadcasting
img  = img - mean                          # per-channel mean subtraction
```

**Common mistake:**
```python
# Works
torch.ones(4) + torch.ones(3, 4)

# Fails – shapes (3,) and (3, 4) are incompatible (3 ≠ 4)
torch.ones(3) + torch.ones(3, 4)

# Fix: make it (3, 1)
torch.ones(3).unsqueeze(-1) + torch.ones(3, 4)
```

### 1.9 Aggregation / Reduction

```python
x = torch.randn(4, 5)

x.sum()           # scalar – sum of all elements
x.sum(dim=0)      # shape (5,) – sum over rows
x.sum(dim=1)      # shape (4,) – sum over columns
x.mean(dim=-1)    # shape (4,) – mean over last dimension
x.max(dim=0)      # returns namedtuple (values, indices)
```

### 1.10 In-place Operations (Use Carefully)

In-place ops end with an underscore and modify the tensor **in memory**:

```python
x.add_(y)     # x = x + y
x.mul_(2.0)   # x = x * 2
```

**Critical rule:**  
**Never** perform in-place operations on tensors that participate in the autograd graph (`requires_grad=True`). Doing so raises a runtime error because the computational graph would become invalid.

### 1.11 NumPy Interoperability

```python
import numpy as np

np_arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(np_arr)   # zero-copy, shares memory
back = t.numpy()               # zero-copy back to NumPy
```

- Shared memory: changing one changes the other.
- Only works for **CPU** tensors. GPU tensors must be moved with `.cpu()` first.

## 2. Autograd: Automatic Differentiation – How PyTorch Learns

### 2.1 The Central Problem

Training a neural network = **minimising a scalar loss** \( L(\mathbf{w}) \) by adjusting the parameters \(\mathbf{w}\).

Gradient descent requires the gradient
\[
\frac{\partial L}{\partial w_i}
\]
for every parameter \(w_i\).

A modern model can have **billions** of parameters. Computing each partial derivative by hand is impossible.  

**PyTorch’s autograd** solves this by applying the **chain rule** automatically through a computational graph.

### 2.2 How Autograd Works – The Computational Graph

Every tensor operation creates a node in a **Directed Acyclic Graph (DAG)**.

- **Forward pass:** Compute the numerical values **and** record the operations (build the graph).
- **Backward pass:** Starting from a scalar loss, traverse the graph in reverse, applying the chain rule at every node.

**Terminology:**

| Term | Meaning |
|------|---------|
| **Leaf tensor** | Tensor created by the user (model weights, input data). Has a `.grad` attribute after `.backward()`. |
| **Non-leaf tensor** | Result of an operation. Does **not** retain `.grad` by default (saves memory). |
| **`requires_grad=True`** | Instructs PyTorch to track all operations on this tensor. |

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/a64f9999-429a-4728-b08d-211cf22dc699" />

### 2.3 Triggering Gradient Computation – Concrete Example

```python
w = torch.tensor(2.0, requires_grad=True)   # leaf – track this
x = torch.tensor(3.0)                       # constant – not tracked

y = w * x + w**2                            # graph is built here
L = y                                       # already a scalar
# or L = y.sum() if y were a vector

L.backward()                                # compute gradients
print(w.grad)                               # tensor(7.)
```

**Manual derivation (step-by-step):**

\[
\begin{align*}
y &= w \cdot x + w^{2} \\
L &= y \\
\frac{\partial L}{\partial w} &= \frac{\partial}{\partial w}(w x + w^{2}) \\
&= x + 2w \\
&= 3 + 2\cdot 2 = 7
\end{align*}
\]

PyTorch arrives at the same answer automatically via the reverse-mode chain rule.

### 2.4 Gradient Flow Rules

| Rule | Explanation |
|------|-------------|
| `requires_grad=True` | Only leaf tensors with this flag receive gradients |
| Float dtype only | `int` / `bool` tensors cannot have `requires_grad=True` |
| Propagation | If **any** input of an operation has `requires_grad=True`, the output also does |
| `torch.no_grad()` | Context manager that **disables** graph building (used in inference) |
| `tensor.detach()` | Returns a new tensor that shares data but has **no** gradient history |

**Mental model:**  
Autograd is a **tape recorder**.  
- Forward pass = record every operation onto the tape.  
- Backward pass = replay the tape in reverse, multiplying local derivatives (chain rule).

### 2.5 Dynamic vs Static Computational Graphs

| Aspect | TensorFlow 1.x / Theano | PyTorch |
|--------|-------------------------|---------|
| Graph type | Static (define-then-run) | Dynamic (define-by-run) |
| Construction | Graph defined once, then data is fed | Graph built **fresh** on every forward pass |
| Control flow | Limited / special constructs | Native Python `if`, `for`, `while` work naturally |
| Debugging | Difficult | Standard Python debugger / `print` work |
| Variable-length sequences | Hard | Natural – graph shape changes with input |

**Advantage of dynamic graphs:** The model can contain arbitrary Python control flow and the graph will adapt automatically.

### 2.6 Gradient Accumulation

```python
# Gradients are ADDED into .grad every time .backward() is called
loss1.backward()
loss2.backward()   # .grad now contains grad1 + grad2

# Must clear before the next optimisation step
optimizer.zero_grad()   # or param.grad.zero_()
```

**Intentional accumulation:**  
Sum gradients over \(N\) mini-batches before calling `optimizer.step()`.  
This simulates a batch size \(N\times\) larger **without** needing \(N\times\) GPU memory.

### 2.7 `torch.no_grad()` vs `.detach()`

```python
# Block-level – no graph is built at all
with torch.no_grad():
    y = model(x)          # fast, no gradient tracking
# Typical use: evaluation / inference

# Tensor-level – break the gradient flow at a specific point
z = encoder(x).detach()   # stop gradients here
out = decoder(z)          # decoder can still be trained
# Typical uses: freezing an encoder, certain GAN tricks
```

### 2.8 Advanced: `retain_graph=True`

By default the computational graph is **freed** after `.backward()` (to save memory).  
Setting `retain_graph=True` keeps the graph alive:

```python
loss1.backward(retain_graph=True)
loss2.backward()               # can reuse the same graph
```

Needed for:
- Multi-loss training that shares intermediate activations
- Higher-order gradients (gradients of gradients)

**Cost:** Higher memory consumption.

### 2.9 Gradient of a Non-Scalar

`.backward()` expects a **scalar**. If the tensor is not scalar you must supply a gradient argument:

```python
y.backward(torch.ones_like(y))   # equivalent to y.sum().backward()
```

This computes the **vector-Jacobian product**.  
**Best practice:** Always reduce your loss to a scalar (e.g. `.mean()` or `.sum()`) before calling `.backward()`.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/f1b9712b-0360-4b1b-bb47-e27c20161455" />


## 3. Segment 1 Summary

| Key Takeaway | Explanation |
|--------------|-------------|
| Tensors | N-dimensional arrays with `dtype`, `shape` and `device`. Every arithmetic / reshape / matmul produces a new tensor. Broadcasting expands shapes automatically. |
| `requires_grad=True` | Tells PyTorch to track operations and build a dynamic computational graph during the forward pass. |
| `.backward()` | On a scalar loss traverses the graph in reverse, applying the chain rule, and writes gradients into the `.grad` attribute of every leaf tensor that requires grad. |
| `torch.no_grad()` | Disables graph construction — mandatory for inference / evaluation to save memory and compute. |
| `optimizer.zero_grad()` | Must be called every training step; otherwise gradients accumulate across iterations. |
| Dynamic graph | Rebuilt from scratch on every forward pass → natural Python control flow and variable-length inputs are first-class. |

## Practical Recommendations (from the lecture)

1. Always experiment with different shapes, dtypes and devices in a notebook.
2. After every reshape / permute / broadcast, print `.shape` to verify the result.
3. Never use in-place operations (`add_`, `mul_`, …) on tensors that require gradients.
4. For inference, wrap the forward pass in `torch.no_grad()`.
5. Remember that `.grad` accumulates — call `optimizer.zero_grad()` (or `model.zero_grad()`) at the start of every iteration.
6. Prefer reducing the loss to a scalar before `.backward()`.
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/4e5e032a-76cd-4e29-8a79-058e0a777c69" />
