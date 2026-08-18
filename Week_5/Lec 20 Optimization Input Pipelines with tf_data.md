
# Week 5 · Segment 2 — Optimizing Input Pipelines with tf.data

**NPTEL: Applied Accelerated Artificial Intelligence**
Instructor: Dr. Satyajit Das · Dept. of Computer Science and Engineering · IIT Guwahati

*Compiled from the segment's slides, lecture transcript, and accompanying code handout.*

## Contents
1. Quick Recap: Eager vs. Graph Execution
2. Why Input Pipelines Matter
3. The tf.data API at a Glance
4. Creating Datasets — Three Ways
5. Core Transformations
6. `.cache()` in Practice
7. Parallel Preprocessing with `num_parallel_calls`
8. More Optimization Techniques
9. TFRecord: The Preferred On-Disk Format
10. Putting It All Together: The Recommended Pipeline
11. Cheat Sheet
12. Key Takeaways

---

This segment picks up right where Segment 1 left off, briefly revisits eager vs. graph execution, then spends the rest of its time on the real subject: building fast `tf.data` input pipelines.

## 1. Quick Recap: Eager vs. Graph Execution

Segment 1 ended with `tf.GradientTape` and a minimal training loop: input data held as `tf.constant` tensors, trainable parameters as `tf.Variable`, a loss computed inside the tape's context, gradients pulled out with `tape.gradient(...)`, and applied via `optimizer.apply_gradients(...)`. Because that loop ran one Python line at a time, it was using **eager execution** — the default in TensorFlow 2.x, where every op runs immediately, just like NumPy.

Eager mode is why TF2 is easy to debug — you can `print()` any intermediate tensor and step through line by line. But it isn't the fastest way to run a model in production.

**Graph execution** trades that debuggability for speed. Wrapping a function in `@tf.function` tells TensorFlow to *trace* the Python function once and compile it into a graph that runs in the C++ runtime, without going back through the Python interpreter on every call.

```python
import tensorflow as tf
import time

def heavy_computation(x):
    for _ in range(50):
        x = tf.matmul(x, x, transpose_b=True)
    return x

graph_fn = tf.function(heavy_computation)
data = tf.random.normal((64, 64))

# Warm-up: the first call to a tf.function traces the graph — exclude it from timing
_ = graph_fn(data)

# Eager timing
t0 = time.perf_counter()
_ = heavy_computation(data)
eager_ms = (time.perf_counter() - t0) * 1000

# Graph timing
t0 = time.perf_counter()
_ = graph_fn(data)
graph_ms = (time.perf_counter() - t0) * 1000

print(f"Eager : {eager_ms:.2f} ms")
print(f"Graph : {graph_ms:.2f} ms")
```

In the lecture demo, eager execution took **≈1.5 ms** and the compiled graph took **≈0.8 ms** for the same computation — roughly a **2× speedup**. The first call to any `tf.function` always pays a one-time tracing/warm-up cost, which is why it's excluded from the timing comparison — otherwise you'd be measuring compilation time, not execution time.

**Rule of thumb:** develop and debug in eager mode; wrap the final training step (and anything performance-critical) in `@tf.function` for production and inference.

### Inspecting a compiled graph

A `@tf.function` is only a *template* until it knows the exact shapes and dtypes it will receive. You can lock in a specific input signature with `tf.TensorSpec` and pull out a **concrete function** — a graph specialized to that signature — then inspect it directly:

```python
@tf.function
def forward(x, w, b):
    hidden = tf.nn.relu(tf.matmul(x, w) + b)
    return tf.reduce_mean(hidden)

concrete_fn = forward.get_concrete_function(
    x=tf.TensorSpec(shape=[None, 8], dtype=tf.float32),
    w=tf.TensorSpec(shape=[8, 8],   dtype=tf.float32),
    b=tf.TensorSpec(shape=[8],      dtype=tf.float32),
)

graph = concrete_fn.graph
print("Graph inputs :", graph.inputs)
print("Graph outputs:", graph.outputs)
for op in graph.get_operations():
    print(" ", op.type, op.name)
```

*(The exact shapes above are illustrative — the technique is the point: define a signature once with `TensorSpec`, then use `get_concrete_function()` to obtain and inspect the real compiled graph — its placeholders, ops, and outputs.)*

This is the bridge into the rest of the segment: once a model is graph-compiled, the **next bottleneck is almost always how fast you can feed it data** — which is exactly what `tf.data` solves.

---

## 2. Why Input Pipelines Matter

A modern GPU can chew through **thousands of images per second**. The question the rest of this segment answers is: *can you actually feed it data that fast?*

- If the CPU can't load and preprocess examples quickly enough, the GPU **stalls** waiting for the next batch.
- A poorly-built input pipeline can waste **40–70% of available GPU time** — expensive accelerator hardware sitting idle.
- Data has to travel disk → CPU (preprocessing) → **PCIe bus** → GPU memory, and PCIe bandwidth is itself a hard ceiling. If every stage runs strictly one after another, that transfer time is pure GPU idle time.

### The naive (sequential) pipeline

```
[ Load batch from disk ] → [ Preprocess on CPU ] → [ Transfer over PCIe ] → [ GPU forward/backward pass ]
        I/O wait                 CPU compute              bandwidth-limited        ← only step where GPU is busy
```

Every batch repeats this chain from scratch. The GPU is idle for three of the four steps, every single iteration.

### The optimized (overlapped) pipeline

The fix is **prefetching**: while the GPU trains on batch *N*, the CPU concurrently loads and preprocesses batch *N+1*.

```
CPU:  [ Load+Preprocess batch 1 ] [ Load+Preprocess batch 2 ] [ Load+Preprocess batch 3 ] ...
GPU:                              [   Train batch 1   ]        [   Train batch 2   ]      ...
```

With enough overlap, GPU idle time approaches **zero**. Three techniques make this possible, and they map directly onto `tf.data` methods used throughout this segment:

| Technique | What it does | `tf.data` method |
|---|---|---|
| Prefetching | CPU prepares the *next* batch while GPU trains on the current one | `.prefetch(AUTOTUNE)` |
| Parallel reads | Multiple threads load files concurrently instead of one at a time | `.map(..., num_parallel_calls=AUTOTUNE)` / `.interleave(...)` |
| Vectorized ops | Transform whole batches, not one example at a time | `.batch()` combined with `.map()` |

> **Connecting to PyTorch:** this is the same problem `DataLoader(num_workers=N, pin_memory=True)` solves in PyTorch. The difference is that `tf.data` usually lets you hand the worker count over to **`AUTOTUNE`**, which watches CPU load and core count at runtime and picks the parallelism for you, instead of requiring you to hand-tune `num_workers`.

---

## 3. The tf.data API at a Glance

`tf.data` is TensorFlow's unified data-loading and preprocessing API. A few defining properties:

- Pipelines are **lazy and composable** — chaining `.map()`, `.batch()`, `.shuffle()`, etc. builds up a plan; nothing runs until you actually iterate over the dataset.
- Execution happens in the **C++ runtime**, not the Python interpreter.
- It can source data from in-memory arrays, Python generators, or files on disk (including remote/networked storage).

**Core abstraction — `tf.data.Dataset`:** a sequence of elements that all share the same structure (shape + dtype). You construct one from:
- `tf.data.Dataset.from_tensor_slices(...)` — in-memory tensors/arrays
- `tf.data.Dataset.from_generator(...)` — a Python generator function
- File-backed sources: `tf.data.TFRecordDataset`, `TextLineDataset`, `FixedLengthRecordDataset`

**The canonical pipeline pattern:**

```python
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(1000).batch(32).map(preprocess_fn)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

`tf.data.AUTOTUNE` shows up everywhere in this API — it tells the runtime to dynamically tune buffer sizes and parallelism at runtime rather than you guessing a fixed number.

---

## 4. Creating Datasets — Three Ways

The handout demonstrates the three most common ways to construct a `Dataset`, each suited to a different situation:

| Method | Best for | Notes |
|---|---|---|
| `from_tensor_slices()` | Small datasets that fit in RAM | Fastest — everything already lives in memory; won't scale to GB-sized datasets |
| `from_generator()` | Streaming / custom loading logic | Requires an explicit `output_signature` (via `tf.TensorSpec`) describing shape & dtype |
| `range()` | Synthetic data, quick tests, index-driven pipelines | Produces `int64` scalars; often combined with `.map()` to generate/load real data per index |

```python
import numpy as np
import tensorflow as tf

# ── Method 1: from in-memory NumPy arrays ──
images = np.random.randn(100, 14, 14, 1).astype(np.float32)
labels = np.random.randint(0, 10, 100).astype(np.int32)

ds_numpy = tf.data.Dataset.from_tensor_slices((images, labels))
print(f"from_tensor_slices → {ds_numpy.element_spec}")

# ── Method 2: from a Python generator ──
def data_generator():
    for i in range(30):
        yield np.random.randn(14, 14, 1).astype(np.float32), np.int32(i % 10)

ds_gen = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(14, 14, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(),          dtype=tf.int32)
    )
)
print(f"from_generator      → {ds_gen.element_spec}")

# ── Method 3: range / synthetic ──
ds_range = tf.data.Dataset.range(100)
print(f"Dataset.range(100)  → {ds_range.element_spec}")

# ── Peek at elements ──
print("\nFirst 3 samples from ds_numpy:")
for img, lbl in ds_numpy.take(3):
    print(f"  image shape: {img.shape}  label: {lbl.numpy()}")
```

**Output:**
```
from_tensor_slices → (TensorSpec(shape=(14, 14, 1), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None))
from_generator      → (TensorSpec(shape=(14, 14, 1), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None))
Dataset.range(100)  → TensorSpec(shape=(), dtype=tf.int64, name=None)

First 3 samples from ds_numpy:
  image shape: (14, 14, 1)  label: 7
  image shape: (14, 14, 1)  label: 9
  image shape: (14, 14, 1)  label: 5
```

`.take(n)` is the standard way to peek at the first *n* elements of any dataset without materializing the whole thing — handy for sanity-checking a pipeline before a full training run.

---

## 5. Core Transformations

Four methods do most of the work in a `tf.data` pipeline:

**`.map(fn, num_parallel_calls=AUTOTUNE)`**
Applies `fn` to every element. Set `num_parallel_calls=tf.data.AUTOTUNE` to run it across multiple CPU threads instead of one element at a time. Tip from the lecture: decorate `fn` with `@tf.function` so the preprocessing itself runs as a compiled graph.

**`.batch(batch_size, drop_remainder=True)`**
Groups consecutive elements into batches. `drop_remainder=True` guarantees every batch has exactly `batch_size` elements — important on TPUs, which need fixed shapes to compile efficiently.

**`.shuffle(buffer_size)`**
Fills a buffer of `buffer_size` elements and samples randomly from it. A bigger buffer means better randomness at the cost of memory. `reshuffle_each_iteration=True` gives a fresh shuffle order every epoch instead of repeating the same one.

**`.cache()` and `.repeat()`**
`.cache()` stores the dataset (in RAM, or on disk if you pass a filename) the first time it's produced, so later epochs skip re-computing everything upstream of the cache. `.repeat()` cycles the dataset for multiple epochs instead of stopping after one pass.

Here's all four working together:

```python
@tf.function
def preprocess(image, label):
    """Normalize to [-1, 1] and apply random noise augmentation."""
    image  = tf.cast(image, tf.float32) / 127.5 - 1.0
    image  = image + tf.random.normal(tf.shape(image), stddev=0.01)
    image  = tf.clip_by_value(image, -1.0, 1.0)
    return image, label

ds = (
    tf.data.Dataset.from_tensor_slices((images, labels))
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)    # parallel CPU workers
    .shuffle(buffer_size=200, reshuffle_each_iteration=True) # randomise order
    .batch(16, drop_remainder=True)                          # group into batches
    .prefetch(tf.data.AUTOTUNE)                               # CPU preps next batch while GPU trains
)

print("Pipeline element spec:")
for spec in ds.element_spec:
    print(f"  {spec}")

for img_b, lbl_b in ds.take(1):
    print(f"\nBatch → images: {img_b.shape}  labels: {lbl_b.shape}")
    print(f"  image value range : [{img_b.numpy().min():.3f}, {img_b.numpy().max():.3f}]")
```

**Output:**
```
Pipeline element spec:
  TensorSpec(shape=(16, 14, 14, 1), dtype=tf.float32, name=None)
  TensorSpec(shape=(16,), dtype=tf.int32, name=None)

Batch → images: (16, 14, 14, 1)  labels: (16,)
  image value range : [-1.000, -0.954]
```

Notice the shape change: each element started as a single `(14, 14, 1)` image; after `.batch(16)`, every element pulled from the pipeline is a `(16, 14, 14, 1)` tensor — a whole batch at once.

---

## 6. `.cache()` in Practice

Without `.cache()`, every method upstream of it — including `.map()` — re-runs **every single epoch**, even when the transformation is deterministic and produces the same result each time. `.cache()` runs that upstream work once, stores the result, and serves it directly from memory (or disk) on every later epoch.

```python
N_ITEMS = 200
ds_raw  = tf.data.Dataset.range(N_ITEMS)

def preprocess_int(x):
    return tf.cast(x, tf.float32) * 2.0

ds_no_cache = ds_raw.map(preprocess_int).batch(20)
ds_cached   = ds_raw.map(preprocess_int).cache().batch(20)

def time_epochs(ds, n_epochs=4, name=''):
    times = []
    for ep in range(n_epochs):
        t0 = time.perf_counter()
        for _ in ds: pass
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  {name:25s}: {[f'{t:.1f}ms' for t in times]}")
    return times

print("Epoch times over 4 epochs (cache fills on epoch 1):")
t_no = time_epochs(ds_no_cache, name='Without .cache()')
t_c  = time_epochs(ds_cached,   name='With .cache()')

print(f"\nEpoch 2 speedup from .cache(): {t_no[1]/max(t_c[1], 0.001):.1f}×")
```

**Output:**
```
Epoch times over 4 epochs (cache fills on epoch 1):
  Without .cache()         : ['16.5ms', '11.2ms', '11.5ms', '11.1ms']
  With .cache()            : ['22.8ms', '3.1ms', '4.2ms', '3.2ms']

Epoch 2 speedup from .cache(): 3.7×
```

Read that table carefully — it tells the whole story:
- **Epoch 1 with cache is actually *slower*** (22.8ms vs 16.5ms): you're paying the normal preprocessing cost *plus* the overhead of writing everything into the cache.
- **From epoch 2 onward, cached epochs are ~3.7× faster** (3–4ms vs 11ms), because `.map()` never runs again — the pipeline just replays stored results.

Since any real training run does many epochs, that one-time epoch-1 cost is paid back almost immediately. Whenever preprocessing is deterministic (no randomness upstream of the cache) and the data fits in memory, `.cache()` is close to a free win.

---

## 7. Parallel Preprocessing with `num_parallel_calls`

`num_parallel_calls` on `.map()` is the direct analogue of `num_workers` in a PyTorch `DataLoader`: it controls how many CPU threads run your preprocessing function concurrently. Setting it to `1` processes elements strictly one at a time; setting it to `tf.data.AUTOTUNE` lets the runtime pick a worker count based on available cores and current load.

```python
N = 400
x_data = np.random.randn(N, 14, 14, 1).astype(np.float32)
y_data = np.random.randint(0, 10, N).astype(np.int32)

@tf.function
def augment(img, lbl):
    img = img + tf.random.normal(tf.shape(img), stddev=0.05)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.clip_by_value(img, -3.0, 3.0)
    return img, lbl

# Sequential map
ds_seq = (tf.data.Dataset.from_tensor_slices((x_data, y_data))
          .map(augment, num_parallel_calls=1).batch(32).prefetch(1))

# Parallel map (AUTOTUNE picks optimal worker count)
ds_par = (tf.data.Dataset.from_tensor_slices((x_data, y_data))
          .map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE))

def time_pipeline(ds, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in ds: pass
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times)

t_seq = time_pipeline(ds_seq)
t_par = time_pipeline(ds_par)
print(f"Sequential map (workers=1)  : {t_seq:.1f} ms")
print(f"Parallel map (AUTOTUNE)     : {t_par:.1f} ms")
print(f"Parallelism speedup         : {t_seq/max(t_par,0.001):.2f}×")
```

**Output:**
```
Sequential map (workers=1)  : 100.2 ms
Parallel map (AUTOTUNE)     : 78.2 ms
Parallelism speedup         : 1.28×
```

A 1.28× speedup looks modest — because `augment()` here is a cheap, three-line function, so there isn't much CPU work to parallelize in the first place. The point generalizes with the cost of your preprocessing: heavier pipelines (JPEG decode, resizing, complex augmentation stacks) see much larger gains from parallel `.map()`, since there's proportionally more work to spread across cores.

---

## 8. More Optimization Techniques

Beyond `.cache()` and parallel `.map()`, three more techniques round out a fast pipeline:

**Prefetching — the single most impactful change.** `dataset.prefetch(tf.data.AUTOTUNE)` overlaps the *last* stage of pipeline production with model training, so the CPU is already assembling the next batch while the GPU is still working on the current one. In the ideal case, the GPU is never waiting on data at all.

**Parallelizing file reads.** For datasets that live in many separate files:
```python
files = tf.data.Dataset.list_files(pattern)
ds = files.interleave(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
```
`.interleave()` reads from multiple files concurrently and hides I/O latency — especially valuable on networked storage (NFS, GCS, S3), where a single file read can be slow.

**Vectorized mapping — batch before you map.** The examples above call `.map()` *before* `.batch()`, so the preprocessing function sees one example at a time. There's a complementary pattern worth knowing: call `.batch()` *before* `.map()`, and write the function to operate on an entire batch tensor at once (using ops that broadcast over the batch dimension). That replaces *N* small function calls with **one** call over *N* elements, cutting Python/graph call overhead. Which order is better depends on whether your preprocessing function is written per-example or per-batch — both are valid `tf.data` patterns, and it's worth being deliberate about which one your `map` function assumes.

---

## 9. TFRecord: The Preferred On-Disk Format

Reading many small individual files (thousands of JPEGs in a directory, say) has real overhead: every file open/close/seek costs something, and that overhead multiplies by however many files exist. **TFRecord** — a binary record format built into TensorFlow — avoids this:

- Sequential reads that saturate disk bandwidth instead of being dominated by per-file overhead
- No per-file filesystem overhead — one file can hold an entire dataset
- Native to TF's C++ input pipeline, avoiding a Python decode step for every record
- Structured records are described with `tf.train.Example`, a flexible key→feature schema

```python
import os, tempfile, shutil

tmpdir   = tempfile.mkdtemp()
tfr_path = os.path.join(tmpdir, 'demo.tfrecord')

# ── Write TFRecord ──
N_RECORDS = 100
with tf.io.TFRecordWriter(tfr_path) as writer:
    for i in range(N_RECORDS):
        img   = np.random.randint(0, 255, (14, 14, 1), dtype=np.uint8)
        lbl   = i % 10
        feature = {
            'image':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.astype(np.float32).tobytes()])),
            'label':  tf.train.Feature(int64_list=tf.train.Int64List(value=[lbl])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[14])),
            'width':  tf.train.Feature(int64_list=tf.train.Int64List(value=[14])),
        }
        ex = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(ex.SerializeToString())

print(f"Written {N_RECORDS} records  →  {os.path.getsize(tfr_path)/1024:.1f} KB")

# ── Read TFRecord ──
feature_desc = {
    'image':  tf.io.FixedLenFeature([], tf.string),
    'label':  tf.io.FixedLenFeature([], tf.int64),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width':  tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(serialized):
    parsed = tf.io.parse_single_example(serialized, feature_desc)
    image  = tf.io.decode_raw(parsed['image'], tf.float32)
    image  = tf.reshape(image, [14, 14, 1])
    return image, parsed['label']

ds_tfr = (
    tf.data.TFRecordDataset(tfr_path)
    .map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(200)
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
)

for img_b, lbl_b in ds_tfr.take(1):
    print(f"TFRecord batch → images: {img_b.shape}  labels[:5]: {lbl_b.numpy()[:5]}")

shutil.rmtree(tmpdir)
print("\nTFRecord pipeline verified ✔")
```

**Output:**
```
Written 100 records  →  85.1 KB
TFRecord batch → images: (16, 14, 14, 1)  labels[:5]: [0 1 3 3 8]

TFRecord pipeline verified ✔
```

Notice the read side is just another `tf.data.Dataset` source (`TFRecordDataset`) — every transformation from the earlier sections (`.map()`, `.shuffle()`, `.batch()`, `.prefetch()`) plugs in exactly the same way regardless of whether the data started life as a NumPy array or a `.tfrecord` file on disk.

---

## 10. Putting It All Together: The Recommended Pipeline

The lecture's closing recommendation is a fixed ordering that combines everything above:

```
source → .cache() → .shuffle() → .map(AUTOTUNE) → .batch() → .prefetch(AUTOTUNE)
```

(with `TFRecordDataset` as the `source` when data lives on disk as TFRecords).

```python
x_all = np.random.randn(400, 14, 14, 1).astype(np.float32)
y_all = np.random.randint(0, 10, 400).astype(np.int32)

@tf.function
def augment_v2(img, lbl):
    img = img + tf.random.normal(tf.shape(img), stddev=0.03)
    return tf.clip_by_value(img, -2.0, 2.0), lbl

optimal_ds = (
    tf.data.Dataset.from_tensor_slices((x_all, y_all))
    .cache()                                                   # ① cache raw data
    .shuffle(buffer_size=400, reshuffle_each_iteration=True)   # ② shuffle
    .map(augment_v2, num_parallel_calls=tf.data.AUTOTUNE)      # ③ parallel augmentation
    .batch(32, drop_remainder=True)                            # ④ batch
    .prefetch(tf.data.AUTOTUNE)                                # ⑤ prefetch
)

count = 0
t0 = time.perf_counter()
for img_b, lbl_b in optimal_ds:
    count += 1
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"Optimal pipeline: {count} batches of 32 in {elapsed_ms:.1f} ms")
print(f"Throughput: {400 / (elapsed_ms/1000):,.0f} samples/sec")
```

**Output:**
```
Optimal pipeline: 12 batches of 32 in 39.6 ms
Throughput: 10,110 samples/sec
```

| Stage | Benefit |
|---|---|
| `.cache()` | Stores preprocessed data in RAM → skip re-reading disk on epoch 2+ |
| `.shuffle(large_buffer)` | Larger buffer → more random order → better generalization |
| `.map(AUTOTUNE)` | Uses all available CPU cores for preprocessing in parallel |
| `.batch(N, drop_remainder)` | Groups N samples → vector ops faster than scalar ops |
| `.prefetch(AUTOTUNE)` | CPU prepares batch N+1 while GPU trains on batch N |

One ordering detail worth internalizing: `.cache()` sits *before* `.shuffle()` and `.map()` here — so it caches the **raw, un-augmented** data, and shuffling/augmentation still happen fresh every epoch. That's deliberate: caching *after* augmentation would freeze one fixed augmented version of the dataset forever, defeating the point of random augmentation.

---

## 11. Cheat Sheet

| Call | Purpose | Key argument(s) |
|---|---|---|
| `Dataset.from_tensor_slices((x, y))` | Build a dataset from in-memory arrays/tensors | — |
| `Dataset.from_generator(gen, output_signature=...)` | Build a dataset from a Python generator | `output_signature`: `tf.TensorSpec` per output |
| `Dataset.range(n)` | Synthetic integer dataset `0..n-1` | — |
| `TFRecordDataset(path)` | Read `.tfrecord` files | usually paired with `.map(parse_fn)` |
| `.map(fn, num_parallel_calls=AUTOTUNE)` | Per-element (or per-batch) transformation | `num_parallel_calls` |
| `.shuffle(buffer_size, reshuffle_each_iteration=True)` | Randomize order | bigger buffer = more random, more RAM |
| `.batch(size, drop_remainder=True)` | Group elements into batches | `drop_remainder` for fixed shapes (TPUs) |
| `.cache()` | Store results after first epoch | optional filename → cache to disk instead of RAM |
| `.repeat(n)` | Cycle dataset for `n` epochs (or forever if omitted) | — |
| `.prefetch(AUTOTUNE)` | Overlap CPU prep with GPU training | almost always the last call in a pipeline |
| `.interleave(load_fn, num_parallel_calls=AUTOTUNE)` | Read multiple files concurrently | pair with `list_files(pattern)` |
| `tf.function` | Compile a Python function into a graph | use `.get_concrete_function(...)` to inspect it |

---

## 12. Key Takeaways

- **The GPU is only as fast as the pipeline feeding it.** A naive, sequential load→preprocess→transfer→train loop can waste 40–70% of GPU time.
- **Prefetching is the single highest-leverage fix** — it overlaps CPU data prep with GPU compute so the accelerator is rarely (ideally never) idle.
- **`AUTOTUNE` is the default answer** for both `num_parallel_calls` and prefetch buffer sizes — let the runtime decide rather than hand-picking constants.
- **`.cache()` pays for itself after one epoch** whenever preprocessing is deterministic and the data fits in memory — cache before random augmentation, not after.
- **TFRecord beats thousands of loose files** for anything disk-bound, since it avoids per-file overhead and supports fast sequential reads.
- **Eager mode is for developing and debugging; `@tf.function` graph mode is for training and inference speed** — this segment's ~2× speedup on a toy example scales up meaningfully on real models.
- The recommended full ordering: **source → `.cache()` → `.shuffle()` → `.map(AUTOTUNE)` → `.batch()` → `.prefetch(AUTOTUNE)`**.
