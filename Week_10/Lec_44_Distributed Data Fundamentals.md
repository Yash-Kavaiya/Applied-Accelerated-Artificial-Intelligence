
# Week 10: Distributed Data Fundamentals: Detailed Notes

> **Course context:**  It consolidates containers, Kubernetes, profiling, distributed training and packaging, and links them to ETL, Spark and Dask from other courses. The session is conceptual.


## 1. The Big Picture: Why This Session Exists

Training datasets keep growing. At some point a single machine runs out of **disk, RAM, or patience**.

**Analogy: the stadium kitchen.** Scaling a small kitchen recipe to feed a stadium is not "more of the same ingredients". You need:
- multiple kitchens (machines)
- a plan for who cooks what (partitioning and scheduling)
- a way to combine the results (aggregation)

Distributed data processing is that plan for datasets too big for one machine.

### Three core ideas

| # | Idea | Explanation |
|---|------|-------------|
| 1 | **Data outgrows one machine** | Modern datasets run from TBs to PBs, bigger than any single disk or RAM. |
| 2 | **Splitting data is the hard part** | How you split it (the *partition*) decides whether later steps are fast or painfully slow. |
| 3 | **Moving data between machines is expensive** | The **shuffle**, which regroups data across workers, is the most expensive operation in the whole pipeline. |

**Chain of consequence:** the way you split data leads to how much data moves between machines, which leads to shuffle cost, which leads to execution speed. So programs should be designed to **keep shuffles minimal**.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/5facbf9f-e2b3-4a4c-8596-1b3896836fe2" />


## 2. Key Terms

| Term | Definition | Intuition |
|------|------------|-----------|
| **Partition** | One chunk of a dataset, small enough for one worker to handle | One "batch of pages" from the phone directory |
| **Worker** | One process (usually on one machine) doing a slice of the overall work | One cook |
| **DAG (task graph)** | Directed acyclic graph of steps and their dependencies: what must happen before what | The recipe flowchart |
| **Lazy evaluation** | Build a plan first and run it only when a result is needed | Read the whole order before cooking |
| **Shuffle** | Moving data between workers to regroup it by a key | Cooks passing ingredients across kitchens |
| **Lineage** | The record of how a piece of data was derived, used to recompute it if lost | The recipe history for each dish |

## 3. Why Distributed Data Matters for ML

| Use case | Scale | Why distribution is needed |
|----------|-------|----------------------------|
| **Pre-training corpora** | 1–15 TB of compressed text | Doesn't fit on one machine's disk, let alone RAM |
| **Image/video datasets** | e.g. LAION-5B is about 240 TB | Must **stream** from object storage or a **parallel file system** |
| **Batch inference** | Scoring about 10 B rows nightly | One machine takes about a week, a cluster takes hours |
| **Feature engineering** | Group-bys and joins over billions of events | "Spark territory" |

**Tie-in to earlier weeks:** containers, Kubernetes and multi-GPU distributed training (done with 2 GPUs) all scale up when you add more GPUs or nodes. Dask, Ray and Spark from other courses build on the same ideas.

## 4. Partitioning: The Single Most Important Choice

> A dataset = **N partitions**, one or more per worker. The partition choice determines everything downstream: speed, shuffle cost, and balance.

Related allocation ideas include cyclic, static and dynamic distribution of data to workers. The lecture did not go deep on these.

### 4.1 The three main strategies

| Strategy | How it works | Strengths | Weaknesses | Best for |
|----------|--------------|-----------|------------|----------|
| **By row range** | Contiguous blocks of rows go to each worker | Simple and balanced sizes | Joins and group-bys trigger a **full shuffle** | Random ID lookups, no pattern |
| **By HASH(key)** | `hash(key)` picks the worker | Joins on that key need **no shuffle**; load spreads evenly | **Loses ordering**; a hot key can hotspot one worker | Group-by or join on a key (e.g. `user_id`) |
| **By RANGE(key)** | Sorted key ranges go to each worker | **Ordered scans and range queries** stay local | **Imbalanced if the key is skewed** | Queries by date, ZIP, alphabetical order |

### 4.2 Hash partitioning equation

$$
\text{worker}(k) = \operatorname{hash}(k) \bmod N
$$

- $k$ is the partition key (e.g. `user_id`, recipient name)
- $N$ is the number of workers (or partitions)
- All rows with the same $k$ land on the **same** worker, so group-bys and joins on $k$ are local.

### 4.3 The telephone directory example

Suppose a directory has 100 pages and 5 workers, and the query is *"How many people named Kumar?"*

- **Row range:** each worker gets 20 pages. Every worker finds its own Kumars, and the partial results must travel to one collector. This is data movement, and for general group-bys or joins it is a **full shuffle**.
- **Hash on name:** all "Kumar" rows are hashed to the same worker, so the query is answered at **one place** with no cross-worker collection.
- **Range on name:** rows are in alphabetical order, so ordered scans (A–D, E–H, and so on) are easy. Balance depends on how names are distributed.

> **Nuance (my addition):** for a simple count, engines often do *partial aggregation* locally before combining, which is cheap. The lecture's point is stronger for group-bys and joins on non-partition keys, where full rows must be regrouped.

### 4.4 The mail-sorting analogy

- Sorting mail by **ZIP code (range)** works great **if requests come in by ZIP**.
- If people ask by **recipient name**, use a **hash of the name**. It spreads load evenly but loses ordering.
- If you need **ordering** (alphabetical, chronological, ZIP-based), use **range partitioning**.

### 4.5 Rule of thumb

> **Partition by the column you'll GROUP or JOIN on most. Then check for skew.**

### 4.6 Skew

**Skew** means some partitions are much larger than others, so one worker becomes the bottleneck while the others sit idle. A simple measure (my addition):

$$
\text{skew ratio} = \frac{\max_i |P_i|}{\frac{1}{N}\sum_{i=1}^{N} |P_i|}
$$

- A ratio of about **1** means balanced.
- A ratio much greater than 1 means a **hotspot**. Total runtime is set by the slowest worker.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/9ab34694-ef3c-473a-96de-54131564ff52" />

## 5. DAGs and Lazy Evaluation

Your code builds a **plan (DAG)**. The engine optimises the plan and executes it **only when you ask for results**.

### Eager vs lazy

| | **Pandas (eager)** | **Dask / Spark (lazy)** |
|--|--------------------|-------------------------|
| Execution | Every line runs immediately | Nothing runs until a result is requested |
| Optimisation | None across steps | Engine sees the whole plan and optimises it |
| Scale | Single machine, in-memory | Distributed across workers |

```python
# PANDAS (EAGER) — every line runs immediately
df = pd.read_csv('events.csv')
df = df[df.type == 'click']
result = df.groupby('hour').count()

# DASK/SPARK (LAZY) — nothing runs until .compute()
df = dd.read_csv('events.csv')
df = df[df.type == 'click']
result = df.groupby('hour').count()
result.compute()   # NOW it runs
```

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/d15475db-d550-4c70-8f52-abf304c0c713" />


> In Spark the trigger is an **action** (`.collect()`, `.count()`, `.show()`, `.write`), while in Dask it is `.compute()`.

### Optimisations lazy evaluation enables

| Optimisation | What it does |
|--------------|--------------|
| **Predicate pushdown** | Filters rows at the disk/format level (Parquet, ORC), so unneeded rows are never loaded |
| **Projection pushdown** | Reads only the columns you'll actually use |
| **Parallel execution** | Splits the plan across workers automatically |

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/166e0ace-137d-443f-add7-206755b4d37c" />


## 6. The Shuffle: The Expensive Operation

> A shuffle moves data between workers to group it by key. It is **by far** the most expensive operation in a distributed pipeline.

### What triggers a shuffle

- `GROUP BY` on a **non-partition** column
- `JOIN` on a **non-partition** column
- **Global sort**
- **Repartitioning**

### Why it's costly

1. **Serialise every row** (CPU cost)
2. **Network transfer** between N workers
3. **Spill to disk** if memory is tight

### Cost estimate (my addition)

If data is spread randomly across $N$ workers and a shuffle must regroup it by key, each row has a $1/N$ chance of already being on the right worker:

$$
\text{fraction of data moved} \approx \frac{N-1}{N}
$$

$$
V_{\text{shuffled}} \approx D \cdot \frac{N-1}{N} \quad\Rightarrow\quad T_{\text{shuffle}} \approx \frac{V_{\text{shuffled}}}{B_{\text{network}}}
$$

- $D$ is the total data size
- $B_{\text{network}}$ is the effective network bandwidth

As $N$ grows, nearly **all** the data moves, which is why network cost dominates.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/92b5d53a-9072-4b20-88b0-c9e514c48ee3" />


### How to minimise shuffles

| Technique | Explanation |
|-----------|-------------|
| **Pre-partition on keys** | Partition by the keys you'll group or join on |
| **Broadcast small tables** (under about 100 MB) | Copy the small table to every worker instead of shuffling both sides of a join |
| **Filter before you group** | Fewer rows means less data to move |

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/e4bc2f9e-0343-437e-84ef-f62b17ce9beb" />


## 7. Fault Tolerance via Lineage

> Lost a partition? **Re-derive it from its parents.** Usually no checkpoints are needed.

**Analogy:** if you lose your notes for step 5 of a recipe but know steps 1–4 exactly, you just redo step 5. You don't need to photograph every step.

### What happens when a worker dies

| Step | Event |
|------|-------|
| 1 | Worker 7 dies mid-`groupBy`. The scheduler notices it **stopped heartbeating** after a few seconds. |
| 2 | The scheduler looks up **which partitions worker 7 was producing**. |
| 3 | It **re-schedules** those partitions on other workers and re-runs the DAG **from the last cached stage**. |
| 4 | The failed stage finishes and the job continues, with **no manual checkpoint step needed**. |

### Caveat

Long DAGs get expensive to recompute. For deep pipelines, manually call **`.persist()` or `.cache()`** at key stages. This shortens the recompute chain, keeps DAGs short, and saves state at meaningful points.

A rough recompute-cost view (my addition):

$$
\text{recompute cost} \propto \sum_{s \,\in\, \text{stages since last cache}} t_s
$$

so caching at key stages resets the sum.

---

## 8. Worked Example: Choosing a Partition Strategy

| Pipeline | Best scheme | Why |
|----------|-------------|-----|
| Clickstream grouped by `user_id` | **Hash(`user_id`)** | Zero shuffle on that group-by/join |
| Sensor readings queried by date | **Range(`date`)** | Range scans stay local |
| Random ID lookups, no pattern | **Row range** | Simplicity wins; nothing to optimise for |
| Clickstream where 1 bot = 40% of rows | **Hash + salting** | Plain hash would hotspot one worker |

### Salting (preview of Session 3)

Plain hashing sends all of a hot key's rows to one worker. **Salting** adds a random suffix to spread that key across several workers:

$$
\text{salted key} = (k,\; r), \qquad r \sim \text{Uniform}\{0, 1, \dots, S-1\}
$$

$$
\text{worker} = \operatorname{hash}(k, r) \bmod N
$$

- $S$ is the number of salt buckets, chosen as the number of workers the hot key should spread over.
- You aggregate per salted key first, then do a small second aggregation to merge the $S$ partial results.

> **Key insight:** *partitioning and skew-handling are two sides of the same coin.* Hash alone does not guarantee zero shuffle or balance, so skew handling (salting) is needed.

## 9. Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Partitioning by the **wrong column** | Every group-by or join triggers a shuffle | Partition by the most frequent group/join key |
| Calling `.compute()` **too often** | Plan is rebuilt and re-executed repeatedly, which kills optimisation | Build the full plan, compute once at the end |
| **Ignoring skew** | One worker becomes the bottleneck | Check the skew ratio; use salting |

**Why this matters for cost:** in the cloud you pay by compute time, and for LLM workloads by tokens processed. Poor partitioning means wasted time and money.

## 10. Quick Revision Sheet

- **Splitting data is hard.** The partition choice determines downstream speed.
- **Shuffle is the most expensive operation** (serialise, network, disk spill). Minimise it.
- **Hash** is best for group/join on a key, **Range** for ordered scans and range queries, **Row range** for simplicity.
- **Lazy evaluation** (Dask, Spark) enables predicate pushdown, projection pushdown and automatic parallelism. Pandas is eager.
- **Lineage** gives fault tolerance by recomputing from parents. **Cache or persist** at key stages of long DAGs.
- **Broadcast** tables under about 100 MB, **filter before grouping**, and **pre-partition** on join keys.
- **Salting** fixes hotspots such as a bot generating 40% of rows.

### Self-check questions

1. Why is a shuffle more expensive than a local computation?
2. For a `JOIN` on `customer_id`, which partitioning scheme would you choose and why?
3. What is the difference between predicate pushdown and projection pushdown?
4. A worker dies mid-job. How does the system recover without manual checkpoints?
5. When would plain hash partitioning fail, and what is the fix?

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/06a9176c-1818-41b2-be96-6ea648f19f40" />
