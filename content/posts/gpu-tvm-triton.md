---
title: "NOTES: GPU Architecture, TVM, and Triton"
date: 2025-10-06T14:00:00-07:00
draft: false
description: "My reading notes on GPU hardware, TVM compilation, and Triton JIT for efficient tensor computation."
tags: ["notes","machine-learning","gpu","compilers","tvm","triton"]
author: "Me" # For multiple authors, use: ["Me", "You"]

# Metadata & SEO
canonicalURL: "https://canonical.url/to/page"
hidemeta: false
searchHidden: true

# Table of contents
showToc: true
TocOpen: false
UseHugoToc: true

# Post features
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
comments: false

# Syntax highlighting
disableHLJS: false # set true to disable Highlight.js
disableShare: false

# Edit link
editPost:
  URL: "https://github.com/t-avil/blog/tree/main/content"
  Text: "Suggest Changes"   # edit link text
  appendFilePath: true      # append file path to the edit link
--------------------

If TensorFlow showed us how to scale ML across entire data centers, and PyTorch 2.0 showed us how to compile dynamic Python code, GPUs and compiler stacks like TVM and Triton show us how to squeeze the absolute last drop of performance out of hardware. This post is my attempt to tie together GPU architecture, TVM’s tensor-level compilation, and Triton’s clever tiling JIT strategy, all in one nerdy, slightly dorky package.

---

## GPU Architecture: Warps, SMs, and Parallel Execution

Hardware instructions on modern GPUs are parallelized into multiples of **32 threads**. These chunks of 32 threads are called **warps**, but the instructions inside a warp can differ - though it’s more efficient if they’re all the same. Each worker, or **Streaming Multiprocessor (SM)**, can execute about **4 warps simultaneously**.

Tasks, also called **blocks**, are assigned to SMs by the GPU scheduler. The kernel function determines how many tasks to launch and how many instructions each contains. Let’s run an example to see this in action:

Suppose we have **2 different kernels**, each with 51 blocks, and **100 SMs**. Most blocks (say 50 from each kernel) contain 8 warps. Each SM executes 4 warps per cycle, so these blocks take **2 cycles** each. The last two blocks (7 and 9 warps) are queued, requiring **Max(ceil(7/4), ceil(9/4)) = 3 cycles**. Total execution: **5 cycles**.

---

### Warp Instruction & Memory Table

| Type                         | Example                                                              | Warp-level execution                                              | Memory hierarchy / latency               |
| ---------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------- |
| 1-cycle / fast               | add, sub, fadd, fmul, fma, bitwise ops (and, or, xor), compare (cmp) | 1 cycle per warp                                                  | Registers; fastest, no memory dependency |
| Slow / multi-cycle           | idiv, fdiv, sqrt, rsqrt                                              | Multiple cycles per warp, pipelined                               | Registers; some compute-bound            |
| Global memory                | ld.global, st.global                                                 | High latency (~300–600 cycles) per warp; scheduler switches warps | DRAM                                     |
| Shared memory                | ld.shared, st.shared                                                 | Low latency (~1–10 cycles) per warp; can bank conflict            | SM-local shared memory                   |
| Texture / special units      | ld.texture, atomic                                                   | Depends on unit; pipelined                                        | Read-only caches / special memory        |
| Matrix multiply / tensor ops | wmma, dp4a, fma                                                      | 1 cycle throughput per warp on tensor cores                       | Registers or shared memory for operands  |

**Key notes:**

* **Global memory:** shared across kernels; multiple kernels can read/write the same chunk simultaneously. Latency is high (~300–600 cycles).
* **Shared memory:** private to each block/SM; fast (~1–10 cycles).
* **Registers:** private to each thread; fastest (<1 cycle).

So, global memory is like a shared kitchen, while shared memory and registers are your private meal prep stations.

---
 
## TVM: From Computation Graphs to Hardware Execution

[TVM](https://tvm.apache.org) starts by taking in an **IR**, basically a computation graph that’s detached from any specific execution schedule. It runs optimizations over this graph, then compiles it for any target hardware.

Before actual code generation, the model is converted into **Tensor Expressions (TE)** - the highest level of abstraction where even scalar operations are represented as tensor computations. From there, TVM progressively lowers the code down to hardware-native operations:

* **TPUs:** tensor instructions
* **CPUs:** scalar code
* **GPUs:** vectorized code

**Vectorization** is essentially mapping your data elements to threads so that each warp executes the same operation simultaneously.

Even at the tensor level, TVM applies several techniques within the TE IR:

* **Tiling:** breaks large loops into smaller blocks that fit better into caches or GPU thread blocks.
* **Unrolling:** expands short loops into straight-line code to reduce overhead and expose more instruction-level parallelism.
* **Parallelism:** maps independent loop iterations to multiple CPU threads or GPU blocks so they can run at the same time.

This transforms nested loops into **full grid-like execution on GPUs**, where each tile or loop block maps to a thread block or warp expected to run in parallel on an SM.

Finding the most optimal execution schedule isn’t straightforward - the search space is massive, and likely **not even a P problem**. There are many tunable knobs:

* tiling sizes
* unroll factors
* thread and block mappings
* vectorization width
* memory layouts

Because of this, TVM uses **machine-learning-based autotuning** to predict which optimization combinations will give the best runtime, constantly refining its schedule through iterative search and real hardware feedback.


**Conceptually:** TVM receives a graph of computation, optimizes it, converts it into imperative tensor operations with a default schedule (how to unwrap and execute the loops), and fuses those two together.
Then it compiles and runs the generated code.
A background ML tuner observes runtime performance, tweaks the schedule parameters, re-fuses and recompiles, and repeats - gradually learning about the underlying hardware (treated as a black box) and version-controlling the best-found schedule.

---

## Triton: Tiles, JIT, and Memory Locality

[Triton](https://arxiv.org/abs/2003.06106) takes a slightly different, complementary approach. Instead of fully automated schedule search like TVM, Triton introduces **tiles** - small sub-tensors that divide computations into manageable chunks.

**How it works:**

* Tiles map naturally to **thread blocks** (groups of warps on an SM).
* Compiler analyzes tile **iteration spaces, liveness, and memory dependencies**.
* Each tile can be loaded into **shared memory or cache**, improving memory locality.
* Triton JIT automatically generates GPU code: partitioning tiles, mapping threads/warps, vectorization, and memory reuse.

**Interesting quirk:** Triton leaves **tile sizing** to the programmer but handles everything else automatically (which is impressive taking into account that the paper demonstrates remarkable efficiency for large tensor operations without manually writing CUDA), which seems like a small jump conceptually but actually simplifies codegen and improves performance drastically.

---

## Closing Thoughts

Reading GPU hardware, TVM, and Triton together is like zooming from transistor-level execution to high-level tensor graphs. GPUs define the **rules of parallelism**, TVM automates **how loops and tensor ops map to hardware**, and Triton shows how **tiles + JIT + memory locality** unlock blazing speed.

Together, they tell the story: *hardware is complicated, but smart compilation and scheduling can make Python code run like native C++*. For anyone who loves digging under the hood of ML performance, these papers are an absolute goldmine.

* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)
* [Triton: An Intermediate Representation and Compiler for Writing Efficient GPU Code](https://arxiv.org/abs/2003.06106)
