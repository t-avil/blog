---

# Basic info

title: "NOTES: training the distributed scale?"
date: 2025-10-21T04:52:54-07:00
draft: false
description: "Summary of DDP, FlexFlow, ZeRO, FSDP, activation checkpointing and ZeRO-Infinity for GPU/ML inference and training systems."
tags: ["notes", "distributed-training", "large-models", "data-parallel", "model-parallel", "memory-optimization"]
author: "Me"


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

disableHLJS: false
disableShare: false

# Edit link
editPost:
  URL: "https://github.com/t-avil/blog/tree/main/content"
  Text: "Suggest Changes"   
  appendFilePath: true      
--------------------

### Key definitions

* **Data parallelism**: every GPU (worker) holds a full copy of the model and processes a different slice of the training data; after computing gradients they are combined (synchronised).
* **Model parallelism**: the model itself is split across devices (layers or parameters), so different GPUs handle different parts of the model.
* **Parameter / optimizer state / gradient**: model *parameters* = weights; *gradients* = derivatives computed in back-prop; *optimizer state* = extra per-parameter data (momentum, Adam’s moments).
* **Collective communication / all-reduce**: when multiple devices coordinate data exchange, e.g., an *all-reduce* sums up a tensor across all devices then distributes the result back.
* **Sharding**: dividing a tensor or state among workers so each stores only a part (instead of full replication).
* **Offloading**: moving data (parameters, states) to slower but larger memory (CPU RAM or SSD/NVMe) when it isn’t actively needed on GPU.

---

## 1. [PyTorch Distributed Data Parallel (DDP)](https://arxiv.org/abs/2006.15704?utm_source=chatgpt.com)

There is a problem of training large deep-neural-network models efficiently across many GPUs, so we used DDP, which replicates the model on each GPU, processes different samples in parallel, and then synchronises gradients via collective operations, achieving high scalability. The main takeaway / innovation / addition / insight is **efficient overlap of computation and communication** - it was implemented through gradient “bucketing” (grouping gradient tensors) and overlapping gradient reduction during backward pass, and enabled by the process-group abstraction (in `torch.distributed`) and collective backends such as NCCL. The second takeaway is **using a process-group abstraction that hides details of inter-GPU communication** - implemented via `torch.distributed.ProcessGroup`, and enabled by NCCL or MPI based libraries that manage all-reduce/ broadcast under the hood. The third insight is **reducing redundant copies of model parameters and buffers across processes** - rather than each process fully copying every buffer, DDP uses shared storage for model parameters in multi-process mode (via `torch.multiprocessing` + shared CUDA memory) so that intra-node clones reuse memory rather than duplicating it. And the last additional interesting fact is that DDP achieves near-linear scalability (up to ~256 GPUs) when configured properly (bucketing, overlapping) as shown in the PyTorch paper. ([arXiv][1])

**Clarification on the “shared memory between processes” point**: In PyTorch DDP when using multi-process on one node, each process holds its own CUDA context but the CPU memory region can use `torch.multiprocessing` with shared memory segments, and PyTorch also uses the NCCL backend so that intra-node GPUs can share parameter memory without full duplication when possible. It’s not that Python’s GIL blocks that sharing - each process has independent Python interpreter, and low-level CUDA/NCCL primitives handle memory sharing and communication. So the insight is: instead of naively replicating full model state in each process, DDP reuses underlying memory (especially on a node) and minimizes inter-process redundant copies, thereby saving GPU/CPU memory and reducing pressure.

---

## 2. [FlexFlow](https://arxiv.org/abs/1807.05358?utm_source=chatgpt.com)

There is a problem of finding the optimal parallelism strategy for DNN training (beyond simple data or model parallelism) so we built FlexFlow, which automatically searches a broader space of parallel strategies (Sample, Operator, Attribute, Parameter) and achieved up to ~3-4× speed‐up over prior systems. The main takeaway is the introduction of the **SOAP search space** (Sample, Operator, Attribute, Parameter) for parallelism - implemented via a simulator that predicts execution cost, and enabled by modelling both computation and communication accurately in that simulator. The second takeaway is **guided randomised search (e.g., MCMC) through the SOAP space** - implemented using the simulator + search algorithm, and enabled by the fact that the simulator is ~1000× faster than executing the full strategy. The third insight is **adapting to heterogeneous GPU clusters and layer/operator-specific splits** - implemented by allowing partitioning not just by samples or weights but also by operators and attributes (e.g., splitting internal tensor attributes), enabled by the flexible runtime that maps operations across devices. And the last additional interesting fact is that FlexFlow sometimes picks unexpected parallelisation patterns (e.g., splitting attributes inside a tensor) that outperform standard data/model parallelism by up to 3.8×. ([arXiv][2])

**Bonus: Other two dimensions besides “load model on different devices & sync gradients”**: In addition to Sample‐parallel (data parallel) and Parameter/model‐parallel, FlexFlow supports:

* Operator parallelism: splitting the work of a single operator (e.g., a matrix‐multiply) across devices (the “O” in SOAP).
* Attribute parallelism: splitting within the tensor attributes (e.g., partitioning along channel/feature dimension) (the “A” in SOAP).
  So FlexFlow explores all four dimensions (S,O,A,P) to find novel configurations.

---

## 3. [ZeRO (Zero Redundancy Optimizer)](https://arxiv.org/abs/1910.02054?utm_source=chatgpt.com)

There is a problem of GPU memory constraints when training extremely large models (billions-to-trillions of parameters), so we used ZeRO, which shards optimizer states, gradients, and parameters across GPUs to achieve drastic memory reduction and enable much bigger models. The main takeaway is **optimizer-state sharding** (Stage 1) - implemented by partitioning the optimizer states so each device only stores a fraction of the momentum/Adam states, and enabled by distributed coordination of update steps and communication to fetch needed states. The second takeaway is **gradient and parameter sharding** (Stages 2 and 3) - implemented via splitting gradients and weights across devices so no device holds full copies, and enabled by efficient collective operations (all-gather, reduce-scatter) that aggregate results across shards. The third insight is **minimizing communication overhead while scaling memory linearly with devices** - implemented by designing communication primitives that only exchange minimal necessary data instead of full copies, and enabled by high-bandwidth interconnects + optimized collective algorithms. And the last additional interesting fact is that ZeRO required *no changes to model code* (you could plug it in to many existing PyTorch models) and has been used to train >100B parameter models. ([arXiv][3])

**About the “communication trick / all-reduce”**: All-reduce is a collective that sums a tensor across all devices and returns the result to each device. ZeRO optimizes this by using *reduce-scatter* (each device receives part of the sum) then *all-gather* (reconstruct full tensor) rather than full all-reduce on large full copies. This dramatically reduces communication volume and overlap time. ([Engineering at Meta][4])

---

## 4. [PyTorch Fully Sharded Data Parallel (FSDP)](https://engineering.fb.com/2021/07/15/open-source/fsdp/?utm_source=chatgpt.com)

There is a problem of memory inefficiency and scalability limits in standard DDP (full replication of parameters/gradients/optimizer states) when training very large models, so we use FSDP, which shards all model parameters, gradients, and optimizer states across GPUs and overlaps communication intelligently, achieving large-model training with much lower per-GPU memory. The main takeaway is **“flat parameter” sharding plus on-the-fly load/unload of parameter shards** - implemented by each FSDP unit wrapping a module, creating a FlatParameter from its weights, dynamically all-gathering when needed on forward/backward and then freeing immediately, enabled by PyTorch’s tensor view mechanics and caching allocator. The second takeaway is **overlap of communication with forward/backward** - implemented by decomposing all-reduce into reduce‐scatter + all‐gather and scheduling that to run concurrently with computation, enabled by the CUDA memory caching allocator and asynchronous streams by PyTorch. The third insight is **nestable/sharded wrapping of submodules for fine granularity** - implemented by allowing nested FSDP units per layer or group, and enabled by PyTorch’s modular `nn.Module` API and dispatcher system for parameter handling. And the last additional interesting fact is that FSDP provides a drop-in replacement for DDP in PyTorch and is now used in production at large scale. ([Engineering at Meta][4])

---

## 5. [Activation Checkpointing](https://pytorch.org/docs/stable/checkpoint.html?utm_source=chatgpt.com)

There is a problem of GPU memory being consumed by storing every activation (intermediate layer output) for back-propagation in deep networks, so we use activation checkpointing, which drops or only stores some activations and recomputes them during the backward pass, achieving substantial memory savings (at the cost of some extra compute). The main takeaway is **memory-vs-compute trade-off** - implemented by marking certain layer outputs as “checkpoint” so they aren’t saved, and recomputing them in backward pass, enabled by frameworks (e.g., PyTorch) that support re-running forward subgraphs and backward hooks. The second takeaway is **scalability for very deep models** - implemented through selective checkpointing of expensive layers (transformers, CNN stacks) and enabled by custom backward scheduling logic. The third insight is **compatibility with sharded/parallel training** - implemented by combining checkpointing with e.g., ZeRO or FSDP to further reduce memory, and enabled by runtime memory planners that coordinate recompute with communications. And the last additional interesting fact is that checkpointing can reduce activation memory by ~80% in some deep transformer stacks, albeit with ~10-30% compute overhead.

---

## 6. [ZeRO-Infinity](https://arxiv.org/abs/2104.07857?utm_source=chatgpt.com)

There is a problem of training trillion+ parameter models on limited GPU memory, so we use ZeRO-Infinity, which extends ZeRO by offloading optimizer states, gradients and parameters to CPU or NVMe storage tiers when idle, achieving virtually unlimited model size while still using GPU compute efficiently. The main takeaway is **hierarchical memory management with offload tiers** - implemented by keeping hot data on GPU, warm data on CPU, and cold data on NVMe, and enabled by asynchronous data movement overlapped with computation. The second takeaway is **communication and bandwidth optimisation across tiers** - implemented by smart prefetching, compression, and asynchronous overlap, and enabled by the runtime scheduler of DeepSpeed. The third insight is **scalable training from few GPUs through large clusters using mixed offload** - implemented by hybrid parallelism (data + tensor + pipeline) plus offload, and enabled by ZeRO-Infinity’s integration into DeepSpeed. And the last additional interesting fact is that ZeRO-Infinity has been used to train >100 billion-parameter models on limited GPU counts by leveraging CPU + NVMe offloading.

---

### Key Takeaways 

- We can offload data and model parts from GPU to CPU or NVMe to save memory.  
- We can use checkpoints to save progress or skip them and recompute when needed.  
- We can sync only parameters that were updated, reducing unnecessary communication.  
- We can shard models by weights, gradients, and optimizer states to save memory and balance load across the network.  
- We can automatically find the best parallelization strategy using a simulator.  
- We can overlap computation and communication so GPUs stay busy while syncing.  

[1]: https://arxiv.org/abs/2006.15704?utm_source=chatgpt.com "PyTorch Distributed: Experiences on Accelerating Data Parallel ..."
[2]: https://arxiv.org/abs/1807.05358?utm_source=chatgpt.com "Beyond Data and Model Parallelism for Deep Neural Networks"
[3]: https://arxiv.org/abs/1910.02054?utm_source=chatgpt.com "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
[4]: https://engineering.fb.com/2021/07/15/open-source/fsdp/?utm_source=chatgpt.com "Fully Sharded Data Parallel: faster AI training with fewer GPUs"
---