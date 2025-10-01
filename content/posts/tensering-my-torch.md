---
title: "NOTES: TensorFlow, TorchDynamo, and TorchInductor"
date: 2025-09-30T23:23:37-07:00
draft: false
description: "My reading notes on two generations of ML system papers: TensorFlow’s distributed execution model and PyTorch 2.0’s compiler stack (TorchDynamo + TorchInductor)."
tags: ["notes","machine-learning","distributed-systems","compilers","tensorflow","pytorch"]
author: "Me"

canonicalURL: "https://canonical.url/to/page"
hidemeta: false
searchHidden: true

showToc: true
TocOpen: false
UseHugoToc: true

ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
comments: false

disableHLJS: false
disableShare: false

editPost:
  URL: "https://github.com/t-avil/blog/tree/main/content"
  Text: "Suggest Changes"
  appendFilePath: true
---

If TensorFlow was about making large-scale distributed ML programmable, PyTorch 2.0 is about making dynamic Python code compilable without losing flexibility. The TensorFlow paper walked through distributed dataflow graphs, parameter servers, checkpointing, and execution placement. In contrast, the PyTorch 2.0 paper focused on TorchDynamo and TorchInductor, showing how dynamic graphs can be captured and compiled into fused kernels that run nearly as fast as handwritten CUDA. Below are my notes on the history, design, and impact of these systems.

---

## TensorFlow: Distributed ML via Dataflow Graphs

**History:**  
Released in 2015 (paper at OSDI 2016), TensorFlow was Google’s successor to DistBelief, but with the key difference of being open source and user-facing.

**Key ideas:**
- **Dataflow graphs**: nodes are ops, edges are tensors. Easy to parallelize and reason about.
- **Device placement**: runtime decides whether ops run on CPU, GPU, or TPU, with flexibility for heterogeneous clusters.
- **Parameter servers**: split model state management from computation; workers compute gradients, parameter servers update global state.
- **Checkpointing and recovery**: periodic backups of model state for fault tolerance and restart.
- **Mutations**: even small ops like `+=` were highlighted for their efficiency in a distributed setting.

**Impact:**
- Brought distributed ML into mainstream practice.  
- Made concepts like parameter servers widely known.  
- Inspired entire ecosystems of frameworks (MXNet, Horovod, PyTorch Distributed).  
- Also exposed challenges like **stale gradients**, a problem still not fully solved.  

---

## TorchDynamo: Graph Capture for Dynamic PyTorch

**History:**  
Introduced with PyTorch 2.0 (2023), TorchDynamo is the front-end of the new compiler stack. It uses CPython hooks to intercept execution and turn PyTorch programs into graphs.

**Key ideas:**
- **Frame evaluation hooks**: intercept Python bytecode at runtime.  
- **FX graph IR**: TorchDynamo lowers programs into FX graphs, a PyTorch-native intermediate representation.  
- **Guards**: attach conditions (tensor shapes, dtypes, control flow) so graphs are only reused when valid.  
- **Minimal optimization**: Dynamo itself does light simplifications but mainly produces a clean graph for backends.  

**Impact:**
- First system to make dynamic Python code reliably compilable in PyTorch.  
- FX graphs became the “lingua franca” of PyTorch compilers.  
- Established the bridge between eager-mode flexibility and compiled execution.  

---

## TorchInductor: Fused Kernels and Code Generation

**History:**  
Also part of PyTorch 2.0, TorchInductor is the default backend compiler that takes FX graphs and lowers them to optimized code.

**Key ideas:**
- **Operator fusion**: combine sequences of ops into a single kernel to cut down launch overhead and improve locality.  
- **Portable kernel generation**: emits C++/OpenMP for CPUs, Triton for GPUs.  
- **Dynamic shapes support**: designs execution plans that adapt without speculative recompilation.  
- **Seamless use**: users just call `torch.compile`, no rewriting needed.  

**Impact:**
- Turned PyTorch 2.0 into a “compiled PyTorch” without losing dynamic semantics.  
- Gave researchers nearly hand-tuned performance with no CUDA required.  
- Strongly boosted distributed training frameworks (DDP, FSDP, DeepSpeed) by improving single-node throughput.  
- Made Triton a hot topic in ML systems by showing it could serve as a production GPU kernel generator.  

---

## Closing Thoughts

Reading these two papers back to back felt like looking at two eras of ML systems design. TensorFlow focused on *distributed execution and system scalability*, while PyTorch 2.0 doubled down on *compilation and single-node performance*. TensorFlow brought parameter servers, checkpoints, and the dataflow model into the vocabulary of ML practitioners. PyTorch 2.0 brought graph capture, FX IR, and kernel fusion into the mainstream. Together they map the shift from "how do we scale models across data centers?" to "how do we make dynamic Python code run like C++?"

---
