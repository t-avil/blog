---
# Basic info
title: "NOTES: Memory of Forgotten"
date: 2025-12-21T23:20:02-08:00
draft: false
description: "From registers to cloud-scale storage: deep notes on memory, latency, hierarchy, and systems research."
tags: ["notes","memory","architecture","numa","storage","systems"]
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
---
> Love, *CPU Memory Access*, FreeBSD Project - [https://people.freebsd.org/~lstewart/articles/cpumemory.pdf](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)

> *HeMem: Scalable Tiered Memory Management for Big Data Applications and Real NVM*, SOSP 2021 - [https://doi.org/10.1145/3477132.3483550](https://doi.org/10.1145/3477132.3483550) ([Technion][1])

> *Assise: Performance and Availability via Client-local NVM in a Distributed File System*, OSDI 2020 - [https://www.usenix.org/system/files/osdi20-anderson.pdf](https://www.usenix.org/system/files/osdi20-anderson.pdf) ([USENIX][2])

> *Don’t Be a Blockhead: Zoned Namespaces Make Work on Conventional SSDs Obsolete*, HotOS 2021 - [https://doi.org/10.1145/3458336.3465300](https://doi.org/10.1145/3458336.3465300) ([ACM SIGOPS][3])

---

Memory systems in computing are a **study of distance and control** more than bits. Every level-from CPU registers to the cluster fabric-exists because **cost (latency, energy) rises with distance and abstraction**. What follows is a deep, chronological and conceptual journey from the smallest unit of state to systems that span racks, guided by both physics and software research.

---

## 1. Registers and the First Illusion: Flat Memory

At the very core of a CPU pipeline lie **registers**-architectural state local to the execution units. These are not “memory” in the conventional sense; they are **bindings for values in-flight**, allocated by the instruction scheduler and renamed to avoid false dependencies. Access to registers happens in **a fraction of a nanosecond** (≈0.25 ns, roughly one cycle on modern CPUs), and there is no latency to hide: data is present or it is not.

Registers are ephemeral. Once another instruction needs a slot, values spill to cache. This spill is the first time memory begins to matter.

---

## 2. Caches: SRAM, Separate Instruction/Data Paths, and Coherence

Modern microarchitectures layer **caches** between registers and main memory, constructed from **Static RAM (SRAM)**. Unlike DRAM, SRAM doesn’t need refresh, making it fast but expensive and lower density.

The hierarchy is subtle:

* **L1 Instruction (I-cache)** and **L1 Data (D-cache)** are *segregated*. They serve different domains: fetched instructions and data operands. This separation is why self-modifying code and instruction fetch pipelines require explicit synchronization; they operate in **different physical units**. Love’s analysis shows that these distinctions carry performance implications that software rarely sees but hardware formalizes. ([Awesome Papers][4])

* **L2 cache** often consolidates data and instructions for each core, trading size for latency.

* **L3 cache** is a shared, inclusive or non-inclusive substrate that enforces coherence across cores.

Coherence protocols (MESI/MOESI) are distributed state machines: a load miss in L1 can trigger remote probes, turning what looked like a microsecond-scale access into a **micro-protocol across cores**. The FreeBSD CPU memory paper emphasizes that **cache misses are topology events**, not simple timing numbers. ([Awesome Papers][4])

---

## 3. NUMA: Latency with Geography

In multisocket servers, memory has **an accent**. Each socket has its own memory controller and channels. An access to local DRAM may be ≈100 ns, but going across a socket can double that, and fetching a cache line from another core on a remote socket can take hundreds of nanoseconds.

NUMA (Non-Uniform Memory Access) emerges not as an optimization but as an architectural necessity: we physically cannot centralize all memory without paying severe latency and bandwidth penalties. To manage this, OSes and runtime systems must be **NUMA-aware**-pinning pages and threads to local memory, migrating hot pages, and orchestrating data locality. Failure to do so makes “shared memory” feel like remote I/O.

---

## 4. DRAM: The Silent Middle

Dynamic RAM (DRAM) is the workhorse of main memory. It stores bits in capacitive cells that leak charge and need periodic refresh. DRAM’s latency (~80-120 ns) comes from *row activation, precharge, and sense amplification*: operations invisible to software but fundamental to performance.

Crucially, DRAM is **volatile**. It forgets everything the moment power is removed. This fact influenced decades of OS design: everything important must be safely stored on stable media and brought into DRAM for processing.

---

## 5. Hard Disk Drives: The Tyranny of Mechanics

Hard Disk Drives (HDDs) predate all of the above. Their performance is dictated by **mechanical physics**:

1. **Seek time** to position the head over the track (≈5-10 ms).
2. **Rotational latency** waiting for the platter to spin (≈8.5 ms at 7200 RPM).
3. **Transfer time** once the head is on target.

These orders-of-magnitude latencies made **sequential access the holy grail**: once the head was placed, the cost of reading many bytes was amortized. `fsck`, FFS, NTFS, ext*, and their spatial allocation policies arise from this tyranny: co-locate related blocks to exploit hardware characteristics.

**Shingled Magnetic Recording (SMR)** took this further by overlapping tracks to increase density. SMR’s tracks behave like **zones**; writing to one track overwrites neighbors, forcing **sequential, host-managed writes**. It reshapes the classical block model and foreshadows zoned SSDs. ([Wikipedia][5])

---

## 6. SSDs and Operational Asymmetry

SSD’s flash memory removed moving parts but introduced a new constraint: **erase-before-write**. Flash is composed of:

* Small **read/write pages** (≈4 KB)
* Large **erase blocks** (≈1-8 MB)
* Write operations that **must be to empty pages**
* Erase operations that clear whole blocks

The result is **operational asymmetry**: reads and writes are cheap, erases are expensive and erode lifetime. To present a conventional block interface, SSDs employ a **Flash Translation Layer (FTL)**, which dynamically remaps logical blocks to physical pages and hides physical realities.

The problem is **Garbage Collection (GC)**: when the drive runs out of clean blocks, the FTL must:

* Identify a victim erase block
* Copy its live pages elsewhere
* Erase it before reuse

This **hidden GC work** causes unpredictable tail latency-a nemesis of storage performance. The OS cannot see inside the FTL, so it can’t plan around these spikes.

---

## 7. Zoned SSDs: Reclaiming Control

Stavrinos et al.’s *Don’t Be a Blockhead: Zoned Namespaces Make Work on Conventional SSDs Obsolete* argues for **exposing flash characteristics to the host** instead of hiding them behind a block layer. ZNS SSDs divide storage into zones that **must be written sequentially and explicitly reset** by the host, removing device-side GC and giving control to software. ([ACM SIGOPS][3])

With ZNS:

* The host knows about physical zone boundaries.
* Garbage collection becomes an explicit action.
* Software can place hot and cold data to minimize erasure.
* Write amplification can be reduced because the system can group writes by temperature.

This is a **major shift**: from opaque devices that manage everything internally to devices that require **host-managed placement**, analogous to how SMR forced host-managed zones on HDDs.

---

## 8. Tiered Memory: PCM and NVM

Emerging Non-Volatile Memory (NVM) technologies like **Phase Change Memory (PCM)** and Intel Optane DC blur the line between memory and storage. They are:

* **Byte-addressable**
* **Persistent**
* Slower than DRAM (≈175-200 ns) but faster than SSDs
* Higher density and cheaper per bit

This disrupts the long-standing assumption that memory must be **volatile and homogeneous**. NVM invites us to rethink memory hierarchies: perhaps memory is not a flat space but a **tiered continuum** from registers to persistent storage.

---

## 9. HeMem: Software Tiered Memory Management

*HeMem: Scalable Tiered Memory Management for Big Data Applications and Real NVM* tackles this directly. The paper identifies that conventional hardware and OS approaches to tiered memory (e.g., Intel Optane DRAM cache or naive OS-level NUMA-like placement) fail when real NVM is part of the hierarchy. ([Technion][1])

Instead of piggybacking on page tables, page faults, or simplistic heuristics, HeMem:

* Uses **CPU performance counters** to sample memory access patterns asynchronously.
* Offloads **page migration with DMA**, which avoids CPU intervention and cache pollution.
* Accounts for **asymmetric bandwidth** between DRAM and NVM.
* Places **policy in user space**, letting applications customize heat-aware placement.

HeMem’s insights are deep: **hotness is a signal, not a static stat**, and migration costs must be amortized and overlapped with application execution. The result is improved runtime and reduced tail latency for big data workloads on real tiered hardware.

---

## 10. CXL and Disaggregated Memory

Compute Express Link (CXL) decouples memory from the CPU socket, ushering in **memory pooling and disaggregation**. CXL memory can be coherent across devices and processors yet sits off the traditional DRAM bus. This enables:

* Elastic memory expansion
* Shared memory pools
* Coherent access over an interconnect

However, latency costs rise: CXL-attached memory is slower than slot-attached DIMMs but faster than networked block storage-creating **another tier** in the latency hierarchy.

Managing this continuum demands systems like HeMem or more advanced OS policies that treat memory as **elastic fabric**, not static banks.

---

## 11. Assise: Distributed File Systems With Client-local NVM

*Assise: Performance and Availability via Client-local NVM in a Distributed File System* applies tiered memory thinking to distributed storage. Instead of a central server managing storage IO, Assise uses **client-local persistent memory** to cache and serve file system state directly. It builds the first persistent, replicated cache coherence layer (CC-NVM) that provides:

* **Crash consistency** via write-logging
* **Linearizability** across nodes
* **Low tail latency** by maximizing local access
* **Fast failover** using hot replicas ([USENIX][2])

Assise shows that **distributed memory and storage should not be separate**: locality and consistency are cheaper to maintain if we treat memory as a first-class, persistent tier at every node.

---

## 12. From Memory to Storage: A Conceptual Synthesis

The evolution of data management is a **move from static geometry to dynamic placement**:

* HDDs taught us **physical locality matters**
* SSDs taught us **operational asymmetry dictates latency**
* ZNS SSDs and PCM/NVM taught us **host-managed placement is inevitable**
* CXL taught us **memory is fabric**
* HeMem and Assise taught us **software must manage tiered heat and coherence**

Traditional abstractions (flat memory, block interfaces, opaque devices) worked when hardware was slow and simple. As hardware becomes more complex-with multiple performance surfaces and asymmetries-**software must regain control**.

---

## 13. Conclusion

Memory is not a flat space. Every layer imposes cost, and every abstraction hides complexity. To build predictable systems, we must make the invisible **visible to software**-the hidden GC, coherence traffic, remote latencies, and tier boundaries-and manage them with informed policies, not hopes.

---

[1]: https://cris.technion.ac.il/en/publications/hemem-scalable-tiered-memory-management-for-big-data-applications/?utm_source=chatgpt.com "HeMem: Scalable Tiered Memory Management for Big Data Applications and Real NVM - Technion - Israel Institute of Technology"
[2]: https://www.usenix.org/system/files/osdi20-anderson.pdf?utm_source=chatgpt.com "[PDF] Assise: Performance and Availability via Client-local NVM in a ..."
[3]: https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s07-stavrinos.pdf?utm_source=chatgpt.com "[PDF] Zoned Namespaces Make Work on Conventional SSDs Obsolete"
[4]: https://paper.lingyunyang.com/reading-notes/conference/sosp-2021/hemem?utm_source=chatgpt.com "HeMem: Scalable Tiered Memory Management for Big Data Applications and Real NVM | Awesome Papers"
[5]: https://en.wikipedia.org/wiki/Shingled_magnetic_recording?utm_source=chatgpt.com "Shingled magnetic recording"




