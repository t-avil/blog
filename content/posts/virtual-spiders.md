---
title: "NOTES: vSpiders or how to virtualize a network"
date: 2025-10-10T14:00:00-07:00
draft: false
description: "A backend engineer’s deep dive into how modern cloud networks work: from Koponen et al.’s classic SDN paper to Google’s Andromeda and SNAP systems."
tags: ["notes","networking","cloud","virtualization","andromeda","snap"]
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

Cloud networking is about **squeezing determinism from chaos** - hundreds of thousands of tenants, each expecting a private, secure, and high-performance network that doesn’t even “feel” shared.


This post takes you from the ground up: we’ll start with what *virtualization* means in networking, explain **overlay networks**, **fabric topologies**, and **control planes**, and then we’ll walk through three real systems: 
**(1)** Koponen et al., *Network Virtualization in Multi-Tenant Datacenters*, NSDI 2014 - [PDF from USENIX](https://www.usenix.org/system/files/conference/nsdi14/nsdi14-paper-koponen.pdf)
**(2)** Dalton et al., *Andromeda: Performance, Isolation, and Velocity at Scale*, NSDI 2018 - [PDF from USENIX](https://www.usenix.org/system/files/conference/nsdi18/nsdi18-dalton.pdf)
**(3)** SNAP: *Snap: a Microkernel Approach to Host Networking*, SOSP / Google - [Official web page (includes PDF) at Google Research](https://research.google/pubs/snap-a-microkernel-approach-to-host-networking/) 

---

## 1. Starting From Zero: What Is Network Virtualization?

Let’s start with what problem we’re solving.

In a datacenter, thousands of virtual machines (VMs) from different tenants share the same physical network. Each tenant expects their own **private network**, with their own IPs, firewalls, and routers - even though physically, all of them are plugged into the same switches.

This illusion is achieved using **network virtualization**: abstracting one physical network into many *logical* ones. It’s like giving every tenant a fake Ethernet cable that “feels” dedicated but is actually software-defined.

### Overlay and Underlay

The real, physical network (switches, routers, cables) is called the **underlay**.
The virtual network that the VMs see - isolated, programmable, and logically routed - is called the **overlay**.

To connect the two, each packet from a VM is **encapsulated**: wrapped in another packet with special headers, then tunneled across the underlay. When it reaches the destination host, the outer header is stripped, and the original packet is delivered to the target VM.

This wrapping process uses protocols like **VXLAN** or **GRE**. Conceptually, it’s the same idea as a shipping box - you can ship arbitrary data safely, even if the delivery network is shared.

---

## 2. The Fabric and the Planes

In large datacenters, the physical topology looks like a **Clos fabric** - layers of leaf and spine switches that provide uniform bandwidth between any two racks.

Over this hardware, networking is divided into three conceptual layers:

* **Data plane:** the layer that moves packets (e.g., a vSwitch on a host).
* **Control plane:** decides how packets should be routed and installs those rules into the data plane.
* **Management plane:** orchestrates everything globally (for example, when a new tenant is created).

This separation is the foundation of **Software-Defined Networking (SDN)**. The control plane is centralized and “smart”; the data plane is distributed and “dumb but fast.”

---

## 3. Koponen et al. (NSDI 2014): The OG SDN Design

In 2014, Teemu Koponen and colleagues proposed one of the first scalable SDN architectures for **multi-tenant datacenters**. Their system was based on **Open vSwitch (OVS)** and **tunneling**.

Each hypervisor host ran a **vSwitch**, a small virtual Ethernet switch that handled all packet forwarding for the VMs on that host. The vSwitch stored forwarding rules programmed by a **central controller** - a logically centralized service that understood all tenants’ topologies.

Here’s how it worked:

1. A VM sends a packet.
2. The vSwitch checks its flow table (like a hash map of “if packet matches → do this”).
3. If the flow rule is missing, the packet is sent to the controller.
4. The controller computes what to do and pushes a new rule back to the vSwitch.
5. Future packets of that flow now follow the cached rule.

The control plane (the central brain) knows everything: which VMs exist, which tenants they belong to, and how they should be connected. The data plane (vSwitches) simply enforces those decisions.

This architecture solved *isolation* and *mobility* (VMs could move between servers without changing IPs). But it had a cost: every packet passed through multiple software layers, context switches, and kernel calls, limiting throughput to a few hundred thousand packets per second per host.

---

## 4. Andromeda: Scaling the Dream (Google NSDI 2018)

Fast-forward to Google Cloud Platform (GCP). By 2018, the scale problem had exploded. The same ideas as Koponen’s system still applied, but now the bar was higher:

* Millions of VMs.
* Microsecond-level latency budgets.
* Zero downtime feature rollout.
* Strong performance isolation between tenants.

### The Core Idea: Hierarchical Flow Processing

Andromeda’s big innovation is the **flow hierarchy** - different processing “paths” depending on the flow’s requirements:

* **Fast Path:** pure userspace, OS-bypass packet processing for high-throughput, low-latency traffic (like VM-to-VM in the same cluster).
* **Coprocessor Path (Slow Path):** CPU-heavy tasks like encryption, NAT, or firewalling.
* **Hoverboard Path:** a “catch-all” for tiny, short-lived flows, using shared gateways.

Instead of processing every packet with full software stack overhead, Andromeda sends 99% of traffic down the fast path, which is basically a tight loop in userspace reading and writing directly to NIC queues via **shared memory**.

### OS Bypass and Shared Memory Queues

Normally, packet I/O goes through the OS kernel: system calls, interrupts, scheduler wakeups - slow.
Andromeda uses **busy-polling** and **shared memory rings** to directly read packets from VM NIC queues and write them to host NIC queues. It’s like cutting the operating system out of the hot path entirely.

This userspace networking stack evolved across versions:

* *Andromeda 1.0:* used modified OVS in kernel.
* *Andromeda 2.0:* moved to full userspace dataplane.
* *Andromeda 2.1:* bypassed the VMM entirely.
* *Andromeda 2.2:* added **DMA offload** using Intel QuickData Engines.

DMA (Direct Memory Access) means copying packets directly from VM memory to NIC buffers without using the CPU, freeing cycles for computation.

### Hoverboard: Scaling Control Plane

In the original SDN model, the control plane installed one flow rule per connection. That breaks at Google scale. Hoverboard fixes this by sending small, short-lived flows through shared **gateways** that already know how to forward traffic, so no per-flow setup is needed.

This design drastically reduces control-plane churn and lets GCP spin up thousands of VMs in seconds.

### Isolation and Hardware Offload

Because GCP is multi-tenant, Andromeda enforces **performance isolation**. Fast and slow paths are assigned different cores or NIC queues, so one tenant’s heavy traffic doesn’t interfere with another’s. Hardware offloads (encryption, checksums, segmentation) ensure that CPU time isn’t wasted on repetitive work.

**Reference:**
Dalton et al., *Andromeda: Performance, Isolation, and Velocity at Scale in Cloud Network Virtualization*, USENIX NSDI 2018.

---

## 5. SNAP: The Next Generation (Google, 2021)

By 2021, Google had evolved Andromeda into **SNAP (Scalable Network Architecture Platform)** - a re-architecture for even larger scale and faster innovation.

### Key Difference: Decoupling Features From Fast Path

In Andromeda, feature growth (like new firewall rules or telemetry) still touched the dataplane code. SNAP separates these concerns entirely:

* **Fast Path:** minimal, stable, performance-critical core.
* **Service Modules:** plug-in features (load balancing, encryption, flow export) that can be updated independently.
* **Remote Packet I/O (RPIO):** moves NIC access to dedicated “network service hosts,” letting compute hosts focus purely on user VMs.

This modularization allows GCP to roll out new network features without risking downtime or performance regressions in the dataplane.

### Infrastructure Design

Each SNAP “host” runs multiple **network service threads**, each pinned to CPU cores, directly communicating with **hardware queues** on smart NICs.
Packet metadata is cached in **shared memory regions**, and inter-core communication is done via **lock-free ring buffers**.

Feature modules attach through a **service graph** - a chain of packet-processing functions like a microservice pipeline but running in-process. Each module declares its compute and latency requirements so the scheduler can pin it optimally.

This design makes SNAP both **faster** and **safer to evolve** - it’s a clean separation between network plumbing and value-added features.

**Reference:**
Hock et al., *SNAP: A Microkernel Approach to Cloud Network Virtualization*, USENIX NSDI 2021.

---

## 5.5. Nitro: AWS approach with custom NIC

AWS takes a fundamentally different approach with the Nitro System, providing hardware-enforced isolation for EC2 instances through custom NIC. Nitro Cards handle networking, storage, and management functions, while the Nitro Security Chip establishes a hardware root of trust, verifies firmware integrity, and ensures secure boot - basically most of the stuff needed for HAVEN/INTEL CGX access control. The remaining Nitro Hypervisor is deliberately minimal, focusing solely on CPU and memory allocation and mapping I/O virtual functions, reducing the attack surface and minimizing interference between tenants. By pinning CPU cores and memory exclusively to individual instances and delegating I/O to hardware, Nitro provides near bare-metal performance with strong isolation guarantees. Networking is handled via the Elastic Network Adapter (ENA) on dedicated Nitro hardware, eliminating shared software dataplanes and reducing noisy neighbor effects. This architecture also enables advanced features like Nitro Enclaves, which create isolated execution environments within an instance for sensitive workloads, providing memory and vCPU isolation without network connectivity and preventing even root access from reaching the enclave.


---

## 6. Applied Summary: From Research to Practice

If you’re a backend engineer, here’s the intuitive way to think about the whole progression:

| Layer           | Koponen et al. (2014)  | Andromeda (2018)                | SNAP (2021)                  |
| --------------- | ---------------------- | ------------------------------- | ---------------------------- |
| Concept         | SDN for VMs            | Datacenter-scale virtualization | Modular network microkernel  |
| Data Plane      | vSwitch (kernel)       | Userspace OS-bypass             | Multi-core service graph     |
| Control Plane   | Centralized controller | Hierarchical, distributed       | Modular API-based            |
| Performance     | Flexible, slow         | Hardware-offloaded, fast        | Hardware + modular isolation |
| Scale           | Thousands of hosts     | Hundreds of thousands           | Millions of endpoints        |
| Feature Updates | Global rollout         | Safe partial deployment         | Hot-swappable modules        |

Think of **Koponen** as the “Linux kernel” moment for networking - the foundation.
**Andromeda** is like DPDK meets Kubernetes - highly optimized, software-defined, yet dynamic.
**SNAP** is the “microkernel” rearchitecture - breaking up the monolith to scale feature velocity.

---

## 7. Final Thoughts

Network virtualization isn’t just about packets - it’s about **turning the network into an API**.

Koponen gave us the first blueprint.
Andromeda made it real at hyperscale.
SNAP turned it into a living, evolving platform.

Together, they form the story of how Google - and the industry at large - turned the network into software.
