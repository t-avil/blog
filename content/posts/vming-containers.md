---
title: "NOTES: VMing the Containers - The Latency-Availability Tradeoff"
date: 2025-10-05T14:30:00-07:00
draft: false
description: "Exploring the tradeoffs between containers and virtual machines, from trap-and-emulate to modern orchestration systems. Spoiler: they're both here to stay."
tags: ["notes","virtualization","containers","docker","distributed-systems","kubernetes"]
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

Before we dive into containers vs VMs, let's talk about what virtualization actually *is*. If you're coming from an OOP background, think of virtualization as the ultimate facade pattern: you're presenting a clean interface (a "virtual machine") that hides the messy reality underneath (the actual hardware and hypervisor). 

There are three main levels of virtualization, each with different tradeoffs:

1. **Trap-and-emulate**: The classic approach where privileged instructions from the guest OS "trap" into the hypervisor, which then emulates them. This is what early VMware and QEMU did in software mode.

2. **Paravirtualization**: The guest OS knows it's virtualized and makes explicit calls to the hypervisor instead of using privileged instructions. Xen pioneered this approach, and it's more efficient than pure trap-and-emulate but requires modifying the guest OS.

3. **Hardware-assisted virtualization**: Intel VT-x and AMD-V added CPU extensions that let hypervisors run guest OSes at native speed without software tricks. This is what modern VMs use under the hood.

Now, with that context in mind, let's talk about the container vs VM debate.

---

## The Great Debate

There's no clear winner between virtualization and containers. If one of them were clearly worse, we wouldn't still be studying both.

In my opinion, the real tradeoff here is between latency and availability.

Here I will be speaking about Docker as I am most versed in it. If we imagine a gradient with containers on one side and virtual machines (VMs) on the other, the key difference lies in how the state of the system is managed. For example, real VMs are abstracted enough to capture the entire state of the guest OS. This allows for pausing, checkpointing, kernel upgrades, and migration. On the other hand, container runtimes like Docker (runc or any variant that suppresses the shim API) don't even support migration and barely support checkpointing. The tradeoff in performance is significant tho. As shown in [*Container-based Operating System Virtualization: A Scalable, High-performance Alternative to Hypervisors*](https://dl.acm.org/doi/10.1145/1272996.1273025) by Stephen Soltesz et al. (EuroSys 2007), the overhead in containers is almost 2x better than hypervisors for many workloads.

That said, the paper seems a bit outdated when it talks about containers' security, networking, and isolation issues, as Docker has addressed most of these. The paper outlines several limitations in container-based operating systems (COS) like Linux-VServer, Virtuozzo, and Solaris Zones, including issues with fault, resource, and security isolation. Containers such as VServer offer limited fault isolation, where a shared kernel can result in a system-wide failure if a fault occurs within one container. Additionally, resource isolation is not foolproof, leading to "cross-talk" between containers that can cause performance inefficiencies, especially when scaling. Security isolation is also weak, as containers share system resources like the PID space, which undermines the strict isolation provided by hypervisors. Networking issues arise from VServer's inability to fully virtualize the network stack, limiting its ability to configure independent network settings. Finally VServer lacks support for live migration and seamless kernel updates.

---

## Docker's Modern Improvements

I believe Docker has addressed these limitations. To enhance fault isolation, Docker uses Linux kernel features like [namespaces](https://docs.docker.com/engine/security/) (as far as I know kernels got big upgrades since VServer times - mentioned in paper) and [control groups (cgroups)](https://docs.docker.com/engine/security/). I think the biggest upgrade was that namespaces that now ensure that each container operates in its isolated environment, preventing faults from propagating between containers, while cgroups allocate and limit resource usage. For improved security, Docker employs features such as Enhanced Container Isolation (ECI), which isolates containers at the user namespace level and integrates [seccomp profiles](https://docs.docker.com/engine/security/seccomp/) and [AppArmor](https://docs.docker.com/engine/security/apparmor/) to restrict system calls. Docker also enables [rootless containers](https://docs.docker.com/engine/security/rootless/), reducing the security risks associated with privileged access. 

In terms of networking, Docker provides network namespaces, allowing containers to have their own isolated network stack, complete with independent IP addresses and routing tables. This prevents cross-container network interference. Docker also supports flexible kernel configurations and runtime environments, offering improved container migration capabilities through image spec. This is crucial for kernel updates and system maintenance, minimizing downtime (though rebuilding from the image is still required). Additionally, it's possible to switch to a different runtime, like Sysbox (instead of runc) for added security. 

Docker has found a second dimension to the gradient tradeoff. It detaches the container's state, meaning that if a container is restarted, its state is effectively lost. However, this allows containers to be packaged into images that can be easily transferred and redeployed on different hosts.

---

## Images vs Snapshots

Another key point is the idea of images versus snapshots. Docker uses standardized, declarative images to configure environments, while VMs typically rely on snapshotting. In a VM, the entire system is configured starting from a bare minimum and hopes to preserve the state. The question then becomes: which is faster for app recovery - bootstrapping a VM from a checkpoint on a new machine, or letting a container orchestrator recognize the dead container and restart it elsewhere? Even if they were tied, the argument is that VM checkpoints preserve the state, but the latency cost is high. Docker's persistent volumes allow for state preservation in a serialized way, managed through the application's business logic layer.

---

## Do We Actually Need Full Isolation?

Now, the real question is: do we actually need hardware emulation or total isolation? Data centers absolutely need these for hard isolation requirements and to emulate a variety of hardware configurations. But for most developers, "secure enough" isolation within a homogeneous stack is sufficient. The largest system I've seen had 250 images, which could translate into roughly 3000 containers. Given that an average server has fewer than 50 cores, this would require about 60 machines, which is easily achievable with hyperscale providers like AWS hosting hundreds of thousands of servers in one datacenter.

So, my point is: they're two different tools. Modern containers, with advances in orchestration and performance, cover most use cases. For example, I've seen containers boot up in a Kubernetes cluster in under 30ms. VMs, however, are about ensuring compatibility with legacy systems. Whether you need to run Apple1-focused software or Nintendo games, VMs give you the ability to do that while maintaining the state - something crucial probably for some scientific computations. 

There's also a middle ground between these two extremes, like KVM, which allows for VM-like workloads to run inside containers. Managing all of these depending on needs is the future. As documented in [AWS's migration history](https://www.allthingsdistributed.com/2020/09/reinventing-virtualization-with-nitro.html), AWS used [Xen](https://perspectives.mvdirona.com/2021/11/xen-on-nitro-aws-nitro-for-legacy-instances/) for most of the time, then migrated to proprietary [AWS Nitro](https://www.allthingsdistributed.com/2020/09/reinventing-virtualization-with-nitro.html) and now even running some nodes on [Firecracker microVM](https://aws.amazon.com/blogs/aws/firecracker-lightweight-virtualization-for-serverless-computing/) which seems to be built on KVM - all of which were mostly probably used as a base layer to host docker images which itself does run a lightweight VM whenever there is a host mismatch.

---

## Closing Thoughts

Having said all of that - the real tradeoff here is between latency (containers are fast) and availability (as heavy VMs can easily checkpoint and migrate) but mostly these 2 techniques will be stacked on each other in practice and it is worth understanding both. The future isn't about choosing one or the other - it's about knowing when to use which tool, and increasingly, how to use them together in hybrid architectures that give us the best of both worlds.

---