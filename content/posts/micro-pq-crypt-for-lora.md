---
# Basic info
title: "PROJECT: Post-Quantum Cryptography in Meshtastic"
date: 2025-09-04T02:21:17-07:00
draft: false
description: "How I Stuffed Kyber Into an ESP32."
tags: [
  "project",
  "esp32",
  "lora",
  "meshtastic",
  "post-quantum cryptography",
  "kyber",
  "quantum-safe",
  "iot security",
  "embedded systems",
  "off-grid communication",
  "mesh networks",
  "resource-constrained devices",
  "encryption performance"
]
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


{{< figure src="/images/micro-pq-crypt-for-lora.jpg" attr="Meshtastic  (lora) firs message over quantum encryption" target="_blank" >}}


This post explains how I integrated post-quantum cryptography into Meshtastic firmware. Specifically, I replaced the classical Curve25519 key exchange with **Kyber-512**, a lattice-based key encapsulation mechanism, while maintaining compatibility with legacy nodes. I specifically wanted to apply this [research paper](https://arxiv.org/html/2503.10207v1#S4) and see how the system wil behave with a couple of mesh nodes upgraded.  
Along the way, I will explain the components: **Meshtastic**, **LoRa**, and the **ESP32**, and provide references to two recent research papers that explore PQ cryptography in IoT contexts.

By the end, readers should understand why post-quantum cryptography matters for mesh networks and how it can run even on tiny embedded devices.

## What is Meshtastic

Meshtastic is an open-source mesh network firmware for **off-grid, low-power communication**. A mesh network is a decentralized network where every node can communicate with nearby nodes and forward messages for distant nodes. This makes it ideal for outdoor adventures, emergency communication, or any situation without cell towers.

Meshtastic nodes usually consist of a **microcontroller**, typically the ESP32, paired with a **LoRa radio** for long-range communication.

## LoRa: Long-Range Radio Technology

**LoRa** stands for Long Range. It is a radio modulation technique designed for low-power, long-distance communication. Unlike Wi-Fi or Bluetooth, LoRa can transmit small amounts of data over several kilometers while using very little battery.

LoRa is ideal for mesh networks because:

* It tolerates noisy environments
* Nodes can forward messages to neighbors, forming multi-hop networks
* It minimizes bandwidth usage, making it compatible with constrained microcontrollers

## ESP32: The Embedded Brain

The **ESP32** is a small, low-cost microcontroller that includes:

* A dual-core processor (tens of megahertz range)
* Wi-Fi and Bluetooth radios
* Hardware acceleration for cryptographic algorithms like AES and SHA
* Limited RAM and flash memory

This makes it powerful enough to run complex algorithms like Kyber, but we have to manage memory and CPU carefully.

Recent research has shown that Kyber can be semi-efficiently implemented on ESP32 by hand-partitioning the algorithm across its dual cores and leveraging cryptographic coprocessors. This improves execution time significantly. See [Efficient Kyber on ESP32](https://arxiv.org/html/2503.10207v1#S4).

## The Quantum Threat

Modern public-key cryptography like **Curve25519** relies on mathematical problems that classical computers find hard to solve. But **quantum computers** can solve these problems quickly using algorithms like Shor's algorithm.

This puts all ECC-based key exchanges at risk.

Post-quantum cryptography (PQC) uses problems that quantum computers cannot solve efficiently. For example, **Kyber** relies on lattice problems that are believed to be quantum-resistant.

One challenge is that PQC can be computationally heavy, which matters for tiny IoT devices like ESP32. A recent survey explores the performance of PQC on constrained devices and emphasizes the need for optimization standards. See [PQC in Resource-Constrained IoT](https://arxiv.org/html/2401.17538v1).

## Classical Meshtastic Encryption

Before PQC, Meshtastic used:

* **AES-256** for message encryption
* **Curve25519** for key exchange
* Pre-shared keys for channel encryption

Flow:

```
1. Node A generates Curve25519 keypair
2. Node A shares public key with Node B
3. Node B computes shared secret using ECDH
4. Derive AES session key
5. Encrypt message
6. Send message with authentication tag
```

This is fast and efficient but vulnerable to quantum attacks.

## Adding Kyber-512

Kyber-512 is a **lattice-based KEM** that provides \~128-bit security against both classical and quantum attacks. Key sizes are larger than Curve25519:

* Public key: 800 bytes
* Private key: 1632 bytes
* Ciphertext: 768 bytes
* Shared secret: 32 bytes

To fit these keys into LoRa packets, we fragment them into smaller pieces (\~200 bytes each) and reassemble them on the receiving node.

Encryption flow:

```
1. Node A generates Kyber keypair
2. Node A broadcasts PQ capability
3. Node A sends key fragments
4. Node B reassembles fragments and verifies hash
5. Node B performs Kyber encapsulation to derive shared secret
6. Derive AES session key from shared secret
7. Encrypt message
8. Send encrypted payload with Kyber ciphertext fragment
```

## Hybrid Crypto Engine

To maintain compatibility with older nodes, Meshtastic now runs **two crypto systems in parallel**:

```cpp
class CryptoEngine {
    bool encryptCurve25519(...);
    bool encryptKyber(...);
    bool decryptCurve25519(...);
    bool decryptKyber(...);

    uint8_t pq_public_key[800];
    uint8_t pq_private_key[1632];
};
```

The router decides at runtime which method to use based on node capabilities.

### Runtime Encryption Decision

In Meshtastic, every outgoing message passes through the Router, which determines which encryption method to use. The decision is made dynamically, based on:

1. Whether the **sending node has Kyber keys** already generated.
2. Whether the **receiving node supports post-quantum cryptography** (PQ capabilities).
3. Whether the **network prefers PQ**, falling back to classical Curve25519 if PQ is unavailable.

This ensures smooth interoperability, even in a mixed network of legacy and PQ-capable devices.

#### Example

```cpp
EncryptionDecision selectEncryption(NodeNum toNode) {
    NodeInfo* peer = nodeDB->getMeshNode(toNode);

    // Broadcast messages or unknown nodes use channel encryption
    if (!peer || peer->user.public_key.size == 0)
        return USE_CHANNEL_ENCRYPTION;

    bool peerSupportsPQ = (peer->user.has_pq_capabilities &&
                           peer->user.pq_capabilities & PQ_CAP_KYBER_SUPPORT);
    bool weHavePQKeys = pqKeyExchangeModule->hasValidPQKeys(toNode);
    bool weSupportPQ = crypto->hasValidKyberKeys();

    if (peerSupportsPQ && weSupportPQ && weHavePQKeys)
        return USE_PQ_PRIMARY;         // Use Kyber-512
    else if (peerSupportsPQ && weSupportPQ && !weHavePQKeys)
        return INITIATE_PQ_EXCHANGE;  // Start key exchange first
    else
        return USE_CLASSICAL_PKI;     // Fallback to Curve25519
}
```

* Node A wants to send a message to Node B.
* Node B advertises PQ capabilities.
* Node A already has Kyber keys for Node B.
* Router chooses **USE\_PQ\_PRIMARY** and encrypts the message with Kyber.

If Node B didn’t yet have keys, the Router initiates a key exchange while sending messages with classical encryption until the PQ keys are ready. This approach avoids breaking the network mid-transition.

---

### Narrative Conclusion

The journey of integrating post-quantum cryptography into Meshtastic is as much about engineering intuition as it is about algorithms. Each challenge-from oversized keys and memory constraints to out-of-order LoRa fragments-taught lessons that extend beyond this single implementation. We learned that:

1. **Security doesn’t happen in isolation**. Memory management, message fragmentation, and CPU scheduling all intersect with cryptography. You can’t just drop in Kyber-512 and expect it to behave.
2. **Hybrid systems ease adoption**. Running PQ and classical encryption side by side ensures backward compatibility while gradually introducing stronger security.
3. **Performance is a negotiation**. Every millisecond counts on an ESP32, but careful caching, lazy key exchange, and leveraging dual cores can keep latency within acceptable limits.
4. **Every reset is a lesson**. Debugging embedded systems often involves unexpected crashes. Each reset forced a closer look at assumptions and helped refine the system into something resilient.


#### Challenges (a.k.a. Things That Made Me Swear at My ESP32)
1. nanopb Shenanigans: Protobuf bytes fields without a size limit = every message allocates an 800-byte buffer. Oops. Fixed with (nanopb).max_size.
2. Stack vs Heap: Kyber eats kilobytes of temporary space. ESP32 task stack ≈ 8KB. I punted to static buffers in .bss to avoid random stack overflows.
3. Performance: Curve25519 runs in <1ms. Kyber keygen = ~47ms, encaps = 8ms, decap = 12ms. Manageable, but caching and lazy exchanges were necessary.

#### Testing
1. Unit tests for crypto roundtrips, fragmentation reassembly, capability negotiation.
1. Multi-node mesh: PQ↔PQ, PQ↔legacy, PQ-only.
2. Benchmarks: +15% RAM usage, ~800B per key exchange, ~764B per PQ packet.
Break-even for PQ overhead: ~1.3KB of encrypted messages per peer.

---

## Final thoughts 


The successful deployment of post-quantum cryptography in Meshtastic demonstrates that the quantum-safe future of communications is not merely theoretical, but practically achievable today.


So, can you cram post-quantum cryptography into a tiny ESP32 mesh node? Turns out: yes, with enough protobuf hacks, static buffers, and late-night coffee.

Meshtastic now has a quantum-resistant mode that interoperates seamlessly with classical nodes. It’s not just a proof-of-concept - it’s running in real test meshes.

This work is a step toward a future where mesh communications stay private even in the face of quantum computers. And along the way, it proved that PQ crypto isn’t just for servers in datacenters - it can live in low-power radios strapped to your backpack.


---

*Full code available on this [GitHub branch](https://github.com/t-avil/meshtastic-firmware-pq/tree/pq-introduction), and way too many comments explaining design decisions. Because if you're going to overthink a project, you might as well document the overthinking process.*
