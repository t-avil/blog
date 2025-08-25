---
# Basic info
title: "MMORPGs From BE Eng Perspective"
date: 2025-03-05T14:42:25-07:00
draft: false
description: "A technical, continuous train-of-thought on building an MMO backend: edge computing, server topologies and authority, clock sync, spatial partitioning with quadtrees, and a deep dive into cloth sync."
tags: ["game-dev","mmorpg","netcode","edge-computing","spatial-partitioning","cloth","backend"]
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
  URL: "https://github.com/CrustularumAmator/blog/tree/main/content"
  Text: "Suggest Changes"   # edit link text
  appendFilePath: true      # append file path to the edit link
---

Building an MMORPG backend is like building a tiny distributed country where every player is a citizen, every NPC a bureaucrat, and latency is the weather. You design for scale, consistency, fairness, and - most importantly - fun. Below is one continuous, slightly nerdy walkthrough of the system-level decisions I’d make (and the gotchas I’d sweat about at 3 AM): edge compute, server topologies and authority, clock synchronization, spatial partitioning with quadtrees, and a focused deep-dive on cloth synchronization (yes, that ugly yet sexy problem).

> Note: some concepts and short references below mirror the “lag reduction” notes I carried into the project (edge/local zone patterns, load balancing strategies, lockstep vs rollback, client/server authority, etc.).&#x20;

---

## The shape of the problem: correctness vs responsiveness

Before we design anything: an MMO must answer two competing demands:

1. **Correctness (authoritative state)** - players shouldn't be able to cheat by lying about their position or health.
2. **Responsiveness (feel)** - actions should feel instant even when your RTT is non-zero.

Solving both means a layered approach: authoritative servers + client prediction + smart networked correction + edge placement.

---

## Edge computing: bring the server to the player

Edge compute reduces RTT by moving session-affine services closer to players. Practical building blocks:

* **Local Zones / Outposts**: run full or partial game services in AWS Local Zones or Outposts when low-latency presence is needed in a metro. Useful for voice, matchmaking, and session brokering.&#x20;
* **Netflix OCAs-style appliances / on-prem edges**: for very large events or regions with special constraints, local appliances can cache assets and reduce WAN hops.&#x20;
* **Edge responsibilities**:

  * **Session brokering & matchmaking** (global service that picks a nearby edge host).
  * **Proximity servers** - short-lived, regionally spawned authoritative instances for combat zones, raids, or sharded areas.
  * **Stateless frontends** (auth, cosmetic inventory, marketplace) can run at the cloud edge with global DBs; heavy simulation runs on local stateful nodes.

Edge placement minimizes RTT for the critical loops (player input → server tick → response) while leaving durable, global services in centralized regions.

---

## Server topologies & authority modes

There are multiple topologies. Pick one (or mix them) based on game mechanics:

1. **Central authoritative servers (classic)**

   * Single authoritative instance per game region/zone.
   * Pros: simple consistency; server enforces rules.
   * Cons: higher latency for distant players; scaling by adding more regions/instances.

2. **Sharded / zoning (cell-based)**

   * The world is split into shards / cells; each cell has an authoritative server. Players move between cells (handoff). Cells can be assigned by a control plane (quadtrees, see below).
   * Pros: horizontal scaling, locality.
   * Cons: handoff complexity and cross-cell consistency.

3. **Client-authoritative with server reconciliation (not for PvP)**

   * Clients lead; servers validate. Good for low-cheat-risk elements like cosmetic sims. Not recommended for combat.

4. **P2P / hybrid**

   * Rare for MMOs due to trust/cheat surface. Often used for LAN/low-latency co-op.

5. **Relay / regional authority**

   * Clients connect to a local relay that aggregates inputs and forwards to a regional authority - useful when you want to reduce global WAN bandwidth but maintain a single authoritative state.

**Authority settings** determine who finalizes state for a given domain:

* **Server-authoritative** for game-critical things (position for hit detection, inventories, combat).
* **Client-predicted, server-validated** for movement: client shows immediate movement, server later corrects.
* **Client-authoritative** for purely cosmetic, local-only simulations (local cloth visual only if correctness not required).

---

## Clock synchronization & ticks: making time meaningful

Time is the lingua franca of multiplayer. You need consistent ticks, determinism where possible, and latency-aware mapping.

### Tick model

* **Server tick** is the canonical clock (e.g., 20-60Hz depending on simulation fidelity). Every authoritative simulation step runs at server tick `T`.
* **Client tick** runs at a higher rate for rendering and local input sampling.

Clients map server ticks to local time using periodic **server snapshots** with server timestamp `S_ts` and server tick `S_tick`. Client computes `clock_offset = local_now - S_ts - network_latency_estimate/2` and uses it to align.

### Time synchronization techniques

* **Ping/Pong RTT estimation (NTP style)**: measure RTT and estimate one-way delay; maintain offset with exponential smoothing. This is the simplest and often sufficient.
* **Adaptive smoothing / Kalman filters** to stabilize jittery latency.
* **PTP-like methods** for tighter sync if hosts are on the same LAN or data center - rarely practical across the public internet.
* **Logical clocks & sequence numbers** for ordering events (tx\_id style) when absolute wall-clock sync is not required.

### Using clocks for networking

* **Server-side authoritative timestamping**: server stamps accepted inputs with server\_tick and broadcasts snapshots keyed by tick.
* **Client interpolation / extrapolation**: client renders server snapshot for tick `T - buffer` (buffer = 1-3 ticks) and interpolates between ticks to hide jitter. For fast actions, client extrapolates forward (prediction) and later reconciles with authoritative corrections.

Pseudo timeline for client:

1. Receive snapshot for server\_tick `T` at local time `R`.
2. `render_target_tick = latest_server_tick - render_buffer`.
3. Interpolate between snapshots for `render_target_tick`.

---

## Network techniques: interpolation, extrapolation, and rollback

* **Interpolation**: keep a short buffer and interpolate between known states. Great for position smoothing, avoids visible teleporting on low packet loss.
* **Extrapolation / prediction**: client simulates forward using local inputs. Server corrects if divergence exceeds threshold (snapback or smooth correction).
* **Rollback (for deterministic sims / lockstep games)**: store history of local inputs and simulation states, apply late-arriving inputs by rolling simulation forward - used in fighting games and lockstep RTS when determinism is strict. For MMOs, rollback is expensive due to state size, so we usually use reconciliation instead.

Lockstep vs rollback vs server-authoritative:

* **Lockstep** (deterministic): all peers run the same deterministic sim and exchange inputs. Low bandwidth but needs bit-exact determinism - hard for cloth and physics.
* **Rollback**: apply late inputs retroactively; requires checkpointing and often used in peer-fighting games.
* **Server-authoritative** with client prediction + reconciliation: most robust for MMO scale.

---

## Spatial partitioning: quadtrees, sharding, and interest management

Partitioning the world by location is essential for scaling and interest management.

### Quadtrees for location-based partitioning

Quadtrees recursively subdivide space into four quadrants until each leaf contains a manageable number of entities or covers a small area. Use-cases:

* **Interest management**: query quadtree for radius or view-frustum to determine which entities a client needs.
* **Server sharding**: map quadtree leaves to authoritative servers or worker pools. When a leaf grows too hot, split it; when sparse, merge.
* **Load balancing**: use heatmaps to migrate hot leaves to more provisioned hosts.

Quadtrees are simple and spatially adaptive. Implementation notes:

* Use **loose quadtrees** to avoid thrashing entities across boundaries.
* Store metadata per node: entity count, CPU load estimate, bandwidth usage.
* Handoff when a player crosses a node boundary: a short "handoff protocol" transfers authority and state and uses overlap regions during handoff to prevent gaps.

### Interest management & multicast

* Compute a **relevance set** per client using quadtree queries + LOD rules (NPCs beyond X meters get less frequent updates).
* Use **multicast groups or topic-based pub/sub** to only send deltas to clients that care (spatial topics like `zone/leaf/123`).

---

## Server-side simulation: partitioning & authority strategies

How to split simulation responsibilities:

* **Cell servers**: authoritative for a spatial cell; responsible for physics & authoritative gameplay.
* **Service workers**: handle non-latency-critical tasks - AI pathfinding batch jobs, economy processing, persistence.
* **Replica/reader nodes**: read-only copies for analytics, leaderboards, or for cheap, eventual-consistent queries.

When scaling, prefer **stateful cells + stateless frontends**. Cells are sticky to players; frontends route input to the correct cell. A matchmaking/control plane assigns cells and manages lifecycle.

---

## Load balancing & provisioning modes

From the slide notes: choose servers based on usage profile - reserved, on-demand, spot - and plan for autoscaling & pre-warming:

* **Reserved servers** for persistent high-capacity regions/peak hours.
* **On-demand** for elasticity.
* **Spot instances** for cheap but replaceable workers (non-critical batch processing, analytics).
* **Pre-warming** or warm pools for game instances that must start fast (avoid spin-up latency during a raid).

Load balancers should be proximity-aware and sticky at the session level when needed, but not for authoritative instances that must reject unauthenticated reroutes.

(See the “lag reduction” notes for a reminder that load balancing and edge placement are cornerstones of latency reduction.)&#x20;

---

## Cloth sync - deep dive (the spicy one)

Cloth simulation is tricky: it's high-frequency, high-degree-of-freedom, and visually sensitive. You usually cannot naively stream per-vertex positions from server to clients - bandwidth explodes. Here’s a practical, layered strategy.

### 1) Decide the authority model for cloth

Options:

* **Client-only (visual-only)** - the server ignores cloth entirely. Good for cosmetics where gameplay is unaffected.
* **Server-authoritative physics** - server runs cloth sim and sends authoritative states; needed if cloth affects gameplay (e.g., entanglement, hitboxes).
* **Hybrid** - server authorizes key constraints/anchors; clients run local sims for rendering and reconcile to authoritative constraints.

For MMORPGs, hybrid is usually the sweet spot: server maintains anchors and critical collisions; clients predict local cloth motion for smooth visuals.

### 2) Represent cloth efficiently

Network transfers should avoid per-frame full-vertex dumps. Use these techniques:

* **Anchor/constraint sync**: send only anchor points and constraint updates (rest lengths, impulses). Clients simulate cloth locally using those anchors.
* **Lodded vertex sets**: send a high-frequency small core set of “driver” vertices and low-frequency deltas for the rest.
* **Principal component / PCA compression**: encode cloth frames in a low-rank basis (approximate by a few coefficients per frame). Great if cloth moves in a constrained manner (capes, flags).
* **Delta + predictive coding**: send deltas relative to predicted local sim to save bandwidth.

### 3) Determinism vs stochasticity

* Use **fixed timestep** local sims (e.g., 120Hz physics loop) and deterministic integrators (semi-implicit Euler, Verlet with fixed order) so client sim mirrors server sim as closely as possible.
* Use shared RNG seed for stochastic forces (wind gusts) so clients and server can reproduce pseudo-random influences deterministically.

### 4) Correction & reconciliation

Even with careful design, client sim drifts. Reconciliation strategy:

* **Authority snapshots**: server sends periodic authoritative "keyframes" with a server\_tick timestamp and a small correction envelope.
* **Blend corrections**: clients smoothly blend from local state to authoritative state over `k` frames rather than snapping. Use velocity-preserving blends when possible to avoid pops.
* **Constraint projection**: server sends constraint corrections (e.g., anchor position corrections or per-constraint stretching penalties). Clients apply these as impulses rather than overwriting positions.

Snippet (pseudo) for client correction blend:

```js
// acolor: authoritative positions; local: predicted
for i in 0..num_verts:
  delta = authoritative[i] - local[i]
  local[i] += delta * clamp( alpha, 0, 1 )  // alpha small, e.g., 0.1
```

### 5) Multi-resolution streaming

* **High-prio**: anchor vertices, collision-critical verts (near player weapon), per-frame.
* **Mid-prio**: silhouette vertices, every N frames.
* **Low-prio**: interior vertices, sparse updates or PCA-coded updates.

### 6) Bandwidth estimate & sampling

Tune:

* Anchor-only mode: few KB/s per cloth.
* Full per-vertex sync (bad): MB/s per client.
* PCA/delta approach: tens to hundreds of KB/s depending on frequency.

### 7) Example hybrid workflow

1. Server computes anchor positions + collision impulses each authoritative tick.
2. Clients simulate local cloth at render rate, sampling server anchors on arrival.
3. On authoritative tick, server sends corrections for anchors & important constraints; clients apply blended corrections.
4. Periodic keyframes (vector-compressed or PCA) re-sync full shape if drift exceeds threshold.

This gets visually-close cloth with manageable bandwidth and server CPU. If cloth affects gameplay, raise server tick and authority; if cosmetic, favor client-only sim and occasional server nudges.

---

## Practical network patterns & conservative heuristics

* **Interest-based update frequency**: players near a cloth source get full updates; distant players get coarse LOD.
* **Thresholded corrections**: only send full state if drift > epsilon.
* **Predictive drift compensation**: server provides velocity/acceleration for anchors so clients can predict until next update.
* **Compression + binary protocols**: use compact binary frames (VarInt, delta-encoding) and consider protocol buffers / custom COBS to minimize packets.

---

## Persistence, reconcilation & authoritative storage

For MMO scale, persistence strategy:

* **Authoritative snapshots** persisted periodically (checkpoints) to durable storage.
* **Event sourcing** for important game mutations (player trades, inventory changes) so you can replay and backfill.
* **Soft state for visuals** such as cloth usually does not persist beyond session, unless tied to gameplay.

Backfills and rollbacks must be deterministic and replayable. If a shard fails, replay CDC-like event logs to bring replicas up-to-date.

---

## Monitoring & metrics to watch closely

* **Server tick time** (ms) and drops (missed ticks).
* **Network jitter & packet loss** per client region.
* **State divergence metrics** (client vs server error histograms).
* **Memory use for state backends** (e.g., per-cell cloth cache).
* **Hotspot detection** to trigger dynamic quadtree splits.

---

## TL;DR: architecture checklist

* Use **edge/local zones** for low-latency session brokering and proximity servers.&#x20;
* Pick **server topology**: cell-based authoritative servers + stateless frontends is a pragmatic choice.
* Sync time by **server ticks + ping-based offset estimation**; render with interpolation and prediction.
* Partition world with **loose quadtrees** for adaptive sharding and interest management.
* Build **handoff protocols** for smooth cross-node migration.
* For cloth: prefer **hybrid authority** (anchors server, local sim client), compress state (anchor-only, PCA, deltas), and reconcile with blended corrections.
* Use **reserved + on-demand + spot** provisioning strategies and warm pools for fast scaling.&#x20;

---

## Final note (real talk)

The art of MMO backend engineering is picking the parts you can operationally own. Don’t try to make every system deterministic across all edge cases - instead, identify *what must be right* (combat hits, inventory), *what must feel right* (movement, cloth, VFX), and *what can be eventual* (leaderboards, analytics). Combine server-authoritative systems for correctness with prediction and local sims for feel. Use edge compute and quadtrees for locality. Make cloth a hybrid problem: clever compression + deterministic cores + smooth corrections. And, most importantly, instrument everything - your best sleep comes from good metrics, not superstition.
