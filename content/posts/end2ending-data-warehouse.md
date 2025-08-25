---
# Basic info
title: "End to ending Data Warehouses as a concept"
date: 2024-10-01T23:23:37-07:00
draft: false
description: "A slightly practical walkthrough of designing a CDC-powered data pipeline: ingestor, bronze lake, silver warehouse, and dbt-powered golden marts with batch DAGs, streaming joins, state management, and chunked reducers."
tags: ["data-engineering","CDC","data-warehouse","streaming","batch","dbt","system-design"]
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
disableHLJS: false # set true to disable Highlight.js
disableShare: false

# Edit link
editPost:
  URL: "https://github.com/CrustularumAmator/blog/tree/main/content"
  Text: "Suggest Changes"   # edit link text
  appendFilePath: true      # append file path to the edit link
---



If you care about analytics at scale, the day will come when you need to build a data pipeline that *remembers* everything and still gives you answers fast. This post walks the whole path: ingesting CDC, landing in a bronze data lake, cleaning & denormalizing into a silver warehouse, then producing golden data marts with dbt. I’ll show the nitty-gritty: exactly how CDC needs to be treated, how a control plane composes map-shuffle-reduce DAGs for SQL, how to make reducers scale (chunking and streaming), and how streaming joins keep marts fresh without recomputing the world.

---

## Architecture overview - the five parts

1. **Ingestor (CDC)** - capture change events (inserts/updates/deletes) reliably and in order.
2. **Bronze tier (data lake)** - raw immutable events, append-only, schema + metadata (offsets, tx id). Persistent volumes with replication (RF=3).
3. **Silver tier (warehouse)** - cleaned, denormalized, query-friendly state (often columnar tables).
4. **Golden tier (data marts / dbt)** - curated models, tests, docs; materialized incrementally or full-refresh via dbt.
5. **Control plane** - tracks chunks, metadata, computes DAGs, schedules map/shuffle/reduce tasks, reconciliation/lineage.

---

## CDC: not just "row snapshots" - treat it like transactional logs

CDC comes in flavors. Two important distinctions:

* **State-capture snapshots**: occasional full table snapshots.
* **Transaction log (logical decoding)**: ordered events exactly as they happened - *this is what you want for correctness*.

Why the difference matters: if your CDC is transactional (Postgres WAL, MySQL binlog), events arrive as ordered, atomic units (tx boundaries, commit/abort). If you treat them as out-of-order or as naive key-value overwrites, you can compute wrong aggregates, miss tombstones (deletes), or break foreign-key replays.

Practical CDC best practices:

* Use tools like **Debezium** / logical decoding to emit change events including: `table`, `op` (c/u/d), `before`, `after`, `tx_id`, `lsn`/`offset`, `timestamp`.
* Include **commit markers** or transaction IDs so you can apply events in commit order when reconstructing state.
* Emit **tombstones** for deletes to allow downstream compaction/merge logic to drop rows.
* Keep schema evolution under a registry (Avro/Protobuf/JSON Schema) and version events - never assume the payload shape is stable.

Example CDC event (pseudo-JSON):

```json
{
  "tx_id": "0xabc123",
  "lsn": 123456,
  "op": "U",
  "table": "orders",
  "before": { "status": "pending" },
  "after":  { "status": "paid" },
  "ts": "2025-08-25T09:00:00Z",
  "schema_version": 4
}
```

---

## Bronze: raw immutable events & durable storage

Bronze is the single source of truth. Requirements:

* **Append-only** storage (PV with RF=3, or S3 for object stores). RF=3 gives resilience: 2 node failures tolerated.
* Each event writes with metadata: topic/partition, offset, tx\_id, arrival\_ts, schema\_version.
* **Compaction policy**: keep raw events for N days/years, but also allow compacted "state snapshots" for faster rehydration.

Implementation tips:

* Put CDC into a message broker (Kafka) or append-store. Consumers (batch/stream) read from durable offsets.
* Persist raw event files in a consistent layout (e.g., `/bronze/{table}/{yyyy}/{mm}/{dd}/part-{shard}.avro`) with checksums for integrity.

---

## Control plane: SQL → DAG → tasks

When a SQL query arrives (ad-hoc report or scheduled job), the control plane must:

1. **Parse SQL** into a logical plan (scan, filter, join, aggregate).
2. **Rewrite** into a physical plan: break into map/shuffle/reduce stages optimized for partitioning keys and data locality.
3. **Emit tasks**: map tasks that read chunks/partitions, shuffle to reducers by key, and reduce tasks that do joins/aggregations.
4. **Track lineage & retries**: store per-task metadata, checkpointing tokens, and ability to restart failed tasks deterministically.

Very high-level pseudocode for DAG generation:

```text
logical = parse(sql)
physical = plan(logical)   // choose join orders, hash vs sort merge
stages = partition_into_stages(physical)
for stage in stages:
  tasks = create_tasks(stage, based_on_chunks)
  schedule(tasks)
```

---

## Map → Shuffle → Reduce: details & guarantees

**Map**: read bronze/silver chunk files, extract per-record keys, emit `(key, payload)` to shuffler. Map workers must implement:

* **Retries** & idempotency (process-by-offset or use idempotent output paths).
* **Checkpoints** (on success write completion marker).
* **Backpressure** for overload.

**Shuffle**: network or disk transfer that groups records by key. Make sure partitioning function is stable (hash(key) % R). Shuffler should support spilling to disk when the in-memory buffer exceeds limits.

**Reduce**: receives all records for a partition (key range). Reducer workloads:

* **Join** multiple tables’ records on the key. Preferred algorithms:

  * **Hash join**: build hash table for smaller relation, probe with larger-a good default.
  * **Sort-merge**: for large sorted inputs; better when inputs are pre-sorted or when memory is limited.
  * **Block nested-loop with chunking**: when one side is too large; stream chunks and iterate.
* **Chunking strategy**: if incoming partition is too big to fit memory, load the first chunk of table A into memory, stream chunks from table B and iterate; when exhausted, evict and next chunk from A. This is essentially *external* join with bounded memory.

Chunked reducer pseudo:

```text
for chunkA in stream_chunks(tableA):
  build_hash(chunkA)
  for chunkB in stream_chunks(tableB):
    for r in chunkB:
      if hash_lookup(r.key):
         emit(joined_row)
  free(chunkA)
```

---

## Streaming vs Batch: when to pick what

**Batch** (map-shuffle-reduce) is great for full recompute, complex multi-way joins, and backfills. **Streaming** is best for incremental updates, near-real-time dashboards, and when you want to avoid full recomputes.

Streaming patterns to implement:

* **Incremental aggregations**: maintain keyed state (counts, sums) and update per event.
* **Stateful joins**: keep one stream as keyed state (persisted in a state backend like RocksDB); incoming events from the other stream are joined against that state - this produces incremental joins and avoids recomputing past windows.
* **n−1 state**: when joining multiple tables (n inputs), streaming engines must maintain persisted state for n−1 tables and apply incoming updates from the nth.

Important streaming considerations:

* **State backend & TTL**: RocksDB (Flink) or managed state, with TTL/compaction to bound storage.
* **Watermarks & late events**: define watermarks to bound lateness; implement logic for retractions if late updates arrive.
* **Exactly-once**: use checkpointing + two-phase commit sinks where possible to get end-to-end exactly-once semantics (Flink, Kafka transactions). If you can't, make consumers idempotent.

Example streaming join behavior (simplified):

1. Stream A (left) persisted keyed state `stateA[key]`.
2. New event from Stream B with key `k` arrives: lookup `stateA[k]`, emit joined records for current stateA. This is *incremental* - it does not recompute when `stateA` later expands (unless you implement replays/retractions).

---

## Silver: cleaned, denormalized, query-optimized

Silver is where messy events become usable rows:

* **Materialized snapshots** or tables with denormalized joins (user + latest\_address + order\_summary).
* **Columnar formats** (Parquet/ORC) with partitioning (date, customer\_id % N) for fast scans.
* **Schema evolution**: reconcile schema versions from bronze; fold fields with default/null rules.

Design for efficient incremental runs:

* Use **upserts** by primary key (using MERGE) when applying cleaned events.
* Maintain **change tracking** so dbt or ETL jobs can run incrementals by modified ranges.

---

## Golden: dbt, tests, and analytics engineering

dbt sits on top:

* Builds models (SQL), materializes tables/views, supports incremental models, docs, and tests.
* dbt DAG maps dependencies - upstream silver tables trigger downstream recalculations.
* Use dbt for tests (`unique`, `not_null`, `relationships`) to catch data drift early.
* CI/CD: run dbt tests on PRs; schedule nightly full-refreshes where necessary.

---

## Operational notes & real-world gotchas

* **Monitoring**: instrument the pipeline. Track lag, task failure rates, shuffle spill rates, state sizes, and restore time.
* **Backfills**: design a deterministic replay mechanism from bronze with tx\_id ordering.
* **Schema drift**: enforce contracts via a schema registry and pre-deploy migrations.
* **Cost & storage**: RF=3 persistent volumes are expensive; consider object storage with periodic state snapshots if you can tolerate eventual consistency.
* **Testing**: use synthetic workloads to test worst-case partition skew and reducer memory/IO patterns.

---

## Final, practical checklist (TL;DR)

* Capture **CDC with tx boundaries**; keep tombstones and schema versioning.
* Store raw events in bronze (RF=3) with consistent layouts.
* Build a **control plane** to generate map-shuffle-reduce DAGs and track chunk metadata.
* Implement **map retries, checkpoints, and idempotency**.
* Make shuffler/partitioning stable; spill to disk if needed.
* Reducers: prefer hash or sort-merge; use chunked external joins when memory limits hit.
* For streaming: use keyed state, watermarks, TTLs, and checkpointing for correctness.
* dbt for golden: tests, DAGs, docs, incremental models.
* Automate backfills & verify restore regularly.


---

##### now you know how motherduck.com 's of this world are build!