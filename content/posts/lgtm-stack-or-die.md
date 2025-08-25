---
# Basic info
title: "Lgtm Stack or Die"
date: 2025-01-13T08:15:04-03:00
draft: false
description: "Alloy, Mimir, Loki, Tempo, Phlare, S3, and a lot of hair-pulling"
tags: ["observability","grafana","mimir","loki","tempo","phlare","opentelemetry","prometheus","golang","datadog-migration"]
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

I learned that migrating off a SaaS like Datadog is less a single migration task and more a personality transplant for your infra team. You’re trading a polished one-click thing for a constellation of components that each have opinions, needs, and a flair for dramatic failure modes. But if you survive the awkward first semester, you end up with vendor-neutral observability, much cheaper S3 storage bills, and dashboards you actually control.

This is the continuous thread of how that migration goes and what I wish someone had told me before I began wrestling with buckets, rings, and ingestion gateways.

---

## The components (the cast)

If you replace Datadog with the Grafana open-source stack, the minimal cast looks like this:

* **Grafana Alloy** - the unified OpenTelemetry ingestion gateway. Everything (metrics, logs, traces, profiles) can flow into Alloy which then routes to the right backend. Think of it as the friendly receptionist of your observability office.
* **Mimir** - Prometheus-compatible metrics store and scraper. Stores TSDB blocks in S3 (cold storage) but keeps some local state and needs a cluster of replicas. Uses Cortex-style ring/consensus to coordinate ingesters and queriers. Partially stateful.
* **Loki** - logs backend; designed to be effectively stateless when backed by object storage. Push logs via Fluent Bit or an OpenTelemetry Collector; Loki writes to S3 for long-term storage.
* **Tempo** - distributed traces; stateless with S3-backed storage so you can scale by adding more replicas.
* **Grafana Phlare** - continuous profiler; runs ingesters/queriers/store-gateways and persists to S3. Can also be fed by Pyroscope agents that convert to OTLP.
* **S3-compatible object storage** - the cornerstone for cold/durable storage (cheap, durable, and enables stateless operation for Loki/Tempo/Phlare/Mimir TSDB blocks).

---

## How Go apps fit in (the developer side)

If you write Go microservices, the wiring is straightforward but worth doing right:

* **Metrics**: instrument with `prometheus/client_golang` and expose `/metrics`. Mimir scrapes Prometheus endpoints (it’s Prometheus-compatible).
* **Logs**: push via **Fluent Bit** or the **OpenTelemetry Collector** to Alloy, which routes to Loki. Don’t ship raw logs directly to S3 - use the pipeline for structure and redaction.
* **Traces**: instrument with the **OpenTelemetry Go SDK** (`go.opentelemetry.io/otel`) and push spans to Alloy; Alloy forwards to Tempo.
* **Profiling**: deploy **Grafana Phlare agents** or use **Pyroscope** with an OTLP converter and push profiles through Alloy to Phlare.

Tiny example - Prometheus counter in Go (pseudo):

```go
var reqs = prometheus.NewCounter(prometheus.CounterOpts{
  Name: "http_requests_total",
  Help: "Total HTTP requests",
})

http.Handle("/metrics", promhttp.Handler())
```

And initializing OpenTelemetry spans (very high level):

```go
tp := otel.TracerProvider(/* exporter config that points to Alloy */)
otel.SetTracerProvider(tp)
tracer := otel.Tracer("my-service")
ctx, span := tracer.Start(ctx, "handleRequest")
defer span.End()
```

---

## Scaling and state: who’s stateful, who pretends not to be

* **Mimir** is *partially stateful*. It writes TSDB blocks to S3, but it also keeps local state and uses a ring/consensus model (Cortex-style) to coordinate ingesters/queriers. You need multiple replicas and to understand the ring behavior. There is no single master, but you can't treat it like a totally stateless service.
* **Loki** and **Tempo** are effectively *stateless* when configured with object storage - add replicas behind a load balancer and scale horizontally.
* **Phlare** scales horizontally with ingesters, queriers, and store gateways, all backed by S3.
* **Alloy** should be deployed with multiple replicas behind a load balancer - it’s the single unified entry point and a potential bottleneck/RPO if under-provisioned.

Replication guidance: a deployment with **1-3 replicas per service** usually supports moderate volumes (more on data estimates below), assuming proper CPU/memory/network sizing.

---

## Storage & cost rough math (two-year window, a real-world-ish estimate)

For a moderately noisy legacy Rails app plus \~10 Golang microservices, expect wildly variable numbers depending on verbosity and sampling:

* **Logs**: \~50-200 GB/day ⇒ **\~36-146 TB** on S3 over two years. (Logs dominate storage unless you aggressively sample/retention them.)
* **Metrics**: with Mimir scraping hundreds of endpoints every 15-30s ⇒ **\~1-3 TB/year** of compressed TSDB blocks.
* **Traces**: sampling-dependent. Medium load might be **100-500 GB/month** ⇒ **\~2-12 TB** over two years.
* **Profiling**: tens to hundreds of GB/year depending on sampling frequency and retention.

These are not exact; they’re *planning* numbers. Adjust retention, sampling, and aggregation to control cost.

---

## Operational advice & gotchas

* **Use S3-compatible object storage** for cheap cold storage and to enable stateless operation for most components. It makes horizontal scaling simple.
* **Automate ingestion and redaction** at Alloy/collector level - don’t rely on developers to scrub logs manually.
* **Backups & verification** - Mimir stores TSDB blocks in S3 but requires local state; verify restores/test queries regularly.
* **Ring architecture awareness** - Mimir's ring and ingesters need proper configuration; small misconfig can lead to “where did my time series go?” panics.
* **Replica sizing** - Loki/Tempo benefit from many small stateless replicas; Mimir needs thoughtful replica counts and resource planning.
* **Alloy HA** - run multiple Alloy replicas so a single node failure doesn’t stop ingestion.

---

## The migration payoff & why I’d do it again

Moving off Datadog gave us vendor neutrality (no vendor lock), big S3 cost wins, and full control over data pipelines. We also gained the ability to debug with local copies of data and to iterate on retention policies without calling sales. The tradeoff: more operational work, more components to monitor, and the occasional late-night spelunking into ring metadata.

---

## Polling vs pushing (the practical verdict)

In general: **polling** (Mimir scraping Prometheus endpoints) provides better reliability and operational simplicity for long-running services. **Pushing** (client or agent pushes metrics/logs/traces) is reserved for edge cases - short-lived, fire-and-die jobs must push before they die. If a job can crash mid-push, you want a push path; otherwise prefer scrape/polling.

---

## Final (slightly sentimental) note

This stack is not magic - it’s a set of well-engineered pieces that work together if you invest in configuration, S3, and sane defaults (retention, sampling, RBAC). The migration is equal parts plumbing and patience: there will be nights of head-scratching; there will also be mornings when you realize your monthly bills dropped and you can finally afford snacks for standups. That is the small, beautiful victory of running your own observability.
