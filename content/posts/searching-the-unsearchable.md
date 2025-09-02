---

# Basic info

title: "NOTES: Searching the unsearchable"
date: 2025-09-01T04:52:54-07:00
draft: false
description: "Levenshtein automata, inverted indexes, BM25, segment merges, ANN vectors, CDC pipelines, and why Elasticsearch is basically three PhDs stapled together with JSON."
tags: [ "elasticsearch", "search", "system-design", "distributed-systems", "vectors", "fuzzy", "ai" ]
author: "Me"


# Metadata & SEO

canonicalURL: "[https://canonical.url/to/page](https://canonical.url/to/page)"
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
  URL: "https://github.com/CrustularumAmator/blog/tree/main/content"
  Text: "Suggest Changes"   
  appendFilePath: true      
--------------------

## Fuzzy Search is More Than Just Levenshtein Distance  

Let’s start from the ground level. If you mistype "pizzq" but still want "pizza", how does a search engine know? At the simplest level you could calculate **Levenshtein distance** between two strings: the minimum number of insertions, deletions, or substitutions required to turn one string into another.  

Here’s a simple dynamic programming implementation in TypeScript:

```ts
function levenshtein(a: string, b: string): number {
  const dp: number[][] = Array.from({ length: a.length + 1 }, () =>
    Array(b.length + 1).fill(0)
  )

  for (let i = 0; i <= a.length; i++) dp[i][0] = i
  for (let j = 0; j <= b.length; j++) dp[0][j] = j

  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1]
      } else {
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1,    // deletion
          dp[i][j - 1] + 1,    // insertion
          dp[i - 1][j - 1] + 1 // substitution
        )
      }
    }
  }
  return dp[a.length][b.length]
}

console.log(levenshtein("pizza", "pizzq")) // 1
```

This works, but if you have millions of documents it’s way too slow to compare every string pairwise.

### Enter Levenshtein Automata

Instead of comparing against every document, you build a **deterministic finite automaton (DFA)** that represents all words within edit distance k of your query. Then you run the DFA against the index.

The states of the automaton represent "how many edits we have left" and "where in the word we are". A simplified DFA construction for edit distance 1 might look like:

```ts
type State = { index: number; edits: number }

function nextStates(state: State, char: string): State[] {
  const { index, edits } = state
  const states: State[] = []

  // Match without edit
  states.push({ index: index + 1, edits })

  if (edits > 0) {
    // Substitution
    states.push({ index: index + 1, edits: edits - 1 })
    // Insertion
    states.push({ index, edits: edits - 1 })
    // Deletion
    states.push({ index + 1, edits: edits - 1 })
  }

  return states
}
```

In production Lucene builds these automata efficiently using special algorithms, but the concept is the same: instead of checking one string at a time, you check all possible close strings in parallel.

---

## Inverted Indexes and Compression Tricks

Search engines don’t store documents like databases do. They flip the structure into an **inverted index**:

```
pizza -> [doc1, doc3, doc7]
pizzeria -> [doc2, doc5]
pepperoni -> [doc1, doc4, doc7]
```

Now queries become intersections of lists. To make this efficient, the lists are compressed.

Example: storing document IDs `[100, 101, 103, 110]`. Instead of storing raw numbers, store **gaps**:

```
[100, 1, 2, 7]
```

Then apply variable-length encoding:

```ts
function encodeVarint(nums: number[]): number[] {
  const bytes: number[] = []
  for (const n of nums) {
    let value = n
    while (value >= 128) {
      bytes.push((value & 0x7f) | 0x80)
      value >>= 7
    }
    bytes.push(value)
  }
  return bytes
}

console.log(encodeVarint([100, 1, 2, 7]))
// -> [100, 1, 2, 7] but each as compressed bytes
```

This allows Elasticsearch to store millions of postings in memory and still intersect them quickly.

---

## Scoring: TF-IDF vs BM25

Both TF-IDF and BM25 are ways to rank documents based on how relevant they are to the query.

* **TF-IDF**:

  * Term Frequency (TF): how many times the word appears in the document.
  * Inverse Document Frequency (IDF): how rare the word is across the collection.
  * Score = TF \* IDF.

The problem: TF grows linearly. A word appearing 50 times is not 50x more relevant.

* **BM25**:
  BM25 adds two important ideas:

  * **Saturation**: additional term frequency gives diminishing returns.
  * **Normalization**: longer documents should not always get higher scores just because they have more words.

The BM25 formula looks like this:

```
score = IDF * ( (tf * (k+1)) / (tf + k * (1 - b + b * (docLen / avgDocLen))) )
```

Where k controls saturation and b controls length normalization.

---

## Query Execution Strategies

When combining postings lists, there are two strategies:

* **Term-at-a-time (TAAT)**: Process one query term’s postings at a time, updating partial scores.
* **Document-at-a-time (DAAT)**: Walk all postings lists in sync, scoring one document fully before moving on.

Example: Query = "pizza OR pasta".

* TAAT:

  * Scan postings for "pizza" → update scores.
  * Scan postings for "pasta" → update scores.
* DAAT:

  * Look at doc1 → check if it has pizza or pasta → assign final score.
  * Move to doc2 → repeat.

DAAT is usually better for top-k queries because you can use a heap and stop early.

---

## Vector Search and HNSW Graphs

Elasticsearch supports vector search with **HNSW graphs (Hierarchical Navigable Small Worlds)**.

Imagine each document as a point in high-dimensional space. Searching means finding nearest neighbors. Brute-force is O(n), too slow.

HNSW builds a multi-layer graph:

* Upper layers have long links (skip connections).
* Lower layers have dense local connections.

Search walks top-down: start at the top layer, use long edges to get close, then drop down layers until you reach the closest nodes. Complexity \~log N.

In Elasticsearch you encode text into vectors using a model, then store them as `dense_vector`. At query time, you provide a query vector and ES traverses the HNSW graph to find nearest docs.

In our case all of this will be used for semantic search (ANN) where found information will be ranked by the distance between the query vector and found items within the HNSW graph space.

---

## Hybrid Search: Lexical Recall, Semantic Precision

Hybrid retrieval = combine two worlds:

* **Lexical search (BM25)**: high recall, because it matches exact tokens.
* **Semantic search (ANN)**: high precision, because it captures meaning. 
  
Workflow:

1. Use BM25 to retrieve top N candidates fast.
2. Use embeddings + ANN search to re-rank them by semantic similarity.

Example:

* Query: "cheap Italian food"
* BM25 finds docs with tokens "cheap", "Italian", "food".
* ANN embedding search reranks so "affordable pizza place" ranks above "expensive Italian furniture".


## The Cluster is Alive (System Design Magic)  

People think Elasticsearch is “just a search box.” Nope. It’s a distributed system disguised as a JSON API.  

Every index is sharded. Each shard is a Lucene index. Replicas exist not just for HA, but also to spread query load. Queries fan out from a **coordinator node** to the right shards, then results fan back in. Cluster state? That’s kept by the **master node(s)**, who do all the boring leader-election / metadata wrangling.  

Then there’s **segment merging.** Every refresh creates new segments (like immutable SSTables). Too many segments = sad query latency. So ES merges them in the background. But merging is expensive, which means you’re always trading **indexing throughput vs query latency.**  

Caches help but lie to you:  
- Query cache → only for repeated identical queries.  
- Shard request cache → helps aggregations, not term lookups.  
- FST-based autocomplete → literally stores prefix/suffix tries as finite state transducers in RAM.  

If you ever wondered why “near real-time” in ES means ~1s, it’s because of the **refresh interval.** You can lower it, but then merges & memory pressure will ruin your day.  

And yes, you can wire ES to your OLTP database via **CDC (Change Data Capture)**. Tools like Debezium stream binlogs → Kafka → ES, giving you low-latency snapshot updates. That’s how you keep your search index within a second or two of reality without rewriting your entire app.  

---

## Autocomplete Strategies

ElasticSearch autocomplete is not a single feature but a set of clever data structures and algorithms that map a user’s partial input to complete terms efficiently. The simplest approach, edge n-grams, works by splitting each indexed word into all its prefixes. For example, the word “pizza” becomes “p”, “pi”, “piz”, “pizz”, and “pizza”. Conceptually, this can be represented as a prefix tree, or trie, where each node corresponds to a prefix and leaves correspond to complete terms. In TypeScript, one might build it like this:

```ts
type TrieNode = { children: Map<string, TrieNode>, isWord: boolean }

function insert(root: TrieNode, word: string) {
  let node = root
  for (const char of word) {
    if (!node.children.has(char)) node.children.set(char, { children: new Map(), isWord: false })
    node = node.children.get(char)!
  }
  node.isWord = true
}

function autocomplete(root: TrieNode, prefix: string): string[] {
  let node = root
  for (const char of prefix) {
    if (!node.children.has(char)) return []
    node = node.children.get(char)!
  }
  const results: string[] = []
  function dfs(n: TrieNode, path: string) {
    if (n.isWord) results.push(path)
    for (const [c, child] of n.children) dfs(child, path + c)
  }
  dfs(node, prefix)
  return results
}
```

Edge n-grams are simple and fast to query, but they consume a lot of index space. For a more memory-efficient solution, ElasticSearch uses **completion suggesters backed by finite state transducers (FSTs)**. FSTs compress common prefixes and store outputs such as document IDs or weights on edges. Traversing the FST from root to leaf enumerates all completions efficiently, essentially providing O(k) lookup time for prefixes of length k. Context suggesters extend this idea by attaching metadata to completions, like location or category, allowing queries like “pizza near Seattle” to return filtered autocomplete results without rebuilding the index.


ElasticSearch also handles **typos in autocomplete using fuzzy matching**, which is built on **Levenshtein automata**. Crucially, fuzzy matching is **applied on top of the prefix / edge n-gram structure or FST**. Each indexed prefix node is effectively a candidate, and the automaton enumerates all paths in the prefix tree that are within the allowed edit distance. For example, if a user types “piza” with a maximum edit distance of 1, the Levenshtein automaton explores the trie paths, allowing one insertion, deletion, or substitution, and still finds “pizza” as a valid completion. This means that fuzzy matching does not ignore the prefix structure; rather, it **traverses the prefix tree or FST while tolerating small deviations**, combining the efficiency of prefix search with the flexibility of typo tolerance.

---
## Highlighting in ElasticSearch

It is about extracting and presenting the portions of text that match a query. There are three main strategies: plain highlighter, unified highlighter, and fast vector highlighter. The plain highlighter is simple, it re-analyzes the document and locates matches. The unified highlighter uses offsets from the inverted index to locate terms precisely and efficiently. The fast vector highlighter leverages **pre-stored term vectors** to avoid re-analysis. Conceptually, each document stores term positions and offsets, and the highlighter simply retrieves the spans for matched terms instead of scanning the text. In TypeScript pseudo-code:

```ts
type TermVector = { term: string, positions: number[], offsets: [number, number][] }

function fastVectorHighlight(termVectors: TermVector[], queryTerms: string[]): [number, number][] {
  const spans: [number, number][] = []
  for (const tv of termVectors) {
    if (queryTerms.includes(tv.term)) {
      spans.push(...tv.offsets)
    }
  }
  return spans
}

// Example: highlights could then be mapped to document text
const docTermVectors: TermVector[] = [
  { term: "pizza", positions: [1], offsets: [[7, 12]] },
  { term: "pasta", positions: [3], offsets: [[17, 22]] }
]

console.log(fastVectorHighlight(docTermVectors, ["pizza"]))
// [[7, 12]]
```
---
## Aggregates getting called out

Aggregations in ElasticSearch are **distributed map-reduce operations**, but understanding the mechanics can help you avoid surprises in CPU and memory usage. Each shard computes aggregations **per segment** and returns partial results, which the coordinating node merges into the final output. Even a simple count can spike CPU if shards have many segments or high-cardinality fields. Key points to keep in mind:

* **Segment-level work:** Each shard has multiple Lucene segments; aggregation runs on each segment. More segments = more work.
* **Memory usage:** High-cardinality terms or large hash tables can blow up memory on shards. Use `shard_size`, `composite aggregations`, or pre-aggregated counts to reduce load.
* **Merging results:** Coordinating node combines shard results. Simple sums are cheap, but top-N terms require sorting across shards.
* **Filters matter:** Apply filters before aggregations to reduce the data processed. Use `filter` aggregations or query-time filters strategically.

Regarding memory allocation: ElasticSearch does not let you assign memory to specific aggregations, but you can influence resource availability by configuring your cluster. For aggregation-heavy workloads, it is common to deploy **dedicated data nodes or coordinating nodes** with larger heap sizes so they can handle the shard-level computations and merging without hitting memory limits. In cloud-managed clusters, you can also provision **spot or on-demand nodes with more RAM during peak hours** to absorb temporary spikes in aggregation load. Additionally, splitting very large aggregations into **smaller composite aggregations** reduces peak memory per shard and keeps operations predictable. With these strategies, aggregations become a **tunable and reliable part of your search infrastructure**, rather than a black box that unexpectedly consumes CPU or memory.

---

## When Theory Meets Production  

This is where things get fun:  

- **Hot-warm-cold tiers**: keep fresh stuff on SSDs, dump old logs on slow spinning rust.  
- **Cross-cluster search**: federated queries across data centers. Yes, global search is a thing.  
- **Multi-tenancy problems**: one noisy client with wildcard queries can starve everyone.  
- **Security**: never expose ES to the public internet unless you like ransomware.  

In the end, ES is basically a mashup of:  
- search engine theory (inverted indexes, automata, scoring),  
- distributed systems design (shards, replicas, coordination),  
- and AI-modern glue (embeddings, re-ranking).  

That’s why it feels magical. You can start with “find me pizza near Seattle” and end up deploying a **hybrid lexical-semantic search system with CDC updates and ANN vectors** - all in one stack.  

---