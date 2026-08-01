# Recommendation System — High-Level Design

## System Purpose
A two-stage retrieval-then-rank architecture that serves personalized item recommendations to users on a marketplace platform. Optimizes for GMV as north star, with catalog coverage and seller fairness as guardrail metrics.

---

## Architecture Overview

```
Anchor Item (PDP view)
     │
     ▼
Retrieval Service ──── ANN Index (precomputed item embeddings)
     │                 anchor embedding lookup only -- no live
     │                 model inference, no user identity involved
     ▼
Top 200 Candidates + Similarity Scores
     │
     ▼
Ranking Service ──── Item Feature Store (batch lookup, all candidates)
     │            ── User Feature Store (if user known)
     │
     ▼
Ranked Candidates (scored)
     │
     ▼
Re-ranking Layer
     │
     ▼
Final List (10–20 items) → API → User
```

---

## Stage 1: Retrieval

**Goal:** Narrow millions of catalog items to ~200 plausible candidates fast.
**Model:** Siamese item encoder (content-based, item-item similarity -- no separate user tower)
**Latency budget:** < 30ms
**Evaluation metric:** Recall@200

---

### Offline Path

#### Item Side
- Item encoder features are precomputed offline and stored as dense vectors in an ANN index.
- **New item trigger:** The catalog service fires an event the moment a new item is created. That event triggers an embedding job — takes the new item's features, runs them through the frozen item encoder, produces a vector, writes it directly into the ANN index without rebuilding the whole index.
- **ANN index rebuild:** Not triggered per new item insert — too expensive. Incremental inserts handle new items in real time. Full rebuilds run when index drift (gap between current index recall and a fresh build) crosses a defined threshold. Index drift is the trigger, not a fixed weekly schedule, because ANN recall degrades unpredictably with volume of incremental inserts.

---

### Online Path

When a PDP request arrives at the retrieval service (anchor item ID + k):

1. Look up the anchor item's precomputed embedding directly from the ANN index — no live model inference, no user identity involved. The item encoder only ever runs offline, at indexing time.
2. Query the ANN index with that embedding.
3. ANN index returns top 200 candidate items with similarity scores, excluding the anchor itself.
4. Candidate IDs and scores forwarded to the ranking service.

Personalization enters at the ranking stage, not here — see Stage 2.

---

### Feature Store Architecture

| Store | Purpose | Latency |
|---|---|---|
| Offline: Snowflake | Raw interaction logs, serving-time logged feature values, assembled point-in-time-correct training datasets | Query-time only, not on the serving path |
| Online: Redis (ElastiCache) | Precomputed User Daily Features + Candidate Daily Features, written daily by the Spark batch job, read at ranking time | Sub-millisecond P99 |

At inference time, retrieval reads only from the precomputed ANN index — no feature-store lookup at all. Ranking reads only from the online feature store (Redis). Neither ever reads directly from the data warehouse.

---

### Fallback & Degradation Chain

| Level | Condition | Behavior |
|---|---|---|
| Primary | ANN index healthy | Full ANN search on the anchor item's precomputed embedding |
| Fallback 1 | ANN index unavailable | Same-category/brand popular items from the item catalog (catalog is current-state, not time-sensitive -- no live store needed) |
| Fallback 2 | Full retrieval system unavailable | Country-level bestsellers served from a pre-computed static list cached in memory |

The final fallback requires no model, no feature store, no ANN search. It never goes down.

**Cold start handling:**
- Cold start items: handled naturally — the item encoder uses content features (category, brand, description embedding) so new items with zero interactions still get a meaningful embedding from content alone.
- Cold start users: retrieval is item-anchored and never touches user identity, so there's no cold-start case at this stage at all. Personalization for unknown users degrades gracefully at ranking instead (see Stage 2).

---

### Monitoring

| Signal | Why it matters |
|---|---|
| P99 retrieval latency | Average latency hides the tail. 1% of requests at 800ms is felt by users even if average is 50ms |
| New item coverage (last 7 days) | If new items stop appearing in any candidate sets, seller fairness guardrail is already broken |
| ANN recall vs. brute-force | Periodically compare ANN results against exact search on a sample of requests. Growing gap = index needs rebuild |

---

## Stage 2: Ranking

**Goal:** Score 200 retrieval candidates and select a final ordered list of 10–20 items.
**Model:** LambdaMART (gradient-boosted trees, learning-to-rank)
**Latency budget:** < 70ms (the remaining end-to-end budget after retrieval's <30ms -- total request latency stays under 100ms p99)
**Evaluation metric:** NDCG@K

---

### Online Path

When the ranking service receives 200 candidate IDs and similarity scores from retrieval:

1. Ranking service fires two parallel fetches:
   - Batch item feature lookup for all 200 candidates from the item feature store (batch key-value, not single-item lookup)
   - User aggregate features from the user feature store, if the user is known (unknown/anonymous users simply skip this fetch — see Fallback & Degradation Chain below)
2. Cross features computed from fetched values:
   - Item price vs. user average spend
   - Item brand vs. user preferred brands
   - Has user purchased from this seller before?
   - Has user bought in this subcategory before?
3. All features (categorical + continuous + cross features) plus the retrieval similarity score are assembled into a single feature vector per candidate and passed through the LambdaMART model.
4. Model outputs a relevance score for each of the 200 candidates.
5. Re-ranking layer applied in order (see Re-ranking section).
6. Final ordered list of 10–20 items returned to the API layer.

---

### Offline Path

- Daily batch job updates aggregate user features in the user feature store.
- Model retraining runs on a defined schedule (weekly) or triggered by monitoring alerts.
- New model versions deployed via shadow mode → canary rollout before full traffic shift.

---

This system has no real-time session store — no in-session click/view/search stream is fetched or fed into ranking. Every feature ranking reads (item, candidate, user) is a precomputed batch snapshot from the daily feature pipeline (see Feature Store Architecture above). Train/serve skew is avoided the same way LLD.md's Training/Serving Preprocessing Parity section describes: point-in-time joins against those same snapshot tables at both training and serving time, not by logging and replaying live-computed features. If session-level personalization is added later, it would need the log-and-join pattern (log the exact feature values used at serving time, join to them at training time) — but that's future scope, not part of this design.

---

### Re-ranking Layer

Re-ranking imposes business constraints on top of the ranking model's relevance scores. The model optimizes for predicted purchase probability — it does not know about novelty or exploration. Re-ranking handles these as explicit rules applied after the model scores.

Both steps draw only from the top 200 candidates. No new items are pulled from outside the candidate pool. The pool is fixed after retrieval.

#### Order of Application

```
Ranking model scores
        │
        ▼
1. Novelty injection
        │
        ▼
2. Exploration slots
        │
        ▼
Final ordered list
```

**Why this order matters:** The ranking model first establishes the best relevance ordering. Novelty is applied first, replacing a small number of slots with relevant items the user hasn't previously seen. Exploration comes after, reserving a controlled number of positions for uncertain items without overwhelming the list — running it first could let exploration slots crowd out the novelty budget before novelty gets a chance to run. (A category/brand diversity step is not currently specified — if added later, it would need its own mechanism defined before it can be placed in this ordering.)

---

#### 1. Novelty Injection

**What it does:** Replaces a fixed number of final list slots with the highest-scored candidates from the top 200 that do not appear in this user's impression or purchase history.

**Mechanism:** Fixed slot allocation, not score boosting. Score boosting creates a calibration problem — hard to tune without distorting the entire ranked list. Fixed slots give precise control over exactly how many novel items appear, independent of score magnitudes.

**Key distinction:** Novelty is user-specific, not item-specific. An item is novel to John if John hasn't seen it. The same item is not novel to Mary if Mary saw it last week. Novelty check queries the specific user's impression log — not a global "new items" list.

**Tradeoff:** Novelty slots convert worse short-term. Accepted because users who discover new things they like retain longer. Retention is a guardrail metric.

---

#### 2. Exploration Slots

**What it does:** Replaces a fixed number of final list slots with the highest-scored candidates from the top 200 that have high score uncertainty — new sellers, new categories, items with thin interaction history.

**Key distinction:** Exploration targets high uncertainty, not low scores. Low score = model is confident this item is not relevant. High uncertainty = model doesn't have enough signal to know either way. Spending a slot on a low-scored item collects nothing useful. Spending a slot on an uncertain item collects signal that improves future recommendations.

**Mechanism:** Identify candidates below an interaction count threshold, or whose confidence interval around the predicted score is wide. Allocate fixed slots. Fill with highest-scored items from that uncertain subset.

**Critical:** Log exploration slots separately from regular recommendation slots. Without separate logging you cannot measure what the system learned from exploration, cannot distinguish exploration-driven interactions from organic recommendation interactions, and cannot tell whether catalog coverage improvements came from exploration or organic discovery.

**Tradeoff:** Trading certain conversion for information gain on under-explored items. Without exploration the feedback loop closes and catalog coverage collapses over time.

---

### Fallback & Degradation Chain

| Level | Condition | Behavior |
|---|---|---|
| Primary | All systems healthy | Full LambdaMART ranking with all features + re-ranking rules |
| Fallback 1 | Ranking model unavailable | Rank by retrieval similarity score only, still apply re-ranking rules |
| Fallback 2 | Full ranking system unavailable | Retrieval candidates sorted by popularity, re-ranking rules skipped |

Each level trades personalization quality for system availability.

---

### Monitoring

| Signal | Why it matters |
|---|---|
| Ranking P99 latency (separate from retrieval) | Identifies which stage is the bottleneck when end-to-end latency spikes |
| CTR by position | Detects whether ranker is learning relevance or just learning position effects. Watch for deviation from expected position decay curve |
| Recommendation-attributed CTR | Click-bait detection signal |
| Recommendation-attributed purchase rate | Direct conversion signal |
| Recommendation-attributed GMV | North star signal |
| Catalog coverage | Guardrail — are we surfacing a healthy spread of the catalog |
| Per-version metric comparison | Canary rollout health check |

---

## Model Versioning & Rollout

### Versioning
- Every model deployed to production gets a version tag.
- Serving infrastructure routes traffic to a specific version.
- New model versions sit in staging until rollout is approved.
- Previous version always available for instant rollback.
- Never overwrite a production model in place.

### Rollout Strategy

```
New model trained
        │
        ▼
Shadow mode
New model runs on live traffic.
Outputs logged but not served.
Compare candidate sets and scores against current production model.
        │
        ▼ (if shadow outputs look healthy)
Canary rollout — 5% of traffic
Monitor CTR, purchase rate, GMV per group.
        │
        ▼ (if canary metrics hold or improve)
Gradual traffic increase → 100%
        │
        If metrics degrade at any point:
        Route canary traffic back to old model immediately.
        Zero downtime. Zero user impact.
```

---

## End-to-End Request Flow (Happy Path)

```
1. User opens a PDP; the recommendation widget requests similar items for the anchor item
2. Request hits retrieval service
3. Retrieval looks up the anchor item's precomputed embedding from the ANN index
4. ANN search returns top 200 candidates + similarity scores (excluding the anchor)
5. Candidates forwarded to ranking service
6. Ranking service batch-fetches item features for all 200 candidates
7. Ranking fetches user aggregate features in parallel, if the user is known
8. Cross features computed
9. LambdaMART model scores all 200 candidates
10. Re-ranking: novelty → exploration
11. Final 10–20 items returned to API
12. API serves response to user
13. Impression event logged: user ID (if known), item IDs, positions, device, timestamp, surface
14. Interaction events logged as they occur: click, add-to-cart, purchase, dwell time
15. Logs flow to data warehouse for next training cycle
```
