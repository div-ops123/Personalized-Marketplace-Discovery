# Recommendation System — High-Level Design

## System Purpose
A two-stage retrieval-then-rank architecture that serves personalized item recommendations to users on a marketplace platform. Optimizes for GMV as north star, with catalog coverage and seller fairness as guardrail metrics.

---

## Architecture Overview

```
User Request
     │
     ▼
Retrieval Service ──── ANN Index (item embeddings)
     │                      │
     │                 Item Feature Store
     │
     ▼
Top 500 Candidates + Similarity Scores
     │
     ▼
Ranking Service ──── Item Feature Store (batch lookup)
     │            ── User Feature Store
     │            ── Real-time Session Store
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

**Goal:** Narrow millions of catalog items to ~500 plausible candidates fast.
**Model:** Two-Tower Neural Network
**Latency budget:** < 100ms
**Evaluation metric:** Recall@500

---

### Offline Path

#### Item Side
- Item tower features are precomputed offline and stored as dense vectors in an ANN index.
- **New item trigger:** The catalog service fires an event the moment a new item is created. That event triggers an embedding job — takes the new item's features, runs them through the frozen item tower, produces a vector, writes it directly into the ANN index without rebuilding the whole index.
- **ANN index rebuild:** Not triggered per new item insert — too expensive. Incremental inserts handle new items in real time. Full rebuilds run when index drift (gap between current index recall and a fresh build) crosses a defined threshold. Index drift is the trigger, not a fixed weekly schedule, because ANN recall degrades unpredictably with volume of incremental inserts.

#### User Side
- Daily batch job reads interaction logs from the data warehouse, computes aggregate user features per user, writes to the user feature store.
- Aggregate features: purchased categories (last 90 days), average price point of past purchases, preferred brands, time since last purchase.
- These are stable signals. Daily refresh is sufficient.

#### Real-Time Session Features
- What the user has clicked or searched in the last 10 minutes of their current session.
- Cannot live in a daily batch pipeline — too stale.
- Events streamed from the website into a stream processor (Kafka + Flink or similar), aggregated per session, written to a low-latency session store readable at inference time.

---

### Online Path

When a user request arrives at the retrieval service:

1. Retrieval service fires two parallel fetches:
   - Aggregate user features from the user feature store
   - Real-time session features from the session store
2. Both arrive in milliseconds — pre-computed, not computed on the fly.
3. Features concatenated into a user feature vector.
4. User feature vector passed through the user tower model → produces user embedding.
5. User embedding used to query the ANN index.
6. ANN index returns top 500 candidate items with similarity scores.
7. Candidate IDs and scores forwarded to the ranking service.

---

### Feature Store Architecture

| Store | Purpose | Latency | Examples |
|---|---|---|---|
| Data warehouse | Raw interaction logs for training | Seconds–minutes | BigQuery, Redshift |
| Feature store | Pre-computed features for inference | < 10ms | Redis, DynamoDB, Feast |
| Session store | Real-time session features | < 10ms | Redis, Flink output |

At inference time the retrieval service reads only from the feature store and session store — never directly from the data warehouse.

---

### Fallback & Degradation Chain

| Level | Condition | Behavior |
|---|---|---|
| Primary | All systems healthy | Full two-tower retrieval with aggregate + session features |
| Fallback 1 | Session store slow or unavailable | Retrieval with aggregate features only, skip session features |
| Fallback 2 | Feature store unavailable | Popular items filtered by user's known category preferences |
| Fallback 3 | Full retrieval system unavailable | Country-level bestsellers served from pre-computed static list cached in memory |

The third fallback requires no model, no feature store, no ANN search. It never goes down.

**Cold start handling:**
- Cold start items: handled naturally — item tower uses content features (category, brand, description embedding) so new items with zero interactions still get a meaningful embedding from content alone.
- Cold start users: no data in feature store → serve country-level bestsellers segmented by user's detected country.

---

### Monitoring

| Signal | Why it matters |
|---|---|
| P99 retrieval latency | Average latency hides the tail. 1% of requests at 800ms is felt by users even if average is 50ms |
| Candidate diversity per request | Diversity collapse = early warning of feedback loop bias before it appears in business metrics |
| New item coverage (last 7 days) | If new items stop appearing in any candidate sets, seller fairness guardrail is already broken |
| ANN recall vs. brute-force | Periodically compare ANN results against exact search on a sample of requests. Growing gap = index needs rebuild |

---

## Stage 2: Ranking

**Goal:** Score 500 retrieval candidates and select a final ordered list of 10–20 items.
**Model:** Wide & Deep
**Latency budget:** < 100ms (total end-to-end target: < 200ms)
**Evaluation metric:** NDCG@K

---

### Online Path

When the ranking service receives 500 candidate IDs and similarity scores from retrieval:

1. Ranking service fires three parallel fetches:
   - Batch item feature lookup for all 500 candidates from item feature store (batch key-value, not single-item lookup)
   - User aggregate features from user feature store
   - Real-time session features from session store
2. Cross features computed from fetched values:
   - Item price vs. user average spend
   - Item brand vs. user preferred brands
   - Has user purchased from this seller before?
   - Has user bought in this subcategory before?
3. All features + retrieval similarity scores concatenated and passed through the ranking model.
4. Model outputs a relevance score for each of the 500 candidates.
5. Re-ranking layer applied in order (see Re-ranking section).
6. Final ordered list of 10–20 items returned to the API layer.

---

### Offline Path

- Daily batch job updates aggregate user features in the user feature store.
- Stream processor continuously updates session features.
- Model retraining runs on a defined schedule (weekly) or triggered by monitoring alerts.
- New model versions deployed via shadow mode → canary rollout before full traffic shift.

---

session store collects only user-item interaction event streams. Clicks, views, add-to-carts, searches. Nothing about items themselves. 

Examples of what gets computed from the stream:

- Items clicked in the last N minutes
- Categories browsed in the last N minutes
- Searches typed in the last N minutes
- Current product being viewed right now
- Number of items viewed without clicking (signals browsing behavior vs. intent behavior)

These feed into cross features like "has the user shown interest in this category in the current session."

## Training-serving feature mismatch 

The correct answer is: you must log session features to the data warehouse and include them in training.
Here's how it works:

At serving time, when a request arrives, you fetch session features from the session store and use them for inference. At the same time, you log those exact session feature values — the ones actually used for that request — alongside the impression event into your data warehouse. Not the raw event stream, the computed feature values that went into the model.

When you retrain, you join your interaction logs with those logged feature values. Now your training data includes the same session features your serving pipeline uses. The model learns from W+X because training data contains W+X, exactly as serving sees it.

If you skip this logging step, you have two bad options: train without session features entirely (model never learns from real-time signals), or try to reconstruct session features from raw logs during training (you'll get approximations that don't match exactly what serving computed, introducing subtle skew). Both options degrade model quality.

The pattern has a name: **log-and-join training pipeline**. At serving time, log the exact feature values used for inference alongside a request ID. At training time, join interaction labels to those logged features using the request ID. This guarantees training and serving see identical feature distributions.

This is also why your feature store and session store need to be designed with logging in mind from day one — not retrofitted later.

Log the actual values used, not the raw values before handling.

---

### Re-ranking Layer

Re-ranking imposes business constraints on top of the ranking model's relevance scores. The model optimizes for predicted purchase probability — it does not know about diversity, novelty, or exploration. Re-ranking handles these as explicit rules applied after the model scores.

All three steps draw only from the top 500 candidates. No new items are pulled from outside the candidate pool. The pool is fixed after retrieval.

#### Order of Application

```
Ranking model scores
        │
        ▼
1. Diversity filtering (MMR)
        │
        ▼
2. Novelty injection
        │
        ▼
3. Exploration slots
        │
        ▼
Final ordered list
```

**Why this order matters:** The ranking model first establishes the best relevance ordering. Diversity is applied early to prevent category concentration among the highest-ranked items. Novelty is then introduced by replacing a small number of slots with relevant items the user hasn't previously seen. Exploration comes later to reserve a controlled number of positions for uncertain items without overwhelming the list. If exploration were applied before diversity, exploration slots could cluster in the same category, forcing diversity to remove or demote them — wasting the exploration budget and collecting less useful feedback.

---

#### 1. Diversity (MMR — Maximum Marginal Relevance)

**What it does:** Reorders the existing top 500 candidates so no single category dominates the final list. Does not add new items or remove items from the pool.

**Mechanism:** Build the final list one slot at a time. For each next slot, apply a penalty to every remaining candidate proportional to its similarity to items already selected. A tuning parameter (lambda) controls the relevance-diversity balance. Lambda close to 1 = relevance dominates. Lambda close to 0 = diversity dominates. Similarity computed in item embedding space — same embeddings produced by the item tower.

**Tradeoff:** Trading some predicted purchase probability for a list that doesn't feel repetitive. A list of ten identical items converts worse than a diverse list even if each individual item scored higher.

---

#### 2. Novelty Injection

**What it does:** Replaces a fixed number of final list slots with the highest-scored candidates from the top 500 that do not appear in this user's impression or purchase history.

**Mechanism:** Fixed slot allocation, not score boosting. Score boosting creates a calibration problem — hard to tune without distorting the entire ranked list. Fixed slots give precise control over exactly how many novel items appear, independent of score magnitudes.

**Key distinction:** Novelty is user-specific, not item-specific. An item is novel to John if John hasn't seen it. The same item is not novel to Mary if Mary saw it last week. Novelty check queries the specific user's impression log — not a global "new items" list.

**Tradeoff:** Novelty slots convert worse short-term. Accepted because users who discover new things they like retain longer. Retention is a guardrail metric.

---

#### 3. Exploration Slots

**What it does:** Replaces a fixed number of final list slots with the highest-scored candidates from the top 500 that have high score uncertainty — new sellers, new categories, items with thin interaction history.

**Key distinction:** Exploration targets high uncertainty, not low scores. Low score = model is confident this item is not relevant. High uncertainty = model doesn't have enough signal to know either way. Spending a slot on a low-scored item collects nothing useful. Spending a slot on an uncertain item collects signal that improves future recommendations.

**Mechanism:** Identify candidates below an interaction count threshold, or whose confidence interval around the predicted score is wide. Allocate fixed slots. Fill with highest-scored items from that uncertain subset.

**Critical:** Log exploration slots separately from regular recommendation slots. Without separate logging you cannot measure what the system learned from exploration, cannot distinguish exploration-driven interactions from organic recommendation interactions, and cannot tell whether catalog coverage improvements came from exploration or organic discovery.

**Tradeoff:** Trading certain conversion for information gain on under-explored items. Without exploration the feedback loop closes and catalog coverage collapses over time.

---

### Fallback & Degradation Chain

| Level | Condition | Behavior |
|---|---|---|
| Primary | All systems healthy | Full Wide & Deep ranking with all features + re-ranking rules |
| Fallback 1 | Session store slow | Ranking with offline features only, skip real-time session features |
| Fallback 2 | Ranking model unavailable | Rank by retrieval similarity score only, still apply diversity rules |
| Fallback 3 | Full ranking system unavailable | Retrieval candidates sorted by popularity with diversity filtering only |

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
| Diversity distribution | Guardrail — are re-ranking diversity rules holding |
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
1. User opens recommendation widget
2. Request hits retrieval service
3. Retrieval fetches user aggregate features + session features in parallel
4. User tower produces user embedding
5. ANN search returns top 500 candidates + similarity scores
6. Candidates forwarded to ranking service
7. Ranking service batch-fetches item features for all 500 candidates
8. Ranking fetches user aggregate + session features in parallel
9. Cross features computed
10. Wide & Deep model scores all 500 candidates
11. Re-ranking: diversity → novelty → exploration
12. Final 10–20 items returned to API
13. API serves response to user
14. Impression event logged: user ID, item IDs, positions, device, timestamp, surface
15. Interaction events logged as they occur: click, add-to-cart, purchase, dwell time
16. Logs flow to data warehouse for next training cycle
17. Stream processor aggregates session events in real time for next request
```

---

## Key Design Principles

- **Latency constraint determines feature availability.** Features requiring real-time computation only at ranking stage. Retrieval relies on pre-computed cached features to meet sub-100ms budget.
- **Retrieval learns coarse distinctions.** Relevant vs. irrelevant. Optimize for recall — missing a relevant item is unrecoverable downstream.
- **Ranking learns fine distinctions.** Better vs. slightly worse among already-plausible candidates. Optimize for NDCG — position of relevant items matters.
- **The gap between stages is your isolation layer for debugging.** Relevant item in retrieval candidates but ranked low → ranker problem. Never appeared in candidates → retrieval problem. Log both sides.
- **Re-ranking separates model decisions from business decisions.** The model optimizes relevance. Re-ranking imposes diversity, novelty, and exploration as explicit product rules on top.
- **Every fallback level is a design decision, not an afterthought.** Each level trades personalization quality for system availability in a controlled, predictable way.
        
