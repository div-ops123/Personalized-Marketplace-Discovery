# Recommendation System Metric Ladder

## 1. Business Outcome (North Star)

**Revenue attributable to recommendation surfaces**

> Gross Merchandise Value (GMV) generated from products that users clicked through a recommendation widget and purchased within a defined attribution window.

**Why?**

This directly measures the business value created by the recommendation system rather than overall company revenue.

---

## 2. Product Success Metric

**Recommendation-attributed Purchase Rate (Conversion Rate)**

> Percentage of recommendation clicks that result in a purchase within the attribution window.

**Why?**

If users purchase more items after interacting with recommendations, recommendation-attributed GMV should increase.

---

## 3. Model Evaluation Metrics

### Offline

#### Retrieval

**Recall@K**

**Purpose**

Measures whether the retrieval stage successfully retrieves the relevant items that the user eventually interacted with.

**Relevant items**

Explicitly defined as items the user eventually purchased (or another predefined engagement signal).

**Choosing K**

K is determined by:

- the ranker's latency budget (e.g., p99 ≤ 100 ms)
- the catalog size

The retrieval stage should return enough candidates for effective ranking while allowing the ranker to satisfy production latency constraints.

---

#### Ranking

**NDCG@K**

K is determined by the number of recommendation slots available on the recommendation surface.

**Why NDCG?**

NDCG supports **graded relevance**.

For example:

- Purchase > Wishlist > Click

Unlike MAP, NDCG rewards placing the most valuable user interactions closer to the top of the ranked list.

---

### Online

After deployment:

- Recommendation-attributed CTR
- Recommendation-attributed Purchase Rate
- Recommendation-attributed GMV

These are measured through online A/B testing before a full rollout.

---

## 4. Feature & Data Quality Metrics

### Completeness

Percentage of missing or default feature values during both training and serving.

Poor completeness results in biased model inputs.

---

### Freshness

Measures how up-to-date features are when the model serves recommendations.

Fresh features ensure the recommender reacts quickly to recent user behavior.

---

### Training-Serving Skew

Verifies that feature computation is identical during training and inference.

Different feature pipelines can silently degrade production performance.

---

### Label Attribution

Ensures training labels originate only from recommendation surfaces.

For example:

✓ Purchased after clicking a recommendation

✗ Purchased after search

✗ Purchased through direct navigation

Without proper attribution, the model learns general purchasing behavior rather than recommendation effectiveness.

---

# Guardrails

While optimizing the North Star metric, the following guardrails must remain healthy.

## Catalog Coverage

Ensure recommendations are distributed across a sufficiently large portion of the catalog rather than repeatedly recommending only popular items.

---

## Diversity Distribution

Monitor category concentration.

If recommendations become overly concentrated in a small number of categories, introduce a reranking stage to improve diversity while preserving relevance.

---

## Recommendation CTR

Monitor CTR as a click-bait detection signal.

CTR should improve together with purchase rate.

A rising CTR with flat or declining purchases may indicate the model is optimizing for clicks rather than genuine user value.

---

# Hard Constraints

These should never be violated.

- Recommendation latency must satisfy the production SLA (e.g., p99 ≤ 100 ms).
- Training-serving skew must remain within acceptable limits.
- Recommendation attribution must use correctly attributed labels from recommendation surfaces only.