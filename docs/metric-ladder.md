# Recommendation System Metric Ladder

## 1. Business Outcome (North Star)

**Revenue attributable to recommendation surfaces**

> Gross Merchandise Value (GMV) generated from products that users clicked through the Product Detail Page Similar Items recommendation widget and purchased within the attribution window (see definition below).

**Why?**

This directly measures the business value created by the recommendation system rather than overall company revenue.

---

## Attribution Window Definition

> A recommendation-attributed purchase is defined as a purchase occurring within 24 hours of a click on an item in the Similar Items widget. 

The 24-hour window is chosen as a reasonable starting assumption for a general e-commerce marketplace and should be validated using historical conversion data.
The appropriate attribution window may vary by product category and business domain.

---

## 2. Product Success Metric

**Recommendation-attributed Purchase Rate (Conversion Rate)**

> Percentage of recommendation clicks that result in a purchase within the attribution window.

**Why?**

If users purchase more items after interacting with recommendations, recommendation-attributed GMV should increase.

---

## 3. Model Evaluation Metrics

### Training Label Definitions

**Retrieval stage**

- Positive: recommendation impressions that were clicked.
- Negative: recommendation impressions shown but not clicked before the widget refreshed or the impression ended.

These labels are used to train the retrieval (candidate generation) model on click signal, since purchase events are too sparse at the impression level to train on directly. This is distinct from the Recall@K offline evaluation below, which measures against purchased items as ground truth — training on abundant click signal, evaluating against the rarer, business-aligned purchase signal.

**Ranking stage**

- Positive: recommendation-attributed purchases (see Attribution Window Definition above).
- Negative: shown but not purchased — includes both clicked-but-not-purchased and shown-but-not-clicked impressions.

---

### Offline

#### Retrieval

**Recall@K**

**Purpose**

Measures whether the retrieval stage successfully retrieves the relevant items that the user eventually interacted with.

**Relevant items**

Explicitly defined as items the user eventually purchased (or another predefined engagement signal).

**Choosing K**

K is determined by:

- the ranker's latency budget
- the catalog size

K is selected experimentally as the smallest candidate set that achieves near-saturated Recall while satisfying the end-to-end serving latency budget.

---

#### Ranking

**NDCG@K**

**Relevant item**

A relevant item is defined as a recommendation-attributed purchase within the defined attribution window.

**Choosing K**

K is determined by the number of recommendation slots available on the recommendation surface.

**Why NDCG?**

The ranking model is trained to predict the probability of purchase. Therefore, offline evaluation should measure whether purchased items are ranked as highly as possible.

NDCG rewards placing relevant (purchased) items closer to the top of the ranked list while assigning less credit when they appear lower in the recommendations. This aligns with the ranking model's objective, since users are more likely to interact with items shown near the top of the recommendation widget.

Binary relevance is used:

* Relevant (1): Recommendation-attributed purchase
* Not Relevant (0): All other recommended items

---

### Online

After deployment:

- Recommendation-attributed CTR
- Recommendation-attributed Purchase Rate

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

## Catalog Coverage — Deferred

Not implemented as an enforced guardrail in this iteration. Revisit once the Similar Items widget has production traffic data — for a widget conditioned on a single anchor item, this would need to be redefined as per-anchor-item coverage (avoiding always recommending the same popular substitutes for a given item) rather than global catalog %.

---

## Recommendation CTR

Monitor CTR as a click-bait detection signal. CTR is not a primary optimization target for this system (see problem-framing.md).

CTR should improve together with purchase rate.

A rising CTR with flat or declining purchases may indicate the model is optimizing for clicks rather than genuine user value.

---

# Hard Constraints

These should never be violated.

- Recommendation latency must satisfy the production SLA.
- Training-serving skew must remain within acceptable limits.
- Recommendation attribution must use correctly attributed labels from recommendation surfaces only.