# Recommendation System Metric Ladder

## 1. Business Outcome (North Star)

**Revenue attributable to recommendation surfaces**

> Gross Merchandise Value (GMV) generated from products that users clicked through the Product Detail Page Similar Items recommendation widget and purchased within the attribution window (see definition below).

**Why?**

This directly measures the business value created by the recommendation system rather than overall company revenue.

---

## Attribution Window Definition

> A purchase counts as recommendation-attributed if it occurs within 24 hours of a click on an item from the Similar Items widget, provided no other attribution event (e.g., a subsequent search or direct navigation touch) overrides it.

24 hours is a starting assumption, not a validated business constant — buying cycles vary by product category (e.g., groceries vs. furniture). The precedence rule for what counts as an "overriding" event is also not yet fully specified (e.g., last-touch vs. any-touch across channels). Revisit both with business/product input once category-level purchase-cycle data is available.

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

**Class imbalance:** purchase events are rare relative to impressions (e.g., 1 purchase per 100–500 impressions). Negatives are downsampled during training to avoid a degenerate "always predict negative" model.

Caution: downsampling negatives biases the model's raw output probability upward relative to true production rates. If the ranking score is used anywhere as a calibrated probability rather than purely for relative ordering, apply a calibration correction (e.g., log-odds correction for the known sampling ratio) at serving time.

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