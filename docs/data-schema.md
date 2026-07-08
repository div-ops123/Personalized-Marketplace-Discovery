# Recommendation System — Data Schema & Training Design

---

## Stage 1: Retrieval

**Goal:** Narrow millions of items to ~500 plausible candidates fast.

### Item Features (Item Tower)
| Feature | Type | Notes |
|---|---|---|
| Item ID | Categorical | Learned embedding |
| Category | Categorical | Top-level (e.g. Footwear) |
| Subcategory | Categorical | Fine-grained (e.g. Sneakers) |
| Brand | Categorical | Learned embedding |
| Price tier | Categorical | Bucketed (budget / mid / premium) |
| Tags | Multi-hot | Style, occasion, material etc. |
| Description embedding | Dense vector | Pre-encoded offline (e.g. sentence-transformer) |

### User Features (User Tower)
| Feature | Type | Notes |
|---|---|---|
| User ID | Categorical | Learned embedding |
| Purchased categories (last 90 days) | Multi-hot | Reflects recent taste |
| Average price point of past purchases | Continuous | Price affinity signal |
| Preferred brands | Multi-hot | Top N brands by purchase frequency |
| Time since last purchase | Continuous | Recency signal — also used as training weight |
| Device type | Categorical | Mobile / desktop / tablet |
| Country / State | Categorical | Geo context |
| Current product being viewed | Categorical | arrives directly in the request payload from the website. |


### Labels
**Positives:** Purchase and Add-to-Cart events only.
Clicks excluded at retrieval stage — too noisy, too weak to teach the model what users genuinely want.

**Positive sampling:** All confirmed purchases and add-to-cart events from recommendation widgets. Also include organic purchases and add-to-carts — behavior revealed preference regardless of how the user got there. Log surface attribution separately so you can filter by source during evaluation.

**Negative sampling strategy:** In-batch negatives with popularity correction.
- Other users' positive items in the same training batch serve as negatives.
- Apply frequency-based down-weighting to popular items — they appear as negatives disproportionately often across batches, which trains the model to under-rank them without correction.
- Add a small proportion of random negatives early in training for stability (curriculum learning — start easy, introduce difficulty as model matures).
- Do NOT use hard negatives at retrieval stage. The model is learning coarse distinctions. Hard negatives introduce fine-grained difficulty before the model has learned basic structure — this destabilizes training.

### Recency Decay
Applied two ways:
1. **As a training weight (before training):** Recent interactions weighted higher during gradient updates. Interactions older than 90 days receive decayed sample weight. Prevents stale historical behavior from dominating the model.
2. **As a feature (inside the model):** Time-since-last-purchase, time-since-last-category-interaction. Lets the model learn how recency affects preference rather than hardcoding it externally.

---

## Stage 2: Ranking

**Goal:** Score ~500 retrieval candidates and select final list shown to user.

### Item Features
| Feature | Type | Notes |
|---|---|---|
| Item ID | Categorical | Learned embedding |
| Price | Continuous | Absolute price |
| Category | Categorical | Same as retrieval |
| Subcategory | Categorical | Same as retrieval |
| Brand | Categorical | Same as retrieval |

### User Features
| Feature | Type | Notes |
|---|---|---|
| User ID | Categorical | Learned embedding |
| Device | Categorical | Mobile / desktop / tablet |
| Country | Categorical | Geo context |
| Page context | Categorical | Homepage / Product page / Cart page |

### User-Item Cross Features (Wide Component)
These are hand-engineered feature crosses fed into the wide (memorization) component.
| Feature | Notes |
|---|---|
| Has user purchased from this seller before? | Trust / familiarity signal |
| Has user bought in this category before? | Category affinity |
| Item price vs. user average spend | Price fit signal |
| Item brand vs. user preferred brands | Brand affinity match |

### Event & Impression Features
| Feature | Type | Notes |
|---|---|---|
| Impression timestamp | Timestamp | Time of widget render |
| Position in widget | Integer | Critical for position bias correction |
| Surface | Categorical | Recommendation widget only |
| Current product being viewed | Categorical | arrives directly in the request payload from the website. |
| Dwell time | Continuous | Time spent on product page after click |
| Event type | Categorical | Click / add-to-cart / purchase / save |

### Labels
Graded relevance score (0–5), combining multiple implicit signals:

| Signal | Score | Reasoning |
|---|---|---|
| Purchase | 5 | Strongest confirmed preference signal |
| Add to cart | 4 | Strong purchase intent |
| Save / Wishlist | 3 | Acknowledged future interest |
| Click + dwell time above threshold | 2 | Genuine engagement |
| Click + immediate bounce | 1 | Weak — could be misclick or price shock |
| Exposed, no interaction | 0 | Confirmed negative — user saw it and passed |

### Negative Sampling Strategy
**Primary:** Confirmed exposed negatives — items shown to the user in the recommendation widget at impression time T that the user did not interact with.
- Negative label ties to the impression event, not a fixed time window. When the widget renders at timestamp T showing items A–J, log that exact impression. Widget refresh at T+3 minutes is a separate impression event.
- Filter by false negative check: before labeling any item as a negative, verify it does not appear in the user's full interaction history across all sessions and all surfaces. A purchase from organic search last week disqualifies an item from being a negative today.

**Secondary:** Hard negatives — items the current retrieval model scores high for this user but that the user did not interact with, filtered by confirmed exposure. These reflect the exact difficulty the ranker faces in production: fine-grained discrimination between already-plausible candidates.

**Do NOT use:**
- Items recommended but never exposed to user (position bias — user never had a fair chance)
- Random catalog items (ranker never sees these in production, training on them teaches nothing useful)

### Position Bias Correction
Log position, device, and viewport for every impression. Apply Inverse Propensity Scoring (IPS) — weight each interaction by the inverse probability that the item at that position was actually seen. A click at position 8 on mobile carries more signal than a click at position 1, because surviving position 8 to get a click is a stronger evidence of genuine preference.

---

## Feedback Loop & Logging

**Interaction logging:** Collect from all surfaces — recommendation widgets, organic search, direct navigation. Log surface attribution on every event so you can filter by source.

**Why collect from all surfaces:** Behavior reveals preference regardless of how the user got there. Use all interactions for building user features and taste profiles. Use only recommendation-attributed interactions for evaluating whether the recommender itself is working.

**Label delay:** Click and add-to-cart recorded immediately. Purchase may have a checkout delay — log checkout completion timestamp separately and join on session/order ID. Do not assume purchase = click timestamp.

---

## Key Design Principles

**Latency constraint determines feature availability.** Features requiring real-time computation (current product being viewed, session context) are only usable at ranking stage where you have milliseconds. Retrieval stage must rely on pre-computed, cached features to meet sub-100ms latency. This forces the algorithm choice — not the other way around.

**Retrieval learns coarse distinctions.** Relevant vs. irrelevant across millions of items. Sampling strategy: easy negatives, in-batch, popularity corrected.

**Ranking learns fine distinctions.** Better vs. slightly worse among already-plausible candidates. Sampling strategy: hard negatives from confirmed exposure, false negative filtered.

**The gap between stages is your isolation layer for debugging.** If a relevant item was in retrieval candidates but ranked low → ranker problem. If it was never in candidates at all → retrieval problem. Without logging both sides of this gap you cannot make that cut.
