# Recommendation System — Data Schema & Training Design

---

## Stage 1: Retrieval

**Goal:** Narrow millions of items to ~500 plausible candidates fast.

**Model:** Shared-weight Siamese (twin) item encoder. A single item-encoding network is run independently on the anchor item and on each candidate item — there is no User Tower. Similarity is the dot-product/cosine between the two independently-produced embeddings. This is an item-to-item design, matching the PDP "Similar Items" task ("which products are similar?") — personalization is deferred entirely to Stage 2.

### Item Encoder Features (shared weights — same encoder run on anchor and candidate)
| Feature | Type | Notes |
|---|---|---|
| Item ID | Categorical | Learned embedding |
| Category | Categorical | Top-level (e.g. Footwear) |
| Subcategory | Categorical | Fine-grained (e.g. Sneakers) |
| Brand | Categorical | Learned embedding |
| Price tier | Categorical | Bucketed (budget / mid / premium) |
| Tags | Multi-hot | Style, occasion, material etc. |
| Image embedding | Dense vector | Raw pre-encoded image-model output, fed as encoder input — not a precomputed pairwise similarity score |
| Text / description embedding | Dense vector | Raw pre-encoded text-model output (e.g. sentence-transformer), fed as encoder input — not a precomputed pairwise similarity score |

Both embeddings must be per-item vectors computed independently of any specific pairing, so each item can be encoded once, cached, and indexed for ANN search. A precomputed pairwise similarity score would require the pair to already be known and is incompatible with this architecture.

**Serving-time consequence:** because there is no User Tower, the anchor item's embedding is already precomputed and cached at indexing time. Serving a request is an embedding lookup + ANN search over the item index — no live encoder inference is needed at request time, which is even more favorable to the sub-100ms budget than a personalized two-tower design would be.

### Training Pairs & Labels
**Positive pair:** (anchor item, recommended item) where the recommended item was clicked from the Similar Items widget.
**Negative pair:** (anchor item, recommended item) where the recommended item was shown but not clicked before the impression ended.
**Additional training negatives:** In-batch negatives.

#### Why use in-batch negatives?
Recommendation impressions provide only a limited number of exposed negatives per anchor. During contrastive training, positive pairs from other anchors in the same mini-batch are reused as additional negatives. This greatly increases the number and diversity of negative examples without requiring extra sampling from the catalog, improving training efficiency while keeping memory and computation manageable.

⚠️ Accepts the usual risk that some in-batch negatives may actually be semantically related (false negatives).

The retrieval model's goal isn't to mimic a text/image embedding model — it's to learn *behavioral* similarity: which items users actually treat as interchangeable or worth exploring together. Text/image embeddings are inputs to the encoder; the click-pair signal is the training target.

**Cold start / bootstrap:** before the Similar Items widget has production traffic, no click-pairs exist yet. Bootstrap the v0 encoder using content-based semantic similarity (image/text embedding distance, category/brand agreement) as a weak positive-pair heuristic, then transition to training on behavioral click-pairs once the widget is live and logging impressions.

---

## Stage 2: Ranking

**Goal:** Score ~200 retrieval candidates and select the final list shown to the user.

**Model:** Gradient-boosted trees (LightGBM/XGBoost) with the **LambdaMART** ranking objective. Features are primarily structured/tabular, which gradient-boosted trees handle well without requiring embeddings. LambdaMART combines GBDTs with LambdaRank's gradient formulation, which optimizes directly for NDCG improvements — weighting updates by how much correcting a pair would improve NDCG, rather than treating every pairwise mistake equally (as plain RankNet does). This concentrates model capacity on getting the top of the ranked list right, matching how users actually consume the widget.

### What does the Ranker train on?

Exactly this same retrieval candidate set.
Not random catalog items.
Not items retrieval never produced.

### User Features
| Feature | Type | Notes |
|---|---|---|
| Preferred brands | Multi-hot | Top N brands by purchase frequency |
| Average purchase price | Continuous | Price affinity signal |
| Historical category affinity | Multi-hot | Reflects recent taste |
| Country | Categorical | Geo context |
| Device | Categorical | Mobile / desktop / tablet — retained as a predictive/context signal (conversion propensity varies by device) independent of its separate role in position-bias logging below |

User ID is deliberately excluded: LightGBM/XGBoost don't jointly learn ID embeddings the way a neural model would, so personalization is carried through behavioral aggregates instead of a raw ID.

### Anchor Item Features
(the item currently being viewed on the PDP)
| Feature | Type | Notes |
|---|---|---|
| Category | Categorical | |
| Subcategory | Categorical | |
| Brand | Categorical | |
| Price | Continuous | |

### Candidate Item Features
| Feature | Type | Notes |
|---|---|---|
| Candidate ID | Categorical | Identifier only — not embedded |
| Category | Categorical | |
| Subcategory | Categorical | |
| Brand | Categorical | |
| Price | Continuous | |
| Historical recommendation CTR | Continuous | Candidate-level (see below), point-in-time correct |
| Historical recommendation CVR | Continuous | Candidate-level (see below), point-in-time correct |
| Historical recommendation impressions | Continuous | Count of prior impressions for this candidate — indicates reliability of the CTR/CVR estimates; low-impression candidates need smoothing/backoff toward a global prior |

### Anchor ↔ Candidate Cross Features
| Feature | Notes |
|---|---|
| Retrieval similarity score | Score from the Stage 1 item encoder |
| Same brand? | Boolean |
| Same category? | Boolean |
| Same subcategory? | Boolean |
| Price ratio | candidate_price / anchor_price |
| Absolute price difference | abs(candidate_price − anchor_price) |

### Example Training Row
| User Avg Price | User Preferred Brand | Historical Category Affinity | Anchor Price | Anchor Brand | Anchor Category | Candidate ID | Candidate Category | Candidate Subcategory | Candidate Brand | Candidate Price | Candidate Historical CTR | Candidate Historical CVR | Historical Impressions | Similarity Score | Same Brand | Same Subcategory | Price Ratio | Abs Price Diff | Label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

### Point-in-Time Correctness (Historical Rate Features)
Every historical rate/count feature is computed using only recommendation interactions observed **before** the current recommendation's timestamp — never the current impression or any future interaction:
- **Historical CTR** must reflect clicks/impressions before this impression, not lifetime totals.
- **Historical CVR** must only use purchases observed before this recommendation.
- **Historical impressions** must count impressions seen before recommendation time, not lifetime impressions.

Computing these without a strict as-of cutoff leaks future outcome information into training — the model will look correct offline and fail in production (label leakage, not just ordinary training-serving skew).

### Candidate-Level, Not Pair-Level, Historical Rates
Historical CTR/CVR are aggregated at the **candidate-item level**, across all anchors it has ever been recommended under — not per (anchor, candidate) pair. Pair-level rates would be too sparse: with up to 2M items, the anchor-candidate pair space is enormous and most pairs would have near-zero historical impressions.

Example: `CTR(candidate) = Clicks(candidate) / Impressions(candidate)`, aggregated across every anchor item that ever surfaced this candidate — so every recommendation of, say, an Adidas Ultraboost contributes to its estimate, regardless of which anchor item it was shown under.

### Training Pairs & Labels
**Positive:** Recommendation-attributed purchases (see metric-ladder.md for Attribution Window Definition).

**Negative:** Candidate items shown in the recommendation impression that were not purchased within the attribution window.
This includes both clicked-but-not-purchased and shown-but-not-clicked candidates.

**Reason:** The ranking model trains on the same candidate set produced by the retrieval stage. Since retrieval has already narrowed millions of products to a small set of plausible candidates, the ranker learns to distinguish between realistic alternatives presented to users in production. This mirrors serving behavior and removes the need for additional random negative sampling.

### Position Bias Correction
Log position and device for every impression. Apply Inverse Propensity Scoring (IPS) — weight each interaction by the inverse probability that the item at that position was actually seen. A click at position 8 on mobile carries more signal than a click at position 1, because surviving position 8 to get a click is stronger evidence of genuine preference.

### Event & Impression Features
| Feature | Type | Notes |
|---|---|---|
| Impression timestamp | Timestamp | Time of widget render |
| Position in widget | Integer | Critical for position bias correction |
| Dwell time | Continuous | Time spent on product page after click |
| Event type | Categorical | Click / add-to-cart / purchase / save |

---

## Feedback Loop & Logging

**Label delay:** Click and add-to-cart recorded immediately. Purchase may have a checkout delay — log checkout completion timestamp separately and join on session/order ID. Do not assume purchase = click timestamp.

---

## Key Design Principles

**Latency constraint determines feature availability.** With no User Tower, Stage 1 requires zero real-time feature computation — item embeddings are precomputed and cached, and serving is an embedding lookup + ANN search. Stage 2 has milliseconds to spend, which is where real-time/session-context features (current anchor item, device, position) are usable.

**Retrieval learns coarse distinctions.** Similar vs. dissimilar, across millions of items. Sampling strategy: easy negatives, in-batch, popularity corrected.

**Ranking learns fine distinctions.** Better vs. slightly worse among already-plausible candidates. Sampling strategy: hard negatives from confirmed exposure, false-negative filtered.

**The gap between stages is your isolation layer for debugging.** If a relevant item was in retrieval candidates but ranked low → ranker problem. If it was never in candidates at all → retrieval problem. Without logging both sides of this gap you cannot make that cut.
