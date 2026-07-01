The pain is never "our model is bad." 
The pain is always: buyers don't find what they'd want, so the business loses revenue, conversion, and repeat purchase. 
The ML system is the mechanism. Revenue is the measurement.

# Problem Statement

Given a buyer's interaction history (clicks, purchases, etc), their real-time session context (current page, query, time of day, etc), and a catalog of available items — predict the ranked list of items most likely to result in a click or purchase, for each unique buyer, at every page load across homepage, listing, and cart surfaces, under a strict serving latency of under 100ms at p99 — in order to increase add-to-cart rate, and repeat purchase rate, without sacrificing catalog coverage or over-indexing on already-popular items.

---

# Constraints you must negotiate, in priority order
1. Latency vs. quality. You cannot run a deep neural network at query time on a catalog of 1M+ items. This is why retrieval-then-rank exists.

2. Popularity bias vs. catalog coverage. A naïve model learns that popular items convert, so it shows only popular items. This creates a feedback loop: popular items get shown → they get clicks → they train as "relevant" → they get shown more. The long tail dies. Your design must name this, and your solution must show you've thought about it — whether through exploration mechanisms, coverage constraints in ranking, or diversity-aware loss functions.

3. Cold start vs. relevance. New buyers have no interaction history. New items have no engagement data. Both break a pure collaborative filtering approach. Your design must state your cold-start strategy for both users and items.

4. Online metrics vs. offline metrics. Offline NDCG doesn't move revenue. Multiple postings make this explicit. Your project must define both — the offline metric you track during development, and the online business metric that decides whether the model ships.

5. Feedback loop correctness vs. convenience. If you train on logged clicks without correcting for position bias, you're training the model to recommend whatever you showed at position 1 — not what users actually preferred. This is propensity scoring. inDrive names it directly. Your design needs to acknowledge it and state your approach, even if the approach is "log position and use inverse propensity weighting in training."

---

# Scale parameters for a Series B/C marketplace

* 500K–2M monthly active buyers
* 200K–2M items in catalog
* 20–100M interaction events per month (clicks, saves, purchases, dwell)
* Sub-100ms p99 serving latency required

---

# Define "DONE"

"Done" = a model that beats a popularity baseline on CTR in an A/B test, with catalog coverage above a defined floor, serving under 100ms, with a monitoring dashboard that catches degradations before the user does.

* A/B testing + causal inference
* Translating model output into business metrics
* Designing the experiment, not just running it
* Multi-objective tradeoff thinking

