The pain is never "our model is bad." 
The pain is always: buyers don't find what they'd want, so the business loses revenue, conversion, and repeat purchase. 
The ML system is the mechanism. Revenue is the measurement.

# Problem Statement

Given a buyer viewing an item on the Product Detail Page — their interaction history, and the attributes of the item they're currently viewing — retrieve and rank the items most similar to that item which this buyer is most likely to click and purchase, served as the "Similar Items" widget on the Product Detail Page, under a strict serving latency of under 100ms at p99 — in order to increase recommendation-attributed conversion rate.

---

# Scale parameters for a Series B/C marketplace

* 500K–2M monthly active buyers
* 200K–2M items in catalog (design reference: 2M items)
* 20–100M interaction events per month (clicks, saves, purchases, dwell)
* Sub-100ms p99 serving latency required

---

# Define "DONE"

"Done" = a model that beats a popularity baseline on recommendation-attributed conversion rate in an A/B test, without the gain coming from CTR alone (see click-bait guardrail in metric-ladder.md), serving under 100ms, with a monitoring dashboard that catches degradations before the user does. Catalog coverage is not an enforced guardrail in this iteration (see metric-ladder.md).

* A/B testing + causal inference
* Translating model output into business metrics
* Designing the experiment, not just running it
* Multi-objective tradeoff thinking
