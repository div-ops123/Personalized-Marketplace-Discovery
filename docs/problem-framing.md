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

===

UPDATE:

One surface. One widget. Built completely end to end.

Which surface and which widget:
Product detail page.
The user is looking at item X. Show them items most likely to convert given that context.

**Widget shown to the user:** 
"Similar Items"

**Recommendation strategy:**
Retrieve and rank products that are most similar to the currently viewed item and most likely to be purchased.

**Primary metric**
Recommendation-attributed CVR — Do recommendations result in purchases?

**Guardrail:**
Recommendation-attributed CTR — monitored to catch click-bait (rising CTR with flat or declining CVR). Not a primary optimization target. See metric-ladder.md.

**Stage 1 — Retrieve similar items (Item-Item tower)**

Examples of signals:
image embedding similarity
text embedding similarity
category
brand
tags
price range

This answers:
"Which products are similar?"

**Stage 2 — Rank them**

Among those similar products, predict
"Which one is this user most likely to click or buy?"

example signals:
retreval similarity score
features from current item, 
candidate item id
user preferences(brand, average purchase price, average clicked price, device type)
historical Query/context CTR of candidate item
ex: Viewing running shoes -> How often is THIS candidate clicked?
historical conversion rate of candidate item


The model outputs one score:
Predicted probability that this user purchases this candidate, given this current item (recommendation-attributed purchase).

Training labels (full definitions in metric-ladder.md):
- Positive: recommendation-attributed purchase
- Negative: shown but not purchased (includes both clicked-no-purchase and shown-no-click impressions), downsampled to manage class imbalance

Attribution window = 24 hours from click (starting assumption, pending business input on product-specific buying cycles). See metric-ladder.md.