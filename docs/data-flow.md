## Raw Logs

Every time a user does anything on the platform, an event fires. 
That event contains: user_id, item_id, event_type (click/view/purchase/add-to-cart), timestamp, position in widget, device, page context, session_id, surface. 
This is raw. It's one row per event. 
A user who had 50 interactions today generates 50 rows.

These raw events land in **Snowflake** continuously via **Kinesis**.

---

**What the Spark batch job actually computes.**

Spark reads the raw event table from Snowflake and computes per-user aggregations:

- Categories purchased in the last 90 days → scan all purchase events for this user in the last 90 days, collect the categories
- Average price point → average the item prices across all purchases
- Preferred brands → count brand occurrences across purchases, take the top N
- Time since last purchase → find the most recent purchase event timestamp

One row per user comes out. These aggregations get written to Redis (online store) so the retrieval service can fetch them in under 10ms at serving time.

---

**point-in-time correctness.**

You're training the model on features that didn't exist at the time of the interaction. This is called **training-serving skew** caused by temporal leakage.

---

**The problem: training-serving skew**
**The fix: the log-and-join training pipeline.**

At serving time, when your retrieval service fetches John's features from Redis to make a recommendation, you also log those exact feature values alongside a request_id to Snowflake. Not recomputed later. The exact values the model used at that moment.

What gets logged per serving event:
- request_id
- user_id
- timestamp
- The exact feature vector used: purchased_categories at that moment, avg_price_point at that moment, preferred_brands at that moment
- item_ids shown to the user
- positions each item appeared at

Then separately, interaction events keep flowing in naturally — clicks, purchases, add-to-carts.

At training time, you join these two tables using request_id and timestamp:

"For this request at timestamp T, John's features were X, items shown were Y, and he clicked item Z."

That assembled row is one training example. Features are correct for the moment the model actually made the prediction. Labels come from what actually happened afterward.

---

**So what does training data actually look like?**

One row per impression event. Columns:
- User features at serving time (from the logged feature values)
- Item features (fetched from item feature table by item_id)
- Cross features (computed at training time from the joined data)
- Label: did the user interact? What type? Graded relevance score.

You don't retrain on raw events directly. You retrain on this assembled, point-in-time correct dataset.

---

**Your encoder question — this is correct thinking.**

Yes. Any preprocessing that happens before the model — categorical encoding, normalization, tokenization of text fields — must be identical between training and serving. If you normalize item price during training, you must normalize it the same way at serving time.

The standard pattern: save your preprocessing pipeline as a versioned artifact alongside the model in the model registry. When you retrain with new data, you either reuse the existing preprocessing pipeline or version a new one alongside the new model. Never let the preprocessing pipeline and the model version drift from each other.

For your two-tower model specifically: item description embeddings are pre-encoded offline using a frozen sentence transformer. That encoder object gets versioned. The normalization applied to price tiers gets versioned. The category encoding mappings get versioned. All travel together with the model version.

---

**What gets logged after serving — complete picture.**

Three separate things get logged, independently:

**1. Serving logs (for training data assembly):** request_id, user_id, timestamp, exact feature values used, item_ids shown, positions, device, surface. Written to Snowflake immediately at serving time.

**2. Interaction events (labels):** user_id, item_id, event_type, timestamp, session_id. Written to Snowflake via Kinesis continuously as users interact.

**3. Prediction scores (for monitoring):** request_id, item_id, retrieval similarity score, ranking model score. Written to Snowflake or S3 for offline evaluation and drift detection.

At training time, Spark joins (1) with (2) on user_id and timestamp to produce labeled training examples with point-in-time correct features. (3) is used separately for monitoring — comparing model score distributions over time to detect drift.

---

**Now the offline store question answers itself.**

You asked if Snowflake is the offline store. Yes. At Series B/C there's no meaningful distinction between your data warehouse and your offline feature store — they're the same system. Snowflake stores raw events, serves as the source for Spark batch feature computation, and stores the assembled training datasets. Dedicated offline feature stores (like Feast's offline layer backed by a data lake) add value when you have many teams computing many feature sets and need a catalog to track them. At your scale, Snowflake directly is perfectly sufficient.

**The offline store is Snowflake. It stores:**
- Raw interaction events (ground truth)
- Logged serving-time feature values (for training data assembly)
- Assembled training datasets (feature-label pairs, point-in-time correct)
- Model evaluation results

---

## **Complete data flow, stated cleanly:**

User interacts → Kinesis captures event → lands in Snowflake raw events table.

Simultaneously at serving time → retrieval service logs exact features used → lands in Snowflake serving logs table.

Spark batch job daily → reads Snowflake raw events → computes user aggregates → writes to Redis online store (for next day's serving).

Spark training job (weekly or triggered) → joins Snowflake serving logs with Snowflake interaction events → assembles point-in-time correct training dataset → trains new model → registers to model registry.

New model deployed → shadow mode → canary → full traffic.

---


## Training window

Rolling 90-day window for training data assembly. Matches your feature schema lookback. Recent enough to capture current behavior. Long enough to have sufficient positive examples even for low-frequency purchasers. Computationally bounded — your Spark training job always reads the same volume of data regardless of how long the system has been running.

One nuance worth naming: within that 90-day window, apply recency decay as sample weights. You already designed this into your data schema. A purchase from 5 days ago gets weight 1.0. A purchase from 85 days ago gets weight 0.2. The window is 90 days but the model is effectively learning more from recent behavior. This gives you the coverage of a long window with the recency emphasis of a short one.