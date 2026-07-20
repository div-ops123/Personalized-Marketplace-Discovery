# Low-Level Design — PDP Similar Items

## Data Pipelines

### Training/Serving Preprocessing Parity
Any preprocessing applied before the model — categorical encoding, normalization, tokenization of text fields — must be identical between training and serving. If item price is normalized during training, it must be normalized the same way at serving time. Standard pattern: the preprocessing pipeline is saved as a versioned artifact alongside the model in the model registry. When retraining, either reuse the existing preprocessing pipeline or version a new one alongside the new model — the preprocessing pipeline and the model version must never drift apart.

### Batch & Training Jobs
- Spark batch job (daily) reads Snowflake raw events, computes per-user and per-candidate aggregates, writes to the Redis online store for the next day's serving, and write to offline store(User Daily Features, and Candidate Daily Features).
- Spark training job (weekly or triggered) joins Snowflake serving logs with Snowflake interaction events, assembles a point-in-time-correct training dataset, trains a new model, and registers it to the model registry.
- New model deployment proceeds shadow mode → canary → full traffic.

Training uses a 90-day rolling window with recency decay applied as sample weights: a purchase from 5 days ago gets weight 1.0, a purchase from 85 days ago gets weight 0.2. This gives the coverage of a long window with the recency emphasis of a short one.

Raw events land in Snowflake continuously via Kinesis.

---

## Cold Start Handling

**New items:** surfaced by retrieval regardless of interaction history, because the item encoder is content-based (image/text embeddings, category, brand) — a new item gets an embedding and enters the ANN index as soon as it's indexed, with zero click history required. Ranking also has signal on new items through their catalog/content features even before behavioral aggregates (historical CTR/CVR) accumulate; those default to a global prior via the smoothing/backoff already defined in data-schema.md.

**New / anonymous users:** retrieval is entirely item-item and unaffected by user identity. Ranking degrades gracefully — user-level features (preferred brands, avg purchase price, category affinity) are simply unavailable, so ranking falls back to anchor/candidate/cross features only. The widget still functions; it's just not personalized for that request.

---

## Class Imbalance

**Retrieval:** uses contrastive learning, where multiple negatives per positive are expected. In-batch negatives provide a large, diverse negative set, so class imbalance isn't treated as a modeling problem here.

**Ranking:** each impression naturally contains far more non-purchased than purchased candidates (roughly 1 purchase per 200–500 impressions). LambdaMART is designed for learning-to-rank under this kind of imbalance, so the initial implementation trains on the full retrieved candidate set as-is. If dataset size becomes computationally prohibitive, negative downsampling can be introduced as a training optimization without changing the serving architecture.

---

## Business Constraints
- Item catalog size: 2M
- New item volume: 5K new items/day
- QPS: 5,000 peak (tied to page loads, not every click — a recommendation request fires once per PDP view)
- Latency: retrieval P99 < 30ms, so end-to-end (retrieval + ranking + re-ranking + API overhead) stays under the 100ms p99 budget
- Retrieval quality target: 95%+ Recall@200 (industry standard at this scale)
- Memory: 2M items × 128 dims × 4 bytes = ~1GB; even at 256 dims, ~2GB — comfortably fits in memory on a single machine with 16GB RAM, with headroom

---

## Tooling & Justification

### ANN Index

The memory calculation above is the decision-maker: even at 2M items and 256 dimensions, the full embedding set is ~2GB, which fits on a single machine. The tradeoff against a distributed vector database is:
- **Gain from distributed:** horizontal scaling, replication, high availability.
- **Give up:** operational complexity, network latency, infrastructure cost.

At this scale, a distributed vector database is overkill — a single in-memory FAISS index is sufficient, deployed redundantly via multiple ECS tasks rather than a distributed store.

Within FAISS, the choice is **HNSW**, rejecting **IVF**:

- **IVF** partitions the vector space into clusters (Voronoi cells) and at query time only searches a subset of clusters (`nprobe`). If the nearest neighbors live in a cluster that wasn't searched, they're missed — permanently. `nprobe` can be increased to recover recall, but that costs more compute per query and erodes the speed advantage. Given the 95%+ recall requirement, this is an unacceptable failure mode.
- **HNSW** builds a navigable graph connecting similar vectors, giving strong recall by construction. At 2M items, insert cost is O(log n) per vector — with 5,000 new items/day (0.06 inserts/second) against a 2M-node graph, this is negligible: each insert takes roughly 5–20ms and is infrequent enough relative to read (query) volume to create no measurable pressure on serving latency.
- **The actual tradeoff:** HNSW isn't tuning-free — it trades IVF's `nprobe` knob for its own `ef_search` knob, which governs the same recall-vs-latency tradeoff at query time. The difference that matters is the failure mode: IVF risks permanently missing neighbors that live in an unsearched cluster, while HNSW's `ef_search` only trades speed for recall and never silently drops a reachable neighbor. Given the 95%+ recall requirement, HNSW's tradeoff is the safer one — the real implementation task is finding the `ef_search` sweet spot against the p99 latency budget, not eliminating tuning altogether.

**Distance metric: cosine similarity.** Euclidean distance measures absolute distance in vector space, which penalizes a long vector and a short vector pointing in the same direction for their length difference even when their direction (meaning) is identical. Cosine similarity measures the angle between vectors, ignoring magnitude — and in embedding space, direction encodes semantic meaning while magnitude is largely a training artifact. FAISS doesn't natively support cosine similarity as a metric the way it supports inner product and L2, so the standard trick is to L2-normalize all embeddings at index-build time and at query time, then use inner product search — which becomes equivalent to cosine similarity on unit-length vectors.

### Cloud
**Choice: AWS as primary cloud.** Dominant cloud among market. Deepest ML ecosystem at Series B/C scale. Existing team knowledge reduces ramp-up cost.

**Cost principle:** use AWS-managed services where they reduce operational burden without introducing meaningful cost or lock-in risk; use best-of-breed open source where the managed AWS equivalent is significantly more expensive or worse. Decision rule: if the managed version saves meaningful engineering time without dramatically higher cost, use it; if it's 3–5x more expensive for something reliably run on a standard EC2 instance, don't.

**Rejected SageMaker (training/serving layer specifically):** proprietary APIs create migration pain. At Series B/C, training on EC2 with PyTorch directly, MLflow for tracking, and FastAPI on ECS for serving gives the same capability with zero lock-in. SageMaker earns its keep only if its specific features (built-in distributed training, managed auto-scaling endpoints) are actually needed — not the case yet at this stage.

**Rejected EKS:** Kubernetes control-plane overhead isn't justified for three services. Revisit when service count exceeds 10–15 or cross-service traffic patterns require mesh routing.

**Rejected Lambda:** no persistent memory between invocations — can't hold the FAISS index or a PyTorch model in memory. Cold-start latency is incompatible with the sub-50ms retrieval budget.

### Storage
**Data warehouse: Snowflake**, chosen for being a multi-cloud, low-maintenance platform with instant elastic scaling — storage and compute are decoupled, compute spins up on demand for the daily feature pipeline and auto-suspends afterward, so cost tracks actual pipeline runtime rather than a standing cluster. Cleaner multi-team access — data scientists, ML engineers, and analytics can run concurrent queries on separate virtual warehouses without contention. Integrates with Airflow for orchestration and dbt for transformation. Dominant choice across Group A targets at Series B/C.

**Rejected Redshift:** single-cloud (AWS-only), and scaling requires more manual cluster/capacity management compared to Snowflake's instant elastic compute.

**Object storage: S3.** No meaningful alternative on AWS. Stores model artifacts, raw log files, training datasets, ANN index snapshots.

### Streaming & Batch Processing
**Event ingestion:** Kinesis — reliably captures every interaction event, buffers it, delivers it to the processing layer without loss.

**Batch processing (daily features):** Spark on EMR. The daily feature computation job reads from Snowflake, aggregates interaction logs, and writes user/candidate feature aggregates back to the feature store. Runs on schedule via Airflow.

### Feature Store
**Offline store: Snowflake.** No separate offline feature store needed at Series B/C — Snowflake serves as both data warehouse and offline feature layer, storing raw interaction logs, serving-time logged feature values, and assembled point-in-time-correct training datasets. Spark reads from Snowflake for both batch feature computation and training-data assembly. A dedicated offline feature store (Feast, Tecton) would add operational complexity without meaningful benefit at this scale.

**Online store: Redis on AWS ElastiCache.** Stores precomputed User Daily Features and Candidate Daily Features, written daily by the Spark batch job on EMR, read at inference time by the ranking service. Sub-millisecond P99 read latency keeps feature fetch well within the ranking service's latency budget. Data structure: one Redis Hash per user (`user_features:{user_id}`), all features fetched in a single `HGETALL`. Session data (device, position) shares the same ElastiCache cluster under a separate key namespace — at Series B/C session volume (well under 500MB) it doesn't justify a second cluster.

**Rejected DynamoDB:** the sub-100ms end-to-end latency budget leaves the ranking service only a few milliseconds for feature fetch after retrieval, ANN search, and model inference are accounted for. ElastiCache's sub-millisecond P99 fits inside that remaining margin; DynamoDB's disk-backed read path does not leave the same margin. Given how tight the latency budget already is, that alone is decisive.

**Retrieval item features:** item embeddings are precomputed by the item encoder at indexing time and held in the FAISS index / cache for lookup — not recomputed at request time (see ANN Index above and Components below).

### Experiment Tracking & Model Registry
**Experiment tracking: MLflow on EC2**, backed by S3 for artifacts and PostgreSQL RDS for run metadata. Each run logs dataset S3 path and timestamp, feature schema version, git commit hash, hyperparameters, training loss per epoch, Recall@200 and NDCG@K on the validation set, training duration/cost, and preprocessing pipeline artifacts. MLflow autologging covers optimizer parameters and loss curves; business-critical context is logged explicitly.

**Rejected Weights & Biases:** stronger experiment-tracking UX, but SaaS-only — training data and model metadata would leave the AWS environment. For a marketplace handling user behavioral data, keeping ML artifacts inside owned infrastructure is the safer default. Revisit if team size and collaboration overhead justify external tooling.

**Model registry: MLflow Model Registry.** Stages: Staging → Production → Archived. Each version includes model weights, preprocessing artifacts (encoders, normalizers), and a model card (dataset description, offline metrics, feature schema, known limitations). Promoting a new production model archives the old one, available for instant rollback without retraining.

**Promotion gate:** the training script runs offline evaluation after training. A model registers to Staging only if Recall@200 ≥ threshold AND NDCG@K ≥ the current production model's score. Manual review is required before promotion from Staging to Production.

### Orchestrator
**Choice: Airflow.** These are scheduled, non-interactive batch jobs — the dominant constraint is reliably orchestrating production workflows, not pipeline aesthetics. Airflow's retry and alerting behavior is the actual production value: every task in a DAG can be configured with retry count, retry delay, and timeout, and that retry-before-alert, alert-before-giving-up pattern is what makes the system maintainable by a small team.

**Rejected AWS Step Functions:** state machines authored in Amazon States Language (JSON) are harder to version, diff, and code-review than Python DAGs, and Step Functions has weaker native tooling for orchestrating long-running Spark/EMR jobs and EC2 GPU training runs than Airflow's existing operators. Step Functions' AWS-native integration is a real advantage, but Airflow's authoring and retry/backfill ergonomics matter more for this pipeline shape.

#### DAG Structure
**DAG 1 — Daily Feature Pipeline.** Trigger: 2am daily schedule. Read interaction events from Snowflake → Spark feature computation on EMR → validate features (null rates, coverage, value ranges, row count) → branch: pass → write to Snowflake offline store → write to Redis online store → notify success; fail → alert on-call, halt pipeline, do not write to Redis.

**DAG 2 — Training Pipeline.** Trigger: drift-detection alert or manual trigger. Read serving logs and interaction events from Snowflake → Spark log-and-join assembly (90-day rolling window, recency decay) → validate training dataset → train on EC2 GPU spot instance → offline evaluation (Recall@200, NDCG@K) → branch: metrics pass threshold → save model + preprocessing artifacts to S3 → register to MLflow → trigger DAG 3 via Airflow REST API → notify team; fail → log failure, alert team, halt.

**DAG 3 — Deployment Pipeline.** Trigger: explicit call from DAG 2. Load registered model from MLflow → deploy to ECS shadow environment → run shadow evaluation (compare candidate sets and scores against production on live traffic) → notify team with shadow metrics via Slack → wait for manual approval → on approval: canary rollout at 5% traffic → monitor canary metrics for a defined window → gradual traffic increase → full promotion → archive previous production model.

### CI/CD for ML
The DAGs above are the ML-specific validation layer (Layer 2). GitHub Actions adds a code-change trigger layer (Layer 1) on top of them; ECS + CloudWatch shadow/canary routing is the production-validation layer (Layer 3).

**Layer 1 — GitHub Actions (minutes).** Training code PRs: unit tests on feature engineering functions, schema validation tests, a model smoke test (train a tiny model on a small sample, check output shape/dtype), lint/type-check. Serving code PRs: unit tests on serving logic (e.g. re-ranking/MMR diversity, fallback-chain behavior when Redis is unavailable), an integration test against a local Docker container with a tiny FAISS index and mocked Redis, and a container build check. On merge to main: training code triggers DAG 2 via the Airflow REST API; serving code builds and pushes the image to ECR and triggers DAG 3.

**Layer 2 — Airflow DAGs (hours).** DAG 2 validates data quality and model performance; DAG 3 validates production behavior via shadow deployment with a manual approval gate before canary.

**Layer 3 — ECS + CloudWatch (hours to days).** Canary routing validates real-traffic behavior; metrics are compared per traffic split; gradual promotion or instant rollback follows from the monitoring signals.

The key difference from traditional software CI/CD: it asks "did the tests pass?" ML CI/CD asks "is this model actually better than what's in production?" — the offline-metric threshold gate in DAG 2 and the shadow-evaluation comparison in DAG 3 are the ML-specific steps with no equivalent in standard software deployment.

### A/B Testing Framework
**User assignment:** consistent hashing of `user_id` to experiment bucket, enforced at the API gateway before the request reaches the retrieval service. Redis-backed bucket-assignment table (`user_id → experiment_group`) for O(1) lookup, guaranteeing the same user always sees the same model version.

**Traffic routing:** the API gateway reads the bucket assignment and routes to the corresponding ECS task definition (control or treatment); both versions run simultaneously.

**Metric collection:** every interaction is tagged with `experiment_group` in the serving logs. A Spark batch job reads Snowflake, computes per-group metrics (recommendation-attributed CTR, purchase rate, GMV), and runs a two-sample significance test. Results are reviewed over a 2–4 week window — a novelty-effect CTR lift in week one can be misleading if it doesn't persist to purchase rate.

**Feedback-loop protection:** every interaction is tagged with the model version that served it. At training time, serving logs are filtered by surface and model version so treatment-group interactions don't corrupt the control model's retraining data, and vice versa.

**Rejected Optimizely / LaunchDarkly:** adds SaaS cost and an external dependency for a bucketing problem that a Redis hash lookup and a consistent-hash function already solve at Series B/C scale.

### Monitoring Infrastructure (Tooling)
- **System health (second-to-second):** CloudWatch — ECS container metrics (CPU, memory), request latency P99 per service, error rates, Redis connection saturation, FAISS query latency. CloudWatch alarms page on-call via PagerDuty.
- **ML quality (hour-to-day):** Grafana dashboards backed by CloudWatch custom metrics. A Spark daily job computes candidate diversity per request, new-item coverage (last 7 days), CTR by position, recommendation-attributed purchase rate and GMV, and catalog coverage, pushed to CloudWatch as custom metrics. Grafana overlays control limits for anomaly detection.
- **Pipeline health (daily):** Airflow task-level monitoring — feature freshness, training dataset row count, feature null rates, Kinesis consumer lag, serving log volume. Failed validation tasks alert via Slack.
- **Rejected Datadog as primary tool:** per-host and per-custom-metric billing grows fast at Series B/C with dozens of ML-specific metrics. CloudWatch + Grafana gets equivalent observability at significantly lower cost. Datadog is a plausible upgrade once team size exceeds 5–6 engineers and the overhead of running three tools exceeds Datadog's pricing.

---

## Components & How They Interact

Three independently deployable, independently auto-scaled ECS services, communicating over the internal AWS VPC (no public traffic between them):

**API Gateway / Orchestrator.** Receives the raw request from the frontend, calls the retrieval service, passes candidates to the ranking service, returns the final list to the frontend. Thin — no ML model, minimal memory footprint. Also owns fallback routing when retrieval or ranking is unavailable, and is where A/B bucket assignment is read and traffic routed to the correct model version.

**Retrieval Service.** FastAPI application. Loads the FAISS index from S3 on container startup; an ECS health check gates traffic until the index is fully loaded (index load can take 30–60s for a large index, and the container must report unhealthy until it's ready, then healthy once ECS starts routing to it — this directly shapes deployment and auto-scaling behavior). At request time: looks up the anchor item's precomputed embedding, queries FAISS, returns 200 candidates with similarity scores. Memory footprint for the FAISS index is under ~2GB at 2M items / 256 dims.

**Ranking Service.** FastAPI application. Loads the LightGBM/XGBoost model on startup. Receives the 200 candidate IDs from the retrieval service, batch-fetches candidate and user features from Redis, reads country/device from the request, computes cross features inline, scores and ranks all 200, and returns the ordered list. Truncation to the top 20 displayed slots happens before the response is logged as impressions.

**Request flow:** User → PDP → API Gateway → Retrieval Service (embedding lookup + ANN search → 200 candidates) → Ranking Service (feature lookups + cross features → score all 200) → truncate to top 20 → respond to user → log Impression Event for the 20 shown. All of the above completes under 100ms p99, with retrieval budgeted at under 30ms p99 of that total.

---

## Monitoring Plan

How data moves from inference, to monitoring, to retraining — where each metric comes from, when it's computed, and what action it triggers.

During inference: after responding to the user, log the Impression Event; log a Click Event for every click in the recommendation widget; purchases are logged through the normal purchase-event path.

1. **System Monitoring** — "Can my service keep serving recommendations?" Monitor P99 latency, QPS, error rate, CPU, memory, GPU utilization. Latency exceeding SLA triggers an alert.

2. **Feature Monitoring** — "Is production data still similar to training data?" Monitor missing values and feature drift.
   - Categorical features: Jensen-Shannon Divergence. *(Rationale not yet fully specified in the original notes — left open rather than invented; fill in before implementation.)*
   - Numerical features: KS test, since it requires no manual binning.
   - Embedding monitoring (image/text/item-encoder outputs): open question, not yet defined.

3. **Prediction Monitoring** — "Has the model started behaving strangely?" Monitor the prediction score distribution over time (not compared against ground truth — ground truth isn't available yet at prediction time).

4. **Online Business Metrics** — computed once outcomes are available: recommendation-attributed CTR, recommendation-attributed CVR, click-bait rate (see metric-ladder.md guardrail).

5. **Offline Evaluation** — computed after ground truth accumulates: Retrieval Recall@200, Ranking NDCG@20.
