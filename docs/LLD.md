
---

Cold start:

Zero interactions → country-level bestsellers is the right call because you genuinely have nothing to work with yet. 
One session event → session + content. 
Enough history → full CF. And A/B testing to find the CF threshold empirically rather than guessing. 

=============

# Business constraints

- Item catalog size = 500k

- new item volume = 5k new items per day.

- QPS = 5,000 peak QPS. This number is lower than you might think because recommendation requests are tied to page loads, not every click.

- latency = Industry standard for retrieval stage is P99 under 30ms so that your total end-to-end recommendation response (retrieval + ranking + re-ranking + API overhead) stays under 150ms. 

- retreval quality = industry standard at this scale is 95%+ recall@500 — meaning your ANN search should return 95% of the items that an exact brute-force search would return in the top 500. The gap is acceptable because the ranking stage provides a safety net.

- Memory calculation — this is the decision-maker:
500K items × 128 embedding dimensions × 4 bytes per float = ~256MB. Even at 256 dimensions: 500K × 256 × 4 = ~512MB. This fits comfortably in memory on a single machine with 16GB RAM, with plenty of headroom. 
At 2M items it's still under 2GB. This single calculation tells you that a distributed vector database is overkill for Series B/C scale.

my exercise:
Now I want to do the tool selection exercise myself for the ANN index:
my teacher said: State your choice and the specific alternative you rejected. For each, name the exact tradeoff — not "simpler" or "too complex" but the actual mechanism of what you gain and what you give up. Use the real numbers above to justify why the tradeoff lands where it does at Series B/C scale.

ans:
well i know vector database is ruled out becos even at 2million items it's still under 2GB. so a 16GB vm is more than enough to hole the item embeddings in memory.
So the tradeoff is:

Gain: horizontal scaling, replication, high availability.

Give up: operational complexity, network latency, infrastructure cost.

so options i am left with:
IVF: i wouldn't use this becos if query lands on the wrong cluster the right items never even make it throught the retreval stage. and i need high recall.
but write is easier here. tradeoff write for recall.
IVF partitions the vector space into clusters (Voronoi cells) and at query time only searches a subset of clusters (controlled by a parameter called nprobe). If the nearest neighbors live in a cluster that wasn't searched, you miss them — permanently. You can increase nprobe to search more clusters and recover recall, but then you're paying more compute per query and the speed advantage shrinks. At 200K-500K items with a recall requirement of 95%+, HNSW gives you that recall without the nprobe tuning game. 

Hierarchical Navigable Small World (HNSW):
this is connects similar vectors to each other. so i can think recall would be high. very fast and i have the memory so good. update is expensive but for our scale.
5,000 new items/day ÷ 86,400 seconds = 0.06 inserts/second. HNSW insert cost is O(log n) per vector because it has to find and wire the new node into the graph at multiple layers. At 0.06 inserts/second against a graph of 500K nodes, that cost is negligible — each insert takes maybe 5-20ms and they're so infrequent relative to reads that they create zero measurable pressure on serving latency.
tradeoff is nothing for this scale. once new vector increases then i can see a reasonable tradeoff.

FAISS (Facebook AI Similarity Search):
this is the toolbox. IVF is one of the tool inside, HNSW is another.

what distance metric to use?
cosine similarity becos i want nearest neighbor not eucline distance becos that uses length too.
Euclidean distance measures absolute distance in vector space — which means a long vector and a short vector pointing in the same direction get penalized for the length difference, even if their direction (meaning) is identical. Cosine similarity measures the angle between vectors, ignoring magnitude. In embedding space, direction encodes semantic meaning (what kind of user, what kind of item) and magnitude is largely an artifact of training. So you want cosine similarity because you care about directional alignment — "this user embedding and this item embedding point in the same conceptual direction" — not absolute proximity in raw space. One practical note: FAISS doesn't natively support cosine similarity as a distance metric in the same way it supports inner product (dot product) and L2 (Euclidean). The standard production trick is to L2-normalize all embeddings at index build time and at query time, then use inner product search — which becomes equivalent to cosine similarity when vectors are unit-length.

# Clean version of your ANN index decision for documentation:
Choice: FAISS with HNSW index, inner product search on L2-normalized embeddings (equivalent to cosine similarity).
Rejected: FAISS with IVF index. IVF partitions the vector space into clusters and only searches a subset at query time, controlled by nprobe. At 95%+ recall requirement, you'd need high nprobe values that erode IVF's speed advantage — and you'd be tuning a parameter to compensate for a structural weakness that HNSW doesn't have.
Rejected: Managed vector databases (Pinecone, Qdrant, Weaviate). At 500K items with 128-256 dimension embeddings, the full index fits in under 512MB — well within a single 16GB VM. Distributing this across managed infrastructure adds network latency per query, monthly managed service cost, and operational complexity to solve a scale problem that doesn't exist at Series B/C.
Why HNSW fits: Connects each vector to its nearest neighbors across multiple graph layers, enabling high-recall graph traversal at query time without exhaustive search. Write cost is O(log n) per insert — at 5,000 new items/day (0.06 inserts/second) against 500K vectors, insert pressure on serving latency is negligible. Recall@500 consistently above 95% at <10ms query latency on commodity hardware. Tradeoff becomes unfavorable only when write throughput scales to thousands of inserts per second — not a Series B/C problem.
Distance metric: Cosine similarity via L2-normalized embeddings + inner product search. Direction encodes semantic meaning in embedding space; magnitude is a training artifact. L2-normalize at index build time and query time, use FAISS IndexFlatIP.

---

# Tooling

## 1. Cloud
Choice: AWS as primary cloud.
Reasoning: Dominant cloud among Group A targets (Constructor, Super.com, Archive, SeatGeek, CookUnity, Uber Freight). Deepest ML ecosystem at Series B/C scale. Existing team knowledge reduces ramp-up cost.
Cost principle: Use AWS managed services where they reduce operational burden without introducing meaningful cost or lock-in risk. Use best-of-breed open source tools where the managed AWS equivalent is significantly more expensive or significantly worse.
The decision rule: if the AWS managed version saves you meaningful engineering time and doesn't cost dramatically more — use it. If the managed version is 3-5x more expensive for something you can run reliably on a standard EC2 instance — don't.
IMPORTANT:
avoid vendor lock-in on the model training and serving layer specifically. SageMaker has proprietary APIs that make migrating away painful. At Series B/C, training on EC2 with PyTorch directly, using MLflow for tracking, and serving via FastAPI on ECS gives you the same capability with zero lock-in. SageMaker makes sense if you need its specific features — built-in distributed training, managed endpoints with auto-scaling — which at Series B/C you probably don't yet.


## 2. Storage
The one genuine advantage Redshift has for your use case:
Native integration with the AWS ecosystem. If you're reading from S3, running training jobs on EC2, using Kinesis for streaming — Redshift has zero-friction integration with all of it. Spectrum lets you query S3 directly without loading data. Snowflake integrates well with AWS too, but it's an external service rather than a native one. At Series B/C this integration friction is small but real.

Object storage: 
S3. No meaningful alternative on AWS. Stores model artifacts, raw log files, training datasets, ANN index snapshots.

Data warehouse: Snowflake.

Rejected: Redshift. Coupled storage and compute means paying for a full cluster 24 hours a day to handle a workload that only needs active compute during daily batch feature computation (estimated 1-2 hours/day). Redshift's cost advantage only materializes at sustained continuous query workloads — not this batch pattern.

Why Snowflake: Decoupled architecture — storage on S3, compute spun up on demand. Daily feature pipeline runs, warehouse auto-suspends, compute billing stops. Pay for compute only when pipelines run. Dominant choice across Group A targets at Series B/C. Cleaner multi-team access — data scientists, ML engineers, and analytics can run concurrent queries on separate virtual warehouses without contention. Integrates with Airflow for pipeline orchestration and dbt for data transformation.

Caveat: If AWS enterprise agreement includes committed Redshift spend, reverse this decision — Redshift becomes effectively free within existing commitment.

## 3. Streaming and batch processing. 

Step by step tool selection:
Step 1: Define what you actually need from each layer.
Event ingestion layer: reliably capture every interaction event from the website, buffer it temporarily, deliver it to the processing layer without loss. This is Kafka or Kinesis.
Stream processing layer: read from the ingestion layer, aggregate session features per user over a N-minute window, write results to the session store. This is Flink, Spark Streaming, or Kafka Streams.
Batch processing layer: read from Snowflake, compute daily aggregate user features, write back to feature store. This is Spark.

Step 2: Kinesis versus Kafka for event ingestion.

Event ingestion: Amazon Kinesis. Managed service, zero cluster operations, native AWS integration, sufficient throughput ceiling for Series B/C event volume (estimated peak 5,000-10,000 events/minute). Rejected Kafka on EC2 and MSK — operational complexity not justified at this event volume and team size.

Stream processing (session features): 
Spark Structured Streaming on EMR. 
Micro-batch with 1-minute windows computes session aggregations (items clicked, categories browsed, searches in last N minutes) and writes to Redis session store. 
Rejected Flink — superior for true sub-second streaming but requires separate cluster alongside Spark batch infrastructure. 1-minute micro-batch accuracy is sufficient for session feature quality at this product scale.

Batch processing (daily features): 
Spark on EMR. Same cluster as stream processing. 
Daily feature computation job reads from Snowflake, aggregates interaction logs, writes user feature aggregates back to the feature store. 
Spark handles this at any scale you'll reach at Series B/C. Already justified by the Spark Streaming decision above — same cluster, same framework.
Runs on schedule via Airflow.

Request-time context: Current product being viewed, device, page context arrive directly in the request payload from the frontend. Not stored in session store or feature store — read off the request at serving time.

## 4. Feature Store(offline + online + real-time)

Offline store: Snowflake.
No separate offline feature store needed at Series B/C. Snowflake serves as both data warehouse and offline feature layer. Stores raw interaction logs, serving-time logged feature values, and assembled point-in-time correct training datasets. Spark reads from Snowflake for both batch feature computation and training data assembly. Adding a dedicated offline feature store (Feast, Tecton) introduces operational complexity without meaningful benefit at this scale.
Online store: Redis on AWS ElastiCache.
Stores pre-computed aggregate user features — purchased categories (last 90 days), average price point, preferred brands, time since last purchase. Written daily by Spark batch job on EMR. Read at inference time by retrieval service and ranking service. Sub-millisecond P99 read latency keeps feature fetch well within the 10ms budget. Data structure: Redis Hash per user (user_features:{user_id}), all features fetched in a single HGETALL command.
Rejected DynamoDB: P99 latency spikes to 5-10ms under load due to disk-backed architecture — consumes the entire feature fetch budget before ANN search or model forward pass runs. Per-request billing compounds at thousands of reads per user per day, making ElastiCache fixed node-hour pricing significantly cheaper at high read QPS.
Real-time session store: Redis on AWS ElastiCache (same cluster, separate key namespace).
Stores per-user session aggregations — items clicked in last N minutes, categories browsed, searches typed, number of items viewed without clicking. Written every 1 minute by Spark Structured Streaming micro-batch job. Read at inference time in parallel with online store lookup. TTL set to 30 minutes per key, reset on every write — active users never lose session state, inactive users' data expires automatically without manual cleanup. Data structure: Redis Hash per session (session_features:{user_id}).
Same ElastiCache cluster as online store, separated by key namespace prefix. At Series B/C session data volume (estimated under 500MB for active sessions) doesn't justify a separate cluster. Revisit when active session count grows beyond single-node memory capacity.

One implementation detail: every time Spark writes updated session features for a user, it also resets the TTL back to 30 minutes. In Redis this happens automatically with the SET command when you include the EX option — each write refreshes the expiry clock. So active users never lose their session data, and inactive users' data cleans itself up without any manual garbage collection.



---

QUESTION:
ranking only uses purchases. that changes daily per user? makes daily do nothing.
should i use othe predictive features? or i should leave as is for now it already show engineering judgement the whole project?

---

## 5. Retrieval model and ranking model 

Retrieval model: Two-tower neural network in PyTorch.
User tower and item tower are independent nn.Module classes. Independence enables offline precomputation of item embeddings. At serving time only the user tower runs — forward pass produces user embedding, ANN search against precomputed FAISS index returns top 500 candidates.
Rejected matrix factorization: uses only user-item interaction signal, no content features, structural cold-start problem for new items.
Rejected NeuMF: replaces dot product with neural interaction function requiring both user and item as joint input, breaking ANN index compatibility — cannot precompute items separately.

Ranking model: Wide & Deep in PyTorch.
Wide component: linear layer over explicit hand-engineered cross features (user-seller purchase history, price vs average spend, brand affinity match). Handles memorization of specific co-occurrence patterns.
Deep component: feedforward neural network consuming user embedding, item embedding, and dense features. Handles generalization across unseen feature combinations and preserves geometric structure of embedding inputs.
Rejected gradient boosting (XGBoost/LightGBM): cannot natively consume raw embedding vectors from retrieval stage without dimensionality reduction that destroys geometric structure. Strong choice if ranking features were purely tabular — not this system.
Rejected transformer ranker: requires significantly more training data to outperform Wide & Deep, higher serving complexity, more compute. Justified at larger scale and team size than Series B/C.
---

Retrieval model inference optimization:

Trained model → TorchScript compilation (removes Python interpreter overhead, enables C++ execution) → INT8 quantization for CPU serving or float16 for GPU serving → ONNX export for ONNX Runtime serving (automatic layer fusion and graph optimization).

Dynamic batching at serving layer: collect requests over 1-5ms window, batch 16-32 user tower forward passes per GPU kernel launch. Reduces per-request GPU overhead at 3-5K peak QPS, improves GPU utilization from ~10% to 60-80%.

User embedding caching: not viable at full tower level due to session features changing per request. Advanced optimization (late feature fusion — pre-compute static aggregate embedding, inject dynamic session features through lightweight layer at serving time) deferred to post-launch when latency profiling confirms user tower as bottleneck.

Item embeddings: precomputed offline, stored in FAISS. Never recomputed at serving time unless item features change (triggered by catalog event pipeline).
---

I have three distinct services running on ECS.

Retrieval service: receives recommendation request, fetches user features from Redis, runs user tower forward pass, queries FAISS, returns 500 candidates to ranking service. Runs on ECS, needs memory for FAISS index (under 512MB at Series B/C item catalog size) and PyTorch user tower model.

Ranking service: receives 500 candidates from retrieval service, fetches item features from Redis batch lookup and user features, computes cross features, runs Wide & Deep forward pass, applies re-ranking rules, returns final ordered list to API layer. Runs on ECS, needs memory for Wide & Deep model.

API gateway / orchestrator: receives raw user request from frontend, calls retrieval service, passes candidates to ranking service, returns final list to frontend. Thin service, no ML model, minimal memory. Also handles fallback routing when retrieval or ranking service is unavailable.

Three ECS services. Each independently auto-scalable. Each independently deployable. Each with its own health checks and circuit breakers.

One more thing: FAISS index loading time. When ECS starts a new retrieval service container, it needs to load the FAISS index from S3 into memory before it can serve requests. That load can take 30-60 seconds for a large index. During that window the container shouldn't receive traffic. ECS health checks handle this — the container reports unhealthy until the index is loaded, then healthy, then ECS starts routing traffic. This is an implementation detail worth naming because it directly affects your deployment and auto-scaling behavior.

Clean version for documentation:

- Model training: EC2 with GPU (p3 instance family) on AWS.

- Standard PyTorch training code with no SageMaker-specific APIs — fully portable across environments. Spot instances for 60-70% cost reduction versus on-demand; checkpoint to S3 every N steps to handle spot interruption and resume. Training triggered by Airflow on schedule (weekly) or monitoring alert (drift detected). Model artifact saved to S3 on completion, registered to MLflow model registry with training metadata (dataset version, hyperparameters, offline evaluation metrics).

- Rejected SageMaker training: proprietary execution environment and APIs create vendor lock-in. Per-instance-hour markup over EC2 baseline not justified by managed features at this scale.

- Model serving: AWS ECS (Elastic Container Service).

- Three containerized services:

- Retrieval service: FastAPI application, loads PyTorch user tower (TorchScript/ONNX) and FAISS index from S3 on startup. ECS health check gates traffic until index fully loaded. Fetches user aggregate features and session features from Redis in parallel, runs user tower forward pass with dynamic batching, queries FAISS, returns 500 candidates with similarity scores.

- Ranking service: FastAPI application, loads Wide & Deep model (TorchScript/ONNX) on startup. Receives 500 candidate IDs from retrieval service, batch-fetches item features from Redis, fetches user features from Redis, computes cross features, runs ranking forward pass, applies re-ranking layer (MMR diversity, novelty slots, exploration slots), returns final ordered list.

- API gateway: thin FastAPI service, orchestrates retrieval → ranking call chain, handles fallback routing if downstream services unavailable, returns final recommendation list to frontend.

- Each service independently auto-scaled via ECS Application Auto Scaling based on CPU utilization and request queue depth. Services communicate via internal AWS VPC — no public internet traffic between services.

- Rejected EKS: Kubernetes control plane operational overhead not justified for three services. Revisit when service count exceeds 10-15 or when cross-service traffic patterns require advanced mesh routing.

- Rejected Lambda: no persistent memory between invocations — cannot hold FAISS index or PyTorch model in memory. Cold start latency incompatible with sub-50ms retrieval budget.


Cold start:

Zero interactions → country-level bestsellers is the right call because you genuinely have nothing to work with yet. 
One session event → session + content. 
Enough history → full CF. And A/B testing to find the CF threshold empirically rather than guessing. 


## 6. Experiment tracking + model versioning and registry

Experiment tracking: MLflow on EC2.

Runs alongside training infrastructure. MLflow server backed by S3 for artifact storage and PostgreSQL RDS for metadata (run parameters, metrics, tags). 
Each training run logs: dataset S3 path and timestamp, feature schema version, git commit hash, hyperparameters, training loss per epoch, Recall@500 and NDCG@K on validation set, training duration and compute cost, preprocessing pipeline artifacts.
MLflow autologging handles optimizer parameters and loss curves. Business-critical context (dataset version, git hash, offline evaluation metrics) logged explicitly.

Rejected Weights & Biases: stronger experiment tracking UX but SaaS-only — training data and model metadata leave your AWS environment. At a marketplace handling user behavior data, keeping ML artifacts inside your own infrastructure is the safer default. Revisit if team size grows and experiment collaboration overhead justifies external tooling.

Model registry: MLflow Model Registry.

Integrated with experiment tracking — single API call promotes a successful run to the registry. 
Registry stages: Staging → Production → Archived. 
Each registered model version includes: model weights, preprocessing pipeline artifacts (encoders, normalizers), model card (training dataset description, offline metrics, feature schema, known limitations).
Old production model moves to Archived on promotion of new model — available for instant rollback without retraining. Run ID links every registry entry back to the exact experiment run that produced it.

Promotion gate: training script runs offline evaluation after training completes. Model registers to Staging only if Recall@500 >= defined threshold AND NDCG@K >= current production model score. Manual review required before promotion from Staging to Production.

## 7.  Orchestrator 

i choose airflow becos jobs are scheduled.
None are particularly interactive.
also becos my dominant constraint isn't making pipelines beautiful.
It's reliably orchestrating production workflows.
Airflow has solved this problem for years.

Airflow's retry and alerting behavior is the actual production value. 
Each task in a DAG can be configured with retry count, retry delay, and timeout. 
This failure handling pattern — retry before alert, alert before giving up — is the operational behavior that makes a production ML system maintainable by a small team.

### DAG structure

DAG 1 — Daily Feature Pipeline
Trigger: 2am daily schedule
Read interaction events from Snowflake → Spark feature computation on EMR → Validate features (null rates, coverage, value ranges, row count) → Branch: pass → Write to Snowflake offline store → Write to Redis online store → Notify success. Branch: fail → Alert on-call, halt pipeline, do not write to Redis.

DAG 2 — Streaming Pipeline
Remove from Airflow. Spark Structured Streaming managed as long-running EMR step. CloudWatch monitors records processed per minute and Kinesis consumer lag. CloudWatch alarms page on-call on throughput drop. Airflow only triggers initial EMR streaming job deployment on pipeline code changes.

DAG 3 — Training Pipeline
Trigger: drift detection alert OR manual trigger
Read serving logs from Snowflake → Read interaction events from Snowflake → Spark log-and-join assembly (90-day rolling window, recency decay, false negative filter) → Validate training dataset → Train model on EC2 GPU spot instance → Offline evaluation (Recall@500, NDCG@K) → Branch: metrics pass threshold → Save model + preprocessing artifacts to S3 → Register to MLflow registry → Trigger DAG 4 via Airflow REST API → Notify team. Branch: metrics fail → Log failure, alert team, halt.

DAG 4 — Deployment Pipeline
Trigger: explicit trigger from DAG 3 via REST API
Load registered model from MLflow → Deploy to ECS shadow environment → Run shadow evaluation (compare candidate sets and scores against production model on live traffic) → Notify team with shadow metrics via Slack → Wait for manual approval (ExternalTaskSensor waiting on manual Promotion DAG trigger) → On approval: canary rollout at 5% traffic → Monitor canary metrics for defined window → Gradual traffic increase → Full promotion → Archive previous production model.


## 8. Model serving 
serving also includes the question of where the model lives in memory, how it gets loaded, and how inference is optimized. 
The workflow: train in PyTorch → export to ONNX → serve with ONNX Runtime inside your FastAPI service.

## Caching:

split the user tower into a static path and a dynamic path. 
Pre-compute the embedding from aggregate features offline and cache it in Redis. 
At serving time, inject session features through a lightweight additional layer rather than running the full tower from scratch. This means only the dynamic portion runs at inference time, not the full tower. This is called a late feature fusion pattern. 
At Series B/C, this is an optimization you'd pursue after the basic system is running and you've measured where latency is actually burning. 
Don't build it upfront — but know it exists.

## CI/CD for ML

The DAGs are the implementation.

Look back at your DAGs. 
DAG 3 has: validate training dataset → train → offline evaluation → metrics threshold gate → register. DAG 4 has: shadow deployment → manual approval → canary → gradual promotion.

this is my ML CI/CD.

The QUESTION: what does a traditional software CI/CD tool like GitHub Actions add on top of what my DAGs already do?
The answer is the code change trigger layer. 
My DAGs handle data validation and model validation. 
GitHub Actions handles code change validation.

The moment an engineer pushes a change to the training code, the serving code, or the feature engineering code, GitHub Actions runs automated checks before that code ever reaches your Airflow DAGs or your ECS services.


### What GitHub Actions specifically runs for my ML system:
**On every pull request to training code:**

- Unit tests on feature engineering functions — does your price normalization function produce the expected output? Does your category encoder handle unseen categories correctly?
- Schema validation tests — does the training dataset schema match what the model expects?
- Model smoke test — train a tiny version of the model on a small data sample, run one forward pass, verify output shape and dtype. Catches code errors before you spin up an EC2 GPU instance.
- Linting and type checking — standard software quality gates

**On merge to main:**

- Trigger DAG 3 in Airflow via REST API — full training pipeline runs with real data
- If DAG 3 succeeds and model is registered, DAG 4 triggers automatically

**On every pull request to serving code (FastAPI services):**

- Unit tests on serving logic — does your re-ranking MMR implementation produce diverse outputs? Does your fallback chain trigger correctly when Redis is unavailable?
- Integration test — spin up a local Docker container with a tiny FAISS index and mock Redis, send a test request, verify the response shape
- Container build test — does the Docker image build without errors?

**On merge to main for serving code:**

- Build and push Docker image to ECR (Elastic Container Registry)
- Trigger DAG 4 for serving code deployment — shadow → canary → production


**The three-layer picture, stated cleanly:**

Layer 1 — GitHub Actions (code validation): runs on every pull request and merge. Validates that code changes don't break the system before anything reaches production. Fast — completes in minutes.

Layer 2 — Airflow DAGs (pipeline orchestration): runs training, evaluation, and deployment pipelines. Validates that data is clean and model is better than production. Slower — hours for full training run.

Layer 3 — ECS + CloudWatch (production validation): shadow mode and canary routing validate that the deployed model behaves correctly on real traffic. Slowest — hours to days of observation before full promotion.

My DAGs handle Layer 2. 
GitHub Actions handles Layer 1. 
Layer 3 is my existing deployment pipeline. 
Together they form my ML CI/CD system.

**CI/CD for ML:** GitHub Actions + Airflow DAGs + ECS deployment pipeline.

**Three validation layers operating at different timescales:**

Layer 1 — GitHub Actions (minutes):
Triggered on pull request and merge to main. Validates code correctness before pipeline execution.
Training code changes: unit tests on feature engineering functions, schema validation tests, model smoke test on small data sample, linting and type checking.
Serving code changes: unit tests on serving logic and re-ranking rules, integration test with local Docker container and mock dependencies, Docker image build validation.
On merge: push Docker image to ECR, trigger Airflow DAG via REST API.

Layer 2 — Airflow DAGs (hours):
DAG 3 validates data quality and model performance. DAG 4 validates production behavior via shadow deployment. Manual approval gate between shadow and canary. Detailed in orchestration section.

Layer 3 — ECS + CloudWatch (hours to days):
Canary routing validates real-traffic behavior. Metrics compared per traffic split. Gradual promotion or instant rollback based on monitoring signals. Detailed in deployment pipeline section.

Key difference from software CI/CD: Traditional CI/CD asks "did the tests pass?" ML CI/CD asks "is this model actually better than what's in production?" The threshold gate in DAG 3 (offline metrics must exceed production model scores) and the shadow evaluation in DAG 4 (candidate sets and score distributions compared against production) are the ML-specific validation steps that have no equivalent in standard software deployment.
---


## A/B testing framework and monitoring infrastructure. 

Naive A/B testing poisons your future training data. 
The fix is surface attribution logging — you already designed this. 
Every interaction is tagged with which model version served it. 
At training time you filter by surface and model version so treatment group interactions don't corrupt the control model's retraining data and vice versa.

**A/B testing framework:**

User assignment: consistent hashing of user_id to experiment bucket, enforced at API gateway before request reaches retrieval service. Redis-backed bucket assignment table (user_id → experiment_group) for O(1) lookup. Guarantees same user always sees same model version.

Traffic routing: API gateway reads bucket assignment, routes request to correct ECS model version (control or treatment). Both versions run simultaneously as separate ECS task definitions.

Metric collection: all interactions tagged with experiment_group in serving logs. Spark batch job reads Snowflake, computes per-group metrics (recommendation-attributed CTR, purchase rate, GMV), runs two-sample statistical significance test. Experiment results reviewed on 2-4 week observation window — novelty effect in week one can produce misleading CTR lift that doesn't persist to purchase rate.

Feedback loop protection: surface attribution on all interactions prevents treatment group behavior from contaminating control model retraining data. At training time, serving logs filtered by model version before assembly.

Rejected external experimentation platforms (Optimizely, LaunchDarkly): adds SaaS cost and external dependency for a bucketing problem solvable with a Redis hash lookup and a consistent hash function at Series B/C scale.

**Monitoring infrastructure:**
- System health (second-to-second): AWS CloudWatch. ECS container metrics (CPU, memory), request latency P99 per service, error rates, Redis connection saturation, FAISS query latency. CloudWatch alarms → PagerDuty for on-call response.
- ML quality (hour-to-day): Grafana dashboards backed by CloudWatch custom metrics. Spark daily job computes: candidate diversity per request, new item coverage (last 7 days), CTR by position, recommendation-attributed purchase rate and GMV, catalog coverage. Pushed to CloudWatch as custom metrics. Grafana visualizes trends with control limit overlays for anomaly detection.
- Pipeline health (daily): Airflow task-level monitoring. Feature freshness, training dataset row count, feature null rates, Kinesis consumer lag, serving log volume. Failed validation tasks alert via Slack integration.
- Rejected Datadog as primary tool: per-host and per-custom-metric billing grows fast at Series B/C with dozens of ML-specific metrics. CloudWatch + Grafana achieves equivalent observability at significantly lower cost. Datadog added as upgrade when team size exceeds 5-6 engineers and operational overhead of three tools exceeds Datadog pricing.


# Feature Store

**Retrieval user features:**

* User ID → from request
* Preferred categories 90 days → offline (Redis)
* Preferred brands 90 days → offline (Redis)
* Purchase frequency 30 days → offline (Redis)
* Time since last purchase → offline (Redis)
* User tenure → offline (Redis)
* Recent viewed categories 30 days → offline (Redis)
* Recent viewed brands 30 days → offline (Redis)
* Average purchase price → offline (Redis)
* Country → from request
* Current product ID being viewed → from request
* Session items clicked last N minutes → online (Redis session store, logged at serving time)
* Session categories browsed last N minutes → online (Redis session store, logged at serving time)

**Retrieval item features:**

* All item features → offline (Redis item store, precomputed from item tower)

**Ranking features:**

* User ID → from request
* Item IDs → from retrieval service response
* Item features (subcategory, price, brand, category) → offline (Redis item store, batch lookup)
* Current session length → online (Redis session store)
* Session items viewed → online (Redis session store)
* Session categories viewed → online (Redis session store)
* Session search query → online (Redis session store)
* Recent search terms 7 days → offline (Redis, updated daily)
* Cart contents → from request payload (frontend sends current cart state with each recommendation request)
* Time since last click → online (Redis session store)
* Session click count → online (Redis session store)
* Has purchased this brand before? → computed at serving time from offline features
* Has purchased this category before? → computed at serving time from offline features
* Item price / average spend ratio → computed at serving time from offline features
* Item brand vs preferred brands → computed at serving time from offline features
* Viewed this brand recently? → computed at serving time from offline features
* Viewed this category recently? → computed at serving time from offline features

**Serving log schema (written to Snowflake at inference time):**

* request_id, user_id, timestamp
* aggregate_features (values read from Redis offline store)
* session_features (values read from Redis session store — point-in-time capture)
* item_ids_shown, positions, device, surface, page_context

```
Kinesis → Spark Structured Streaming → Redis (session store)
                                              ↓
User request → Retrieval service reads Redis → runs inference 
                    → logs {request_id, user_id, timestamp, 
                             aggregate_features, session_features, 
                             items_shown, positions} → Snowflake
```

=============

# Proper LLD covers:

Data pipelines in detail — the exact schema of your interaction logs, how the batch pipeline computes and writes user features, how the stream processor aggregates session events, how the log-and-join training pipeline works.

Model training pipeline — how you go from raw logs to a trained two-tower model and a trained Wide & Deep model. Offline evaluation before any model touches production. Feature engineering steps. Negative sampling implementation.

Serving infrastructure — the retrieval service and ranking service as deployed components. API contracts between them. How the ANN index is loaded and queried. How the feature store is structured and accessed. Latency budgets per component.

Re-ranking implementation — MMR algorithm concretely. How novelty and exploration slots are allocated and logged. How the final list is assembled.

Monitoring and alerting infrastructure — where metrics are computed, how alerts are triggered, what the on-call response looks like.

===


retreval **Positive sampling:** All confirmed purchases and add-to-cart events from recommendation widgets.

ranking positive sampling: Graded relevance score (0–5) events from recommendation widgets.

QUESTION:
so it's when events happen that we get fresh training data? becos that is when labels arrive.
we then join Serving logs with interaction events(labels) from specifically recommendation widgets. to retrain?
that meansnot all Serving logs will have a corresponding interaction events.
that means not all Serving logs would be used for training?
that means there's a labeling delay?
---



---

One surface. One widget. Built completely end to end.

Which surface and which widget:
Product detail page.
The user is looking at item X. Show them items most likely to convert given that context.

**Widget shown to the user:** 
"Similar Items" or "You May Also Like"

**Recommendation strategy:**
Retrieve and rank products that are most similar to the currently viewed item and most likely to be clicked or purchased.

**Primary metrics**
Recommendation-attributed CTR — Are users engaging with the recommendations?
Recommendation-attributed CVR — Do those recommendations result in purchases?

**Guarail:**
watch for click bait.

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
user preferences(brand, average purchase price, average clicked price, country, device type, )
historical Query/context CTR of candidate item
ex: Viewing running shoes -> How often is THIS candidate clicked?
historical conversion rate of candidate item


The model outputs one score, for example:
Predicted probability
that this user clicks
this candidate
given this current item.

Indicate which features are:
computed offline
computed online
cached
refreshed daily

---

**Similar Items**

Goal:
Help users compare products.

Useful when the customer thinks:
"I'm not sure I want this exact item."

Here you're asking:
"Show me alternatives."

Not accessories.

**Frequently Viewed Together**

Goal:
Help users discover related products.

Useful when the customer thinks:
"What else might I need?"





Claude code prompt:
> Polish my lld.md to be precise and clear and neatly written.

> tell me what you think the single weakest part of my design is right now — the thing a senior engineer would push back on hardest in a design review.

> crtique my systemdesign. the problem framing. the architecture. does it align with what the market says they want from a ML engineer to do for them?
what did i not put that is of high value that they need?
what did i put that is of low value?


--

another one is i said to use all data not just the one gotten from recommendation widget.
i said to use it to build user taste profile
then evaluate on only data from recommendation widget

how does that make any sense?
it doesn't blend to me.
is it not only tge data i will use for triniing i should be collecting?

---

i think another thing is that recommendations are updated only after 24 hrs.

question: am i designing solution againt or for the problem?

product detail page recs suppose change based on what product is currently being viwed.
all personalized to user taste profile.

QUESTION:  product detail page if user goes back n clicks same item we can show same recs?
is that my problem or the developer problem?
