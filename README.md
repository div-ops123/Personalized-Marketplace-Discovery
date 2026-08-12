
# Personalized Marketplace Discovery

A two-stage retrieval-then-rank recommender that powers a "Similar Items" widget on a marketplace Product Detail Page (PDP) — built, trained, and served end-to-end locally, with a production-scale system design documented for the pieces not worth running on a laptop.

## Demo

https://github.com/user-attachments/assets/76415683-8938-4185-859b-2764bc0ef339

---

**What this project demonstrates:**

- **System design before code.** [`docs/HLD.md`](docs/HLD.md) and [`docs/LLD.md`](docs/LLD.md) were scoped, written, and reviewed before implementation started — the build order in [`docs/build-phases.md`](docs/build-phases.md) follows from that design, not the other way around. See [Design Decisions](#5-design-decisions--trade-offs) for how that scoping actually shaped what got built.
- **End-to-end production ownership.** Every layer — synthetic data generation, the Airflow feature pipeline, training, the champion/challenger promotion gate, and the serving stack — was designed, built, and run by one person, not handed off between stages.
- **Translating ML metrics to business terms.** Every offline metric reported here (Recall@200, NDCG@20) is tied to a specific product metric and business outcome in [`docs/metric-ladder.md`](docs/metric-ladder.md) — not reported in isolation from the revenue question it's supposed to answer.
- **Scale and latency awareness.** Every infra choice — ANN algorithm, online store, cloud services — is justified against a concrete latency budget and a stated business scale (Series B/C: ~2M items, 5,000 peak QPS, 100ms p99), not picked by default or by hype. See the [trade-offs table](#5-design-decisions--trade-offs) below.

---

## 1. The Problem

Buyers on a marketplace land on a product page and often don't find what they'd actually want next — not because the catalog lacks it, but because nothing surfaces it. That's lost conversion and repeat purchase, not a modeling curiosity.

**Problem statement:** given a buyer viewing an item on the PDP — their interaction history, and the attributes of the item they're currently viewing — retrieve and rank the items most similar to it that this buyer is most likely to click and purchase, under a **100ms p99 serving latency budget**, to increase recommendation-attributed conversion rate.

**Scale this is designed for** (Series B/C marketplace, per [`docs/problem-framing.md`](docs/problem-framing.md)): 500K–2M monthly buyers, up to ~2M catalog items, 20–100M interaction events/month.

**"Done" is defined precisely, not vaguely** — see [Results](#4-results) below for why that matters here.

---

## 2. The Solution

### Architecture

```
Anchor Item (PDP view)
     │
     ▼
Retrieval Service ──── ANN Index (precomputed item embeddings)
     │                 anchor embedding lookup only -- no live
     │                 model inference, no user identity involved
     ▼
Top 200 Candidates + Similarity Scores
     │
     ▼
Ranking Service ──── Item Feature Store (batch lookup, all candidates)
     │            ── User Feature Store (if user known)
     │
     ▼
Ranked Candidates (scored)
     │
     ▼
Re-ranking Layer (novelty injection, exploration slots)
     │
     ▼
Final List (10–20 items) → API → User
```

Full request-level walkthrough, fallback chains, and monitoring signals: [`docs/HLD.md`](docs/HLD.md). Data contracts, storage choices, and the champion/challenger promotion gate: [`docs/LLD.md`](docs/LLD.md).

### Tech stack

| Layer | Choice | Why |
|---|---|---|
| Retrieval model | Siamese item encoder (content-based, weight-shared) | Request is anchored to a specific item (PDP context), not a user — see [Design Decisions](#5-design-decisions--trade-offs) below |
| Ranking model | LambdaMART (LightGBM, learning-to-rank) | Directly optimizes NDCG on tabular categorical + cross features; cheaper to serve at <70ms than a neural ranker at this feature scale |
| ANN index | FAISS | Standard, well-understood, no managed-service dependency for a local build |
| Offline store | Postgres (local stand-in for Snowflake, swapped via one env var) | Same data-access layer works against Snowflake with zero code change — see [What's Built vs. Designed](#6-whats-built-vs-designed-not-deployed) |
| Online feature store | Redis (ElastiCache), daily-batch only | No real-time session store — every serving-time feature is a precomputed daily snapshot, which is also what keeps train/serve skew bounded (point-in-time joins against the same tables both training and serving read) |
| Orchestration | Airflow (LocalExecutor, docker-compose) | Runs the actual daily feature aggregation DAG against Postgres/Spark, not a fixture generator |
| Experiment tracking / registry | MLflow (aliases + tags, MLflow 3.x) | `champion` / `challenger` aliases, not the deprecated Staging/Production/Archived stage API |
| Serving | FastAPI (API Gateway + Retrieval Service + Ranking Service), static exports | Each service mounts a one-time export (FAISS index, LightGBM `model.txt`, JSON feature/catalog snapshot) — no live DB, MLflow, or Redis dependency to demo it |

---

## 3. Synthetic Data

There's no real marketplace behind this — production interaction data was replaced with a deliberately-simulated one, built to exercise the same edge cases and failure modes real data would, not just a happy-path fixture:

- **Item catalog** — a few thousand items with category, subcategory, brand, price, tags, and a text description. A reserved subset is generated with a future "introduced" date and zero interaction history, specifically so the cold-start-item path is something to actually test, not just a code branch that never executes.
- **User population** — a synthetic population with a hidden per-user category/brand affinity structure. This latent preference only ever drives the event simulator's behavior — it's never persisted or exposed to any model, since that would leak ground truth straight into training.
- **90 days of simulated PDP events** — for each simulated PDP view: an anchor item, a ~200-candidate set built from latent affinity + noise (standing in for what retrieval eventually has to learn), impression events for the top 20 with position/device/country, click events with probability driven by affinity and decaying with position (so position-bias correction has something real to correct), and purchase events split across three cases — attributed within 24h of a click, organic with no preceding click, and clicked-but-never-converted — all three needed so the attribution join downstream actually exercises both branches, not just the case where everything lines up.

```bash
# after infra/docker-compose.pipeline.yml is up
uv run python data_gen/generate_reference_data.py   # item catalog + user population
uv run python data_gen/generate_events.py            # 90 days of impressions/clicks/purchases
```

Generation code: [`data_gen/`](data_gen/). Full generation plan: [`docs/build-phases.md`](docs/build-phases.md) (Phases 1–2).

---

## 4. Results

Honest framing: this is a synthetic-data local build, so there's no live A/B test and no real GMV to report — the [problem framing](docs/problem-framing.md) defines "done" as *beating a popularity baseline on recommendation-attributed conversion in an A/B test*, which requires production traffic this project doesn't have. What's actually measured is the offline proxy metrics the online experiment would be gated on:

| Metric | Stage | Result | Floor (gate threshold) |
|---|---|---|---|
| Recall@200 | Retrieval | **0.99** | 0.80 |
| NDCG@20 | Ranking | **0.39** | 0.25 |

Both are measured against a **frozen, held-out test set** (not recomputed per run — see [Design Decisions](#5-design-decisions--trade-offs)), so champion and challenger models are always compared on identical data regardless of when each was trained.

**Guardrails tracked, not just the north star:** recommendation CTR (click-bait detection — should move with purchase rate, not ahead of it), catalog coverage (explicitly deferred as an *enforced* guardrail until there's real production traffic to define per-anchor-item coverage against — see [`docs/metric-ladder.md`](docs/metric-ladder.md)).

---

## 5. Design Decisions & Trade-offs

This project's design work happened almost entirely before any code was written — [`docs/HLD.md`](docs/HLD.md) and [`docs/LLD.md`](docs/LLD.md) exist specifically to force every infra and modeling choice through an explicit "what does this cost, what does it buy" pass, at the scale defined in [`docs/problem-framing.md`](docs/problem-framing.md), rather than defaulting to whatever's popular. A few of those trade-offs, chosen specifically because each one is a direct **latency vs. accuracy vs. speed/cost** call, not a style preference:

| Decision | Trade-off | What was gained | What was given up |
|---|---|---|---|
| **HNSW over IVF** (ANN index) | Accuracy vs. latency | HNSW's `ef_search` knob trades query speed for recall without ever *permanently* dropping a reachable neighbor | IVF's `nprobe` knob is cheaper to tune, but a neighbor in an unsearched cluster is missed for good — unacceptable against the 95%+ Recall@200 target |
| **Single in-memory FAISS index over a distributed vector DB** | Operational complexity/latency vs. scalability | At 2M items × 256 dims (~2GB), the index fits on one machine — no network hop, no distributed-query latency | Gives up horizontal scaling and built-in replication/HA that a distributed store would provide, deliberately, because the memory math says it isn't needed yet |
| **ElastiCache (Redis) over DynamoDB** (online feature store) | Latency, decisively | Sub-millisecond p99 reads fit inside the few milliseconds ranking has left after retrieval + ANN search + inference are accounted for in the 100ms budget | DynamoDB's easier ops/scaling story — rejected because its disk-backed read path doesn't leave the same latency margin, and the budget is tight enough that this alone was decisive |
| **Full candidate set now, negative downsampling reserved as a lever** (ranking class imbalance) | Training speed/compute vs. data fidelity | LambdaMART trains on the full ~200–500:1 imbalanced candidate set as-is, keeping every signal | If dataset size becomes computationally prohibitive, downsampling negatives is the documented next lever — **and this is exactly the trade-off local hardware forced early, in practice**, see [What Didn't Work](#what-didnt-work) below |

Full trade-off reasoning (including rejected AWS services, storage, and orchestration choices) is in [`docs/LLD.md`](docs/LLD.md#tooling--justification).

### Key Decision: Item-Anchored Retrieval, Not a User Tower

**Options considered:**

| Approach | Retrieval-time inputs | Cold-start-user handling | Complexity |
|---|---|---|---|
| Two-tower (user tower + item tower) | User embedding + anchor item | Needs a fallback path for unknown/anonymous users at the retrieval stage | Two models to train, version, and keep in sync; a live user-feature fetch on the retrieval hot path |
| **Item-anchored (Siamese item encoder)** — chosen | Anchor item embedding only | **No cold-start case at retrieval at all** — retrieval never touches user identity | One model; retrieval is a pure embedding lookup + ANN query, no live inference |

**Decision:** item-anchored retrieval; personalization is pushed entirely to the ranking stage.

**Reasoning:**
- The surface being built is a PDP "Similar Items" widget — the request is *already* anchored to a specific item by definition. A user tower would be personalizing a signal (item-item similarity) that the anchor item already constrains well, while adding a live user-feature dependency to the retrieval hot path for a stage with a <30ms budget.
- Item-item similarity has a natural cold-start answer: a new item's content features (category, brand, description embedding) still produce a meaningful embedding with zero interaction history. A user tower has no equivalent — an anonymous or brand-new user has no embedding to retrieve with, forcing a fallback path exactly where the design doesn't need one.
- Collapsing to one model removes an entire class of skew: two towers trained separately can drift out of a shared embedding space; one encoder can't.

**Trade-off accepted:** personalization can't influence *which* 200 candidates get pulled in — only how they're scored and re-ranked afterward. If a user's taste diverges sharply from an item's typical neighbors, that gap only gets corrected at ranking, not retrieval. Documented as an explicit design boundary in [`docs/HLD.md`](docs/HLD.md#stage-1-retrieval), not an oversight.

---

## 6. What's Built vs. Designed (Not Deployed)

Everything below "Built" runs locally, end-to-end, and has been exercised by hand through the full pipeline. Everything under "Designed, not deployed" is specified in [`docs/LLD.md`](docs/LLD.md) / [`docs/HLD.md`](docs/HLD.md) with real reasoning behind each choice, but was never stood up — running it costs money and adds no additional resume signal over documenting it precisely.

**Built and verified locally:**
- Synthetic data generation (item catalog with deliberate cold-start gaps, user population with hidden latent preferences, 90-day event simulation with position-biased clicks and both attributed and organic purchases)
- Airflow DAG computing daily User/Candidate features from raw events into Postgres, with data-quality validation (null rates, coverage, row counts)
- Retrieval + ranking dataset builders (point-in-time joins, no label leakage)
- Training: Siamese item encoder + LambdaMART ranker, tracked in MLflow — retrieval training auto-selects CUDA when available (falls back to CPU otherwise), device logged as an MLflow run param; verified on an NVIDIA GPU locally with Recall@200 matching the CPU-trained baseline
- FAISS index build from trained item embeddings
- **Local validation pipeline** — a champion/challenger promotion gate: evaluates a newly-trained candidate against the current champion on a frozen test set, gates promotion on clearing both an absolute floor and the champion's score, promotes via MLflow aliases (`challenger` → `champion` as an explicit, separate step, not automatic)
- Local serving stack — API Gateway + Retrieval Service + Ranking Service (FastAPI), each mounting a static export, no live DB/MLflow/Redis needed to run it

**Designed, not deployed** (documented decisions, not implementation gaps):
- AWS deployment (Snowflake swap via env var, ElastiCache, EC2-hosted Airflow/Spark run) — the data-access layer already supports the Snowflake swap with zero code change; the actual cloud run was scoped out as cost with no additional resume signal over the working local equivalent
- CI/CD (GitHub Actions around the existing DAGs)
- Shadow-mode / canary rollout for model promotion
- A/B testing harness (needed for the actual "done" bar in [`docs/problem-framing.md`](docs/problem-framing.md), since that requires live traffic this project doesn't have)

---

## 7. What I Learned

**Design and problem-framing discipline:**
- To frame the right problem before writing a line of code — the problem statement, scale assumptions, and definition of "done" in [`docs/problem-framing.md`](docs/problem-framing.md) were written first, and everything else follows from them.
- To design before touching any line of code, and to scope that design to the actual stated business scale (Series B/C, ~2M items, 5,000 peak QPS) rather than either over-building for hyperscale or under-building for a toy.
- To list out the failure modes up front and design for them deliberately — cold-start items/users, ANN index unavailability, ranking model unavailability, full-system fallback — rather than discovering them in an incident later. Every fallback chain in [`docs/HLD.md`](docs/HLD.md) was scoped before the corresponding service was built.

**ML domain knowledge:**
- Ranking algorithms and their evaluation metrics specifically — why Recall@K is the right offline proxy for retrieval and NDCG@K for ranking, and how choosing K itself is a latency/catalog-size decision, not an arbitrary constant.
- Translating ML metrics into business terms — every offline metric in this project ladders up to a specific product metric and business outcome in [`docs/metric-ladder.md`](docs/metric-ladder.md), instead of being reported as an isolated number nobody outside ML would know how to act on.

**Tool judgment:**
- That the right tool depends on business scale, not on what's trending — e.g. rejecting a distributed vector database in favor of a single in-memory FAISS index because the actual memory math (~2GB at 2M items) said a distributed store buys nothing yet; rejecting SageMaker and EKS for the same reason at this team/traffic size. See the [trade-offs table](#5-design-decisions--trade-offs) above.

**Judgment on AI-assisted work (still developing, not claiming mastery):**
- This project was built with heavy use of AI tooling, which made *reviewing and correcting* its output as important a skill as writing the design myself.
<!-- Concrete examples from this repo's own history: catching that the high-level design doc had drifted to describe a live user-tower retrieval path when the system actually built is item-anchored with zero live inference at retrieval time; catching a subtle correctness bug where naively rebuilding the retrieval model's vocabulary instead of reloading its run-specific version would have silently misaligned embeddings and produced scores that looked valid but meant nothing; and, when a re-ranking design referenced an undefined "diversity" step, choosing to flag it as unspecified rather than inventing an algorithm just to make the doc look complete. -->

**Builder mindset over student mindset:**
- Learned Airflow, Spark, and ranking by building with them inside a real system with real constraints, not by studying each in isolation first and assembling the pieces afterward — build-to-learn rather than learn-then-build.

### What Didn't Work

- **The full Airflow backfill never completed.** It stalled partway through for a reason I haven't root-caused yet — flagged to come back and debug, but I didn't block the rest of the build on it and moved forward with what I had running.
- **Building the training dataset over the full 90-day window caused a memory blowup on local hardware.** Scoped the training window down to 5 days of data to keep moving, instead of the full window the design originally called for.

### What I'd Do Differently

- Spend more time on the data side specifically — which features would actually move the model, rather than treating the feature set as fixed once the pipeline was generating it.
- Consider moving training to the cloud earlier, specifically to get past the local RAM bottleneck and train on the full data window instead of the 5-day subset the local memory blowup forced.

---

## 8. Try It

Requires Docker and Python 3.11+. [uv](https://docs.astral.sh/uv/) is optional — recommended if you have it (faster, and matches the exact locked versions in `uv.lock`), but plain `pip` against `pyproject.toml` works too.

```bash
git clone https://github.com/div-ops123/Personalized-Marketplace-Discovery.git
cd Personalized-Marketplace-Discovery
cp .env.example .env   # fill in local Postgres/Airflow credentials
```

**With uv:**

```bash
uv sync --extra datagen --extra training --extra serving --extra dev
```

**With plain pip (no uv):**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[datagen,training,serving,dev]"
```

The commands below use `uv run python ...` — if you went the pip route, drop the `uv run` prefix once your venv is activated (plain `python ...` works the same).

**Full pipeline (data → training → validation gate):**

```bash
docker compose --env-file .env -f infra/docker-compose.pipeline.yml up -d   # Postgres + Airflow + Spark
docker compose --env-file .env -f infra/docker-compose.mlflow.yml up -d     # MLflow tracking server

# 1. generate the synthetic catalog, users, and events -- see Synthetic Data (above)

# 2. backfill the daily feature pipeline DAG (User/Candidate Daily Features, one day per task)
#    note: this backfill never completed cleanly on local hardware for me either --
#    see What Didn't Work. If it stalls, the dataset builders below only need
#    however many days' worth of snapshots actually landed in Postgres.
docker compose -f infra/docker-compose.pipeline.yml exec airflow-scheduler \
    airflow dags backfill daily_feature_pipeline -s 2025-01-01 -e 2025-04-01

# 3. build the retrieval + ranking training datasets from the backfilled features
#    (--days caps history to fit a local Spark driver's heap -- see What Didn't Work)
uv run python pipelines/spark_jobs/run_dataset_builders.py --builder both --days 5

uv run python training/retrieval_train.py
uv run python training/ranking_train.py
uv run python training/validation_pipeline.py --model-type retrieval
uv run python training/validation_pipeline.py --model-type ranking
uv run python training/validation_pipeline.py --model-type retrieval --promote-challenger
uv run python training/validation_pipeline.py --model-type ranking --promote-challenger
```

**Just the serving demo** (no Postgres/Airflow/Spark/MLflow required — static exports only):

```bash
uv run python serving/export_ranking_model.py
uv run python serving/export_feature_snapshot.py
uv run python serving/build_retrieval_index.py

docker compose -f infra/docker-compose.serving.yml up -d
# API Gateway now serving "Similar Items" requests against the exported artifacts
```

**Tests:**

```bash
uv run pytest
```

---

## Docs

- [`docs/problem-framing.md`](docs/problem-framing.md) — problem statement, scale assumptions, definition of "done"
- [`docs/metric-ladder.md`](docs/metric-ladder.md) — business metric → product metric → offline metric chain, guardrails
- [`docs/HLD.md`](docs/HLD.md) — high-level system design, request flow, fallback chains, monitoring
- [`docs/LLD.md`](docs/LLD.md) — low-level design: data contracts, storage, the champion/challenger promotion gate
- [`docs/data-flow.md`](docs/data-flow.md) — point-in-time join logic, train/serve parity
- [`docs/future-improvements.md`](docs/future-improvements.md) — scoped-out ideas and why
