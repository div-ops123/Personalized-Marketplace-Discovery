# Build Plan — Data Strategy & Pipeline Design phase

Phase 0 — Infra scaffolding

1. Repo layout: /data_gen, /pipelines (Spark jobs + Airflow DAGs), /training, /serving, /infra (docker-compose files).
2. docker-compose.pipeline.yml — Postgres (offline-store stand-in for Snowflake), Airflow (official Apache docker-compose, LocalExecutor is enough at this scale), Spark. At your data volume, running PySpark in local mode inside one container is simpler than a master/worker cluster and adds no demo value the cluster topology would — same reasoning as skipping EMR. Your call if you want the cluster look for the resume line.
3. Warehouse connector: one data-access layer, backend selected by env var (WAREHOUSE_BACKEND=postgres|snowflake). Postgres is the default for local/docker-compose; Snowflake only gets wired in during the AWS recording session via env swap, no code change.

Phase 1 — Static reference data

4. Generate Item Catalog directly (it's current-state, not derived — fine to hand-generate): item_id, category, subcategory, brand, price, tags, description, image (left null). Reserve a subset of items as empty cells to indicate "not yet introduced" (future catalog date, zero history) — this is what makes the cold-start demo real later.
5. Pick a scale: something like a few thousand items and a low-thousands user population — enough to produce non-trivial category/brand distributions and a real train/holdout split, small enough to generate and train on fast.
6. Generate a synthetic user population with a hidden latent preference structure (e.g. per-user category/brand affinity weights) — used only inside the event simulator to drive realistic behavior, never persisted or read directly by any model (that would be leaking ground truth into training).

Phase 2 — Raw event simulation (the actual "historical data")

7. Simulate PDP views over a multi-day window (suggest ~90 days, matching the rolling window already defined in LLD.md) — for each: pick an anchor item, build a ~200-candidate set using latent category/brand affinity + noise (this stands in for what retrieval will eventually learn), log Impression Events for the top 20 with position, device, country, retrieval_similarity_score.
8. Generate Click Events: probability driven by user↔candidate latent affinity, decaying with position — this is what makes the IPS position-bias correction demonstrably do something later, instead of being inert code.
9. Generate Purchase Events: some following a click within 24h (attributed), some purchases with no preceding click (organic/unattributed), some clicks that never convert. You need all three cases or the attribution join never exercises its False branch meaningfully.

Phase 3 — The real pipeline (reused later as production DAG code)

11. Write the Spark aggregation job: reads raw events from Postgres, computes User Daily Features and Candidate Daily Features exactly per data-flow.md's schema and point-in-time snapshot semantics (snapshot_date, historized, append-only). This script is not a fixture generator — it's the literal logic DAG 1 runs in production; you're backfilling history with it, not faking output.
12. Add the validation step from LLD.md's DAG 1 (null rates, coverage, value ranges, row counts) with concrete thresholds — easy to set now since you control the synthetic distribution.
13. Wrap steps 11–12 in an actual Airflow DAG 1, running against docker-compose Postgres/Spark. This is your first real, testable "e2e ownership" artifact — runs entirely locally, no AWS needed yet.

Phase 4 — Dataset assembly

14. Build the Retrieval Dataset Builder and Ranking Dataset Builder per data-flow.md's join logic (point-in-time joins on the daily snapshots, current-state join on Item Catalog, in-batch negatives left to the training loop, not materialized).

Phase 5 — Experimentation (local PC)

15. Train the retrieval encoder — Train the LambdaMART ranker. Log both to MLflow (local Docker MLflow).
no cold-start content-similarity bootstrap first, that's just documented thinking. i train the models with behavioral click-pairs we simulated.
16. Build the FAISS/HNSW index from the trained encoder's item embeddings.

Phase 6 — Serving path (the lightweight, visitor-facing docker-compose)

17. Separate docker-compose.serving.yml: API Gateway + Retrieval Service + Ranking Service + Redis + mounted FAISS index + trained model artifacts. This is the profile anyone can clone and click around without touching Postgres/Airflow/Spark at all.

Phase 7 — CI/CD + AWS recording

18. Wire GitHub Actions (Layer 1) around the DAGs already built in Phase 3.
19. For the recording session only: swap the warehouse connector to real Snowflake, stand up ElastiCache, run Airflow/Spark on an EC2 box (same docker-compose files, just pointed at cloud config), run the pipeline once + the serving demo, record, tear everything down.
