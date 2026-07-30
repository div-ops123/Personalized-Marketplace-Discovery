# Model Evaluation: Business Interpretation

Interpretation of the first end-to-end training run of both models
(`training/retrieval_train.py`, `training/ranking_train.py`), translating
offline metrics into what they mean for the business rather than reporting
them as bare numbers. See `docs/metric-ladder.md` for why these particular
metrics were chosen.

**This run is a pipeline-correctness smoke test, not a production result.**
It was trained on a `--days 5` local dataset build (`run_dataset_builders.py`),
with `--val-days 1 --test-days 1` (leaving ~3 days of train data) and, for
retrieval, only 2 epochs. Conclusions below are about *what the numbers mean*
and *whether the pipeline works*, not deployment decisions — see
"What's still needed" at the end.

---

## Ranking model (LambdaMART)

```
best_iteration=42            (early stopping: stopped_iteration=50, patience exhausted at 42)
val_ndcg_at_20=0.3855
valid_0-ndcg_at_20=0.9283    (LightGBM's own internal metric -- do not use, see below)
training_duration_seconds=25.8
```

**Which number to trust.** Two NDCG@20 values were logged for the same
validation set, and they disagree sharply (0.928 vs 0.385). This is not a
bug — they measure different things. LightGBM's own internal
`valid_0-ndcg_at_20` scores every validation query group, including the
large majority that contain **no purchase at all** (purchases are rare:
~1 per 200-500 impressions). A group with no relevant item is trivially
"perfect" to rank — nothing to get wrong — so LightGBM's number is inflated
by groups that aren't actually testing anything. This repo's own
`ndcg_at_k` (`training/ranking_eval.py`) deliberately excludes those
zero-positive groups, so it only averages over query groups where the
model actually had a purchase to rank correctly. **`val_ndcg_at_20=0.3855`
is the number that reflects real ranking quality** and is the one used
below.

**What it means.** NDCG@20 of 0.3855 means that, among impressions that
contained a genuine purchase, the model's ranking of the top 20 candidates
captures about 39% of the maximum possible ranking quality (a perfect
ranking, purchased item first, scores 1.0). The ranking order captures real
signal — it is meaningfully above 0 — but is still far from placing
purchased items consistently near the top.

**Business translation.** Higher NDCG@20 means recommendation-attributed
purchases surface higher in the widget's slots. Since the North Star metric
(`docs/metric-ladder.md`) is recommendation-attributed GMV, and higher
position generally drives more clicks, this offline number is a *leading
indicator*, not a guarantee — actual GMV/CTR/purchase-rate impact can only
be confirmed through online A/B testing (per `metric-ladder.md`'s
Online section).

**On "deploy iteration 42."** `best_iteration=42` is the checkpoint
`booster.predict()` used by default after early stopping, so the model
already being evaluated *is* iteration 42 — this is correct, but should be
made explicit (`num_iteration=booster.best_iteration`) at model-save time
rather than relying on LightGBM's default, to avoid ambiguity if that
default ever changes.

**Sample size caveat.** This run's validation window was 1 day out of a
5-day dataset. Given how rare purchases are, this is a small number of
scored query groups — treat 0.3855 as a rough signal that the pipeline
produces a real ranking (better than random), not as a precise estimate of
production ranking quality.

---

## Retrieval model (Siamese item encoder)

```
train_loss=3.32
val_recall_at_200=0.9906        (105/106 anchors -- see sample size note)
best_val_recall_at_200=0.9906   (best epoch == last epoch)
training_duration_seconds=193.1  (~3.2 min)
```

**Correction:** 0.9906 is **99.06%**, not 92% — an early transcription
error. The 8%/92% split does not appear anywhere in the actual metrics.

**What it means.** Recall@200 of 99.06% means that for 105 of 106
evaluable anchors (anchors with at least one purchased candidate in the
validation window), the item the user actually went on to purchase
appeared somewhere in the encoder's top-200 nearest neighbors out of the
*full* item catalog. Only 1 anchor's purchased item was missed entirely
by candidate generation.

**Business translation.** This is the retrieval stage's job: don't lose
the right answer before the ranker ever sees it. A high Recall@200 means
the ranking stage is very rarely starved of the item the user would have
bought — retrieval is not currently the bottleneck in this pipeline.

**Sample size caveat.** 99.06% comes from exactly 105/106 anchors (a
1-day validation window). One additional miss would drop this to ~98%,
and one fewer would take it to 100% — this is not yet a statistically
stable estimate. A run against the full backfilled history would give a
far more trustworthy number.

**On train_loss and over/underfitting.** `train_loss` alone cannot tell
you whether the model is over- or underfitting — that's true of any
model, not specific to neural networks. What's needed is a training
signal compared against a *validation* signal across epochs. That
validation signal already exists here: `val_recall_at_200` is logged per
epoch (visible as a curve in the MLflow UI, not just the final scalar).
If `train_loss` keeps falling while `val_recall_at_200` plateaus or
declines, that's overfitting; both moving together is healthy fitting.
Early stopping (patience=3) already uses this validation curve, not
`train_loss`, as its stopping signal — so the pipeline already guards
against overfitting even without a dedicated validation-loss metric.
`best_val_recall_at_200 == val_recall_at_200` here simply means training
ended (2 epochs, the smoke-test setting) before any later epoch beat the
best one — not evidence of over- or underfitting either way, since 2
epochs is too few to see a trend.

`train_loss` itself has no direct business meaning — it's an optimization
diagnostic (is gradient descent working, is it still improving), not a
number to report externally. `val_recall_at_200` is the metric with
business meaning.

**Training time.** ~3.2 minutes, but over ~3 days of train data (5-day
build minus 1 val day minus 1 test day), not the full 5.

---

## What's still needed before these are business conclusions

- Retrain on the full backfilled history (or as many days as local memory
  allows — see the `--days` flag added to `run_dataset_builders.py`) with
  a val/test window large enough to contain a statistically meaningful
  number of purchase-labeled query groups/anchors.
- Run retrieval for its full configured epoch budget (20, not 2) with
  early stopping actually engaged, so the train_loss/val_recall curves are
  long enough to diagnose over/underfitting.
- Validate any offline metric improvement against the North Star metric
  (recommendation-attributed GMV) via online A/B testing — offline
  Recall@200 and NDCG@20 are leading indicators, not proof of business
  impact, per `docs/metric-ladder.md`.
