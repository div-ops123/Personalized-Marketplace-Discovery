"""LambdaMART ranker via LightGBM's native `lambdarank` objective."""

import lightgbm as lgb
import pandas as pd

from training.constants import DEFAULT_LGBM_PARAMS, RANKING_EARLY_STOPPING_ROUNDS, RANKING_NUM_BOOST_ROUND


def train_lambdamart(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    group_train: list[int],
    x_val: pd.DataFrame,
    y_val: pd.Series,
    group_val: list[int],
    params: dict | None = None,
    num_boost_round: int = RANKING_NUM_BOOST_ROUND,
    early_stopping_rounds: int = RANKING_EARLY_STOPPING_ROUNDS,
) -> lgb.Booster:
    """Trains a LambdaMART ranker.

    Class imbalance (~1 purchase per 200-500 impressions, per docs/HLD.md's
    Class Imbalance section) is handled natively by the pairwise/listwise
    lambdarank loss -- no downsampling, no is_unbalance/scale_pos_weight
    (those are classification-objective knobs, not applicable to
    lambdarank).

    Args:
        x_train: Training features (encode_ranking_examples's X).
        y_train: Training labels.
        group_train: Per-group row counts for x_train, in row order
            (build_query_groups's output for the train split).
        x_val: Validation features.
        y_val: Validation labels.
        group_val: Per-group row counts for x_val.
        params: LightGBM params; defaults to DEFAULT_LGBM_PARAMS.
        num_boost_round: Max boosting rounds.
        early_stopping_rounds: Stop if val NDCG@K hasn't improved in this
            many rounds.

    Returns:
        lgb.Booster: The trained model, at its best iteration per early
            stopping on validation NDCG@K.
    """
    params = dict(params or DEFAULT_LGBM_PARAMS)
    categorical_features = [c for c in x_train.columns if str(x_train[c].dtype) == "category"]

    train_set = lgb.Dataset(
        x_train, label=y_train, group=group_train, categorical_feature=categorical_features
    )
    val_set = lgb.Dataset(
        x_val,
        label=y_val,
        group=group_val,
        categorical_feature=categorical_features,
        reference=train_set,
    )

    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds)],
    )
