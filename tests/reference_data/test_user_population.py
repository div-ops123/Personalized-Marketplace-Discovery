"""Unit tests for synthetic user population and latent affinity generation."""

import numpy as np
import pytest

from user_population import build_users, sample_user_affinity


def test_build_users_empty():
    """n_users=0 returns an empty DataFrame with the expected columns."""
    df = build_users(0, seed=1)
    assert len(df) == 0
    assert set(df.columns) == {"user_id", "signup_date"}


def test_build_users_row_count():
    """Generates exactly n_users rows."""
    df = build_users(25, seed=1)
    assert len(df) == 25


def test_build_users_no_affinity_columns_leak():
    """Guard: the persisted users table must never contain affinity data."""
    df = build_users(10, seed=1)
    assert set(df.columns) == {"user_id", "signup_date"}


def test_sample_user_affinity_deterministic():
    """Same user_id and seed always produces bit-identical affinity vectors."""
    a1 = sample_user_affinity("user_000001", global_seed=42)
    a2 = sample_user_affinity("user_000001", global_seed=42)
    assert np.array_equal(a1["category_affinity"], a2["category_affinity"])
    assert np.array_equal(a1["brand_affinity"], a2["brand_affinity"])


def test_sample_user_affinity_differs_by_user():
    """Different users get different affinity vectors."""
    a1 = sample_user_affinity("user_000001", global_seed=42)
    a2 = sample_user_affinity("user_000002", global_seed=42)
    assert not np.array_equal(a1["category_affinity"], a2["category_affinity"])


def test_sample_user_affinity_sums_to_one_and_nonnegative():
    """Dirichlet-distributed affinities sum to 1 and have no negative entries."""
    affinity = sample_user_affinity("user_000003", global_seed=42)
    for vector in affinity.values():
        assert np.isclose(vector.sum(), 1.0)
        assert (vector >= 0).all()


def test_sample_user_affinity_empty_user_id_raises():
    """An empty user_id is rejected explicitly."""
    with pytest.raises(ValueError):
        sample_user_affinity("", global_seed=42)
