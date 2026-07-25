"""Regression guard: common/constants.py is the single source of truth
for ATTRIBUTION_WINDOW_HOURS, consumed by both data_gen/event_simulator.py
and pipelines/spark_jobs/daily_features.py.
"""

from common.constants import ATTRIBUTION_WINDOW_HOURS


def test_attribution_window_hours_is_positive():
    assert ATTRIBUTION_WINDOW_HOURS > 0


def test_data_gen_config_reexports_the_same_value():
    from config import ATTRIBUTION_WINDOW_HOURS as dg_window

    assert dg_window == ATTRIBUTION_WINDOW_HOURS
