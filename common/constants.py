"""Constants shared between data_gen/ (synthetic event simulation) and
pipelines/ (the production Spark aggregation job) -- a single source of
truth so the two can never silently drift apart.
"""

# Purchase-attribution window: a purchase is attributed to a click on the
# same item by the same user if the click happened within this many hours
# before the purchase. Used by data_gen/event_simulator.py to simulate
# attributed purchases and by pipelines/spark_jobs/daily_features.py to
# compute recommendation_cvr with the identical join condition.
ATTRIBUTION_WINDOW_HOURS = 24
