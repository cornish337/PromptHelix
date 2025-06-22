
from prometheus_client import Counter, Gauge, Histogram # Added Histogram for potential future use

# --- Genetic Algorithm Metrics ---

# Gauges: Values that can go up and down
GA_CURRENT_GENERATION = Gauge(
    'prompthelix_ga_current_generation_number',
    'Current generation number of the Genetic Algorithm'
)

GA_BEST_FITNESS = Gauge(
    'prompthelix_ga_best_fitness_score',
    'Best fitness score in the current GA generation'
)

GA_AVG_FITNESS = Gauge(
    'prompthelix_ga_average_fitness_score',
    'Average fitness score in the current GA generation'
)

GA_MEDIAN_FITNESS = Gauge(
    'prompthelix_ga_median_fitness_score',
    'Median fitness score in the current GA generation'
)

GA_MIN_FITNESS = Gauge(
    'prompthelix_ga_min_fitness_score',
    'Minimum fitness score in the current GA generation'
)

GA_STD_DEV_FITNESS = Gauge(
    'prompthelix_ga_std_dev_fitness_score',
    'Standard deviation of fitness scores in the current GA generation'
)

GA_POPULATION_SIZE = Gauge(
    'prompthelix_ga_population_size',
    'Current size of the GA population'
)

GA_RUNNING_STATUS = Gauge(
    'prompthelix_ga_running_status',
    'Status of the GA run (1 for running/active, 0 for idle/stopped/completed/error)'
    # Could also use an Enum with states if preferred and supported well by dashboards
)


# Counters: Values that only go up
GA_EVALUATIONS_TOTAL = Counter(
    'prompthelix_ga_evaluations_total',
    'Total number of chromosome fitness evaluations performed'
)

GA_SUCCESSFUL_EVALUATIONS_TOTAL = Counter(
    'prompthelix_ga_successful_evaluations_total',
    'Total number of successful chromosome fitness evaluations'
)

GA_FAILED_EVALUATIONS_TOTAL = Counter(
    'prompthelix_ga_failed_evaluations_total',
    'Total number of failed chromosome fitness evaluations (e.g. timeout, LLM error)'
)

GA_MUTATIONS_TOTAL = Counter(
    'prompthelix_ga_mutations_total',
    'Total number of mutation operations performed on chromosomes'
)

GA_CROSSOVERS_TOTAL = Counter(
    'prompthelix_ga_crossovers_total',
    'Total number of crossover operations performed between chromosomes'
)

GA_ELITE_INDIVIDUALS_TOTAL = Counter(
    'prompthelix_ga_elite_individuals_carried_over_total',
    'Total number of elite individuals carried over to the next generation'
)

# Example Histogram (Optional - can be added if detailed timing/score distributions are needed)
# GA_FITNESS_SCORE_DISTRIBUTION = Histogram(
#     'prompthelix_ga_fitness_score_distribution',
#     'Distribution of fitness scores in GA generations',
#     buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Example buckets
# )

# TODO: Add more metrics as needed, e.g., specific mutation strategy counts,
# LLM call latencies (if instrumented), error counts by type, etc.

# --- Application Level Metrics (Example) ---
# HTTP_REQUESTS_TOTAL = Counter(
# 'prompthelix_http_requests_total',
# 'Total number of HTTP requests made to the application',
# ['method', 'endpoint', 'status_code']
# )

# --- Agent Specific Metrics (Example - could be in agent modules) ---
# AGENT_MESSAGES_PROCESSED_TOTAL = Counter(
# 'prompthelix_agent_messages_processed_total',
# 'Total messages processed by an agent',
# ['agent_id', 'message_type']
# )

def initialize_ga_metrics():
    """
    Initializes GA-specific metrics to a starting state (e.g., 0 or specific values).
    Useful if the application might restart and you want to ensure gauges are reset
    or set to a known state if they are not already registered (Prometheus client handles
    reregistration gracefully but not necessarily resetting values across restarts unless designed so).
    For most counters, this is not strictly needed as they start at 0.
    For gauges, setting them to 0 or a default "not running" state can be useful.
    """
    GA_CURRENT_GENERATION.set(0)
    GA_BEST_FITNESS.set(0)
    GA_AVG_FITNESS.set(0)
    GA_MEDIAN_FITNESS.set(0)
    GA_MIN_FITNESS.set(0)
    GA_STD_DEV_FITNESS.set(0)
    GA_POPULATION_SIZE.set(0)
    GA_RUNNING_STATUS.set(0) # 0 = not running

    # Counters effectively start at 0, but explicit calls don't hurt if one wanted to ensure.
    # However, .inc() is the typical usage. For a true reset (if ever needed, not typical for counters):
    # prometheus_client.REGISTRY.unregister(GA_EVALUATIONS_TOTAL)
    # GA_EVALUATIONS_TOTAL = Counter(...) and then re-register or let it auto-register.
    # This is complex. Usually, counters just accumulate.

# Note: Calling initialize_ga_metrics() could be done at application startup,
# particularly before a GA run might begin if there's a clear "start" point.
# If the GA runs continuously or on demand, these will be updated as it runs.
# For now, we'll rely on them being set during the GA lifecycle.
# If the app restarts, gauges will reset to 0 by default if not set.
# Counters will effectively restart from 0 for the new process.
"""old
This block previously exposed a `/metrics` endpoint and helper
functions for updating GA metrics. The implementation was replaced by
`initialize_ga_metrics` and Prometheus gauges above and is retained
only for reference. All former helper functions such as
``update_ga_metrics`` and ``record_generation`` have been removed as
they are no longer used.
"""
