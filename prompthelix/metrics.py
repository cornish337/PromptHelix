from prometheus_client import Gauge, Counter

# GA metrics

ga_current_generation = Gauge("ga_current_generation", "Current GA generation")

ga_best_fitness = Gauge("ga_best_fitness", "Best fitness score this generation")

ga_population_size = Gauge("ga_population_size", "Population size after generation")

ga_evaluations_total = Counter("ga_evaluations_total", "Total chromosome evaluations")

def record_generation(generation: int, population_size: int, best_fitness: float, evaluated_count: int):
    """Update Prometheus metrics for a GA generation."""
    ga_current_generation.set(generation)
    ga_population_size.set(population_size)
    ga_best_fitness.set(best_fitness)
    ga_evaluations_total.inc(evaluated_count)
