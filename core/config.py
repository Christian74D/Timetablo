from core.crossover_functions import crossover_random, crossover_biological, crossover_graph_based

use_multithreading = False
max_generations = 40
runs_per_setting = 1
mr = 0.06

tuning_param = "crossover"
tuning_values = ["graph_based"]

#mr tuning
tuning_dict = {
    "const": [mr] * max_generations,
    "fuzzy_tuned_cyclic": [0.06 * (1 + 0.5 * (-1) ** i) for i in range(max_generations)],
}

fixed_params = {
    "max_generations": max_generations,
    "population_size": 100,
    "mutation_rate": mr,
    "k": 30,
    "elitism_ratio": 0.1,
    "crossover": crossover_random,
    "replacement_ratio": 0.1,
    "mr_tuning": "const"
}
