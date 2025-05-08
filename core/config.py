from core.crossover_functions import crossover_random, crossover_biological, crossover_graph_based

use_multithreading = True
max_generations = 40
runs_per_setting = 8
mr = 0.06

tuning_param = "mutation_rate"
tuning_values = [0.01, 0.02, 0.03]

#mr tuning
tuning_dict = {
    "const": [mr] * max_generations,
    "fuzzy_tuned_cyclic": [0.06 * (1 + 0.5 * (-1) ** i) for i in range(max_generations)],
}

fixed_params = {
    "max_generations": max_generations,
    "population_size": 100,
    "mutation_rate": mr,
    "k": 3,
    "elitism_ratio": 0.1,
    "crossover": crossover_graph_based,
    "replacement_ratio": 0.1,
    "mr_tuning": "None"
}
