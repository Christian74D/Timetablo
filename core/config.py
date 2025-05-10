import numpy as np


use_multithreading = True
max_generations = 50
runs_per_setting = 10
mr = 0.02

tuning_param = "mutation_rate"
tuning_values = [0.001, 0.01, 0.02, 0.03, 1.0]

#mr tuning
tuning_dict = {
    "const": [mr] * max_generations,
    "cyclic": np.linspace(0.015, 0.025, 5).tolist() * max_generations,
    "linear": np.linspace(0.02, 1, max_generations).tolist(),
}

fixed_params = {
    "max_generations": max_generations,
    "mutation": "mutate_gene",
    "population_size": 100,
    "mutation_rate": mr,
    "k": 1,
    "elitism_ratio": 0.1,
    "crossover": "random",
    "replacement_ratio": 0.1,
    "mr_tuning": "None"
}
