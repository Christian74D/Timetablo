from core.crossover_functions import crossover_random, crossover_biological, crossover_graph_based
import numpy as np

use_multithreading = True
max_generations = 10
runs_per_setting = 6
mr = 0.02

tuning_param = "mr_tuning"
tuning_values = ["const"]

#mr tuning
tuning_dict = {
    "const": [mr] * max_generations,
    "cyclic": np.linspace(0.015, 0.025, 5).tolist() * max_generations,
    "linear": np.linspace(0.02, 1, max_generations).tolist(),

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
