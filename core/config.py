import numpy as np

input_path = 'data/data.xlsx' #'simulations/synthetic_data_20.xlsx'
use_multithreading_setup = False
use_multithreading_main = True
max_generations = 50
runs_per_setting = 4
mr = 0.02

tuning_param = "mutation_rate"
tuning_values = [0.005,0.01, 0.015, 0.02]

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
