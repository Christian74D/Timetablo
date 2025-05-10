from core.crossover_functions import crossover_random, crossover_biological, crossover_graph_based

use_multithreading = True
max_generations = 40
runs_per_setting = 1
mr = 0.02

tuning_param = "mutation_rate"
tuning_values = [.02, .04, .06, .08, .1, .2, .4, .8, 1.0]

#mr tuning
tuning_dict = {
    "const": [mr] * max_generations,
    "cyclic": [mr*.9, mr, mr*1.1] * (max_generations // 2)
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
