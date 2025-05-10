import string
import random
import time
import pickle
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
from tqdm import tqdm
import pickle

from core.time import calculate_time, save_time_to_file
from core.constants import data_path
from core.plot_student_timetable import plot_timetables_for_all_sections
from core.generate_individual import generate_gene
from core.mutate import mutate_gene
from core.fitness_calculator import fitness
from core.crossover_functions import crossover_random, crossover_biological, crossover_graph_based


import time
import numpy as np
from core.EA import EA  # Your EA class
from core.plot_student_timetable import plot_timetables_for_all_sections

def run_ea(fixed_params, tuning_param, tuning_value, run_idx, max_generations, runs_per_setting):
    start_time = time.time()
    fitness_sums = np.zeros(max_generations)

    ea = EA(**fixed_params, **{tuning_param: tuning_value})
    sol, fitness, fitness_history = ea.run() 

    # Create a valid filename string
    filename = f"timetable_fitness_{int(fitness)}_{tuning_param}_{tuning_value}_run_{run_idx}.pdf"
    plot_timetables_for_all_sections(sol, filename)

    fitness_sums[:len(fitness_history)] += fitness_history
    elapsed = time.time() - start_time

    return [
        (run_idx, tuning_value, gen, fitns, elapsed)
        for gen, fitns in enumerate(fitness_history)
    ], fitness_sums, elapsed



"""
    with open("data/heuristic_allocation.pkl", "rb") as f:
        data, encoded_df, section_map, subject_map, staff_map = pickle.load(f)
    
    gene1 = generate_gene(data, section_map)
    plot_timetables_for_all_sections(gene1, section_map, data, "tt.pdf")
    print(fitness(gene1, data))
    
    gene11 = mutate_gene(data, gene1)
    plot_timetables_for_all_sections(gene11, section_map, data, "tt_mutated.pdf")
    print(fitness(gene11, data))

    gene2 = generate_gene(data, section_map)
    plot_timetables_for_all_sections(gene2, section_map, data, "tt2.pdf")
    print(fitness(gene2, data))

    child1, child2 = crossover_graph_based(gene1, gene2)
    plot_timetables_for_all_sections(child1, section_map, data, "tt_child1.pdf")
    plot_timetables_for_all_sections(child2, section_map, data, "tt_child2.pdf")
"""

