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
def main():
    with open("data/heuristic_allocation.pkl", "rb") as f:
        data, encoded_df, section_map, subject_map, staff_map = pickle.load(f)
    
    data, gene = generate_gene(data, section_map)
    plot_timetables_for_all_sections(gene, section_map, data, "tt.pdf")
    print(fitness(gene, data))
    
    data, gene = mutate_gene(data, gene)
    plot_timetables_for_all_sections(gene, section_map, data, "tt_mutated.pdf")
    print(fitness(gene, data))
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    time_str = calculate_time(start, end)
    save_time_to_file(time_str)






