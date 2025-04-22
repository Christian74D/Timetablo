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

from core.time import calculate_time, save_time_to_file
from core.constants import data_path
from core.plot_student_timetable import plot_timetables_for_all_sections
from core.generate_individual import generate_gene
from data.data_processor import process_data


def main():
    encoded_df, section_map, subject_map, staff_map = process_data(data_path)
    data, data_lookup = pickle.load(open(data_path, 'rb'))
    gene = generate_gene(data, section_map)
    plot_timetables_for_all_sections(gene, section_map, data, data_lookup)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    time_str = calculate_time(start, end)
    save_time_to_file(time_str)






