from data.data_processor import process_data
from core.generate_individual import generate_gene
from core.constants import ImpossibleAllocationError, allocation_attempts, heuristic_trials, heuristic_samples

from copy import deepcopy
import pickle
import random
from tqdm import tqdm

def score(seed):
    random.seed(seed)
    data, encoded_df, section_map, subject_map, staff_map = process_data()
    score = 0
    for _ in range(heuristic_trials):
        data_copy = deepcopy(data)  
        try:              
            h = generate_gene(data_copy, section_map, heuristic=True)
            score += h
        except ImpossibleAllocationError:
            score += allocation_attempts + 1 #if allocation fails, add a large number to the score
    #print(f"Seed: {seed}, Score: {score}")
    return score

def generate_heuristic_allocation():
    min_score = float('inf')
    best_seed = 0
    for seed in tqdm(range(heuristic_samples), desc="Finding best seed"):
        random.seed(seed)
        sscore = score(seed)
        if sscore < min_score:
            best_seed = seed
            min_score = sscore
    print(f"Best seed: {best_seed} with score: {score(best_seed)}")
    print("Heuristic Base Allocation Completed")

    random.seed(best_seed)
    return process_data()