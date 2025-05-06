from data.data_processor import process_data
from core.generate_individual import generate_gene
from core.constants import ImpossibleAllocationError, allocation_restarts, heuristic_trials

from copy import deepcopy
import pickle

def generate_heuristic_allocation():
    min_score = float('inf')
    best_gene = None
    for _ in range(heuristic_trials):
        try:
            data, encoded_df, section_map, subject_map, staff_map = process_data()
            score = 0
            for _ in range(allocation_restarts):
                data_copy = deepcopy(data)                
                h = generate_gene(data_copy, section_map, heuristic=True)
                score += h
            print(f"Score: {score}")
            if score < min_score:
                min_score = score
                best_gene = [data, encoded_df, section_map, subject_map, staff_map]
        except ImpossibleAllocationError:
            continue
    
    print("Heuristic Base Allocation Completed")
    return best_gene