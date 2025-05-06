import pickle
import random
import pandas as pd
from core.constants import DAYS, HOURS, allowed_lab_configs, data_path

def load_sections(file_path):
    section_df = pd.read_excel(file_path)
    return section_df['section'].tolist()

def allocate_labs(gene, data, section_data):
    free_slots = {
        sec: {(day, hour) for day in range(DAYS) for hour in range(HOURS)}
        for sec in section_data
    }

    for sec in section_data:
        for day in range(DAYS):
            for hour in range(HOURS):
                if gene[sec][day][hour] is not None:
                    free_slots[sec].discard((day, hour))

    for item in data:
        lab_len = item["lab"]
        if lab_len <= 0:
            continue

        configs = allowed_lab_configs.get(lab_len, [])
        if not configs:
            print(f"No allowed config for lab length {lab_len} in item {item['id']}")
            continue

        possible_blocks = [(day, [c-1 for c in cfg]) for day in range(DAYS) for cfg in configs]
        random.shuffle(possible_blocks)

        success = False
        for day, cfg in possible_blocks:
            if all(all((day, hour) in free_slots[sec] for sec in item["sections"]) for hour in cfg):
                for sec in item["sections"]:
                    for hour in cfg:
                        #print(f"Allocated lab {cfg} for item {item['id']} on day {day}")
                        gene[sec][day][hour] = (item["id"], item["subjects"])
                        free_slots[sec] = {slot for slot in free_slots[sec] if slot[0] != day} #remove all lab slots from that day (one lab per day)

                item["block"] = [(day, hour) for hour in cfg]  
                
                success = True
                break

        if not success:
            print(f"Failed to allocate lab for item {item['id']}")

    return gene

def lab_allocator(input_path, section_path):
    section_data = load_sections(section_path)    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Initialize gene structure
    gene = {
        sec: [[None for _ in range(HOURS)] for _ in range(DAYS)]
        for sec in section_data
    }

    # Fill gene with pre-existing blocks
    # Modify blocks in original data to use 0-based indexing
    for item in data:
        if "block" in item and item["block"]:
            item["block"] = [(day - 1, hour - 1) for day, hour in item["block"]]

    # Now fill gene without subtracting again
    for item in data:
        if "block" in item and item["block"]:
            for day, hour in item["block"]:
                for sec in item["sections"]:
                    gene[sec][day][hour] = (item["id"], item["subjects"])


    # Run lab allocation
    gene = allocate_labs(gene, data, section_data)

    # Save updated data
    

    df = pd.DataFrame(data)
    # Convert list columns to comma-separated strings
    
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

    # Now export to Excel
    df.to_excel("data/data_lunch_lab_allocated.xlsx", index=False)

    #print(f"Lab allocation complete. Saved to: {output_path}")
    #print(f"Data saved to: data/data_lunch_lab_allocated.xlsx")

    with open(data_path, 'wb') as f:
        pickle.dump((data), f)

    return data