import random
from copy import deepcopy
from core.constants import HOURS, DAYS

def mutate_gene(data, gene, mutation_rate):
    gene = deepcopy(gene)  # Create a deep copy of the gene structure
    for item in data:
        if item.get("block") is not None or item["theory"] == 0 or random.random() > mutation_rate:
            continue

        subject_id = item["id"]
        sections = item["sections"]
        current_periods = item.get("period", [])

        # Gather all currently used days for this subject ID
        used_days = {day for (day, _) in current_periods}
        #possible_days = [d for d in range(DAYS) if d not in used_days or d in used_days]

        # Recollect current slots to free them temporarily
        for (day, hour) in current_periods:
            for sec in sections:
                gene[sec][day][hour] = None

        # Build a map of new available hours for each day across all sections
        free_slots_per_day = {
            day: [
                hour for hour in range(HOURS)
                if all(gene[sec][day][hour] is None for sec in sections)
            ]
            for day in range(DAYS)
        }

        # Include original periods as valid slots (reversible)
        #for day, hour in current_periods:
        #    if day in free_slots_per_day:
        #        free_slots_per_day[day].append(hour)

        # Filter out only days with at least one free hour
        valid_days = [d for d in free_slots_per_day if free_slots_per_day[d]]
        #print(used_days, valid_days)
        #if len(valid_days) < len(current_periods):
        #    # Reassign original periods if mutation fails
        #    for day, hour in current_periods:
        #        for sec in sections:
        #           gene[sec][day][hour] = (subject_id, item["subjects"])
        #   continue

        # Choose new days and hours
        chosen_days = random.sample(valid_days, len(current_periods))
        new_periods = []

        for day in chosen_days:
            hour = random.choice(free_slots_per_day[day])
            for sec in sections:
                gene[sec][day][hour] = (subject_id, item["subjects"])
            new_periods.append((day, hour))
        item["period"] = new_periods
    
    return gene
