import random
from core.constants import HOURS, DAYS

def generate_gene(data, section_data):
    SECTIONS = len(section_data)
    # Initialize the gene and free_slots
    gene = {
        sec: [[None for _ in range(HOURS)] for _ in range(DAYS)]
        for sec in section_data
    }

    free_slots = {
        sec: {(day, hour) for day in range(DAYS) for hour in range(HOURS)}
        for sec in section_data
    }
   
    
    for item in data:
        if item["block"] is not None:
            for period in item["block"]:
                day, hour = period
                for sec in item["sections"]:
                    if gene[sec][day][hour]:
                        print(f"Conflict in lab allocation for item {item['id']} on day {day} hour {hour}")
                        print("Current gene:", gene[sec][day][hour])
                        print("Replacing with:", (item["id"], item["subjects"]))
                    gene[sec][day][hour] = (item["id"], item["subjects"])
                    free_slots[sec].remove((day, hour))
               
            item["period"] = item["block"]

    for item in data:
        if item["block"] is None:
            theory, lab = item["theory"], item["lab"]
            sections = item["sections"]

            # Build a map of available hours per day (intersection across all sections)
            free_slots_per_day = {
                day: [
                    hour for hour in range(HOURS)
                    if all((day, hour) in free_slots[sec] for sec in sections)
                ]
                for day in range(DAYS)
            }

            # Get all days that have at least one common free slot across sections
            valid_days = [day for day, hours in free_slots_per_day.items() if hours]
            
            if len(valid_days) < theory:
                print("Not enough valid days for assignment:", item)
                continue

            # Randomly select distinct days equal to the theory hours needed
            chosen_days = random.sample(valid_days, theory)

            assigned_periods = []

            for day in chosen_days:
                available_hours = free_slots_per_day[day]
                if not available_hours:
                    print(f"No available hour on day {day} for item {item}")
                    continue

                hour = random.choice(available_hours)
                for sec in sections:
                    gene[sec][day][hour] = (item["id"], item["subjects"])
                    free_slots[sec].remove((day, hour))

                assigned_periods.append((day, hour))

            item["period"] = assigned_periods
            
    return gene
