import random
from constants import HOURS, SECTIONS, DAYS, data, section_data

def generate_gene():
    # Initialize the gene and free_slots
    gene = {
        sec: [[None for _ in range(HOURS)] for _ in range(DAYS)]
        for sec in section_data["section"]
    }

    free_slots = {
        sec: {(day, hour) for day in range(DAYS) for hour in range(HOURS)}
        for sec in section_data["section"]
    }

    for item in data:
        if item["block"] is not None:  
            for period in item["block"]:
                day, hour = period
                day -= 1
                hour -= 1
                # Check if the slot is available for all sections
                if all((day, hour) in free_slots[sec] for sec in item["sections"]):
                    for sec in item["sections"]:
                        gene[sec][day][hour] = (item["id"], item["subjects"])
                        free_slots[sec].remove((day, hour))
                else:
                    print(f"Clash detected for item {item['id']} at period {period}.")
                    break  

            item["period"] = item["block"]

        else:
            theory, lab = item["theory"], item["lab"] 
            # Theory
            for _ in range(theory):
                for _ in range(100):
                    day = random.randint(0, DAYS - 1)
                    hour = random.randint(0, HOURS - 1)
                    if all((day, hour) in free_slots[sec] for sec in item["sections"]):
                        for sec in item["sections"]:
                            gene[sec][day][hour] = (item["id"], item["subjects"])
                            free_slots[sec].remove((day, hour))
                        item["period"] = (day, hour)
                        break
                else:
                    print("Assignment failed:", item, f"\nremaining assignments: theory {theory} lab {lab}")
                
    return gene
