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
