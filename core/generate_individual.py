import random
from constants import HOURS, SECTIONS, DAYS, data, section_data

def generate_gene():
    gene = {
        sec: [[None for _ in range(HOURS)] for _ in range(DAYS)]
        for sec in section_data["section"]
    }

    free_slots = {
        sec: {(day, hour) for day in range(DAYS) for hour in range(HOURS)}
        for sec in section_data["section"]
    }

    for item in data:
        assigned = False
        attempts = 0
        while not assigned and attempts < 100:
            day = random.randint(0, DAYS - 1)
            hour = random.randint(0, HOURS - 1)
            if all((day, hour) in free_slots[sec] for sec in item["sections"]):
                for sec in item["sections"]:
                    gene[sec][day][hour] = (item["id"], item["subjects"])
                    free_slots[sec].remove((day, hour))
                item["period"] = (day, hour)
                assigned = True
            attempts += 1
        if not assigned:
            print("Assignment failed:", item)

    return gene

