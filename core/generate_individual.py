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

    # Loop through the data items (subjects or classes)
    for item in data:
        assigned = False
        attempts = 0

        if item["block"] is not None:  # If block is not None, assign directly
            clashes = False
            for period in item["block"]:
                day, hour = period
                # Adjust for 1-based indexing (subtract 1)
                day -= 1
                hour -= 1
                # Check if the slot is available for all sections
                if all((day, hour) in free_slots[sec] for sec in item["sections"]):
                    for sec in item["sections"]:
                        gene[sec][day][hour] = (item["id"], item["subjects"])
                        free_slots[sec].remove((day, hour))
                else:
                    clashes = True
                    print(f"Clash detected for item {item['id']} at period {period}.")
                    break  # Stop if any clash occurs

            if not clashes:
                item["period"] = item["block"]
                assigned = True
        else:
            # Random allocation if block is None
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
