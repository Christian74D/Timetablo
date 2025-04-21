import pandas as pd
import random
from math import ceil
from collections import defaultdict

days = 5
lunch_hours = [4, 5]

section_df = pd.read_excel('data/sorted_section_codes.xlsx')
sections = section_df['section'].tolist()

max_lunch_per_hour = ceil(days / len(lunch_hours))

lunch_allocations = {section: [None] * days for section in sections}
section_hour_counts = {section: defaultdict(int) for section in sections}

for day in range(days):
    random.shuffle(sections)
    for section in sections:
        random.shuffle(lunch_hours)
        for hour in lunch_hours:
            if section_hour_counts[section][hour] < max_lunch_per_hour:
                lunch_allocations[section][day] = hour
                section_hour_counts[section][hour] += 1
                break

for day in range(days):
    print(f"\nDay {day + 1}:")
    for hour in lunch_hours:
        secs = [s for s in sections if lunch_allocations[s][day] == hour]
        print(f"  Lunch Hour {hour}: {', '.join(secs) if secs else 'None'}")
