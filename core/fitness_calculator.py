from collections import defaultdict
from core.constants import DAYS, HOURS
from core.config import gap_penalty

def build_data_by_id(data):
    return {entry['id']: entry for entry in data}

def count_teacher_conflicts(gene, data_by_id):
    conflicts = 0
    sections = list(gene.keys())

    for day in range(DAYS):
        for period in range(HOURS):
            staff_to_sections = defaultdict(set)

            for section in sections:
                slot = gene[section][day][period]
                if slot is None:
                    continue

                entry_id, _ = slot
                curr = data_by_id.get(entry_id, {})
                staff_list = curr.get('staffs', [])
                for staff in staff_list:
                    if staff == "nan":
                        continue
                    staff_to_sections[staff].add(entry_id)

            for staff, sec_list in staff_to_sections.items():
                if len(sec_list) > 1:
                    #print(f"Conflict detected for staff {staff} on day {day}, period {period}: {sec_list}")
                    conflicts += len(sec_list) - 1  # One is OK, rest are conflicts
            
            #print(f"Day {day}, Period {period}: Staff to Sections: {staff_to_sections}")

    return conflicts

def count_idle_gaps(gene):
    gaps = 0

    for section, schedule in gene.items():
        for day in range(DAYS):
            day_schedule = schedule[day]
            first = None
            last = None

            # Find the first and last non-empty periods
            for i in range(HOURS):
                if day_schedule[i] is not None:
                    if first is None:
                        first = i
                    last = i

            # Count gaps between first and last
            if first is not None and last is not None:
                for i in range(first, last + 1):
                    if day_schedule[i] is None:
                        gaps += 1

    return gaps

def fitness(gene, data, gap_penalty=gap_penalty):
    data_by_id = build_data_by_id(data)
    teacher_conflicts = count_teacher_conflicts(gene, data_by_id)
    idle_gaps = count_idle_gaps(gene)

    return teacher_conflicts + gap_penalty * idle_gaps

