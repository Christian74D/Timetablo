import pandas as pd
import random
import string

# === MASTER CONTROL ===
NUM_SECTIONS = 20  # 👈 Scale this to adjust everything
     # 👈 Number of clusters for shared multi-section subject

# === RANDOMIZED DERIVED CONSTANTS ===
NUM_ENTRIES = int(NUM_SECTIONS * random.uniform(2, 4))
NUM_SUBJECTS = int(NUM_SECTIONS * random.uniform(2, 4))
NUM_STAFF = int(NUM_SECTIONS * random.uniform(2, 4))
LAB_RATIO = random.uniform(0.1, 0.2)
BLOCKED_ENTRY_COUNT = max(1, int(NUM_SECTIONS / 10) + random.choice([0, 1]))
BLOCKED_SECTIONS_PER_ENTRY = min(NUM_SECTIONS, random.randint(6, 12))
K_CLUSTERS = NUM_SECTIONS // 3

# === ENTRY COMPOSITION LIMITS ===
SUBJ_COUNT_RANGE = (1, 2)
STAFF_COUNT_RANGE = (1, 4)

# === ANONYMIZED IDENTIFIERS ===
def excel_style_labels(n):
    labels = []
    i = 1
    while len(labels) < n:
        label = ""
        x = i
        while x > 0:
            x, r = divmod(x - 1, 26)
            label = chr(65 + r) + label
        labels.append(label)
        i += 1
    return labels

def generate_section_codes():
    return [f"Sec {label}" for label in excel_style_labels(NUM_SECTIONS)]

def generate_subject_codes():
    return [f"XX{label}" for label in excel_style_labels(NUM_SUBJECTS)]

def generate_staff_list():
    return [f"Prof {label}" for label in excel_style_labels(NUM_STAFF)]

# === ENTRY GENERATORS ===
def generate_entries(section_pool, subject_pool, staff_pool):
    entries = []
    used_ids = 1
    cluster_size = NUM_SECTIONS // K_CLUSTERS
    shared_subjects = random.sample(subject_pool, K_CLUSTERS)
    clusters = [section_pool[i * cluster_size:(i + 1) * cluster_size] for i in range(K_CLUSTERS)]

    # Generate one shared-subject multi-section entry per cluster
    for idx, cluster_sections in enumerate(clusters):
        subj = shared_subjects[idx]
        staffs = random.sample(staff_pool, random.randint(*STAFF_COUNT_RANGE))
        entries.append({
            "id": used_ids,
            "sections": ", ".join(cluster_sections),
            "subjects": subj,
            "staffs": ", ".join(staffs),
            "theory": random.randint(2, 4),
            "lab": "",
            "block": ""
        })
        used_ids += 1

    # Fill remaining with mostly single-section entries
    while used_ids <= NUM_ENTRIES:
        is_lab = random.random() < LAB_RATIO
        sec_count = 1  # Only 1 section unless it's in a cluster above

        subj_count = random.randint(*SUBJ_COUNT_RANGE)
        staff_count = random.randint(*STAFF_COUNT_RANGE)

        sections = random.sample(section_pool, sec_count)
        subjects = random.sample([s for s in subject_pool if s not in shared_subjects], subj_count)
        staffs = random.sample(staff_pool, staff_count)

        entries.append({
            "id": used_ids,
            "sections": ", ".join(sections),
            "subjects": ", ".join(subjects),
            "staffs": ", ".join(staffs),
            "theory": random.randint(2, 4) if not is_lab else "",
            "lab": 2 if is_lab else "",
            "block": ""
        })
        used_ids += 1
    return entries

def generate_blocked_entries(section_pool, start_id):
    blocked_entries = []
    allowed_hours = [1, 2, 7, 8]
    used_slots = set()

    for i in range(start_id, start_id + BLOCKED_ENTRY_COUNT):
        sections = random.sample(section_pool, BLOCKED_SECTIONS_PER_ENTRY)
        block_slots = set()

        while len(block_slots) < 4:
            slot = (random.randint(1, 5), random.choice(allowed_hours))
            if slot not in used_slots:
                used_slots.add(slot)
                block_slots.add(slot)

        blocked_entries.append({
            "id": i,
            "sections": ", ".join(sections),
            "subjects": f"BLOCKED_{i}",
            "staffs": "",
            "theory": "",
            "lab": "",
            "block": ", ".join(f"({d}, {h})" for d, h in sorted(block_slots))
        })
    return blocked_entries

# === GENERATE & SAVE ===
section_pool = generate_section_codes()
subject_pool = generate_subject_codes()
staff_pool = generate_staff_list()

normal_entries = generate_entries(section_pool, subject_pool, staff_pool)
blocked_entries = generate_blocked_entries(section_pool, NUM_ENTRIES + 1)

all_entries = normal_entries + blocked_entries
df = pd.DataFrame(all_entries)

filename = f"simulations/synthetic_data_{NUM_SECTIONS}.xlsx"
df.to_excel(filename, index=False)
print(f"✅ Anonymized data generated → Saved to '{filename}'")
