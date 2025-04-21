import pandas as pd
import pickle
import os
from reportlab.lib import colors

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "..", "data")

section_data = pd.read_excel(os.path.join(data_dir, "sorted_section_codes.xlsx"))
staff_data = pd.read_excel(os.path.join(data_dir, "sorted_staff_codes.xlsx"))
subject_data = pd.read_excel(os.path.join(data_dir, "sorted_subject_codes.xlsx"))

SECTIONS = len(section_data["section"].unique())
DAYS = 5
HOURS = 8

with open(os.path.join(data_dir, "timetable_data.pkl"), "rb") as f:
    data = pickle.load(f)

#keeping blocked periods at top and others sorted by number of sections involved
data.sort(key=lambda x: (x["block"] is None, -len(x["sections"])), reverse=False)
data_lookup = {item['id']: item for item in data}


multisec_color = colors.violet
blocked_color = colors.lightgreen

allowed_lab_configs = []