import pandas as pd
import pickle
import os
from reportlab.lib import colors

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "..", "data")

DAYS = 5
HOURS = 8

multisec_color = colors.violet
blocked_color = colors.lightgreen

data_path = 'data/timetable_data.pkl'

days = 5
lunch_hours = [4, 5]

allowed_lab_configs = {2: [(1, 2), (3, 4), (5, 6), (7, 8)]}
