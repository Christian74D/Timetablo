from data.theory_lab_splitter import theory_lab_split
from data.generate_lunches import generate_lunches
from data.lab_allocator import lab_allocator
from data.data_formatter import format_timetable_data
from data.enocode_data import encoded_data
import pandas as pd

def process_data():
    theory_lab_split('data/data.xlsx', 'data/split_data.xlsx')
    encoded_data('data/data.xlsx', 'data/encoded_data.xlsx', 'data/section_codes.xlsx', 'data/staff_codes.xlsx', 'data/subject_codes.xlsx')
    generate_lunches('data/split_data.xlsx', 'data/section_codes.xlsx', 'data/data_with_lunch.xlsx')
    format_timetable_data('data/data_with_lunch.xlsx', 'data/timetable_data_without_labs.pkl')
    data = lab_allocator('data/timetable_data_without_labs.pkl', 'data/section_codes.xlsx')
    encoded_df, section_map, subject_map, staff_map = encoded_data('data/data_lunch_lab_allocated.xlsx', 'data/encoded_data.xlsx', 'data/section_codes.xlsx', 'data/staff_codes.xlsx', 'data/subject_codes.xlsx')
    return data, encoded_df, section_map, subject_map, staff_map