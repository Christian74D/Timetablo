import pandas as pd
import os

base = os.path.dirname(os.path.abspath(__file__))

df = pd.read_excel(os.path.join(base, 'sastra_data.xlsx'))

subject_codes = []
for _, row in df.iterrows():
    subject_codes.extend([s.strip() for s in str(row['subjects']).split(',')])
sorted_subject_codes = sorted(set(subject_codes))
generated_subject_codes = [f"SUB_{str(i).zfill(3)}" for i in range(len(sorted_subject_codes))]
subject_df = pd.DataFrame({'subject_code': sorted_subject_codes, 'generated_code': generated_subject_codes})
subject_df.to_excel(os.path.join(base, 'sorted_subject_codes.xlsx'), index=False)

staff_names = []
for _, row in df.iterrows():
    staff_names.extend([s.strip() for s in str(row['staffs']).split(',')])
sorted_staff_names = sorted(set(staff_names))
generated_staff_codes = [f"FAC_{str(i).zfill(3)}" for i in range(len(sorted_staff_names))]
staff_df = pd.DataFrame({'staff_name': sorted_staff_names, 'generated_code': generated_staff_codes})
staff_df.to_excel(os.path.join(base, 'sorted_staff_codes.xlsx'), index=False)

def section_sort_key(section):
    parts = section.split('_')
    try:
        sem = int(parts[0])
        sec = parts[-1]
    except (IndexError, ValueError):
        sem, sec = 0, ''
    return (sem, sec)

sections = []
for _, row in df.iterrows():
    sections.extend([s.strip() for s in str(row['sections']).split(',')])
sorted_sections = sorted(set(sections), key=section_sort_key)
generated_section_codes = [f"SEC_{str(i).zfill(3)}" for i in range(len(sorted_sections))]
section_df = pd.DataFrame({'section': sorted_sections, 'generated_code': generated_section_codes})
section_df.to_excel(os.path.join(base, 'sorted_section_codes.xlsx'), index=False)

section_map = section_df.set_index('section')['generated_code'].to_dict()
subject_map = subject_df.set_index('subject_code')['generated_code'].to_dict()
staff_map = staff_df.set_index('staff_name')['generated_code'].to_dict()

encoded_rows = []
for _, row in df.iterrows():
    sections = [section_map.get(s.strip(), s.strip()) for s in str(row['sections']).split(',')]
    subjects = [subject_map.get(s.strip(), s.strip()) for s in str(row['subjects']).split(',')]
    staffs = [staff_map.get(s.strip(), s.strip()) for s in str(row['staffs']).split(',')]
    encoded_row = row.copy()
    encoded_row['sections'] = ', '.join(sections)
    encoded_row['subjects'] = ', '.join(subjects)
    encoded_row['staffs'] = ', '.join(staffs)
    encoded_rows.append(encoded_row)

encoded_df = pd.DataFrame(encoded_rows)
encoded_df.to_excel(os.path.join(base, 'sastra_data_encoded.xlsx'), index=False)

print("Data encoded and saved.")
