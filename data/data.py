import pandas as pd
import pickle

def process_timetable_data(input_file, output_pkl='timetable_data.pkl'):
    df = pd.read_excel(input_file)

    data = []
    for _, row in df.iterrows():
        block_raw = str(row['block'])
        if pd.isna(row['block']) or block_raw.strip() == '' or block_raw.lower() == 'nan':
            block = None
        else:
            block = [
                tuple(map(int, item.strip().strip('()').split(',')))
                for item in block_raw.split('),') if item.strip()
            ]

        def safe_int(val):
            return int(val) if pd.notna(val) and str(val).strip() != '' else 0

        record = {
            'id': int(row['id']),
            'sections': [s.strip() for s in str(row['sections']).split(',')],
            'subjects': [s.strip() for s in str(row['subjects']).split(',')],
            'staffs': [s.strip() for s in str(row['staffs']).split(',')],
            'theory': safe_int(row['theory']),
            'lab': safe_int(row['lab']),
            'block': block
        }
        data.append(record)

    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)

    return data

# Example usage
if __name__ == '__main__':
    data = process_timetable_data('sastra_data.xlsx')
    print("Data processed and saved to 'timetable_data.pkl'")  
