import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from generate_individual import generate_gene
from constants import DAYS, HOURS, data, section_data
from tqdm import tqdm

def plot_section_timetable(gene, section_name):
    timetable = gene[section_name]
    
    # Increased the page size
    fig, ax = plt.subplots(figsize=(15, 18))  # Larger page size
    
    ax.set_title(f"Timetable for Section {section_name}", fontsize=14)
    table_data = [
        [
            f"{timetable[day][hour][1]}\n{timetable[day][hour][0]}" if timetable[day][hour] else ""
            for hour in range(HOURS)
        ]
        for day in range(DAYS)
    ]
    
    ax.axis('off')
    table = ax.table(
        cellText=table_data,
        rowLabels=[f"Day {i}" for i in range(DAYS)],
        colLabels=[f"Hour {i}" for i in range(HOURS)],
        loc='center',
        cellLoc='center'
    )
    table.scale(1.5, 1.5)
    
    staff_table_data = []
    for item in data:
        if section_name in item["sections"] and item["block"] is None:
            subject_codes = ", ".join(item["subjects"]) if item["subjects"] else "N/A"
            staff_names_str = ", ".join(item["staffs"]) if item["staffs"] else "N/A"
            theory_str = item["theory"] if item["theory"] is not None else "N/A"
            lab_str = item["lab"] if item["lab"] is not None else "N/A"
            staff_table_data.append([item["id"], subject_codes, staff_names_str, theory_str, lab_str])
    
    # Adjusted the staff table bbox for full width and more height
    staff_table = ax.table(
        cellText=staff_table_data,
        colLabels=["ID", "Subject Codes", "Staff Names", "Theory", "Lab"],
        loc='bottom',
        cellLoc='left',
        bbox=[0.1, -0.8, 0.8, 0.4]  # Full width, more height
    )
    staff_table.auto_set_column_width([0, 1, 2, 3, 4])
    staff_table.scale(1.5, 1.5)
    
    plt.tight_layout()
    return fig

def plot_all_timetables():
    gene = generate_gene()
    print("Printing timetables...")
    with PdfPages("timetables.pdf") as pdf:
        for section in tqdm(section_data["section"], desc="Generating plots"):
            fig = plot_section_timetable(gene, section)
            pdf.savefig(fig)
            plt.close(fig)
    print("All timetables successfully saved to 'timetables.pdf'")

if __name__ == "__main__":
    plot_all_timetables()
