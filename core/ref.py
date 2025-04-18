from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import string
from copy import deepcopy
import networkx as nx
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from concurrent.futures import ProcessPoolExecutor, as_completed

start = time.time()

DAYS = 5          
SECTIONS = 16     
NUM_SUBJECTS = 5  
MAX_SUBJECTS = 3  
MAX_HOURS = 6    
TEACHERS = int(np.ceil(SECTIONS * NUM_SUBJECTS / MAX_SUBJECTS))    # Min no of teachers required

random.seed(42)
SECTIONS_LIST = [f"Section {ch}" for ch in string.ascii_uppercase[:SECTIONS]]

STAFF = random.sample( [
    "Alice Brown", "Bob Carter", "Charlie Davis", "David Evans", "Emma Foster",
    "Frank Green", "Grace Harris", "Henry Irwin", "Isabella Jones", "Jack King",
    "Katherine Lewis", "Liam Morgan", "Mia Nelson", "Nathan Owens", "Olivia Parker",
    "Peter Quinn", "Quentin Rogers", "Rachel Smith", "Samuel Turner", "Tina Underwood",
    "Umar Vance", "Victor White", "Wendy Xavier", "Xavier Young", "Yvonne Zane",
    "Zach Adams", "Adam Brooks", "Bella Carter", "Carl Daniels", "Diana Elliott",
    "Edward Fisher", "Fiona Grant", "George Hall", "Hannah Ingram", "Ian Jackson",
    "Julia Knight", "Kevin Logan", "Laura Mitchell", "Michael Norris", "Nancy Oliver",
    "Oscar Peterson", "Paula Quinn", "Quentin Richards", "Rebecca Stone", "Stephen Taylor",
    "Thomas Upton", "Ursula Vaughn", "Vincent Williams", "Walter Xu", "Xavier York",
    "Yasmine Zeller", "Zachary Allen", "Andrew Benson", "Brenda Collins", "Cameron Doyle",
    "Denise Everett", "Ethan Ford", "Felicia Gonzalez", "Gordon Hunter", "Helen Isaacs",
    "Isaac Jenkins", "Jessica Knight", "Kyle Lambert", "Lydia Martin", "Mason Nichols",
    "Natalie Ortiz", "Owen Patterson", "Penelope Quinn", "Quincy Russell", "Riley Stevens",
    "Sophia Thompson", "Tyler Underwood", "Uma Vaughn", "Victor Walters", "William Xavier",
    "Xena Young", "Yuri Zimmerman", "Zane Adams", "Abigail Brooks", "Benjamin Cooper",
    "Catherine Dawson", "Derek Ellis", "Eleanor Fisher", "Frederick Grant", "Gabriella Harris",
    "Harvey Ingram", "Isla Johnson", "Jonathan King", "Kaitlyn Lee", "Landon Mitchell",
    "Madeline Norton", "Noah Owen", "Olga Peterson", "Patrick Quinn", "Quinn Richards",
    "Raymond Smith", "Scarlett Turner", "Travis Underwood", "Ulysses Vaughn", "Violet Williams",
    "Wesley Xavier", "Xander Young", "Yasmine Zeller", "Zion Adams"
], TEACHERS)

ALL_SUBJECTS = [
    "Data Structures", "Algorithms", "Operating Systems", "Computer Networks",
    "Database Management Systems", "Artificial Intelligence", "Machine Learning",
    "Deep Learning", "Cyber Security", "Web Development", "Software Engineering",
    "Computer Graphics", "Distributed Systems", "Cloud Computing",
    "Internet of Things", "Human-Computer Interaction", "Embedded Systems",
    "Computational Theory", "Blockchain Technology", "Quantum Computing"
]

# Random subject allotment simulation
SUBJECTS = random.sample(ALL_SUBJECTS, NUM_SUBJECTS)
SUBJECT_CODES = {sub: "".join(word[0] for word in sub.split()).upper() for sub in SUBJECTS}
SUBJECT_HOURS = {sub: random.choice([3, 4]) for sub in SUBJECTS}
subject_colors = {SUBJECT_CODES[sub]: color for sub, color in zip(SUBJECTS, mcolors.TABLEAU_COLORS)}

teacher_load = {teacher: [] for teacher in STAFF}
subject_allocation = {}
section_teacher_map = {section: {} for section in SECTIONS_LIST}

for section in SECTIONS_LIST:
    for subject in SUBJECTS:
        available_teachers = [t for t in STAFF if len(teacher_load[t]) < MAX_SUBJECTS and t not in section_teacher_map[section].values()]
        if available_teachers:
            chosen_teacher = random.choice(available_teachers)
            teacher_load[chosen_teacher].append((section, subject))
            subject_allocation[(section, subject)] = chosen_teacher
            section_teacher_map[section][subject] = chosen_teacher

df = pd.DataFrame([(sec, sub, teacher) for (sec, sub), teacher in subject_allocation.items()],
                  columns=["Section", "Subject", "Teacher"])

table_data = df.pivot(index="Section", columns="Subject", values="Teacher")

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("tight")
ax.axis("off")
table = ax.table(cellText=table_data.values, colLabels=table_data.columns, rowLabels=table_data.index, cellLoc="center", loc="center")

table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([i for i in range(len(table_data.columns))])

plt.savefig("Staff_allocation.png")


def generate_gene():
    gene = [[[None for _ in range(SECTIONS)] for _ in range(MAX_HOURS)] for _ in range(DAYS)]

    for sec in range(SECTIONS):
        free_slots_per_day = {day: list(range(MAX_HOURS)) for day in range(DAYS)}

        for subject in SUBJECTS:
            hours_needed = SUBJECT_HOURS[subject]
            valid_days = [day for day in range(DAYS) if len(free_slots_per_day[day]) > 0]
            chosen_days = random.sample(valid_days, hours_needed)

            for day in chosen_days:
                hour = random.choice(free_slots_per_day[day])
                gene[day][hour][sec] = SUBJECT_CODES[subject]
                free_slots_per_day[day].remove(hour)

    return gene


def plot_timetable(gene):
    fig, axes = plt.subplots(1, SECTIONS, figsize=(SECTIONS * 4, 6), sharey=True, dpi = 300)
    if SECTIONS == 1:
        axes = [axes]  

    for sec in range(SECTIONS):
        data = np.full((DAYS, MAX_HOURS), "", dtype=object)
        colors = np.full((DAYS, MAX_HOURS), "white", dtype=object)

        for day in range(DAYS):
            for hour in range(MAX_HOURS):
                subject = gene[day][hour][sec]
                if subject:
                    data[day][hour] = subject
                    colors[day][hour] = subject_colors.get(subject, "gray")

        ax = axes[sec]
        ax.set_title(f"Section {chr(65 + sec)}")
        ax.set_xticks(np.arange(MAX_HOURS))
        ax.set_yticks(np.arange(DAYS))
        ax.set_xticklabels([f"Period {p+1}" for p in range(MAX_HOURS)])
        ax.set_yticklabels([f"Day {d+1}" for d in range(DAYS)])
        
        for i in range(DAYS):
            for j in range(MAX_HOURS):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colors[i, j], ec="black"))
                ax.text(j + 0.5, i + 0.5, data[i, j], ha="center", va="center", fontsize=8)

        ax.set_xlim(0, MAX_HOURS)
        ax.set_ylim(0, DAYS)
        ax.invert_yaxis() 

    plt.tight_layout()
    plt.savefig("Timetable.png")

import random
from copy import deepcopy

def mutate_gene(gene):
    mutated_gene = deepcopy(gene)

    for sec in range(SECTIONS):
        day1 = random.randint(0, DAYS - 1)
        hour1 = random.randint(0, MAX_HOURS - 1)
        sub = mutated_gene[day1][hour1][sec]

        used_days = {d for d in range(DAYS) if sub in [mutated_gene[d][h][sec] for h in range(MAX_HOURS)]}
        possible_days = [d for d in range(DAYS) if d not in used_days or d == day1]

        if not possible_days:
            continue

        day2 = random.choice(possible_days)
        hour2 = random.randint(0, MAX_HOURS - 1)

        mutated_gene[day1][hour1][sec], mutated_gene[day2][hour2][sec] = (
            mutated_gene[day2][hour2][sec], mutated_gene[day1][hour1][sec]
        )

    return mutated_gene


def generate_staff_dependency_graph():
    G = nx.Graph()  
    section_teachers = {sec: {} for sec in SECTIONS_LIST}
    for (section, subject), teacher in subject_allocation.items():
        subject_code = SUBJECT_CODES[subject]  
        if teacher not in section_teachers[section]:
            section_teachers[section][teacher] = set()
        section_teachers[section][teacher].add(subject_code)

    G.add_nodes_from(SECTIONS_LIST)

    for i, sec1 in enumerate(SECTIONS_LIST):
        for sec2 in SECTIONS_LIST[i+1:]:  
            conflict_pairs = []
            for teacher in section_teachers[sec1]:
                if teacher in section_teachers[sec2]:
                    subjects_sec1 = section_teachers[sec1][teacher]
                    subjects_sec2 = section_teachers[sec2][teacher]
                    conflict_pairs.extend((s1, s2) for s1 in subjects_sec1 for s2 in subjects_sec2)

            if conflict_pairs:
                G.add_edge(sec1, sec2, conflicts=conflict_pairs)  

    plt.figure(figsize=(8, 6), dpi=300)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000)

    edge_labels = {(u, v): str(G[u][v]["conflicts"]) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Staff Dependency Graph")
    plt.savefig("Staff_graph.png")

    return G

staff_graph = generate_staff_dependency_graph()


def crossover_random(parent1, parent2):
    child1 = np.array(parent1, copy=True)
    child2 = np.array(parent2, copy=True)

    swap_sections = random.sample(range(SECTIONS), k=random.randint(1, SECTIONS // 2))
    for sec in swap_sections:
        child1[:, :, sec], child2[:, :, sec] = parent2[:, :, sec], parent1[:, :, sec]

    return child1, child2

def crossover_biological(parent1, parent2):
    child1 = np.array(parent1, copy=True)
    child2 = np.array(parent2, copy=True)
    crossover_point = random.randint(1, SECTIONS - 1)
    
    child1[:,:,crossover_point], child2[:,:,crossover_point] = parent2[:,:,crossover_point], parent1[:,:,crossover_point]
    return child1, child2

def crossover_graph_based(parent1, parent2):
    child1 = np.array(parent1, copy=True)
    child2 = np.array(parent2, copy=True)

    start_vertex = random.choice(list(staff_graph.nodes))

    visited = set()
    queue = [start_vertex]
    while queue and len(visited) < len(staff_graph.nodes) // 2:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(neigh for neigh in staff_graph.neighbors(vertex) if neigh not in visited)

    for vertex in visited:
        idx = ord(vertex[-1]) - ord('A')
        child1[:, :, idx], child2[:, :, idx] = parent1[:, :, idx], parent2[:, :, idx]
    for vertex in set(staff_graph.nodes) - visited:
        idx = ord(vertex[-1]) - ord('A')
        child1[:, :, idx], child2[:, :, idx] = parent2[:, :, idx], parent1[:, :, idx]

    return child1, child2

random.seed(42)
def count_teacher_conflicts(gene):
    conflicts = 0

    for sec1, sec2, data in staff_graph.edges(data=True):
        sec1 = sec1[-1]
        sec2 = sec2[-1]
        if sec1 > sec2:
            sec1, sec2 = sec2, sec1  

        sec1_idx = ord(sec1) - ord('A')  
        sec2_idx = ord(sec2) - ord('A')
        for sub1, sub2 in data["conflicts"]:
            positions = [(d, p) for d in range(DAYS) for p in range(MAX_HOURS) if gene[d][p][sec1_idx] == sub1]

            for d, p in positions:
                if gene[d][p][sec2_idx] == sub2:
                    conflicts += 1

    return conflicts

def count_same_day_subject_conflicts(gene):
    def has_duplicates(lst):
        return len(lst) != len(set(lst))

    conflicts = 0

    for day in range(len(gene)): 
        for sec in range(len(gene[0][0])): 
            subjects = [gene[day][hour][sec] for hour in range(len(gene[0])) if gene[day][hour][sec] is not None]
            if has_duplicates(subjects):
                conflicts += 1

    return conflicts


def fitness(gene):
    teacher_conflicts = count_teacher_conflicts(gene)
    subject_conflicts = count_same_day_subject_conflicts(gene)
    return (teacher_conflicts + subject_conflicts)

class EA:
    def __init__(self, population_size, max_generations, elitism_ratio, k=3, mutation_rate=0.3, replacement_ratio=0.0, crossover = "random", mr_tuning = False):
        self.population_size = population_size
        self.max_generations = max_generations
        self.elitism_size = int(elitism_ratio * population_size)
        self.k = k  
        self.mutation_rate = mutation_rate  
        self.population = [generate_gene() for _ in range(population_size)]
        self.best_solution = None
        self.best_fitness = float('inf')  
        self.best_list = []
        self.successful_mutations = [] 
        self.ema_successful_mutations = []  
        self.fitness_changes = []  
        self.ema_fitness_changes = []  
        self.generation = 0
        self.replacement_ratio = replacement_ratio
        self.mr_tuning = tuning_dict[mr_tuning] if mr_tuning != "None" else False
        if crossover == "random":
            self.crossover = crossover_random
        elif crossover == "biological":
            self.crossover = crossover_biological
        else:
            self.crossover = crossover_graph_based
    
        

    def calc_fitness(self):
        self.population_fitness = [fitness(gene) for gene in self.population]

    def select_parents(self):
        self.parents = []
        for _ in range(self.population_size):
            batch = random.sample(range(self.population_size), self.k)
            chosen_one = min(batch, key=lambda idx: self.population_fitness[idx])
            self.parents.append(self.population[chosen_one])

    def next_gen(self):
        if self.mr_tuning:
            self.mutation_rate = self.mr_tuning[len(self.best_list)]
        self.calc_fitness()

        sorted_indices = sorted(range(self.population_size), key=lambda i: self.population_fitness[i])
        best_idx = sorted_indices[0]
        new_best_fitness = self.population_fitness[best_idx]

        fitness_change = - (new_best_fitness - self.best_fitness if self.best_fitness != float('inf') else 0)

        if new_best_fitness < self.best_fitness:
            self.best_fitness = new_best_fitness
            self.best_solution = self.population[best_idx]

        self.best_list.append(self.best_fitness)
        self.fitness_changes.append(fitness_change)

        alpha = 0.5
        ema_value = alpha * fitness_change + (1 - alpha) * self.ema_fitness_changes[-1] if self.ema_fitness_changes else fitness_change
        self.ema_fitness_changes.append(ema_value)

        if self.best_fitness == 0:
            return True

        elites = [self.population[i] for i in sorted_indices[:self.elitism_size]]

        self.select_parents()
        next_gen = elites[:]

        successful = 0
        total_mutations = 0
        
        while len(next_gen) < self.population_size:
            p1, p2 = random.sample(self.parents, 2)
            p1, p2 = np.array(p1), np.array(p2)
            c1, c2 = self.crossover(p1, p2)
            
            total_mutations = 0
            successful = 0
            
            expected_mutations = (MAX_HOURS * DAYS / 2) * self.mutation_rate  
            mutation_count = 0
            
            for _ in range(int(expected_mutations)):  
                mutation_count += 1
                fitness_before = fitness(c1)
                mutated = mutate_gene(c1)
                if fitness(mutated) < fitness_before:
                    successful += 1
                c1 = mutated
            
                mutation_count += 1
                fitness_before = fitness(c2)
                mutated = mutate_gene(c2)
                if fitness(mutated) < fitness_before:
                    successful += 1
                c2 = mutated
            
            if random.random() < (expected_mutations - int(expected_mutations)):  
                mutation_count += 1
                fitness_before = fitness(c1)
                mutated = mutate_gene(c1)
                if fitness(mutated) < fitness_before:
                    successful += 1
                c1 = mutated
            
                mutation_count += 1
                fitness_before = fitness(c2)
                mutated = mutate_gene(c2)
                if fitness(mutated) < fitness_before:
                    successful += 1
                c2 = mutated
            
            total_mutations += mutation_count


            next_gen.extend([c1, c2])

        next_gen = next_gen[:self.population_size]

        random_count = max(1, int(self.replacement_ratio * self.population_size))
        random_individuals = [generate_gene() for _ in range(random_count)]

        next_gen[-random_count:] = random_individuals

        self.population = next_gen
        self.generation += 1

        success_ratio = successful / total_mutations if total_mutations > 0 else 0
        self.successful_mutations.append(success_ratio)

        ema_value = alpha * success_ratio + (1 - alpha) * self.ema_successful_mutations[-1] if self.ema_successful_mutations else success_ratio
        self.ema_successful_mutations.append(ema_value)

        return False


    def run(self):
        while self.generation < self.max_generations:
            if self.next_gen():
                break 
        return self.best_solution, self.best_fitness, self.best_list

use_multithreading = True

max_generations = 100
runs_per_setting = 40
mr = 0.06
tuning_dict = {
    "const": [mr]*max_generations,
    "fuzzy_tuned_cyclic":[0.06720385674931131, 0.06480371900826447, 0.06240358126721764, 0.060003443526170805, 0.057765946722093016, 0.06816385251690309, 0.06664012579492115, 0.06539160616166098, 0.06327248661353306, 0.0612380267691895, 0.07177212112879151, 0.06950395874186752, 0.06701719270863167, 0.06461391780510471, 0.06215413447532757, 0.07248339828937946, 0.07005067720518554, 0.067531858043538, 0.06495549164541263, 0.06228853229426212, 0.07273154089433642, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036, 0.07279614325068873, 0.07019628099173555, 0.06759641873278238, 0.06499655647382921, 0.062396694214876036]#[0.07680440771349863, 0.07200413223140496, 0.06720385674931131, 0.06240358126721764, 0.057765946722093016, 0.07790154573360353, 0.07404458421657904, 0.07042172971255799, 0.06580338607807439, 0.0612380267691895, 0.08202528129004744, 0.07722662082429722, 0.07217236137852642, 0.0671984745173089, 0.06215413447532757, 0.08283816947357651, 0.07783408578353948, 0.0727266163545794, 0.06755371131122914, 0.06228853229426212, 0.08312176102209876, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036, 0.08319559228650139, 0.07799586776859505, 0.07279614325068873, 0.06759641873278238, 0.062396694214876036]#[0.05626976839098052, 0.06189674523007857, 0.06752372206917662, 0.07315069890827466, 0.07904194952647463, 0.058298354597730556, 0.06673329833165689, 0.0759940159507786, 0.08351736934242793, 0.09173904222314755, 0.06640488671124149, 0.0739852611479504, 0.08087496354747256, 0.08851397903515791, 0.09551028579023665, 0.06778264696192575, 0.07519192457259852, 0.08237685564298985, 0.08915335145930378, 0.09558632477786791, 0.06848620146272388, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729, 0.06873023160901949, 0.07560325476992144, 0.08247627793082339, 0.08934930109172534, 0.09622232425262729]#[0.0646044335977222, 0.06891139583757035, 0.07321835807741849, 0.07752532031726664, 0.0820876349107293, 0.08978686254796758, 0.07015378938431435, 0.0775336958944104, 0.08318590121153308, 0.0891845114847488, 0.09485741321748231, 0.10055545257967827, 0.07552204251321373, 0.08097621497417727, 0.08619052147317305, 0.09103450896859629, 0.09650896539230971, 0.10181633008907164, 0.07636321544369096, 0.0812652184940232, 0.08648962295103697, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646, 0.09685097125097125, 0.10194839079049606, 0.07646129309287204, 0.08155871263239685, 0.08665613217192165, 0.09175355171144646]#[0.0646044335977222, 0.06675791471764626, 0.06891139583757035, 0.07321835807741849, 0.07776723307332249, 0.06734014691097567, 0.07249224903045816, 0.0775336958944104, 0.08318590121153308, 0.08918451148474878, 0.07488743148748604, 0.07793047574925066, 0.08055684534742798, 0.08603722841006335, 0.09126055214806557, 0.07586209080716355, 0.07873099808320004, 0.0814530640712573, 0.08654497750284976, 0.09142337080577609, 0.07631437319209144, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645, 0.07646129309287204, 0.07901000286263445, 0.08155871263239685, 0.08665613217192165, 0.09175355171144645]#[0.054434649525558626, 0.056249137843077245, 0.05806362616059587, 0.061692602795633114, 0.07302706077740051, 0.056374190567481525, 0.06028325866294973, 0.06481792569984698, 0.06980539357793379, 0.08361893755193933, 0.06342501571374978, 0.0662441193749581, 0.06854227118705256, 0.07340593199988933, 0.08664106798976196, 0.06474990924346061, 0.06736533820866625, 0.06978546541429949, 0.07415495831827212, 0.0869225661329375, 0.06536563447758147, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518, 0.06556535047444138, 0.06775086215692276, 0.06993637383940414, 0.07430739720436691, 0.08742046729925518]#[0.07257953270074484, 0.061692602795633114, 0.05806362616059587, 0.056249137843077245, 0.05477029558305038, 0.0751655874233087, 0.06611712240452552, 0.06481792569984698, 0.06364609414458669, 0.06271420316395449, 0.08456668761833305, 0.07265484060479276, 0.06854227118705256, 0.06692893799989909, 0.06498080099232147, 0.08633321232461415, 0.07388456448692428, 0.06978546541429949, 0.06761187376077751, 0.06519192459970313, 0.08715417930344195, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138, 0.08742046729925518, 0.07430739720436691, 0.06993637383940414, 0.06775086215692276, 0.06556535047444138]#[0.7257953270074484, 0.6169260279563311, 0.5806362616059587, 0.5624913784307725, 0.5477029558305039, 0.7516558742330871, 0.6611712240452552, 0.6481792569984698, 0.6364609414458668, 0.627142031639545, 0.8456668761833305, 0.7265484060479276, 0.6854227118705255, 0.6692893799989909, 0.6498080099232146, 0.8633321232461415, 0.7388456448692429, 0.6978546541429949, 0.6761187376077752, 0.6519192459970312, 0.8715417930344196, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138, 0.8742046729925518, 0.7430739720436691, 0.6993637383940414, 0.6775086215692276, 0.6556535047444138]
#[0.05268110396898276, 0.05794921436588104, 0.06321732476277932, 0.06848543515967759, 0.07394933909011273, 0.05348924606978396, 0.05973854214888901, 0.06638340267180326, 0.07236287193063011, 0.07858328517897346, 0.05642708988072909, 0.06238247230340885, 0.06812918961893089, 0.0740837174999471, 0.07990546724552083, 0.05697912885144192, 0.06287978710868261, 0.0687119369129529, 0.07444170801976244, 0.08002862268316017, 0.057235681032325604, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414, 0.057318896031017245, 0.06305078563411898, 0.06878267523722069, 0.07451456484032241, 0.08024645444342414]
}
fixed_params = {
    "max_generations": max_generations,
    "population_size": 100,
    "mutation_rate": mr,  
    "k": 30,  
    "elitism_ratio": 0.1,
    "crossover": crossover_random,  
    "replacement_ratio": 0.1,
    "mr_tuning": "fuzzy_tuned_cyclic"
}

tuning_param = "crossover"  
tuning_values = ["graph_based"]  
results = []

fixed_params.pop(tuning_param, None)


def run_ea(tuning_value, run_idx):
    """Run EA and measure execution time."""
    start_time = time.time()
    
    fitness_sums = np.zeros(max_generations)
    ea = EA(**fixed_params, **{tuning_param: tuning_value})
    
    _, _, fitness_history = ea.run()
    fitness_sums[:len(fitness_history)] += fitness_history
    
    end_time = time.time()
    elapsed_time = end_time - start_time  # Time taken

    return [(run_idx, tuning_value, gen, fitns, elapsed_time) for gen, fitns in enumerate(fitness_history)], fitness_sums, elapsed_time

def compute_average(fitness_sums):
    return fitness_sums / runs_per_setting

def track_progress(future_to_run):
    total_threads = len(future_to_run)
    print(f"Total threads to be deployed: {total_threads}")
    completed_threads = 0
    while completed_threads < total_threads:
        completed_threads = sum([future.done() for future in future_to_run])
        progress = (completed_threads / total_threads) * 100
        print(f"Progress: {progress:.2f}% - {completed_threads}/{total_threads} completed.", end='\r')
        time.sleep(1)

results = []
time_records = []

if use_multithreading:
    with ProcessPoolExecutor() as executor:
        future_to_run = {}
        
        for tuning_value in tuning_values:
            for run_idx in range(runs_per_setting):
                future = executor.submit(run_ea, tuning_value, run_idx)
                future_to_run[future] = (tuning_value, run_idx)
        
        track_progress(future_to_run)
        
        for future in as_completed(future_to_run):
            raw_data, fitness_sums, elapsed_time = future.result()
            results.extend(raw_data)
            time_records.append([future_to_run[future][1], future_to_run[future][0], elapsed_time])  # Store time stats

            fitness_avg = compute_average(fitness_sums)
            for gen, avg_fitness in enumerate(fitness_avg):
                results.append([future_to_run[future][1], future_to_run[future][0], gen, avg_fitness, elapsed_time])
else:
    for tuning_value in tqdm(tuning_values, desc="Tuning Progress"):
        for run_idx in tqdm(range(runs_per_setting), desc=f"Runs for tuning {tuning_value}", leave=False):
            raw_data, fitness_sums, elapsed_time = run_ea(tuning_value, run_idx)
            results.extend(raw_data)
            time_records.append([run_idx, tuning_value, elapsed_time])  # Store time stats

            fitness_avg = compute_average(fitness_sums)
            for gen, avg_fitness in enumerate(fitness_avg):
                results.append([run_idx, tuning_value, gen, avg_fitness, elapsed_time])

# **Save results to CSV**
columns = ["Run", tuning_param, "Generation", "Fitness", "Time (s)"]
df = pd.DataFrame(results, columns=columns)
df.to_csv(f"fitness_results_{tuning_param}.csv", index=False)
print(f"Saved all results to fitness_results_{tuning_param}.csv")

# **Plot Fitness Convergence**
df_avg = df.groupby([tuning_param, "Generation"], as_index=False)["Fitness"].mean()
tuning_values_sorted = sorted(df_avg[tuning_param].unique())
colors = cm.viridis(np.linspace(0, 1, len(tuning_values_sorted)))

plt.figure(figsize=(12, 6), dpi=300)
for tuning_value, color in zip(tuning_values_sorted, colors):
    subset = df_avg[df_avg[tuning_param] == tuning_value]
    plt.plot(subset["Generation"], -subset["Fitness"], label=f"{tuning_param}={tuning_value}", linewidth=2, color=color)

plt.xlabel("Generation")
plt.ylabel("Average Best Fitness")
plt.title(f"Average Fitness Convergence for {tuning_param} Values")
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
plt.savefig(f"{tuning_param}_comparison.png")

# **Compute Summary Statistics**
df_time = pd.DataFrame(time_records, columns=["Run", tuning_param, "Time (s)"])

last_generation = df[df["Generation"] == max(df["Generation"])]
summary_stats = last_generation.groupby([tuning_param]).agg(
    avg_fitness=("Fitness", "mean"),
    min_fitness=("Fitness", "min"),
    max_fitness=("Fitness", "max"),
    std_fitness=("Fitness", "std"),
    avg_time=("Time (s)", "mean"),
    min_time=("Time (s)", "min"),
    max_time=("Time (s)", "max"),
    std_time=("Time (s)", "std"),
).reset_index()

summary_stats.to_csv(f"{tuning_param}_summary_stats.csv", index=False)
print(f"Saved summary stats to {tuning_param}_summary_stats.csv")

# time calculation
end = time.time()
total_time  = end - start
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
time_str = f"Total Time: {hours} hours, {minutes} minutes, {seconds} seconds"
with open("time.txt", "w") as file:
    file.write(time_str)
print("Time saved to time.txt")






