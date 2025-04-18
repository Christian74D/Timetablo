import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# Load saved data
with open('timetable_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Preprocess
row_info = []
periods_by_id = {}
for row in data:
    rid = row['id']
    row_info.append({
        'id': rid,
        'sections': row['sections'],
        'staffs': set(row['staffs']),
    })
    periods_by_id[rid] = row['theory'] + row['lab']

# All unique sections
sections = sorted({sec for row in row_info for sec in row['sections']})

# Build graph
G = nx.Graph()
G.add_nodes_from(sections)

# Collect unique IDs per edge
section_pair_ids = defaultdict(set)

for i in range(len(row_info)):
    row_a = row_info[i]
    ida = row_a['id']
    for j in range(i + 1, len(row_info)):
        row_b = row_info[j]
        idb = row_b['id']

        if not row_a['staffs'].intersection(row_b['staffs']):
            continue

        for sec_a in row_a['sections']:
            for sec_b in row_b['sections']:
                if sec_a == sec_b:
                    continue
                key = tuple(sorted((sec_a, sec_b)))
                section_pair_ids[key].add(ida)
                section_pair_ids[key].add(idb)

# Add edges with computed weights
for (sec1, sec2), id_set in section_pair_ids.items():
    weight = sum(periods_by_id[rid] for rid in id_set)
    if weight > 0:
        G.add_edge(sec1, sec2, weight=weight)

# Draw graph
plt.figure(figsize=(10, 8), dpi=600)
pos = nx.spring_layout(G, seed=42)

nx.draw(G, pos, with_labels=True, node_color="lightgreen", node_size=2200, font_size=10)
edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Section Dependency Graph (Unique ID-based Period Sum)")
plt.savefig("dependency_graph.png", dpi=600)

# Save graph
with open("dependency_graph.pkl", "wb") as f:
    pickle.dump(G, f)

print("Graph saved to 'dependency_graph.pkl'")
print("Graph image saved to 'dependency_graph.png'")
