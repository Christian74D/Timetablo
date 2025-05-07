import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

def track_progress(future_to_run):
    total = len(future_to_run)
    completed = 0
    print(f"Total threads to be deployed: {total}")
    while completed < total:
        completed = sum(f.done() for f in future_to_run)
        print(f"Progress: {(completed / total) * 100:.2f}% - {completed}/{total} completed.", end='\r')
        time.sleep(1)

def compute_average(fitness_sums, runs):
    return fitness_sums / runs

def plot_fitness(df, tuning_param):
    df_avg = df.groupby([tuning_param, "Generation"], as_index=False)["Fitness"].mean()
    tuning_values_sorted = sorted(df_avg[tuning_param].unique())
    colors = cm.viridis(np.linspace(0, 1, len(tuning_values_sorted)))

    plt.figure(figsize=(12, 6), dpi=300)
    for val, color in zip(tuning_values_sorted, colors):
        subset = df_avg[df_avg[tuning_param] == val]
        plt.plot(subset["Generation"], -subset["Fitness"], label=f"{tuning_param}={val}", linewidth=2, color=color)

    plt.xlabel("Generation")
    plt.ylabel("Average Best Fitness")
    plt.title(f"Average Fitness Convergence for {tuning_param} Values")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.savefig(f"outputs\{tuning_param}_comparison.png")

def save_summary(df, time_records, tuning_param):
    df_time = pd.DataFrame(time_records, columns=["Run", tuning_param, "Time (s)"])
    last_gen = df[df["Generation"] == df["Generation"].max()]

    summary = last_gen.groupby(tuning_param).agg(
        avg_fitness=("Fitness", "mean"),
        min_fitness=("Fitness", "min"),
        max_fitness=("Fitness", "max"),
        std_fitness=("Fitness", "std"),
        avg_time=("Time (s)", "mean"),
        min_time=("Time (s)", "min"),
        max_time=("Time (s)", "max"),
        std_time=("Time (s)", "std"),
    ).reset_index()

    summary.to_csv(f"outputs\{tuning_param}_summary_stats.csv", index=False)
    print(f"Saved summary stats to {tuning_param}_summary_stats.csv")
