"""
pso.py — PSO-based group matching for Qalib project (Backend Safe Version)
Removed GUI (PyQt5) & plotting for server use.
"""

import os
import math
import numpy as np
import pandas as pd

# -------------------------
# 1️⃣ Load Dataset
# -------------------------
def load_dataset():
    """
    Automatically detect dataset.xlsx in data/ or current folder.
    Keep only numeric columns for PSO fitness calculation.
    """
    possible_paths = [
        os.path.join(os.getcwd(), "data", "dataset.xlsx"),
        os.path.join(os.getcwd(), "uploads", "dataset.xlsx"),
        os.path.join(os.getcwd(), "dataset.xlsx"),
    ]

    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break

    if not path:
        raise FileNotFoundError("dataset.xlsx not found. Place it in the uploads/ or project folder.")

    # Try reading with flexible header detection
    for header_row in range(0, 4):
        df = pd.read_excel(path, header=header_row)
        all_text_headers = all(isinstance(col, str) and col.strip() != "" for col in df.columns)
        if all_text_headers:
            break

    # Keep only numeric columns for PSO
    df_num = df.select_dtypes(include=[np.number])
    if df_num.empty:
        raise ValueError("No numeric columns found. Ensure your dataset has numeric survey responses.")
    
    return df_num


# -------------------------
# 2️⃣ Fitness Function
# -------------------------
def compute_group_score(df, members):
    """Compute group score using mean of numeric columns."""
    if len(members) == 0:
        return 0.0
    return float(df.iloc[members].mean().mean())


def decode_groups(particle, n_mem, n_profile):
    """Convert particle vector into group assignments."""
    ranks = np.argsort(np.argsort(-particle))
    n_grp = max(1, math.ceil(n_profile / n_mem))
    grp_assignments = (ranks % n_grp).astype(int)

    groups = {}
    for g in range(n_grp):
        groups[g] = np.where(grp_assignments == g)[0].tolist()
    return groups


def fitness_function(particle, df, n_mem=3, scoring="min"):
    """Evaluate fitness for one particle."""
    n_profile = df.shape[0]
    groups = decode_groups(particle, n_mem, n_profile)
    group_scores = [compute_group_score(df, members) for members in groups.values()]

    if scoring == "max":
        return max(group_scores)
    elif scoring == "min":
        return np.min(group_scores)
    elif scoring == "mean":
        return np.mean(group_scores)
    elif scoring == "sum":
        return np.sum(group_scores)
    else:
        raise ValueError("Unknown scoring method: choose 'max', 'mean', or 'sum'")


# -------------------------
# 3️⃣ PSO Algorithm
# -------------------------
class Particle:
    def __init__(self, dim, bounds, df, n_mem, scoring):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.df = df
        self.n_mem = n_mem
        self.scoring = scoring

        self.best_position = self.position.copy()
        self.best_value = fitness_function(self.position, df, n_mem, scoring)
        self.current_value = self.best_value

    def update_velocity(self, global_best_position, w, c1, c2):
        r1, r2 = np.random.random(self.position.shape), np.random.random(self.position.shape)
        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.best_position - self.position)
            + c2 * r2 * (global_best_position - self.position)
        )

    def update_position(self, bounds):
        self.position = np.clip(self.position + self.velocity, bounds[0], bounds[1])
        self.current_value = fitness_function(self.position, self.df, self.n_mem, self.scoring)
        if self.current_value > self.best_value:
            self.best_value = self.current_value
            self.best_position = self.position.copy()


def run_pso(num_particles=20, max_iter=1000, n_mem=3, scoring="min", verbose=True):
    """Main PSO function for optimization."""
    df = load_dataset()
    dim = df.shape[0]
    print("No. of participants:", dim)
    print("No. of group:", dim / n_mem)
    bounds = (0, 1)

    particles = [Particle(dim, bounds, df, n_mem, scoring) for _ in range(num_particles)]
    global_best_position = particles[0].best_position.copy()
    global_best_value = particles[0].best_value

    history = []

    for iteration in range(max_iter):
        w = 1 - (iteration / max_iter)
        for p in particles:
            p.update_velocity(global_best_position, w, c1=2.0, c2=2.0)
            p.update_position(bounds)

            if p.best_value > global_best_value:
                global_best_value = p.best_value
                global_best_position = p.best_position.copy()

        history.append(global_best_value)
        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration+1}/{max_iter} | Best value: {global_best_value:.6f}")

    best_groups = decode_groups(global_best_position, n_mem, dim)
    return global_best_position, global_best_value, history, best_groups
