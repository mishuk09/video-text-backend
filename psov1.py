# pso.py — Optimized PSO (same meaning/output)
import os
import math
import numpy as np
import pandas as pd

def load_dataset(path=None):
    """
    Load dataset.xlsx (once). Return numeric-only DataFrame with float32 dtype.
    If path provided, read that file, otherwise search uploads/ or data/.
    """
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        df = pd.read_excel(path, header=0)
    else:
        possible_paths = [
            os.path.join(os.getcwd(), "uploads", "dataset.xlsx"),
            os.path.join(os.getcwd(), "data", "dataset.xlsx"),
            os.path.join(os.getcwd(), "dataset.xlsx"),
        ]
        path_found = None
        for p in possible_paths:
            if os.path.exists(p):
                path_found = p
                break
        if not path_found:
            raise FileNotFoundError("dataset.xlsx not found in uploads/ or data/ or project root.")
        df = pd.read_excel(path_found, header=0)

    # Keep numeric columns only and convert to float32 for speed
    df_num = df.select_dtypes(include=[np.number]).astype(np.float32)
    if df_num.empty:
        raise ValueError("No numeric columns found. Ensure dataset has numeric survey responses.")
    return df_num


def compute_group_score_from_values(arr_values, members):
    """
    arr_values: numpy array shape (n_profiles, n_features)
    members: list/array of member indices
    Returns single float score = mean of all numeric values for chosen members.
    """
    if len(members) == 0:
        return 0.0
    sub = arr_values[members]  # shape (k, n_features)
    # single mean across all elements
    return float(sub.mean())


def decode_groups_from_particle(particle, n_mem, n_profile):
    """
    Reproduce original decode logic:
    ranks = np.argsort(np.argsort(-particle))
    n_grp = max(1, math.ceil(n_profile / n_mem))
    grp_assignments = (ranks % n_grp).astype(int)
    Return dict: {group_index: [member_indices, ...], ...}
    """
    # Keep same ranking behavior to preserve meaning
    ranks = np.argsort(np.argsort(-particle))
    n_grp = max(1, math.ceil(n_profile / n_mem))
    grp_assignments = (ranks % n_grp).astype(int)

    groups = {}
    for g in range(n_grp):
        groups[g] = np.where(grp_assignments == g)[0].tolist()
    return groups


def fitness_from_particle_and_values(particle, arr_values, n_mem=3, scoring="min"):
    """
    Compute fitness given particle and pre-loaded numpy array arr_values.
    This avoids pandas and repeated conversions.
    """
    n_profile = arr_values.shape[0]
    groups = decode_groups_from_particle(particle, n_mem, n_profile)
    # compute group scores
    group_scores = [compute_group_score_from_values(arr_values, members) for members in groups.values()]

    if scoring == "max":
        return max(group_scores)
    elif scoring == "min":
        return np.min(group_scores)
    elif scoring == "mean":
        return np.mean(group_scores)
    elif scoring == "sum":
        return np.sum(group_scores)
    else:
        raise ValueError("Unknown scoring method: choose 'max', 'min', 'mean' or 'sum'")


class Particle:
    __slots__ = ("position", "velocity", "best_position", "best_value", "current_value")
    def __init__(self, dim, bounds, arr_values, n_mem, scoring):
        self.position = np.random.uniform(bounds[0], bounds[1], dim).astype(np.float32)
        self.velocity = np.random.uniform(-1.0, 1.0, dim).astype(np.float32)
        self.best_position = self.position.copy()
        # Evaluate once using fast numpy-based fitness
        self.best_value = fitness_from_particle_and_values(self.position, arr_values, n_mem, scoring)
        self.current_value = self.best_value

    def update_velocity(self, global_best_position, w, c1, c2):
        # use vectorized randoms
        r1 = np.random.random(self.position.shape).astype(np.float32)
        r2 = np.random.random(self.position.shape).astype(np.float32)
        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.best_position - self.position)
            + c2 * r2 * (global_best_position - self.position)
        ).astype(np.float32)

    def update_position(self, bounds, arr_values, n_mem, scoring):
        self.position = np.clip(self.position + self.velocity, bounds[0], bounds[1]).astype(np.float32)
        self.current_value = fitness_from_particle_and_values(self.position, arr_values, n_mem, scoring)
        if self.current_value > self.best_value:
            self.best_value = self.current_value
            self.best_position = self.position.copy()


def run_pso(df, num_particles=20, max_iter=1000, n_mem=3, scoring="min", verbose=False):
    """
    Main PSO function.
    df: pandas DataFrame (numeric-only, dtype float32) already loaded by caller.
    Returns: (best_position, best_value, history_list, best_groups_dict)
    """
    # Convert once to numpy values
    arr_values = df.values  # shape (n_profiles, n_features)
    n_profile = arr_values.shape[0]
    if n_profile == 0:
        raise ValueError("Dataset has 0 rows.")

    dim = n_profile
    bounds = (0.0, 1.0)

    # Initialize particles
    particles = [Particle(dim, bounds, arr_values, n_mem, scoring) for _ in range(num_particles)]

    # initialize global best
    global_best_position = particles[0].best_position.copy()
    global_best_value = particles[0].best_value
    for p in particles:
        if p.best_value > global_best_value:
            global_best_value = p.best_value
            global_best_position = p.best_position.copy()

    history = []

    for iteration in range(max_iter):
        # inertia weight schedule (linearly decreasing)
        w = 1.0 - (iteration / max_iter)
        for p in particles:
            p.update_velocity(global_best_position, w, c1=2.0, c2=2.0)
            p.update_position(bounds, arr_values, n_mem, scoring)

            # check for global improvement
            if p.best_value > global_best_value:
                global_best_value = p.best_value
                global_best_position = p.best_position.copy()

        history.append(global_best_value)
        if verbose and (iteration % 50 == 0 or iteration == max_iter - 1):
            print(f"Iter {iteration+1}/{max_iter} | Best value: {global_best_value:.6f}")

    best_groups = decode_groups_from_particle(global_best_position, n_mem, dim)
    return global_best_position, global_best_value, history, best_groups
