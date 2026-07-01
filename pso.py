# pso.py — Optimized PSO with Word-Model Group Scoring
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


def compute_group_score_word(df, members):
    """
    Word-model group score calculation.
    Qalb block: D, H, T -> each dominated by a different person
    DreamTeam: Hip, Hus, Hac -> each dominated by a different person
    One member can dominate at most 1 Qalb + 1 DreamTeam component.
    
    Args:
        df: DataFrame with columns ['D', 'H', 'T', 'Hip', 'Hus', 'Hac']
        members: list of member indices
        
    Returns:
        tuple: (group_score, leaders_global)
            - group_score: float score for the group
            - leaders_global: dict mapping component to global row index
    """
    if len(members) == 0:
        return 0.0, {}

    # Work only on the members in this group
    group_data = df.iloc[members].reset_index()
    group_data.rename(columns={'index': 'orig_idx'}, inplace=True)

    qalb_comps = ['D', 'H', 'T']
    dt_comps   = ['Hip', 'Hus', 'Hac']
    comps = qalb_comps + dt_comps

    norms = {
        'D':   125.0,
        'H':   115.0,
        'T':   70.0,
        'Hip': 30.0,
        'Hus': 30.0,
        'Hac': 30.0
    }

    # Separate domination counts per block
    qalb_dom_count = {i: 0 for i in range(len(group_data))}
    dt_dom_count   = {i: 0 for i in range(len(group_data))}

    best_vals = {c: 0.0 for c in comps}
    leaders_local = {}  # component -> local index

    # --- QALB BLOCK: D, H, T ---
    for c in qalb_comps:
        sorted_idx = group_data[c].sort_values(ascending=False).index.tolist()

        chosen = None
        for idx in sorted_idx:
            if qalb_dom_count[idx] < 1:  # at most 1 Qalb component per member
                chosen = idx
                break

        # If all already dominate 1 Qalb component, fall back to best
        if chosen is None:
            chosen = sorted_idx[0]

        qalb_dom_count[chosen] += 1
        raw_val = group_data.loc[chosen, c]
        best_vals[c] = raw_val / norms[c]
        leaders_local[c] = chosen

    # --- DREAM TEAM BLOCK: Hip, Hus, Hac ---
    for c in dt_comps:
        sorted_idx = group_data[c].sort_values(ascending=False).index.tolist()

        chosen = None
        for idx in sorted_idx:
            if dt_dom_count[idx] < 1:  # at most 1 DT component per member
                chosen = idx
                break

        # If all already dominate 1 DT component, fall back to best
        if chosen is None:
            chosen = sorted_idx[0]

        dt_dom_count[chosen] += 1
        raw_val = group_data.loc[chosen, c]
        best_vals[c] = raw_val / norms[c]
        leaders_local[c] = chosen

    # Compute match_Q and match_DT from these distributed best_vals
    best_GD   = best_vals['D']
    best_GH   = best_vals['H']
    best_GT   = best_vals['T']
    best_GDT1 = best_vals['Hip']
    best_GDT2 = best_vals['Hus']
    best_GDT3 = best_vals['Hac']

    match_Q  = (best_GD + best_GH + best_GT) / 5.0
    match_DT = (best_GDT1 + best_GDT2 + best_GDT3) / 5.0

    # Similarity factor using all members in the group
    unique_ids = len(set(members))
    if unique_ids == 1:
        ratio = 1.0 / 3.0
    elif unique_ids == 2:
        ratio = 2.0 / 3.0
    else:
        ratio = 1.0

    part_Q  = match_Q * ratio
    part_DT = match_DT * ratio
    group_score = part_Q + part_DT

    # Convert leaders from local indices to global row indices of df
    leaders_global = {
        c: int(group_data.loc[leaders_local[c], 'orig_idx']) for c in leaders_local
    }

    return group_score, leaders_global


def compute_group_score_word_fast(data_arrays, members, norms_array):
    """
    Optimized numpy-only version of word-model scoring for PSO iterations.
    Much faster than pandas version - used during PSO optimization.
    
    Args:
        data_arrays: numpy array shape (n_profiles, 6) for [D, H, T, Hip, Hus, Hac]
        members: numpy array or list of member indices
        norms_array: numpy array [125.0, 115.0, 70.0, 30.0, 30.0, 30.0]
        
    Returns:
        float: group_score only (no leaders for speed)
    """
    if len(members) == 0:
        return 0.0
    
    # Extract member data
    group_data = data_arrays[members]  # shape (k, 6)
    n_members = len(members)
    
    # Qalb: indices 0, 1, 2 (D, H, T)
    # DreamTeam: indices 3, 4, 5 (Hip, Hus, Hac)
    qalb_dom_count = np.zeros(n_members, dtype=np.int32)
    dt_dom_count = np.zeros(n_members, dtype=np.int32)
    
    best_vals = np.zeros(6, dtype=np.float32)
    
    # --- QALB BLOCK: indices 0, 1, 2 ---
    for comp_idx in range(3):
        sorted_indices = np.argsort(-group_data[:, comp_idx])  # descending order
        
        chosen = -1
        for idx in sorted_indices:
            if qalb_dom_count[idx] < 1:
                chosen = idx
                break
        
        if chosen == -1:
            chosen = sorted_indices[0]
        
        qalb_dom_count[chosen] += 1
        raw_val = group_data[chosen, comp_idx]
        best_vals[comp_idx] = raw_val / norms_array[comp_idx]
    
    # --- DREAM TEAM BLOCK: indices 3, 4, 5 ---
    for comp_idx in range(3, 6):
        sorted_indices = np.argsort(-group_data[:, comp_idx])
        
        chosen = -1
        for idx in sorted_indices:
            if dt_dom_count[idx] < 1:
                chosen = idx
                break
        
        if chosen == -1:
            chosen = sorted_indices[0]
        
        dt_dom_count[chosen] += 1
        raw_val = group_data[chosen, comp_idx]
        best_vals[comp_idx] = raw_val / norms_array[comp_idx]
    
    # Compute matches
    match_Q = (best_vals[0] + best_vals[1] + best_vals[2]) / 5.0
    match_DT = (best_vals[3] + best_vals[4] + best_vals[5]) / 5.0
    
    # Similarity factor
    unique_ids = len(np.unique(members))
    if unique_ids == 1:
        ratio = 1.0 / 3.0
    elif unique_ids == 2:
        ratio = 2.0 / 3.0
    else:
        ratio = 1.0
    
    group_score = (match_Q + match_DT) * ratio
    return float(group_score)


def compute_group_score_from_values(arr_values, members):
    """
    Legacy function maintained for backward compatibility.
    Now returns mean score.
    For word-model scoring, use compute_group_score_word() instead.
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


def fitness_from_particle_and_values(particle, arr_values, n_mem=3, scoring="min", df=None, fast_arrays=None, norms_array=None):
    """
    Compute fitness given particle and pre-loaded numpy array arr_values.
    If fast_arrays provided, uses optimized numpy word-model scoring.
    Otherwise falls back to pandas or legacy scoring.
    
    Args:
        particle: particle position vector
        arr_values: numpy array (used for legacy scoring)
        n_mem: members per group
        scoring: 'min', 'max', 'mean', or 'sum'
        df: optional DataFrame with proper columns for word-model scoring
        fast_arrays: pre-computed numpy arrays for fast scoring
        norms_array: normalization values as numpy array
    """
    n_profile = arr_values.shape[0] if arr_values is not None else len(df)
    groups = decode_groups_from_particle(particle, n_mem, n_profile)
    
    # Use fast numpy scoring if available
    if fast_arrays is not None and norms_array is not None:
        group_scores = [compute_group_score_word_fast(fast_arrays, members, norms_array) 
                       for members in groups.values()]
    elif df is not None:
        # Pandas word-model scoring (slower, for final results)
        group_scores = [compute_group_score_word(df, members)[0] for members in groups.values()]
    else:
        # Legacy scoring
        group_scores = [compute_group_score_from_values(arr_values, members) for members in groups.values()]

    if scoring == "max":
        return max(group_scores) if group_scores else 0.0
    elif scoring == "min":
        return np.min(group_scores) if group_scores else 0.0
    elif scoring == "mean":
        return np.mean(group_scores) if group_scores else 0.0
    elif scoring == "sum":
        return np.sum(group_scores) if group_scores else 0.0
    else:
        raise ValueError("Unknown scoring method: choose 'max', 'min', 'mean' or 'sum'")


class Particle:
    __slots__ = ("position", "velocity", "best_position", "best_value", "current_value")
    def __init__(self, dim, bounds, arr_values, n_mem, scoring, df=None, fast_arrays=None, norms_array=None):
        self.position = np.random.uniform(bounds[0], bounds[1], dim).astype(np.float32)
        self.velocity = np.random.uniform(-1.0, 1.0, dim).astype(np.float32)
        self.best_position = self.position.copy()
        # Evaluate once using fast numpy-based fitness
        self.best_value = fitness_from_particle_and_values(self.position, arr_values, n_mem, scoring, df, fast_arrays, norms_array)
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

    def update_position(self, bounds, arr_values, n_mem, scoring, df=None, fast_arrays=None, norms_array=None):
        self.position = np.clip(self.position + self.velocity, bounds[0], bounds[1]).astype(np.float32)
        self.current_value = fitness_from_particle_and_values(self.position, arr_values, n_mem, scoring, df, fast_arrays, norms_array)
        if self.current_value > self.best_value:
            self.best_value = self.current_value
            self.best_position = self.position.copy()


def run_pso(df, num_particles=20, max_iter=1000, n_mem=3, scoring="min", verbose=False, use_word_model=True):
    """
    Main PSO function with optional word-model scoring.
    
    Args:
        df: pandas DataFrame (numeric-only, dtype float32) already loaded by caller.
        num_particles: number of particles in swarm
        max_iter: maximum iterations
        n_mem: members per group
        scoring: 'min', 'max', 'mean', or 'sum'
        verbose: print progress
        use_word_model: if True and df has proper columns, use word-model scoring
        
    Returns: 
        tuple: (best_position, best_value, history_list, best_groups_dict)
    """
    # Convert once to numpy values
    arr_values = df.values  # shape (n_profiles, n_features)
    n_profile = arr_values.shape[0]
    if n_profile == 0:
        raise ValueError("Dataset has 0 rows.")

    dim = n_profile
    bounds = (0.0, 1.0)

    # Check if we should use word-model scoring and prepare fast arrays
    df_for_scoring = None
    fast_arrays = None
    norms_array = None
    
    if use_word_model:
        required_cols = {'D', 'H', 'T', 'Hip', 'Hus', 'Hac'}
        if required_cols.issubset(df.columns):
            df_for_scoring = df
            # Pre-compute numpy arrays for ultra-fast scoring during PSO
            fast_arrays = df[['D', 'H', 'T', 'Hip', 'Hus', 'Hac']].values.astype(np.float32)
            norms_array = np.array([125.0, 115.0, 70.0, 30.0, 30.0, 30.0], dtype=np.float32)
            if verbose:
                print("✅ Using optimized word-model scoring")
        else:
            if verbose:
                print("⚠️ Word-model columns not found, falling back to legacy scoring")

    # Initialize particles with fast arrays
    particles = [Particle(dim, bounds, arr_values, n_mem, scoring, None, fast_arrays, norms_array) 
                 for _ in range(num_particles)]

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
            p.update_position(bounds, arr_values, n_mem, scoring, None, fast_arrays, norms_array)

            # check for global improvement
            if p.best_value > global_best_value:
                global_best_value = p.best_value
                global_best_position = p.best_position.copy()

        history.append(global_best_value)
        if verbose and (iteration % 50 == 0 or iteration == max_iter - 1):
            print(f"Iter {iteration+1}/{max_iter} | Best value: {global_best_value:.6f}")

    best_groups = decode_groups_from_particle(global_best_position, n_mem, dim)
    return global_best_position, global_best_value, history, best_groups
