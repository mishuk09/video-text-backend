import os
import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog



# 1️⃣ GROUP SCORE FROM WORD DOCUMENT
#    Qalb block:  D, H, T  -> each dominated by a different person
#    DreamTeam:   Hip, Hus, Hac -> each dominated by a different person
#    One member can dominate at most 1 Qalb + 1 DreamTeam component.

def compute_group_score_word(df, members):
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



# 2️⃣ GROUP DECODING & FITNESS

def decode_groups(particle, n_mem, n_profile):
    """
    Convert particle position vector into groups via ranking + modulo.
    """
    ranks = np.argsort(np.argsort(-particle))  # 0 = highest position
    n_groups = math.ceil(n_profile / n_mem)

    assignments = ranks % n_groups
    groups = {
        g: np.where(assignments == g)[0].tolist()
        for g in range(n_groups)
    }
    return groups


def fitness_function(particle, df, n_mem):
    """
    Fitness = min_g Group_score_g, using the word-model formula with
    Qalb and DreamTeam block constraints.
    """
    groups = decode_groups(particle, n_mem, len(df))
    scores = []
    for m in groups.values():
        s, _ = compute_group_score_word(df, m)
        scores.append(s)
    return np.min(scores)



# 3️⃣ PSO CLASSES

class Particle:
    def __init__(self, dim, bounds, df, n_mem):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position.copy()

        self.df = df
        self.n_mem = n_mem

        self.best_value = fitness_function(self.position, self.df, self.n_mem)

    def update_velocity(self, global_best, w, c1=2.0, c2=2.0):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        self.velocity = (
            w * self.velocity +
            c1 * r1 * (self.best_position - self.position) +
            c2 * r2 * (global_best - self.position)
        )

    def update_position(self, bounds):
        self.position = np.clip(self.position + self.velocity, bounds[0], bounds[1])
        val = fitness_function(self.position, self.df, self.n_mem)

        if val > self.best_value:
            self.best_value = val
            self.best_position = self.position.copy()



# 4️⃣ RUN PSO

def run_pso(df, num_particles=30, max_iter=100, n_mem=3):
    dim = len(df)
    bounds = (0, 1)

    particles = [Particle(dim, bounds, df, n_mem) for _ in range(num_particles)]

    global_best = particles[0].best_position.copy()
    global_best_val = particles[0].best_value
    history = []

    for it in range(max_iter):
        w = 1 - it / max_iter

        for p in particles:
            p.update_velocity(global_best, w)
            p.update_position(bounds)

            if p.best_value > global_best_val:
                global_best_val = p.best_value
                global_best = p.best_position.copy()

        history.append(global_best_val)
        print(f"Iter {it+1}/{max_iter} | Best Fitness = {global_best_val:.6f}")

    groups = decode_groups(global_best, n_mem, dim)
    return global_best_val, history, groups



# 5️⃣ MAIN — TXT OUTPUT USING WORD MODEL

if __name__ == "__main__":
    app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(
        None, "Select Excel File", "", "Excel Files (*.xlsx *.xls)"
    )

    if not file_path:
        sys.exit()

    df_raw = pd.read_excel(file_path)
    names = df_raw.iloc[:, 0]

    # FEATURE EXTRACTION (DT1,DT2,DT3 -> Hip,Hus,Hac)
    df_sub = df_raw.iloc[:, 39:140]

    df_new = pd.DataFrame({
        'D':   df_sub.iloc[:, 0:25].sum(axis=1),
        'H':   df_sub.iloc[:, 25:48].sum(axis=1),
        'T':   df_sub.iloc[:, 48:63].sum(axis=1),
        'Hip': df_sub.iloc[:, 77:83].sum(axis=1),
        'Hus': df_sub.iloc[:, 83:89].sum(axis=1),
        'Hac': df_sub.iloc[:, 89:95].sum(axis=1)
    })

    required = {'D', 'H', 'T', 'Hip', 'Hus', 'Hac'}
    if not required.issubset(df_new.columns):
        raise ValueError("Dataset missing required columns")

    n_mem = 3

    best_val, history, groups = run_pso(
        df_new,
        num_particles=30,
        max_iter=100,
        n_mem=n_mem
    )

    print("\nPSO FINISHED")
    print("Best (worst-group) fitness:", best_val)

    # Convergence plot
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("PSO Convergence (Word-Model Group Score)")
    plt.savefig("pso_wordmodel.png", dpi=300)
    plt.close()

    # Compute group scores and leaders
    group_scores = []
    group_leaders = {}
    for g, members in groups.items():
        s, leaders = compute_group_score_word(df_new, members)
        group_scores.append(s)
        group_leaders[g] = leaders

    best_group_score = max(group_scores) if group_scores else 0.0
    min_group_score = min(group_scores) if group_scores else 0.0

    avg_group_score = float(np.mean(group_scores)) if group_scores else 0.0
    median_score = float(np.median(group_scores)) if group_scores else 0.0

    # TXT output
    with open("output_wordmodel.txt", "w", encoding="utf-8") as f:
        f.write("="*100 + "\n")
        f.write("QALIB GROUP MATCHING RESULTS (WORD-DOCUMENT FORMULA)\n")
        f.write("="*100 + "\n")
        f.write(f"Total Participants: {len(df_new)}\n")
        f.write(f"Total Groups Formed: {len(groups)}\n")
        f.write(f"Best Group Score (Word model): {best_group_score:.6f}\n")
        f.write(f"Average Group Score (Word model): {avg_group_score:.6f}\n")
        f.write(f"min Score (Word model): {min_group_score:.6f}\n")
        f.write(f"median Group Score (Word model): {median_score:.6f}\n")
        f.write("="*100 + "\n\n")

        f.write("GROUP SUMMARY\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Group':<8} | {'Score':<12} | Members\n")
        f.write("-"*100 + "\n")

        for g, members in groups.items():
            score, _ = compute_group_score_word(df_new, members)
            member_names = ", ".join(names.iloc[members].astype(str))
            f.write(f"{g:<8} | {score:<12.6f} | {member_names}\n")

        f.write("\n" + "="*100 + "\n")
        f.write("DETAILED GROUP INFORMATION\n")
        f.write("="*100 + "\n\n")

        components = ['D', 'H', 'T', 'Hip', 'Hus', 'Hac']

        for g, members in groups.items():
            score, leaders = compute_group_score_word(df_new, members)

            f.write("="*100 + "\n")
            f.write(f"GROUP {g}\n")
            f.write("="*100 + "\n")
            f.write(f"Group Score (Word model): {score:.6f}\n")
            f.write(f"Number of Members: {len(members)}\n\n")

            f.write("Component leaders (Qalb: D/H/T different, DreamTeam: Hip/Hus/Hac different):\n")
            for c in components:
                leader_idx = leaders[c]
                leader_name = names.iloc[leader_idx]
                leader_score = df_new.loc[leader_idx, c]
                f.write(f"  {c}: {leader_name} (score = {leader_score})\n")
            f.write("\n")

            header = "#  | Name                           | " + " | ".join([f"{c:<6}" for c in components])
            f.write(header + "\n")
            f.write("-"*100 + "\n")

            for idx, m in enumerate(members, start=1):
                row_vals = [df_new.loc[m, c] for c in components]
                row = f"{idx:<2} | {str(names.iloc[m])[:30]:<30} | " + \
                      " | ".join([f"{v:<6}" for v in row_vals])
                f.write(row + "\n")

            f.write("\n")

    print("Files generated successfully:")
    print("  ✔ pso_wordmodel.png")
    print("  ✔ output_wordmodel.txt")
