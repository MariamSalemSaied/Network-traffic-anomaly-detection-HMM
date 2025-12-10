# -*- coding: utf-8 -*-
"""Hidden_Markov
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
from sklearn.cluster import KMeans

"""#Upload cleaned data"""

df = pd.read_csv('cleaned_network_data (3).csv')
df.head()



"""# Return Calculation + State Labeling

##calculating empirical steady state
If the attack pattern continues forever, which emission type     dominates long-term?
"""

seq = df['Emission'].tolist()

def build_transition_matrix_from_sequence(seq):
    symbols = sorted(list(set(seq)))
    n = len(symbols)
    sym_to_idx = {s:i for i,s in enumerate(symbols)}

    A = np.zeros((n,n))

    for i in range(len(seq)-1):
        a = sym_to_idx[seq[i]]
        b = sym_to_idx[seq[i+1]]
        A[a,b] += 1

    A = A / A.sum(axis=1, keepdims=True)
    return A, symbols

"""##Steady State Via Eigen vector"""

def steady_state_eig(A):
    w, v = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(w - 1))
    pi = np.real(v[:, idx])
    pi = pi / np.sum(pi)
    return pi

A_emp, symbols = build_transition_matrix_from_sequence(seq)
pi_emp = steady_state_eig(A_emp)

steady_state_empirical = pd.Series(pi_emp, index=symbols).sort_values(ascending=False)
steady_state_empirical.head(20)

from sklearn.preprocessing import LabelEncoder
import numpy as np

# Turn the string "Emission" column into numbers (0, 1, 2...)

le = LabelEncoder()
df['Observation'] = le.fit_transform(df['Emission'])

# Save the mapping so we know what the numbers mean later
observation_map = dict(zip(le.transform(le.classes_), le.classes_))
print("Encoding Complete. Example mapping:", list(observation_map.items())[:3])

# ---------------------------------------------------------
# PERSON 1 TASK: Sequence Construction
# Group the data into "windows" or "sessions" to create sequences
# ---------------------------------------------------------
def create_sequences(data, seq_length=10):
    sequences = []
    # Loop through the data and cut it into chunks of 10
    for i in range(0, len(data) - seq_length + 1, seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Create the sequences using the new 'Observation' column
# We use a sequence length of 10 (grouping 10 packets together)
observation_sequences = create_sequences(df['Observation'].values, seq_length=10)

print(f"\n--- Sequence Construction Results ---")
print(f"Total Sequences Created: {observation_sequences.shape[0]}")
print(f"Sequence Length: {observation_sequences.shape[1]}")
print(f"Example Sequence (Numbers): {observation_sequences[0]}")

"""# Baum–Welch Training (HMM)"""

n_states = 3
n_obs = df['Observation'].nunique()
print(f"n_states = {n_states}, n_obs = {n_obs}")

pi = np.full(n_states, 1.0 / n_states)
A = np.random.rand(n_states, n_states)
A /= A.sum(axis=1, keepdims=True)

B = np.random.rand(n_states, n_obs)
B /= B.sum(axis=1, keepdims=True)

def forward(obs_seq, pi, A, B):
    T = len(obs_seq)
    N = len(pi)

    alpha = np.zeros((T, N))

    # Initialization
    alpha[0] = pi * B[:, obs_seq[0]]

    # Induction
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs_seq[t]]

    return alpha

def backward(obs_seq, A, B):
    T = len(obs_seq)
    N = A.shape[0]

    beta = np.zeros((T, N))
    beta[-1] = 1  # initialization

    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i] * B[:, obs_seq[t+1]] * beta[t+1])

    return beta

def baum_welch(sequences, pi, A, B, max_iters=10):
    n_states = A.shape[0]
    n_obs = B.shape[1]

    for iteration in range(max_iters):
        print(f"\n____Iteration {iteration+1}____")

        # Accumulators
        pi_new = np.zeros(n_states)
        A_num = np.zeros((n_states, n_states))
        A_den = np.zeros(n_states)
        B_num = np.zeros((n_states, n_obs))
        B_den = np.zeros(n_states)

        for obs_seq in sequences:

            alpha = forward(obs_seq, pi, A, B)
            beta = backward(obs_seq, A, B)

            T = len(obs_seq)
            prob_seq = np.sum(alpha[-1])
            if prob_seq < 1e-300:
                prob_seq = 1e-300

            # Gamma (state probabilities)
            gamma = (alpha * beta) / prob_seq

            # Xi (state transition probabilities)
            xi = np.zeros((T - 1, n_states, n_states))
            for t in range(T - 1):
                denom = np.sum(alpha[t][:, None] * A * (B[:, obs_seq[t+1]] * beta[t+1])[None, :])
                if denom < 1e-300:
                    denom = 1e-300

                xi[t] = (alpha[t][:, None] * A *
                         (B[:, obs_seq[t+1]] * beta[t+1])[None, :]) / denom

            # Update accumulators
            pi_new += gamma[0]
            A_num += np.sum(xi, axis=0)
            A_den += np.sum(gamma[:-1], axis=0)

            for t in range(T):
                B_num[:, obs_seq[t]] += gamma[t]
            B_den += np.sum(gamma, axis=0)

        # Update HMM parameters
        pi = pi_new / np.sum(pi_new)
        A = A_num / A_den[:, None]
        B = B_num / B_den[:, None]

    return pi, A, B

pi_trained, A_trained, B_trained = baum_welch(observation_sequences, pi, A, B)

print("\n___TRAINED HMM PARAMETERS____")
print("π (initial distribution):\n", pi_trained)
print("\nA (transition matrix):\n", A_trained)
print("\nB (emission matrix):\n", B_trained)

#Just for Testing
pi = pi_trained
A = A_trained
B = B_trained

print("pi sum:", pi.sum())
print("pi min/max:", pi.min(), pi.max())
print("any NaN in pi:", np.isnan(pi).any())

row_sums_A = A.sum(axis=1)
row_sums_B = B.sum(axis=1)

print("A row sums :", row_sums_A[:10])
print("any NaN in A:", np.isnan(A).any())

print("B row sums :", row_sums_B[:10])
print("any NaN in B:", np.isnan(B).any())

"""##HMM Steady State
If attacks follow hidden stages, which hidden state is most active long-term?
"""

def steady_state(A, tol=1e-12, max_iter=10000):
    """
    Compute the steady-state distribution for transition matrix A
    using power iteration.
    """
    n = A.shape[0]
    pi = np.ones(n) / n  # start from uniform distribution

    for _ in range(max_iter):
        pi_next = pi @ A
        if np.linalg.norm(pi_next - pi) < tol:
            return pi_next
        pi = pi_next

    return pi  # return last estimate even if not fully converged

pi_bw, A_bw, B_bw = baum_welch(observation_sequences, pi, A, B)

steady = steady_state(A_bw)
print("Steady state distribution:", steady)

"""# Person 5"""




df.head()
seeds = [0, 1, 2]               # seeds to estimate variance
init_types = ['random', 'uniform', 'cluster']
max_iter_per_run = 20           # max EM iterations per run
pilot_iters = 10
outdir = "Initialization_Results"

def safe_normalize_rows(mat, eps=1e-12):
    mat = mat + eps
    return mat / mat.sum(axis=1, keepdims=True)

# ---------------------------
# Initializers
# ---------------------------
def init_random(K, M, seed=None):
    rng = np.random.RandomState(seed)
    pi = rng.dirichlet([1.0]*K)
    A = np.vstack([rng.dirichlet([1.0]*K) for _ in range(K)])
    B = np.vstack([rng.dirichlet([1.0]*M) for _ in range(K)])
    return pi, A, B

def init_uniform(K, M):
    pi = np.ones(K) / K
    A = np.ones((K, K)) / K
    B = np.ones((K, M)) / M
    return pi, A, B

def init_cluster_based(seqs, K, M, seed=None):
    """
    Fast cluster-based initializer:
      - build per-sequence frequency vectors (n_sequences x M)
      - cluster those vectors with KMeans
      - aggregate start counts and emission counts per cluster
      - set A to uniform fallback (safe) — if you need transitions, use per-packet clustering
    """
    n_seq = len(seqs)
    freq = np.zeros((n_seq, M), dtype=float)
    for i, s in enumerate(seqs):
        if len(s) == 0:
            continue
        vals, counts = np.unique(s, return_counts=True)
        freq[i, vals.astype(int)] = counts
    # normalize rows
    row_sums = freq.sum(axis=1, keepdims=True); row_sums[row_sums==0] = 1.0
    freq = freq / row_sums

    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = km.fit_predict(freq)

    start_counts = np.zeros(K)
    emit_counts = np.zeros((K, M))
    for i, s in enumerate(seqs):
        lab = labels[i]
        start_counts[lab] += 1
        if len(s) == 0:
            continue
        vals, counts = np.unique(s, return_counts=True)
        emit_counts[lab, vals.astype(int)] += counts

    pi = (start_counts + 1e-8) / (np.sum(start_counts) + 1e-8 * K)
    A = np.ones((K, K)) / K   # fallback uniform transition matrix
    B = (emit_counts + 1e-8)
    B = B / B.sum(axis=1, keepdims=True)
    return pi, A, B

def sequence_loglikelihood(seqs, pi, A, B):
    total_ll = 0.0
    for s in seqs:
        if len(s) == 0:
            continue
        alpha = forward(s, pi, A, B)   # uses Person-4 forward signature
        # If your forward returns (alpha, scale) change accordingly; here we use old style.
        # If forward returns tuple (alpha, scale) detect and adapt:
        if isinstance(alpha, tuple) or (hasattr(alpha, "__len__") and len(alpha) == 2 and isinstance(alpha[1], np.ndarray)):
            # forward returned (alpha, scale)
            alpha_val = alpha[0]
        else:
            alpha_val = alpha
        prob_seq = np.sum(alpha_val[-1])
        if prob_seq <= 0:
            prob_seq = 1e-300
        total_ll += math.log(prob_seq)
    return total_ll

def train_with_initial(seqs, K, M, initializer, seed=None, max_iter=10, tol=1e-4, verbose=False):
    # Initialize
    if initializer == 'random':
        pi, A, B = init_random(K, M, seed)
    elif initializer == 'uniform':
        pi, A, B = init_uniform(K, M)
    elif initializer == 'cluster':
        pi, A, B = init_cluster_based(seqs, K, M, seed)
    else:
        raise ValueError("Unknown initializer")

    history = []
    prev_ll = None
    t0 = time.time()

    for it in range(max_iter):
        # call Person-4's baum_welch for a single step by setting max_iters=1
        pi_new, A_new, B_new = baum_welch(seqs, pi.copy(), A.copy(), B.copy(), max_iters=1)
        pi, A, B = pi_new, A_new, B_new

        ll = sequence_loglikelihood(seqs, pi, A, B)
        history.append(ll)
        if verbose:
            print(f"[{initializer} seed={seed}] iter={it+1}, loglik={ll:.4f}")
        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    elapsed = time.time() - t0
    return {
        'initializer': initializer,
        'seed': seed,
        'history': history,
        'final_ll': history[-1] if history else sequence_loglikelihood(seqs, pi, A, B),
        'iterations': len(history),
        'time_s': elapsed,
        'pi': pi, 'A': A, 'B': B
    }

def experiment(seqs, K, M, seeds=[0,1,2], max_iter=10):
    results = {it: [] for it in init_types}
    for itype in init_types:
        for sd in seeds:
            res = train_with_initial(seqs, K, M, itype, seed=sd, max_iter=max_iter, tol=1e-6, verbose=False)
            results[itype].append(res)
    return results

def plot_convergence(results, title="Convergence: log-likelihood vs EM iterations"):
    plt.figure(figsize=(9,5))
    for init_type, runs in results.items():
        maxlen = max(len(r['history']) for r in runs)
        data = np.vstack([np.pad(r['history'], (0, maxlen-len(r['history'])), constant_values=np.nan) for r in runs])
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        iters = np.arange(1, len(mean)+1)
        plt.plot(iters, mean, label=f"{init_type} mean")
        plt.fill_between(iters, mean-std, mean+std, alpha=0.2)
    plt.xlabel("EM iterations")
    plt.ylabel("Log-likelihood")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def summary_table(results):
    rows = []
    for init_type, runs in results.items():
        finals = [r['final_ll'] for r in runs]
        iters = [r['iterations'] for r in runs]
        times = [r['time_s'] for r in runs]
        rows.append({
            'initializer': init_type,
            'n_runs': len(runs),
            'final_ll_mean': float(np.mean(finals)),
            'final_ll_std': float(np.std(finals)),
            'iterations_mean': float(np.mean(iters)),
            'time_mean_s': float(np.mean(times)),
        })
    df = pd.DataFrame(rows).sort_values('final_ll_mean', ascending=False)
    return df

# Run a fast pilot (few iterations) to compare initializers
# ---------------------------

pilot_results = experiment(observation_sequences, n_states, n_obs, seeds=seeds, max_iter=pilot_iters)
plot_convergence(pilot_results, title="Pilot: log-likelihood (pilot EM iterations)")
df_pilot = summary_table(pilot_results)
print("\nPilot summary (higher final_ll_mean is better):")

# choose the initializer with highest mean final LL
best_init = df_pilot.iloc[0]['initializer']
print(f"\n[Initialization] Pilot winner: {best_init}")

# Final training on the best initializer
# ---------------------------
print(f"\n[PERSON-5] Running final training with '{best_init}' initializer (max_iter={max_iter_per_run}) ...")
final_results = experiment(observation_sequences, n_states, n_obs, seeds=[seeds[0]], max_iter=max_iter_per_run)
# pick the first run (seed seeds[0]) as final model
final_run = final_results[best_init][0]
pi, A, B = final_run['pi'], final_run['A'], final_run['B']

print("\nFinal run summary:")
print("initializer:", final_run['initializer'], "seed:", final_run['seed'], "iterations:", final_run['iterations'], "time_s:", final_run['time_s'])
print("final log-likelihood:", final_run['final_ll'])
print("Saved final model to globals: pi, A, B")

plot_convergence({best_init: final_results[best_init]}, title=f"Final: {best_init} convergence")



















"""

```
# This is formatted as code
```

# Person 6"""

def viterbi(obs_seq, pi, A, B):
    """
    Viterbi Algorithm (from scratch)
    obs_seq: list or array of observations ([0,1,2,1,...])
    pi: initial distribution
    A: transition matrix
    B: emission matrix
    """
    T = len(obs_seq)
    N = len(pi)

    # Step 1: Initialize DP tables
    delta = np.zeros((T, N))     # highest probability of any path reaching state i at time t
    psi = np.zeros((T, N), dtype=int)  # backpointers

    # Initialization
    delta[0] = pi * B[:, obs_seq[0]]

    # Step 2: Recursion
    for t in range(1, T):
        for j in range(N):  # next state
            probs = delta[t-1] * A[:, j]  # all previous → j
            psi[t, j] = np.argmax(probs)
            delta[t, j] = np.max(probs) * B[j, obs_seq[t]]

    # Step 3: Backtracking
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])  # best last state

    for t in range(T - 2, -1, -1):
        states[t] = psi[t+1, states[t+1]]

    return states, delta

# Test Viterbi on the FIRST sequence
test_seq = observation_sequences[0]

decoded_states, delta = viterbi(test_seq, pi_trained, A_trained, B_trained)

print("\n===== VITERBI DECODING RESULTS =====")
print("Observation Sequence (numbers):")
print(test_seq)

print("\nDecoded Hidden States:")
print(decoded_states)

print("\nMost Likely Final State:", decoded_states[-1])
print("Log Probability of Final Step:", np.log(np.max(delta[-1]) + 1e-300))

print("\n" + "="*60)
print("PERSON 7: ANOMALY DETECTION LOGIC")
print("="*60)



# Step 1: Compute Log-Likelihood for ALL Sequences
# ------------------------------------------------------------
# Using the Forward Algorithm to calculate P(O|λ) for each sequence

def compute_log_likelihood(obs_seq, pi, A, B):
    """
    Compute log P(O|λ) using Forward Algorithm

    Args:
        obs_seq: observation sequence (list/array of integers)
        pi: initial state distribution
        A: transition matrix
        B: emission matrix

    Returns:
        log_likelihood: log probability of the sequence
    """
    if len(obs_seq) == 0:
        return -np.inf

    # Use existing forward function
    alpha = forward(obs_seq, pi, A, B)

    # Handle if forward returns tuple (alpha, scale)
    if isinstance(alpha, tuple):
        alpha = alpha[0]

    # Calculate P(O|λ) = sum of alpha at last time step
    prob_seq = np.sum(alpha[-1])

    # Avoid log(0)
    if prob_seq <= 0:
        prob_seq = 1e-300

    log_likelihood = np.log(prob_seq)
    return log_likelihood


print("\n[Step 1] Computing log-likelihood for all sequences...")
print(f"Total sequences to analyze: {len(observation_sequences)}")

# Calculate log-likelihood for each sequence
log_likelihoods = []
for i, seq in enumerate(observation_sequences):
    ll = compute_log_likelihood(seq, pi_trained, A_trained, B_trained)
    log_likelihoods.append(ll)

    # Show progress every 100 sequences
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(observation_sequences)} sequences...")

log_likelihoods = np.array(log_likelihoods)

print(f"\n✓ Log-likelihood computation complete!")
print(f"  Mean log-likelihood: {np.mean(log_likelihoods):.4f}")
print(f"  Std log-likelihood: {np.std(log_likelihoods):.4f}")
print(f"  Min log-likelihood: {np.min(log_likelihoods):.4f}")
print(f"  Max log-likelihood: {np.max(log_likelihoods):.4f}")

# Step 2: Define Thresholds for Classification
# ------------------------------------------------------------
# Using percentiles to define Normal/Suspicious/Attack boundaries

print("\n[Step 2] Defining classification thresholds...")

# Method 1: Percentile-based thresholds
threshold_attack = np.percentile(log_likelihoods, 5)      # Bottom 5% = Attack
threshold_suspicious = np.percentile(log_likelihoods, 20)  # Bottom 20% = Suspicious

print(f"\n✓ Thresholds defined:")
print(f"  Attack threshold (5th percentile): {threshold_attack:.4f}")
print(f"  Suspicious threshold (20th percentile): {threshold_suspicious:.4f}")
print(f"  Normal threshold: > {threshold_suspicious:.4f}")

# ------------------------------------------------------------
# Step 3: Classify Each Session
# ------------------------------------------------------------

def classify_session(log_likelihood, threshold_suspicious, threshold_attack):
    """
    Classify a session based on its log-likelihood

    Returns:
        'Normal', 'Suspicious', or 'Attack'
    """
    if log_likelihood < threshold_attack:
        return "Attack"
    elif log_likelihood < threshold_suspicious:
        return "Suspicious"
    else:
        return "Normal"


print("\n[Step 3] Classifying all sessions...")

# Classify each sequence
classifications = []
for ll in log_likelihoods:
    label = classify_session(ll, threshold_suspicious, threshold_attack)
    classifications.append(label)

classifications = np.array(classifications)

# Count each category
unique, counts = np.unique(classifications, return_counts=True)
classification_counts = dict(zip(unique, counts))

print(f"\n✓ Classification complete!")
print(f"\nClassification Summary:")
for label in ['Normal', 'Suspicious', 'Attack']:
    count = classification_counts.get(label, 0)
    percentage = (count / len(classifications)) * 100
    print(f"  {label:12s}: {count:5d} sessions ({percentage:5.2f}%)")

# Step 4: Create Results DataFrame
# ------------------------------------------------------------

results_df = pd.DataFrame({
    'Sequence_ID': range(len(observation_sequences)),
    'Log_Likelihood': log_likelihoods,
    'Classification': classifications
})

print(f"\n✓ Results DataFrame created with {len(results_df)} entries")

# Step 5: Visualization - Distribution of Log-Likelihoods
# ------------------------------------------------------------

print("\n[Step 4] Creating visualizations...")

# Plot 1: Histogram with classification zones
plt.figure(figsize=(12, 6))
plt.hist(log_likelihoods, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(threshold_attack, color='red', linestyle='--', linewidth=2, label=f'Attack threshold ({threshold_attack:.2f})')
plt.axvline(threshold_suspicious, color='orange', linestyle='--', linewidth=2, label=f'Suspicious threshold ({threshold_suspicious:.2f})')
plt.xlabel('Log-Likelihood', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Log-Likelihoods with Classification Thresholds', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: Box plot by classification
plt.figure(figsize=(10, 6))
results_df.boxplot(column='Log_Likelihood', by='Classification',
                   patch_artist=True, figsize=(10, 6))
plt.suptitle('')  # Remove default title
plt.title('Log-Likelihood Distribution by Classification', fontsize=14, fontweight='bold')
plt.xlabel('Classification', fontsize=12)
plt.ylabel('Log-Likelihood', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 3: Pie chart of classifications
st.subheader("Classification Pie Chart")

class_counts = results_df["Classification"].value_counts()

if len(class_counts) == 0:
    st.info("No data to plot.")
else:
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(
        class_counts.values,
        labels=class_counts.index,
        autopct='%1.1f%%',
        startangle=140
    )

    ax.set_title("Classification Distribution")
    ax.axis("equal")  # Ensures perfect circle
    
    st.pyplot(fig)

# Plot 4: Scatter plot - Sequence ID vs Log-Likelihood
plt.figure(figsize=(14, 6))
for label, color in zip(['Normal', 'Suspicious', 'Attack'],
                        ['green', 'orange', 'red']):
    mask = results_df['Classification'] == label
    plt.scatter(results_df[mask]['Sequence_ID'],
               results_df[mask]['Log_Likelihood'],
               c=color, label=label, alpha=0.6, s=20)

plt.axhline(threshold_attack, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
plt.axhline(threshold_suspicious, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
plt.xlabel('Sequence ID', fontsize=12)
plt.ylabel('Log-Likelihood', fontsize=12)
plt.title('Anomaly Detection: Sequence Log-Likelihood Over Time', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✓ All visualizations generated successfully!")

# Step 6: Detailed Examples
# ------------------------------------------------------------

print("\n[Step 5] Extracting detailed examples...")

# Get examples from each category
normal_examples = results_df[results_df['Classification'] == 'Normal'].head(3)
suspicious_examples = results_df[results_df['Classification'] == 'Suspicious'].head(3)
attack_examples = results_df[results_df['Classification'] == 'Attack'].head(3)

print("\n" + "="*60)
print("EXAMPLE SEQUENCES:")
print("="*60)

print("\n--- NORMAL SEQUENCES (High Likelihood) ---")
for idx, row in normal_examples.iterrows():
    seq_id = int(row['Sequence_ID'])
    ll = row['Log_Likelihood']
    print(f"\nSequence #{seq_id}: Log-Likelihood = {ll:.4f}")
    print(f"  Observations: {observation_sequences[seq_id][:10]}...")  # First 10 observations

print("\n--- SUSPICIOUS SEQUENCES (Medium Likelihood) ---")
for idx, row in suspicious_examples.iterrows():
    seq_id = int(row['Sequence_ID'])
    ll = row['Log_Likelihood']
    print(f"\nSequence #{seq_id}: Log-Likelihood = {ll:.4f}")
    print(f"  Observations: {observation_sequences[seq_id][:10]}...")

print("\n--- ATTACK SEQUENCES (Low Likelihood) ---")
for idx, row in attack_examples.iterrows():
    seq_id = int(row['Sequence_ID'])
    ll = row['Log_Likelihood']
    print(f"\nSequence #{seq_id}: Log-Likelihood = {ll:.4f}")
    print(f"  Observations: {observation_sequences[seq_id][:10]}...")

# ------------------------------------------------------------
# Step 7: Statistical Summary
# ------------------------------------------------------------

print("\n" + "="*60)
print("STATISTICAL SUMMARY BY CLASSIFICATION:")
print("="*60)

summary_stats = results_df.groupby('Classification')['Log_Likelihood'].describe()
print("\n", summary_stats)

# Step 8: Save Results
# ------------------------------------------------------------

print("\n[Step 6] Saving results...")

# Save to CSV
results_df.to_csv('anomaly_detection_results.csv', index=False)
print("✓ Results saved to 'anomaly_detection_results.csv'")

# Save summary statistics
with open('anomaly_detection_summary.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("ANOMALY DETECTION SUMMARY (PERSON 7)\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total Sequences Analyzed: {len(observation_sequences)}\n\n")
    f.write("Classification Counts:\n")
    for label in ['Normal', 'Suspicious', 'Attack']:
        count = classification_counts.get(label, 0)
        percentage = (count / len(classifications)) * 100
        f.write(f"  {label}: {count} ({percentage:.2f}%)\n")
    f.write(f"\nThresholds Used:\n")
    f.write(f"  Attack threshold: {threshold_attack:.4f}\n")
    f.write(f"  Suspicious threshold: {threshold_suspicious:.4f}\n")
    f.write("\n" + "="*60 + "\n")

print("✓ Summary saved to 'anomaly_detection_summary.txt'")

# Final Output
# ------------------------------------------------------------

print("\n" + "="*60)
print("PERSON 7: ANOMALY DETECTION COMPLETE!")
print("="*60)
print("\nKey Findings:")
print(f"  • {classification_counts.get('Normal', 0)} Normal sessions")
print(f"  • {classification_counts.get('Suspicious', 0)} Suspicious sessions")
print(f"  • {classification_counts.get('Attack', 0)} Attack sessions")
print(f"\n  • Attack threshold: {threshold_attack:.4f}")
print(f"  • Suspicious threshold: {threshold_suspicious:.4f}")
print("\nAll results saved and visualizations generated!")
print("="*60)