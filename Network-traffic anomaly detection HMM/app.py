import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Streamlit page config
st.set_page_config(
    page_title="HMM Network Anomaly Detection",
    layout="wide"
)

st.title("Hidden Markov Model")

st.markdown(
    """
Upload a **cleaned network dataset** .
"""
)

# =========================
# Utility functions
# =========================

def create_sequences(data, seq_length=10):
    sequences = []
    for i in range(0, len(data) - seq_length + 1, seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)


def forward(obs_seq, pi, A, B):
    T = len(obs_seq)
    N = len(pi)
    alpha = np.zeros((T, N))

    alpha[0] = pi * B[:, obs_seq[0]]

    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs_seq[t]]

    return alpha


def backward(obs_seq, A, B):
    T = len(obs_seq)
    N = A.shape[0]
    beta = np.zeros((T, N))
    beta[-1] = 1.0

    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i] * B[:, obs_seq[t+1]] * beta[t+1])

    return beta


def compute_sequence_log_likelihood(obs_seq, pi, A, B):
    if len(obs_seq) == 0:
        return -np.inf
    alpha = forward(obs_seq, pi, A, B)
    prob_seq = np.sum(alpha[-1])
    if prob_seq <= 0:
        prob_seq = 1e-300
    return np.log(prob_seq)


def classify_session(log_likelihood, thr_suspicious, thr_attack):
    if log_likelihood < thr_attack:
        return "Attack"
    elif log_likelihood < thr_suspicious:
        return "Suspicious"
    else:
        return "Normal"


def baum_welch(sequences, n_states, n_obs, max_iters=10, tol=1e-4):
    # Random initialization
    rng = np.random.RandomState(0)
    pi = np.full(n_states, 1.0 / n_states)

    A = rng.rand(n_states, n_states)
    A /= A.sum(axis=1, keepdims=True)

    B = rng.rand(n_states, n_obs)
    B /= B.sum(axis=1, keepdims=True)

    ll_history = []
    prev_ll = None

    for iteration in range(max_iters):
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

            # Gamma
            gamma = (alpha * beta) / prob_seq

            # Xi
            xi = np.zeros((T - 1, n_states, n_states))
            for t in range(T - 1):
                numerator = alpha[t][:, None] * A * (B[:, obs_seq[t+1]] * beta[t+1])[None, :]
                denom = np.sum(numerator)
                if denom < 1e-300:
                    denom = 1e-300
                xi[t] = numerator / denom

            # Update accumulators
            pi_new += gamma[0]
            A_num += np.sum(xi, axis=0)
            A_den += np.sum(gamma[:-1], axis=0)

            for t in range(T):
                B_num[:, obs_seq[t]] += gamma[t]
            B_den += np.sum(gamma, axis=0)

        # Normalize to get new parameters
        pi = pi_new / np.sum(pi_new)
        A = A_num / A_den[:, None]
        B = B_num / B_den[:, None]

        # Compute total log-likelihood
        total_ll = 0.0
        for obs_seq in sequences:
            total_ll += compute_sequence_log_likelihood(obs_seq, pi, A, B)
        ll_history.append(total_ll)

        if prev_ll is not None and abs(total_ll - prev_ll) < tol:
            break
        prev_ll = total_ll

    return pi, A, B, ll_history


# =========================
# Data loading
# =========================

st.sidebar.header("1. Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload cleaned network CSV (must include 'Emission' column):",
    type=["csv"]
)

default_info_placeholder = st.sidebar.empty()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")
else:
    df = None
    st.info(
        "Upload a cleaned dataset CSV to start. It should contain an `Emission` column "
        "similar to the one you generated in your Colab script."
    )

if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head())

    st.markdown("**Columns:** " + ", ".join(df.columns.astype(str)))

    if "Attack category" in df.columns:
        st.write("Attack category distribution:")
        st.bar_chart(df["Attack category"].value_counts())

# =========================
# HMM configuration
# =========================

st.sidebar.header("2. HMM Parameters")

n_states = st.sidebar.slider("Number of hidden states", 2, 8, 3)
seq_length = st.sidebar.slider("Sequence length", 5, 50, 10, step=5)
max_iters = st.sidebar.slider("Max Baum–Welch iterations", 5, 50, 15, step=5)
max_sequences = st.sidebar.number_input(
    "Max number of sequences to use (0 = use all)", 
    min_value=0, step=100, value=0
)

st.sidebar.header("3. Anomaly Thresholds (percentiles)")
attack_pct = st.sidebar.slider("Attack threshold (bottom %)", 1, 20, 5)
suspicious_pct = st.sidebar.slider("Suspicious threshold (bottom %)", 5, 50, 20)

if suspicious_pct <= attack_pct:
    st.sidebar.error("Suspicious percentile must be greater than attack percentile.")

# =========================
# Prepare sequences
# =========================

if df is not None and "Emission" in df.columns and suspicious_pct > attack_pct:
    st.subheader("Sequence construction and encoding")

    # Encode the Emission column into integer observations
    le = LabelEncoder()
    df["Observation"] = le.fit_transform(df["Emission"].astype(str))
    n_obs = df["Observation"].nunique()

    st.write(f"Total observations: {len(df)}")
    st.write(f"Unique emission types: {n_obs}")

    obs_array = df["Observation"].values
    sequences = create_sequences(obs_array, seq_length=seq_length)

    if len(sequences) == 0:
        st.error("Not enough data to create sequences for the chosen sequence length.")
    else:
        if max_sequences > 0 and max_sequences < len(sequences):
            sequences = sequences[:max_sequences]
        st.write(f"Total sequences created: {len(sequences)}")
        st.write(f"Example sequence (first 10 observations): {sequences[0][:10]}")

        # =========================
        # Train HMM
        # =========================
        st.subheader("HMM Training")

        if st.button("Train HMM"):
            with st.spinner("Training HMM with Baum–Welch..."):
                pi, A, B, ll_history = baum_welch(
                    sequences,
                    n_states=n_states,
                    n_obs=n_obs,
                    max_iters=max_iters
                )

            st.success("Training complete.")

            # Show basic info
            st.write("Initial state distribution (π):")
            st.write(pi)

            st.write("Transition matrix (A) – first few rows:")
            st.dataframe(pd.DataFrame(A).round(4).head())

            st.write("Emission matrix (B) – first few rows:")
            st.dataframe(pd.DataFrame(B).round(4).head())

            # Plot log-likelihood convergence (centered, small)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(range(1, len(ll_history) + 1), ll_history, marker="o")
                ax.set_xlabel("EM iteration")
                ax.set_ylabel("Total log-likelihood")
                ax.set_title("Baum–Welch convergence")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, use_container_width=False)

            # =========================
            # Anomaly detection
            # =========================
            st.subheader("Anomaly Detection")

            # Compute log-likelihood for each sequence
            log_likelihoods = []
            for seq in sequences:
                ll = compute_sequence_log_likelihood(seq, pi, A, B)
                log_likelihoods.append(ll)
            log_likelihoods = np.array(log_likelihoods)

            st.write(
                f"Mean log-likelihood: {log_likelihoods.mean():.4f}, "
                f"Std: {log_likelihoods.std():.4f}"
            )

            thr_attack = np.percentile(log_likelihoods, attack_pct)
            thr_suspicious = np.percentile(log_likelihoods, suspicious_pct)

            st.write(
                f"Attack threshold ({attack_pct}th percentile): {thr_attack:.4f} "
                f"Suspicious threshold ({suspicious_pct}th percentile): {thr_suspicious:.4f}"
            )

            # Classify each sequence
            classifications = [
                classify_session(ll, thr_suspicious, thr_attack)
                for ll in log_likelihoods
            ]
            classifications = np.array(classifications)

            # Build results DataFrame
            results_df = pd.DataFrame({
                "Sequence_ID": np.arange(len(sequences)),
                "Log_Likelihood": log_likelihoods,
                "Classification": classifications
            })

            st.write("Classification counts:")
            st.table(
                results_df["Classification"].value_counts()
                .rename("Count")
                .to_frame()
            )

            # Pie Chart of Classifications (centered, small)
            st.subheader("Classification Pie Chart")

            class_counts = results_df["Classification"].value_counts()

            if len(class_counts) == 0:
                st.info("No data to plot.")
            else:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.pie(
                        class_counts.values,
                        labels=class_counts.index,
                        autopct='%1.1f%%',
                        startangle=140,
                        textprops={'fontsize': 8}
                    )
                    ax.set_title("Classification Distribution", fontsize=11)
                    ax.axis("equal")
                    st.pyplot(fig, use_container_width=False)

            # Histogram with thresholds (centered, small)
            st.markdown("### Log-likelihood distribution with thresholds")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                ax_hist.hist(log_likelihoods, bins=40, edgecolor="black", alpha=0.7)
                ax_hist.axvline(thr_attack, color="red", linestyle="--", label="Attack threshold")
                ax_hist.axvline(thr_suspicious, color="orange", linestyle="--", label="Suspicious threshold")
                ax_hist.set_xlabel("Log-likelihood")
                ax_hist.set_ylabel("Frequency")
                ax_hist.legend(fontsize=8)
                ax_hist.grid(True, alpha=0.3)
                st.pyplot(fig_hist, use_container_width=False)

            # Box plot by classification (centered, small)
            st.markdown("### Log-likelihood by classification")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig_box, ax_box = plt.subplots(figsize=(10, 4))
                results_df.boxplot(
                    column="Log_Likelihood",
                    by="Classification",
                    ax=ax_box
                )
                ax_box.set_title("Log-likelihood distribution by class", fontsize=11)
                ax_box.set_xlabel("Classification")
                ax_box.set_ylabel("Log-likelihood")
                fig_box.suptitle("")
                ax_box.grid(True, alpha=0.3)
                st.pyplot(fig_box, use_container_width=False)

            # Scatter plot (centered, small)
            st.markdown("### Sequence ID vs log-likelihood")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig_scatter, ax_scatter = plt.subplots(figsize=(10, 4))
                colors = {"Normal": "green", "Suspicious": "orange", "Attack": "red"}
                for label in ["Normal", "Suspicious", "Attack"]:
                    mask = results_df["Classification"] == label
                    ax_scatter.scatter(
                        results_df.loc[mask, "Sequence_ID"],
                        results_df.loc[mask, "Log_Likelihood"],
                        s=20,
                        alpha=0.6,
                        label=label,
                        c=colors[label]
                    )
                ax_scatter.axhline(thr_attack, color="red", linestyle="--")
                ax_scatter.axhline(thr_suspicious, color="orange", linestyle="--")
                ax_scatter.set_xlabel("Sequence ID")
                ax_scatter.set_ylabel("Log-likelihood")
                ax_scatter.legend(fontsize=8)
                ax_scatter.grid(True, alpha=0.3)
                st.pyplot(fig_scatter, use_container_width=False)

            # Statistical summary
            st.markdown("### Statistical summary")
            st.dataframe(results_df.groupby("Classification")["Log_Likelihood"].describe())

            # Example sequences per class
            st.markdown("### Example sequences")

            for label in ["Normal", "Suspicious", "Attack"]:
                st.markdown(f"**{label} examples**")
                subset = results_df[results_df["Classification"] == label].head(3)
                if subset.empty:
                    st.write("No sequences in this class.")
                else:
                    for _, row in subset.iterrows():
                        seq_id = int(row["Sequence_ID"])
                        ll = row["Log_Likelihood"]
                        seq = sequences[seq_id]
                        st.write(f"Sequence #{seq_id}, log-likelihood = {ll:.4f}")
                        st.write(f"Observations (first 10): {seq[:10]}")
                        st.markdown("---")
        else:
            st.info("Press **Train HMM** to start training and anomaly detection.")
else:
    if df is not None and "Emission" not in df.columns:
        st.error("The uploaded dataframe does not contain an 'Emission' column.")
