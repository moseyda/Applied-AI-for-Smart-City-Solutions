import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import joblib

# ---------------------------
# 1. Load Data & Preprocess
# ---------------------------

# Load scaled energy data and scaler
scaled_data = np.load("scaled_power.npy").flatten()
scaler = joblib.load("scaler.save")

# Define discrete state space
n_states = 10
bins = np.linspace(np.min(scaled_data), np.max(scaled_data), n_states + 1)

# Discretize the continuous data into state indices
state_indices = np.digitize(scaled_data, bins, right=True) - 1
state_indices = np.clip(state_indices, 0, n_states - 1)

# ---------------------------
# 2. Transition Matrix
# ---------------------------

transition_matrix = np.zeros((n_states, n_states))
for (s1, s2) in zip(state_indices[:-1], state_indices[1:]):
    transition_matrix[s1, s2] += 1

# Normalize rows to obtain transition probabilities
with np.errstate(invalid="ignore"):
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

# ---------------------------
# 3. Reward Function
# ---------------------------

# Lower energy consumption states are better (more reward)
rewards = -np.array([
    np.mean(scaled_data[state_indices == s]) if np.any(state_indices == s) else 0
    for s in range(n_states)
])

# ---------------------------
# 4. Value Iteration (Policy Optimisation)
# ---------------------------

gamma = 0.9  # Discount factor
n_iter = 100
V = np.zeros(n_states)
policy = np.zeros(n_states, dtype=int)

for _ in range(n_iter):
    Q = np.zeros((n_states, n_states))
    for s in range(n_states):
        for a in range(n_states):
            Q[s, a] = rewards[a] + gamma * np.dot(transition_matrix[a], V)
    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)

# ---------------------------
# 5. Evaluation & Comparison
# ---------------------------

# Apply policy to each time step's state
policy_actions = policy[state_indices[:-1]]

# Predict values using the policy (average values from selected states)
mdp_predictions = np.array([
    np.mean(scaled_data[state_indices == a]) if np.any(state_indices == a) else 0
    for a in policy_actions
])
mdp_predictions = scaler.inverse_transform(mdp_predictions.reshape(-1, 1)).flatten()

# Load true values and RNN predictions from Part C
true_values = np.load("true_values.npy").flatten()
rnn_predictions = np.load("rnn_predictions.npy").flatten()

# Ensure same length for comparison
min_len = min(len(true_values), len(mdp_predictions), len(rnn_predictions))
true_values = true_values[:min_len]
mdp_predictions = mdp_predictions[:min_len]
rnn_predictions = rnn_predictions[:min_len]

# Compute MSE
mdp_mse = mean_squared_error(true_values, mdp_predictions)
rnn_mse = mean_squared_error(true_values, rnn_predictions)

print("\n--- Evaluation Comparison ---")
print(f"MDP-based Prediction MSE: {mdp_mse:.6f}")
print(f"RNN Prediction MSE:       {rnn_mse:.6f}")

# ---------------------------
# 6. Visualisation
# ---------------------------

plt.figure(figsize=(12, 6))
plt.plot(true_values[:200], label="True Energy Consumption")
plt.plot(mdp_predictions[:200], label="MDP Predicted", linestyle='--')
plt.plot(rnn_predictions[:200], label="RNN Predicted", linestyle='dotted')
plt.title("Energy Consumption: True vs MDP vs RNN")
plt.xlabel("Time Steps")
plt.ylabel("Global Active Power (kilowatts)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# 7. Transition Matrix Heatmap
# ---------------------------

plt.figure(figsize=(8, 6))
sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="viridis")
plt.title("Transition Probability Matrix Heatmap")
plt.xlabel("Next State")
plt.ylabel("Current State")
plt.tight_layout()
plt.show()

# ---------------------------
# 8. Reward Function Bar Plot
# ---------------------------

plt.figure(figsize=(8, 4))
sns.barplot(x=np.arange(n_states), y=rewards, palette="coolwarm")
plt.title("Reward Function by State")
plt.xlabel("State")
plt.ylabel("Reward (Negative Avg Energy Consumption)")
plt.tight_layout()
plt.show()
