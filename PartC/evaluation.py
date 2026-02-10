import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Load processed and normalised data
scaled_data = np.load("scaled_power.npy")
scaler = joblib.load("scaler.save")

# Model definition (should be the same as training)
class LSTMModel(nn.Module):
    def __init__(self, hidden_size=128, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Set device (using CPU to avoid CUDA out of memory)
device = torch.device("cpu")

# Load model weights
model = LSTMModel(hidden_size=128, dropout=0.2)
model.load_state_dict(torch.load("model_seq64_lr0.001_hs128_dr0.1_bs24.pth", map_location=device))
model.to(device)
model.eval()

# Load true values
true_values_tensor = np.load("true_model_seq64_lr0.001_hs128_dr0.1_bs24.npy")
true_values_raw = true_values_tensor

# Sequence length must match training
seq_length = 64

# Create sequences from scaled data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, seq_length)

# Batch prediction to avoid OOM
batch_size = 64
predicted_list = []

with torch.no_grad():
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.float32).view(-1, seq_length, 1).to(device)
        preds = model(batch_tensor).cpu().numpy()
        predicted_list.append(preds)

predicted_values_scaled = np.concatenate(predicted_list, axis=0)

# Inverse scaling
predicted_values = scaler.inverse_transform(predicted_values_scaled)
true_values = scaler.inverse_transform(y.reshape(-1, 1))

# Moving average baseline
def moving_average(data, window=seq_length):
    return np.convolve(data.flatten(), np.ones(window)/window, mode='valid')

ma_preds = moving_average(true_values.flatten(), window=seq_length)
aligned_true = true_values[seq_length:]  # Align length to moving average output

# Fix length mismatch if needed
if len(aligned_true) != len(ma_preds):
    min_len = min(len(aligned_true), len(ma_preds))
    aligned_true = aligned_true[:min_len]
    ma_preds = ma_preds[:min_len]

# Calculate MSE
ma_mse = mean_squared_error(aligned_true, ma_preds)
model_mse = mean_squared_error(true_values, predicted_values)

np.save("rnn_predictions.npy", predicted_values)
np.save("true_values.npy", true_values)

print("\n--- Evaluation Results ---")
print(f"Moving Average MSE: {ma_mse:.6f}")
print(f"LSTM Model MSE: {model_mse:.6f}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(true_values[:200], label="True")
plt.plot(predicted_values[:200], label="LSTM Prediction")
plt.plot(np.arange(seq_length, seq_length + 200), ma_preds[:200], label="Moving Average", linestyle="--")
plt.legend()
plt.title("Energy Consumption Forecasting")
plt.xlabel("Time Steps")
plt.ylabel("Global Active Power (kilowatts)")
plt.grid()
plt.show()
