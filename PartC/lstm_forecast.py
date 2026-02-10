########### ########## #########
########### ########## #########
#     IMPORTANT, READ THIS     #
########### ########## #########
########### ########## #########
#LSTM Forecasting for Energy Consumption 
#  This file is ONLY for the first trained model to test script functionality, not the tuned one, refer to Report documentation, figures 28, 29.

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Load processed and normalised data
scaled_data = np.load("scaled_power.npy")
scaler = joblib.load("scaler.save")

# Model definition
class LSTMModel(nn.Module):
    def __init__(self, hidden_size=128, dropout=0.2):
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

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model weights
model = LSTMModel(hidden_size=128, dropout=0.2)
model.load_state_dict(torch.load("trained_lstm_energy_model.pth"))
model.eval()

# --- Step 4: Prediction ---

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 24
X, y = create_sequences(scaled_data, seq_length)

X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, seq_length, 1)

with torch.no_grad():
    predictions = model(X_tensor).cpu().numpy()

# Invert scaling
predicted_values = scaler.inverse_transform(predictions)
true_values = scaler.inverse_transform(y.reshape(-1, 1))

# --- Step 5: Hyperparameter Tuning (with multiple epochs) ---

def try_hyperparams(hidden_size, dropout, batch_size, epochs=5):
    temp_model = LSTMModel(hidden_size=hidden_size, dropout=dropout)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.001)

    dataset = torch.utils.data.TensorDataset(X_tensor, torch.tensor(y, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    temp_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.view(-1, seq_length, 1)
            optimizer.zero_grad()
            output = temp_model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    temp_model.eval()
    with torch.no_grad():
        test_preds = temp_model(X_tensor).cpu().numpy()
    mse = mean_squared_error(true_values, scaler.inverse_transform(test_preds))
    return mse

# Try hyperparameter grid
grid_results = {}
for hs in [64, 128]:
    for dr in [0.2, 0.3]:
        for bs in [64, 128]:
            print(f"Testing Hidden={hs}, Dropout={dr}, Batch={bs}")
            mse = try_hyperparams(hs, dr, bs, epochs=5)
            grid_results[(hs, dr, bs)] = mse
            print(f"Result: MSE={mse:.6f}\n")

best_params = min(grid_results, key=grid_results.get)
print("\nBest hyperparameters:", best_params, "with MSE =", grid_results[best_params])

# --- Step 6: Baseline Comparison ---

def moving_average(data, window=24):
    return np.convolve(data.flatten(), np.ones(window)/window, mode='valid')

ma_preds = moving_average(true_values, window=24)

aligned_true = true_values[24:]

ma_mse = mean_squared_error(aligned_true, ma_preds)
model_mse = mean_squared_error(true_values, predicted_values)

print("\n--- Evaluation Results ---")
print(f"Moving Average MSE: {ma_mse:.4f}")
print(f"LSTM Model MSE: {model_mse:.4f}")


# Plot results
plt.figure(figsize=(12, 6))
plt.plot(true_values[:200], label="True")
plt.plot(predicted_values[:200], label="LSTM Prediction")
plt.plot(np.arange(24, 224), ma_preds[:200], label="Moving Average", linestyle="--")
plt.legend()
plt.title("Energy Consumption Forecasting")
plt.xlabel("Time Steps")
plt.ylabel("Global Active Power (kilowatts)")
plt.grid()
plt.show()
