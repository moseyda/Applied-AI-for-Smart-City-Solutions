import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

# Reminder: Before running the script, set this environment variable in your shell/command prompt:
# Windows CMD: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Linux/macOS: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM Model
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

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def evaluate_in_batches(model, X_tensor, seq_length, batch_size=128):
    model.eval()
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.view(-1, seq_length, 1).to(device)
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                output = model(batch_X)
            preds.append(output.cpu())
    return torch.cat(preds).numpy()

def train_model(X_tensor, y_tensor, seq_length, hidden_size, dropout, batch_size, epochs=5, learning_rate=0.001):
    torch.cuda.empty_cache()  # Clear GPU cache before training

    model = LSTMModel(hidden_size=hidden_size, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # For mixed precision training

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.view(-1, seq_length, 1).to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                output = model(batch_X)
                loss = criterion(output, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    # Evaluation using batch processing to prevent OOM
    predictions = evaluate_in_batches(model, X_tensor, seq_length, batch_size=128)
    return model, predictions

def moving_average(data, window=24):
    return np.convolve(data.flatten(), np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    # Load preprocessed data and scaler
    scaled_data = np.load("scaled_power.npy")
    scaler = joblib.load("scaler.save")

    ############### === MANUAL HYPERPARAMETERS - EDIT THESE BEFORE EACH RUN === ############### 
    seq_length = 64
    learning_rate = 0.001
    hidden_size = 128
    dropout = 0.1
    batch_size = 24
    epochs = 5
    ############### =========================================================== ############### 

    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Train model with current settings
    model, preds = train_model(X_tensor, y_tensor, seq_length, hidden_size, dropout, batch_size, epochs=epochs, learning_rate=learning_rate)

    # Invert scaling
    preds_inverted = scaler.inverse_transform(preds)
    true_inverted = scaler.inverse_transform(y.reshape(-1, 1))

    # Calculate MSE
    mse = mean_squared_error(true_inverted, preds_inverted)

    # Calculate moving average baseline MSE
    ma_preds = moving_average(true_inverted.flatten(), window=seq_length)
    aligned_true = true_inverted[seq_length:]

    # Align lengths if mismatch (if necessary)
    if len(aligned_true) != len(ma_preds):
        min_len = min(len(aligned_true), len(ma_preds))
        aligned_true = aligned_true[:min_len]
        ma_preds = ma_preds[:min_len]

    ma_mse = mean_squared_error(aligned_true, ma_preds)

    print(f"\nTested params: seq_length={seq_length}, lr={learning_rate}, hidden={hidden_size}, dropout={dropout}, batch={batch_size}")
    print(f"LSTM Model MSE: {mse:.6f}")
    print(f"Moving Average MSE (window={seq_length}): {ma_mse:.6f}")

    # Save the model and results
    filename = f"model_seq{seq_length}_lr{learning_rate}_hs{hidden_size}_dr{dropout}_bs{batch_size}.pth"
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

    # Save predictions and true values for later analysis
    np.save(f"preds_{filename}.npy", preds_inverted)
    np.save(f"true_{filename}.npy", true_inverted)

    # Plot results for first 200 points
    plt.figure(figsize=(12, 6))
    plt.plot(true_inverted[:200], label="True")
    plt.plot(preds_inverted[:200], label="LSTM Prediction")
    plt.plot(np.arange(seq_length, seq_length + 200), ma_preds[:200], label="Moving Average", linestyle="--")
    plt.legend()
    plt.title(f"Energy Forecasting (seq={seq_length}, lr={learning_rate}, hs={hidden_size}, dr={dropout}, bs={batch_size})")
    plt.xlabel("Time Steps")
    plt.ylabel("Global Active Power (kilowatts)")
    plt.grid()
    plt.show()
