import torch
print(torch.__version__)
print("Torch is working!")
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
device
ticker ='AAPL'
df = yf.download(ticker, '2020-01-01')
df
df.Close.plot(figsize=(12,8))
scaler = StandardScaler()

df['Close'] = scaler.fit_transform(df['Close'])
df.Close
# Ensure matching rows
min_len = min(len(x_train), len(y_train))

x_train = x_train[:min_len]
y_train = y_train[:min_len]

if len(y_train.shape) == 1:
    y_train = y_train.view(-1, 1)
seq_length = 30
x_data = []
y_data = []

for i in range(len(df) - seq_length):
    x_data.append(df.Close.iloc[i:i+seq_length])
    y_data.append(df.Close.iloc[i+seq_length])   # next day prediction

x_data = np.array(x_data)
y_data = np.array(y_data)

# reshape tensors
x_train = torch.tensor(x_data).float().unsqueeze(-1)   # [N, 30, 1]
y_train = torch.tensor(y_data).float().unsqueeze(-1)   # [N, 1]

x_train = torch.tensor(x, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)
train_size = int(0.8 * len(data))

x_train = torch.from_numpy(data[:train_size, :-1, :1]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(data[:train_size, :-1, :1]).type(torch.Tensor).to(device)
x_test = torch.from_numpy(data[:train_size:, :-1, :1]).type(torch.Tensor).to(device)
y_test = torch.from_numpy(data[:train_size:, :-1, :1]).type(torch.Tensor).to(device)
x_train
y_train
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("model output shape:", model(x_train).shape)
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )

        # predict 29 output values
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim
        ).to(x.device)

        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim
        ).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])

        return out
model = PredictionModel(
    input_dim=1,
    hidden_dim=32,
    num_layers=2,
    output_dim=1
).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
print(x_train.shape)
print(y_train.shape)
print(model(x_train).shape)
# reshape back to the original (1157, 29, 1)
y_train = y_train.reshape(1157, 29, 1)

# use the last timestep as the label
y_train = y_train[:, -1, :]

print("y_train FIXED:", y_train.shape)
print(x_train.shape)
print(y_train.shape)
print(model(x_train).shape)
# Fix y_train shape
if isinstance(y_train, torch.Tensor):
    # y_train is (1157, 29, 1) → keep only last timestep
    y_train = y_train[:, -1, :]   # now (1157, 1)
else:
    # y_train might be numpy
    y_train = y_train[:, -1].reshape(-1, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32)

print("NEW y_train:", y_train.shape)
num_epochs = 200

for i in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train)

    if i % 25 == 0:
        print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# y_train_pred is a tensor → convert properly
y_train_pred_np = y_train_pred.detach().cpu().numpy()

# y_train is already numpy → use directly
y_train_np = y_train

# y_test_pred is a tensor → convert properly
y_test_pred_np = y_test_pred.detach().cpu().numpy()

# Inverse transform
y_train_pred_inv = scaler.inverse_transform(y_train_pred_np)
y_train_inv = scaler.inverse_transform(y_train_np)
y_test_pred_inv = scaler.inverse_transform(y_test_pred_np)

y_train_pred = model(x_train)
y_test_pred = model(x_test)

# Convert tensor predictions to numpy
y_train_pred = y_train_pred.detach().cpu().numpy()
y_test_pred = y_test_pred.detach().cpu().numpy()

# Inverse transform predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

# Inverse transform true values (already numpy)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
train_rmse = root_mean_squared_error(y_train[:, 0], y_train_pred[:, 0])
test_rmse = root_mean_squared_error(y_train[:, 0], y_test_pred[:, 0])
train_rmse
test_rmse
# Create dates for plotting
dates = df.iloc[-len(y_test):].index

# Check shapes
print("dates length:", len(dates))
print("y_test shape:", y_test.shape)
print("y_test_pred shape:", y_test_pred.shape)

print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print("y_test_pred shape:", y_test_pred.shape)
print("df length:", len(df))
# --- FIX y_test properly ---
# If y_test came from your scaling step, get the correct last 1157 values from df

correct_y_test = df['Close'].values[-1157:]     # or whatever your target column is

# Reshape to (1157, 1)
correct_y_test = correct_y_test.reshape(-1, 1)

# Replace the broken one
y_test = correct_y_test

print("NEW y_test shape:", y_test.shape)
fig = plt.figure(figsize=(12,10))
gs = fig.add_gridspec(4,1)

ax1 = fig.add_subplot(gs[:3, 0])
ax1.plot(df.iloc[-len(y_test):].index, y_test, color = 'blue', label = 'Actual price')
ax1.plot(df.iloc[-len(y_test):].index, y_test_pred, color = 'green', label = 'Predicted price')
ax1.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('price')

ax2 = fig.add_subplot(gs[3,0])
ax2.axhline(test_rmse, color = 'blue', linestyle='--',label='RMSE')
ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_pred), 'r', label = 'Prediction Error')
ax2.legend()
plt.title('Prediction Erro')
plt.xlabel('Date')
plt.ylabel('Error')

plt.tight_layout()
plt.show()
