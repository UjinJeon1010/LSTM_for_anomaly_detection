import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import gc
import os

# ==================== HYPERPARAMETERS ====================
dropout_rate = 0.1
lstm_layers = 3
epochs = 80          # you can start smaller (e.g. 50) to debug
learning_rate = 0.002
batch_size = 64
window_size = 5
horizon = 1

base_path = "/Users/ujinjeon/python/Anomaly_detection"

normal_path = f"{base_path}/training_data_normal.txt"
test_path   = f"{base_path}/test_data.txt"

def load_run(path):
    df = pd.read_csv(path, delimiter="\t", engine="python")
    df.columns = [c.strip() for c in df.columns]
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.rename(columns={
        "RF Time": "Time",
        "For Pwr (W)": "For",
        "Refl Pwr (W)": "Refl",
    })
    return df.reset_index(drop=True)

df_train_normal = load_run(normal_path)
df_test         = load_run(test_path)

df_train_normal_orig = df_train_normal.copy()
df_test_orig         = df_test.copy()

scaler = StandardScaler().fit(df_train_normal[["For", "Refl"]])
df_train_normal[["For", "Refl"]] = scaler.transform(df_train_normal[["For", "Refl"]])
df_test[["For", "Refl"]]         = scaler.transform(df_test[["For", "Refl"]])

def make_windows(df, window_size, horizon):
    X, y, idxs = [], [], []
    n = len(df)
    for i in range(n - window_size - horizon + 1):
        target_idx = i + window_size + horizon - 1
        window = df[["For", "Refl"]].iloc[i:i+window_size].values   # (w,2)
        target = df[["For", "Refl"]].iloc[target_idx].values        # (2,)
        X.append(window)
        y.append(target)
        idxs.append(target_idx)
    return np.array(X), np.array(y), np.array(idxs)

X_train, y_train, idx_train = make_windows(df_train_normal, window_size, horizon)
X_test,  y_test,  idx_test  = make_windows(df_test,         window_size, horizon)

# ==================== CREATE TENSORS / DATALOADERS ====================
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
train_dataset  = TensorDataset(X_train_tensor, y_train_tensor)
train_loader   = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

X_test_tensor  = torch.FloatTensor(X_test)

# ==================== MODEL DEFINITION ====================
class CNNLSTMWithDropoutCell(nn.Module):
    def __init__(self, in_dim, hid_dim, p_drop):
        super().__init__()
        self.p = p_drop
        self.W_i = nn.Linear(in_dim + hid_dim, hid_dim)
        self.W_f = nn.Linear(in_dim + hid_dim, hid_dim)
        self.W_o = nn.Linear(in_dim + hid_dim, hid_dim)
        self.W_g = nn.Linear(in_dim + hid_dim, hid_dim)
        self.register_buffer("mask", None)

    def _locked_mask(self, x):
        if self.training and self.mask is None:
            self.mask = (torch.rand_like(x[:, :1]) > self.p).float() / (1.0 - self.p)
        return self.mask if self.training else 1.0

    def forward(self, x_t, h_c_prev):
        h_prev, c_prev = h_c_prev
        concat = torch.cat([x_t, h_prev], dim=-1)
        m = self._locked_mask(concat)
        concat = concat * m
        i = torch.sigmoid(self.W_i(concat))
        f = torch.sigmoid(self.W_f(concat))
        o = torch.sigmoid(self.W_o(concat))
        g = torch.tanh(self.W_g(concat))
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def reset_mask(self):
        self.mask = None

class VariationalLSTM(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers, p_drop):
        super().__init__()
        self.layers = nn.ModuleList([
            CNNLSTMWithDropoutCell(in_dim if l == 0 else hid_dim, hid_dim, p_drop)
            for l in range(num_layers)
        ])

    def forward(self, x):
        B, T, _ = x.size()
        h = x
        for layer in self.layers:
            layer.reset_mask()
            h_t = c_t = torch.zeros(
                B, layer.W_i.out_features, device=x.device, dtype=x.dtype
            )
            out_steps = []
            for t in range(T):
                h_t, c_t = layer(h[:, t, :], (h_t, c_t))
                out_steps.append(h_t)
            h = torch.stack(out_steps, dim=1)
        return h

class LSTMOnly(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_size=2,
                 num_layers=3, p_drop=0.3):
        super().__init__()
        self.lstm = VariationalLSTM(in_dim=input_size, hid_dim=hidden_size,
                                    num_layers=num_layers, p_drop=p_drop)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc_mean = nn.Linear(64, output_size)   # For + Refl
        self.fc_var  = nn.Linear(64, output_size)
        self.fc_cls  = nn.Linear(64, 1)             # fault_any

    def forward(self, x):
        seq = self.lstm(x)
        h = seq[:, -1, :]
        h = nn.functional.relu(self.fc1(h))
        mean = self.fc_mean(h)
        var  = torch.exp(self.fc_var(h))
        logits = self.fc_cls(h).squeeze(-1)
        return mean, var, logits

model = LSTMOnly(input_size=2, hidden_size=256, output_size=2,
                 num_layers=lstm_layers, p_drop=dropout_rate)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

print("="*70)
print("TRAINING JOINT FOR/REFL MODEL ON NORMAL RUN ONLY")
print("="*70)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        mean_output, var_output, _ = model(batch_X)   # ignore class_output
        var_output = torch.clamp(var_output, min=1e-6, max=1e3)

        pred_loss = 0.5 * torch.mean(((mean_output - batch_y) ** 2) / var_output +
                                     torch.log(var_output))

        l1_lambda = 1e-6
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        total_loss = pred_loss + l1_lambda * l1_norm
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += total_loss.item()

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d} | Loss: {epoch_loss / len(train_loader):.4f}")


def predict_with_uncertainty(model, x, n_samples=50, eps=1e-6, max_std=1e3):
    was_training = model.training
    model.eval()
    preds = []

    with torch.no_grad():
        for _ in range(n_samples):
            mean_out, var_out, _ = model(x)
            var_out = torch.nan_to_num(var_out, nan=1.0, posinf=max_std, neginf=max_std)
            var_out = torch.clamp(var_out, min=eps, max=max_std)
            preds.append(mean_out)

    preds = torch.stack(preds)            # (S, N, 2)
    mean_preds = preds.mean(dim=0).cpu().numpy()
    var_preds  = preds.var(dim=0).cpu().numpy()
    var_preds  = np.maximum(var_preds, 0.1)
    model.train(was_training)
    return mean_preds, var_preds


# ==================== TEST EVALUATION ON test_data.txt ====================
print("\n" + "="*70)
print("EVALUATING ON TEST DATA (test_data.txt)")
print("="*70)

model.eval()
with torch.no_grad():
    mean_test_scaled, var_test_scaled, logits_test = model(X_test_tensor)
    probs_test = torch.sigmoid(logits_test).cpu().numpy()
    preds_fault = (probs_test > 0.5).astype(int)

# ==================== INVERSE TRANSFORM & PLOT ON RAW TIME ====================
print("\nComputing baseline uncertainty on normal training data...")
mean_train_sc, var_train_sc = predict_with_uncertainty(model, X_train_tensor, n_samples=100)
std_train_sc = np.sqrt(var_train_sc)

# 95% CI width in scaled space: width = upper - lower = 2*1.96*std
width_train_sc = 2 * 1.96 * std_train_sc      # shape (N_train, 2)

# Use mean CI width per channel as baseline
baseline_width_for_sc  = width_train_sc[:, 0].mean()
baseline_width_refl_sc = width_train_sc[:, 1].mean()

print("\nPredicting on test_data with uncertainty...")
mean_test_sc, var_test_sc = predict_with_uncertainty(model, X_test_tensor, n_samples=100)
std_test_sc   = np.sqrt(var_test_sc)

# 95% CI (scaled)
lower_test_sc = mean_test_sc - 1.96 * std_test_sc
upper_test_sc = mean_test_sc + 1.96 * std_test_sc
width_test_sc = upper_test_sc - lower_test_sc

mean_test_flat  = mean_test_sc.reshape(-1, 2)
lower_test_flat = lower_test_sc.reshape(-1, 2)
upper_test_flat = upper_test_sc.reshape(-1, 2)

mean_test_orig  = scaler.inverse_transform(mean_test_flat)
lower_test_orig = scaler.inverse_transform(lower_test_flat)
upper_test_orig = scaler.inverse_transform(upper_test_flat)

mean_for_test  = mean_test_orig[:, 0]
mean_refl_test = mean_test_orig[:, 1]
lower_for_test = lower_test_orig[:, 0]
upper_for_test = upper_test_orig[:, 0]
lower_refl_test = lower_test_orig[:, 1]  # if you also want CI for Refl, do analogously
upper_refl_test = upper_test_orig[:, 1]

# True signals at the window targets
time_test = df_test_orig["Time"].iloc[idx_test].values
for_true  = df_test_orig["For"].iloc[idx_test].values
refl_true = df_test_orig["Refl"].iloc[idx_test].values

plasma_on = for_true > 1.0

mean_for_test[~plasma_on] = 0.0
lower_for_test[~plasma_on] = 0.0
upper_for_test[~plasma_on] = 0.0
mean_refl_test[~plasma_on] = 0.0
lower_refl_test[~plasma_on] = 0.0
upper_refl_test[~plasma_on] = 0.0

# ----- Rule 1: CI violation (forward) -----
out_for = (for_true < lower_for_test) | (for_true > upper_for_test)
out_refl = (refl_true < lower_refl_test) | (refl_true > upper_refl_test)

fault_ci_for = out_for & out_refl

# ----- Rule 2: CI width > 1.2 * baseline width (forward) -----
width_test_for_sc = width_test_sc[:, 0]                # scaled width
fault_width_for   = width_test_for_sc > 1.2 * baseline_width_for_sc

# Combine rules (forward)
fault_for = fault_ci_for | fault_width_for

# (Optional) Same for reflected
width_test_refl_sc = width_test_sc[:, 1]
fault_ci_refl   = (refl_true < lower_refl_test) | (refl_true > upper_refl_test)
fault_width_refl = width_test_refl_sc > 1.2 * baseline_width_refl_sc
fault_refl = fault_ci_refl | fault_width_refl

# Overall fault if either channel flags
fault_any = fault_for | fault_refl
print("Total fault points (any):", fault_any.sum())



plt.figure(figsize=(14, 7))

plt.plot(time_test, for_true,  label="Forward Power (True)",
         color="black", linewidth=2, zorder=5)
plt.plot(time_test, refl_true, label="Reflected Power (True)",
         color="tab:blue", linewidth=1.5, zorder=4)

plt.plot(time_test, mean_for_test,  label="Predicted Forward",   color="red",   linewidth=2)
plt.plot(time_test, mean_refl_test, label="Predicted Reflected", color="orange", linewidth=2)

plt.fill_between(time_test, lower_for_test, upper_for_test,
                 color="lightblue", alpha=0.3, label="95% CI (Forward)", zorder=1)

plt.fill_between(time_test, lower_refl_test, upper_refl_test,
                 color="lightblue", alpha=0.3, label="95% CI (refl)", zorder=1)

first_fault = True

for t, f in zip(time_test, fault_any):
    if f:
        plt.axvline(
            x=t,
            color="purple",
            linestyle="--",
            alpha=0.3,
            label="Fault detection" if first_fault else None
        )
        first_fault = False

plt.title("Plasma Test Run – Fault detection", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.xlim(1800,2300)
plt.ylabel("Power (W)", fontsize=16)
plt.legend(fontsize=12, loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


gc.collect()
