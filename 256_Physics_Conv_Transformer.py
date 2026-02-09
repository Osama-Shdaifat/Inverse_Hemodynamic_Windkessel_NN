import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
#              CONFIGURATION
# ==========================================
data_folder = r"C:\Users\A\Desktop\Physics_Ready_Dataset_256"

# These files will ONLY contain the best model found during training
save_flow_model  = r"C:\Users\A\Desktop\specialist_FLOW_best_256.pth"
save_param_model = r"C:\Users\A\Desktop\specialist_PARAM_best_256.pth"

# HYPERPARAMETERS
SEQUENCE_LENGTH = 256    # <--- CHANGED: Matches new file size
BATCH_SIZE      = 64     # <--- INCREASED: Smaller files mean we can batch more
EPOCHS          = 50    # <--- INCREASED: Short files need more epochs to converge
LEARNING_RATE   = 5e-4   # <--- INCREASED: Smaller models learn better with higher LR
DROPOUT         = 0.1   

# NORMALIZATION
P_DIVISOR = 180.0  
F_DIVISOR = 700.0  

# PARAM RANGES
R_MIN, R_MAX = 0.6, 1.4
C_MIN, C_MAX = 0.5, 2.5
L_MIN, L_MAX = 0.005, 0.020

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- TRAINING DEVICE: {device} ---")

# ==========================================
#        1. UNIFIED DATASET
# ==========================================
class PhysicsDataset(Dataset):
    def __init__(self, file_list, seq_len):
        self.file_list = file_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            f_path = self.file_list[idx]
            df = pd.read_csv(f_path)
            
            pressure = df['Pressure'].values.astype(np.float32)
            flow     = df['Flow'].values.astype(np.float32)
            
            r_val = df['Label_R'].iloc[0]
            c_val = df['Label_C'].iloc[0]
            l_val = df['Label_L'].iloc[0]

            if len(pressure) > self.seq_len:
                pressure = pressure[:self.seq_len]
                flow     = flow[:self.seq_len]
            else:
                pad = self.seq_len - len(pressure)
                pressure = np.pad(pressure, (0, pad), 'edge')
                flow     = np.pad(flow, (0, pad), 'edge')

            # Normalize
            p_norm = pressure / P_DIVISOR
            f_norm = flow / F_DIVISOR
            
            # Normalize Targets
            r_norm = (r_val - R_MIN) / (R_MAX - R_MIN)
            c_norm = (c_val - C_MIN) / (C_MAX - C_MIN)
            l_norm = (l_val - L_MIN) / (L_MAX - L_MIN)

            return (torch.tensor(p_norm).unsqueeze(-1),          
                    torch.tensor(f_norm).unsqueeze(-1),          
                    torch.tensor([r_norm, c_norm, l_norm], dtype=torch.float32)) 

        except Exception as e:
            return torch.zeros(self.seq_len, 1), torch.zeros(self.seq_len, 1), torch.zeros(3)

# ==========================================
#       2. FLOW SPECIALIST (LIGHTWEIGHT)
# ==========================================
class ConvFlowSpecialist(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 64  # <--- REDUCED: 128 was too big for 256 points
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), # <--- SMALLER KERNEL
            nn.GELU(),
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQUENCE_LENGTH, d_model))
        
        # Reduced Heads to 2 (Stable for short sequences)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, 
                                                   dim_feedforward=256, 
                                                   dropout=DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3) # <--- REDUCED LAYERS
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1) 
        )

    def forward(self, src):
        x = src.permute(0, 2, 1) 
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1) 
        x = x + self.pos_encoder
        features = self.encoder(x)
        return self.decoder(features)

# ==========================================
#     3. PARAM SPECIALIST (COMPACT TCN)
# ==========================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1) 
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample: residual = self.downsample(residual)
        return self.relu(out + residual)

class CascadeParamSpecialist(nn.Module):
    def __init__(self):
        super().__init__()
        
        # REMOVED DILATION 16 -> It's too wide for 256 samples
        
        # 1. THE RESISTOR (Needs P and Q)
        self.tcn_r = nn.Sequential(
            ResidualBlock1D(2, 16, dilation=1), 
            ResidualBlock1D(16, 32, dilation=2),
            ResidualBlock1D(32, 64, dilation=4),
            ResidualBlock1D(64, 64, dilation=8),
            nn.AdaptiveAvgPool1d(1) 
        )
        self.head_r = nn.Sequential(nn.Linear(64 + 6, 64), nn.GELU(), nn.Linear(64, 1))
        
        # 2. THE CAPACITOR
        self.tcn_c = nn.Sequential(
            ResidualBlock1D(2, 16, dilation=1),
            ResidualBlock1D(16, 32, dilation=2),
            ResidualBlock1D(32, 64, dilation=4),
            ResidualBlock1D(64, 64, dilation=8),
            # dilation 16 removed
            nn.AdaptiveAvgPool1d(1)
        )
        self.head_c = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

        # 3. THE INDUCTOR
        self.tcn_l = nn.Sequential(
            ResidualBlock1D(2, 16, dilation=1),
            ResidualBlock1D(16, 32, dilation=2),
            ResidualBlock1D(32, 64, dilation=4),
            nn.AdaptiveMaxPool1d(1) 
        )
        self.head_l = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, p_src, f_src):
        # Concatenate Pressure and Flow (Channel dim is 2 initially)
        x_combined = torch.cat([p_src, f_src], dim=2) 
        x_in = x_combined.permute(0, 2, 1) # [Batch, 2, Seq]
        
        # Stats Calculation
        raw_mean_p = torch.mean(p_src, dim=1)
        raw_max_p, _ = torch.max(p_src, dim=1)
        raw_min_p, _ = torch.min(p_src, dim=1)
        raw_mean_f = torch.mean(f_src, dim=1)
        raw_max_f, _ = torch.max(f_src, dim=1)
        raw_min_f, _ = torch.min(f_src, dim=1)
        
        stats = torch.cat([raw_mean_p, raw_max_p, raw_min_p, 
                           raw_mean_f, raw_max_f, raw_min_f], dim=1) 
        
        r_pred = self.head_r(torch.cat([self.tcn_r(x_in).squeeze(-1), stats], dim=1))
        c_pred = self.head_c(self.tcn_c(x_in).squeeze(-1))
        l_pred = self.head_l(self.tcn_l(x_in).squeeze(-1))
        
        return torch.cat([r_pred, c_pred, l_pred], dim=1)

# ==========================================
#            4. MAIN TRAINING LOOP
# ==========================================
def main():
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_files: return

    train_files, temp_files = train_test_split(all_files, test_size=0.30, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.50, random_state=42)
    
    train_loader = DataLoader(PhysicsDataset(train_files, SEQUENCE_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PhysicsDataset(val_files, SEQUENCE_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(PhysicsDataset(test_files, SEQUENCE_LENGTH), batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data Split: {len(train_files)} Train | {len(val_files)} Val | {len(test_files)} Test")

    flow_model = ConvFlowSpecialist().to(device)
    param_model = CascadeParamSpecialist().to(device) 

    opt_flow = optim.AdamW(flow_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    opt_param = optim.AdamW(param_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    sched_flow = optim.lr_scheduler.ReduceLROnPlateau(opt_flow, mode='min', factor=0.5, patience=5)
    sched_param = optim.lr_scheduler.ReduceLROnPlateau(opt_param, mode='min', factor=0.5, patience=5)

    criterion_flow = nn.L1Loss()   
    criterion_param = nn.MSELoss() 

    print("\n--- STARTING TRAINING (LIGHTWEIGHT V3) ---")
    
    hist_flow_val, hist_param_val = [], []
    best_param_loss = float('inf') 
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        flow_model.train()
        param_model.train()
        
        for p_in, f_target, p_target in train_loader:
            p_in, f_target, p_target = p_in.to(device), f_target.to(device), p_target.to(device)
            
            # 1. Flow Training
            opt_flow.zero_grad()
            f_pred = flow_model(p_in)
            loss_f = criterion_flow(f_pred, f_target)
            loss_f.backward()
            opt_flow.step()
            
            # 2. Param Training (Student Forcing)
            opt_param.zero_grad()
            noisy_flow_input = f_pred.detach() 
            p_pred = param_model(p_in, noisy_flow_input)
            loss_p = criterion_param(p_pred, p_target)
            loss_p.backward()
            opt_param.step()
        
        # Validation
        flow_model.eval()
        param_model.eval()
        val_f_loss, val_p_loss = 0, 0
        
        with torch.no_grad():
            for p_in, f_target, p_target in val_loader:
                p_in, f_target, p_target = p_in.to(device), f_target.to(device), p_target.to(device)
                
                f_guess = flow_model(p_in)
                val_f_loss += criterion_flow(f_guess, f_target).item()
                val_p_loss += criterion_param(param_model(p_in, f_guess), p_target).item()
        
        avg_val_f = val_f_loss / len(val_loader)
        avg_val_p = val_p_loss / len(val_loader)
        
        sched_flow.step(avg_val_f)
        sched_param.step(avg_val_p)
        
        hist_flow_val.append(avg_val_f)
        hist_param_val.append(avg_val_p)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Flow L1: {avg_val_f:.6f} | Param MSE: {avg_val_p:.6f}")

        # --- SAVE BEST MODEL ---
        if avg_val_p < best_param_loss:
            best_param_loss = avg_val_p
            print(f"   >>> NEW RECORD! Saving Best Model (MSE: {best_param_loss:.6f})")
            torch.save(flow_model.state_dict(), save_flow_model)
            torch.save(param_model.state_dict(), save_param_model)

    # --- FINAL TEST (Using the BEST model, not the last one) ---
    print("\n--- FINAL TEST (Loading Best Models) ---")
    
    flow_model.load_state_dict(torch.load(save_flow_model))
    param_model.load_state_dict(torch.load(save_param_model))
    
    test_f, test_p = 0, 0
    with torch.no_grad():
        for p_in, f_target, p_target in test_loader:
            p_in, f_target, p_target = p_in.to(device), f_target.to(device), p_target.to(device)
            
            f_guess = flow_model(p_in)
            test_f += criterion_flow(f_guess, f_target).item()
            test_p += criterion_param(param_model(p_in, f_guess), p_target).item()
            
    print(f"Flow MAE: {test_f/len(test_loader):.6f}")
    print(f"Param MSE: {test_p/len(test_loader):.6f}")
    print(f"Total Time: {(time.time() - start_time)/60:.2f} min")

    plt.figure(figsize=(10, 5))
    plt.plot(hist_flow_val, label='Flow (L1)', color='blue')
    plt.plot(hist_param_val, label='Param (MSE)', color='red')
    plt.title("Training Loss: Lightweight V3 (Best Saved)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()