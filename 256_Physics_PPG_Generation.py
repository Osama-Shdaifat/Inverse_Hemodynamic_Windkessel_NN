import os
import glob
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed

# ==========================================
#              CONFIGURATION
# ==========================================
input_folder  = r"C:\Users\A\Desktop\Physics_Flow_Waves_256"
output_folder = r"C:\Users\A\Desktop\Physics_Ready_Dataset_256"
n_jobs = 12 

# Physics Constants
Zc_ratio = 0.05       
# CHANGED: Increased from 1.0 to 2.0 to stop the "Red Spikes"
SMOOTHING_SIGMA = 2.0  

# ==========================================
#          3-ELEMENT MATH ENGINE
# ==========================================
def calculate_pressure_3_element(t, flow_raw, dt, R_total, C, L):
    # --- FIX FOR SHORT FILES (WARM-UP) ---
    # We double the signal [Flow, Flow] to let the math settle down
    # before the real data starts. This prevents glitches at Index 0.
    flow_padded = np.concatenate([flow_raw, flow_raw])
    
    # Smooth the padded flow (Sigma 2.0 removes the jagged noise)
    flow = gaussian_filter1d(flow_padded, sigma=SMOOTHING_SIGMA)
    
    Zc = R_total * Zc_ratio       
    Rp = R_total * (1 - Zc_ratio) 
    
    # Gradient (Slope)
    dQ_dt = np.gradient(flow, dt)
    
    # Calculate Components
    P_inertial = L * dQ_dt
    P_resistive = flow * Zc
    
    # Windkessel Integration
    P_stored = np.zeros_like(flow)
    P_stored[0] = np.mean(flow) * Rp 
    decay_factor = 1.0 / (Rp * C)
    
    for i in range(1, len(P_stored)):
        inflow = flow[i-1]
        p_prev = P_stored[i-1]
        dp = (inflow / C) - (p_prev * decay_factor)
        P_stored[i] = p_prev + (dp * dt)

    # Sum Components
    P_total_padded = P_stored + P_resistive + P_inertial
    
    # --- SLICE BACK TO ORIGINAL SIZE ---
    # We only return the second half (the clean, warmed-up part)
    n_samples = len(flow_raw)
    return P_total_padded[n_samples:]

# ==========================================
#              CORE WORKER
# ==========================================
def process_patient(f_path):
    try:
        df = pd.read_csv(f_path)
        t = df['Time'].values
        flow = df['Flow'].values
        dt = np.mean(np.diff(t))

        fname = os.path.basename(f_path)
        
        # Robust Regex: Finds "R_" followed by numbers/dots
        r_match = re.search(r"R_([0-9\.]+)", fname)
        c_match = re.search(r"C_([0-9\.]+)", fname)
        l_match = re.search(r"L_([0-9\.]+)", fname)
        
        if not (r_match and c_match and l_match):
            return f"SKIP: Could not find params in {fname}"

        R_val = float(r_match.group(1).strip('._'))
        C_val = float(c_match.group(1).strip('._'))
        L_val = float(l_match.group(1).strip('._'))

        pressure = calculate_pressure_3_element(t, flow, dt, R_val, C_val, L_val)
        
        # Save Labels INSIDE the CSV for the Transformer
        df_out = pd.DataFrame({
            'Time': t,
            'Pressure': pressure,
            'Flow': flow,
            'Label_R': np.full(len(t), R_val),
            'Label_C': np.full(len(t), C_val),
            'Label_L': np.full(len(t), L_val)
        })
        
        new_path = os.path.join(output_folder, fname)
        df_out.to_csv(new_path, index=False)
        return None 
        
    except Exception as e:
        return f"ERROR in {os.path.basename(f_path)}: {str(e)}"

# ==========================================
#          VISUAL VERIFICATION
# ==========================================
def verify_dataset_visually(folder, num_samples=5):
    print(f"\n--- Running Visual Verification on {num_samples} files ---")
    files = glob.glob(os.path.join(folder, "*.csv"))
    
    if not files:
        print("No files to verify!")
        return

    samples = random.sample(files, min(len(files), num_samples))
    
    for i, f_path in enumerate(samples):
        df = pd.read_csv(f_path)
        fname = os.path.basename(f_path)
        
        r_lbl = df['Label_R'].iloc[0]
        c_lbl = df['Label_C'].iloc[0]
        l_lbl = df['Label_L'].iloc[0]
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Plot Pressure (Input)
        color = 'tab:red'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Input: Pressure (mmHg)', color=color, fontweight='bold')
        ax1.plot(df['Time'], df['Pressure'], color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(40, 180) # Expected BP range
        
        # Plot Flow (Target)
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Target: Flow (mL/s)', color=color, fontweight='bold')
        ax2.plot(df['Time'], df['Flow'], color=color, alpha=0.6, linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"Sample #{i+1}: {fname}\nLABELS: R={r_lbl:.2f}, C={c_lbl:.2f}, L={l_lbl:.4f}")
        plt.grid(True, alpha=0.3)
        plt.show()

# ==========================================
#              MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not files:
        print("No CSV files found!")
        return

    # --- SANITY CHECK ---
    print("--- DIAGNOSTIC CHECK ---")
    first_file = files[0]
    fname = os.path.basename(first_file)
    print(f"Checking first file: {fname}")
    
    r_check = re.search(r"R_([0-9\.]+)", fname)
    if r_check:
        print(f" > R detected: {r_check.group(1)}")
        print("Diagnostic PASS. Running full batch...")
    else:
        print("Diagnostic FAIL. Regex issue.")
        return

    print(f"Processing {len(files)} files...")
    results = Parallel(n_jobs=n_jobs)(delayed(process_patient)(f) for f in files)
    
    errors = [r for r in results if r is not None]
    print(f"\nDone. Success: {len(files) - len(errors)} | Errors: {len(errors)}")
    
    # --- LAUNCH VISUAL CHECK ---
    if len(files) - len(errors) > 0:
        verify_dataset_visually(output_folder)

if __name__ == "__main__":
    main()