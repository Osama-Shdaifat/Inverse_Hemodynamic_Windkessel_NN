import os
import glob 
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
#              CONFIGURATION
# ==========================================
save_path = r"C:\Users\A\Desktop\Physics_Flow_Waves_256"
num_files = 15000        
points_per_file = 256    # <--- CHANGED TO 256
sampling_rate = 80       
total_duration = points_per_file / sampling_rate 

if not os.path.exists(save_path):
    os.makedirs(save_path)

# ==========================================
#        1. CONSTRUCTIVE GENERATOR
# ==========================================
def generate_guaranteed_pulse(t, frequency, target_sv, target_peak, phase_shift=0):
    period = 1.0 / frequency
    # We add phase_shift to t so the cycle doesn't always start at 0
    cycle_t = (t + phase_shift) % period
    center = 0.15 * period 
    
    # Width calculation (Sigma)
    sigma = target_sv / (target_peak * 2.506) 
    
    # Generate
    raw_pulse = np.exp(-0.5 * ((cycle_t - center) / sigma) ** 2)
    raw_pulse[raw_pulse < 0.01] = 0 
    
    # Exact Peak Scaling
    flow = raw_pulse * target_peak
    return flow

# ==========================================
#            2. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"Generating {num_files} Variable-Quality files...")
    t = np.linspace(0, total_duration, points_per_file)

    for i in range(num_files):
        # --- A. Physiology Targets ---
        bpm = np.random.uniform(50, 110)
        target_sv = np.random.uniform(60, 110)
        target_peak = np.random.uniform(350, 650) 
        current_freq = bpm / 60.0 
        
        # --- B. Generate Clean (WITH PHASE SHIFT) ---
        # Randomly shift the start by 0 to 1 full second
        random_shift = np.random.uniform(0, 1.0) 
        clean_flow = generate_guaranteed_pulse(t, current_freq, target_sv, target_peak, phase_shift=random_shift)
        
        # --- C. VARIABLE NOISE INJECTION ---
        quality_roll = np.random.random()
        
        if quality_roll > 0.5:
            # GRADE A: Good Clinical Data
            white_noise_amp = np.random.uniform(2, 5)
            resp_amp = np.random.uniform(0, 10)
            mains_hum = 0
            turb_factor = 0.02
            
        elif quality_roll > 0.2:
            # GRADE B: Average Data
            white_noise_amp = np.random.uniform(5, 12)
            resp_amp = np.random.uniform(10, 30)
            mains_hum = 0
            turb_factor = 0.05
            
        else:
            # GRADE C: "The Nightmare"
            white_noise_amp = np.random.uniform(10, 20)
            resp_amp = np.random.uniform(30, 60)
            mains_hum = np.random.uniform(5, 15)
            turb_factor = 0.10 
            
        # 1. White Noise
        noise = np.random.normal(0, white_noise_amp, len(t))
        
        # 2. Respiration (Baseline Wander)
        # Note: At 3.2s duration, this will look like a "Slope" or "Drift"
        # This is GOOD because it matches the drift artifacts we want to learn to ignore.
        resp_freq = np.random.uniform(0.2, 0.35)
        respiration = resp_amp * np.sin(2 * np.pi * resp_freq * (t + random_shift))
        
        # 3. Turbulence
        turbulence = clean_flow * np.random.normal(0, turb_factor, len(t))
        
        # 4. Mains Hum
        hum = mains_hum * np.sin(2 * np.pi * 20 * t) 
        
        noisy_flow = clean_flow + noise + respiration + turbulence + hum
        
        noisy_flow[noisy_flow < -50] = -50

        # --- D. Save ---
        R_val = np.random.uniform(0.6, 1.4) 
        C_val = np.random.uniform(0.5, 2.5)
        L_val = np.random.uniform(0.005, 0.020)
        
        filename = f"Flow_{i+1:04d}_SV_{target_sv:.1f}_BPM_{int(bpm)}_R_{R_val:.2f}_C_{C_val:.2f}_L_{L_val:.4f}.csv"
        full_path = os.path.join(save_path, filename)
        
        df = pd.DataFrame({'Time': t, 'Flow': noisy_flow})
        df.to_csv(full_path, index=False)
        
        if (i + 1) % 500 == 0:
            print(f"Progress: {i + 1}/{num_files}")

    print(f"Done. Saved to {save_path}")
    
    # --- PLOT ONE "BAD" EXAMPLE TO VERIFY ---
    files = glob.glob(os.path.join(save_path, "*.csv"))
    if files:
        plt.figure(figsize=(10, 5))
        for _ in range(3):
            sample = pd.read_csv(random.choice(files))
            plt.plot(sample['Time'], sample['Flow'], alpha=0.7)
            
        plt.title("Quality Check: 256 Samples (3.2s) with Randomized Start Phases")
        plt.xlabel("Time (s)")
        plt.ylabel("Flow (mL/s)")
        plt.grid(True, alpha=0.3)
        plt.show()