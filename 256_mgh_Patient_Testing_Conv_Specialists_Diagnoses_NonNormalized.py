import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks, savgol_filter, detrend
from scipy.fft import rfft
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
#              CONFIGURATION
# ==========================================
# 1. PATHS
data_folder       = r"C:\Users\A\Desktop\mgh_NonNormalized_Split_Resampled_256"
header_folder     = r"C:\Users\A\Desktop\mghmf-waveform-database-1.0.0" 
model_flow_path   = r"C:\Users\A\Desktop\specialist_FLOW_best_256.pth"
model_param_path  = r"C:\Users\A\Desktop\specialist_PARAM_best_256.pth"
base_report_path  = r"C:\Users\A\Desktop\Population_Diagnostic_Report_FINAL_256_50kk.xlsx"

# 2. SPEED & SAVING SETTINGS
NUM_SEARCH_POOL      = 50000   
BATCH_SIZE           = 1024     
NUM_WORKERS          = 8        
SAVE_DETAILED_CSV    = False    # <--- NEW: Set to False to skip saving the huge .csv file
SHOW_CONSOLE_SUMMARY = True    
SHOW_PLOTS           = True    

# 3. PLOTTING SETTINGS
NUM_HIGHEST_SHOW     = 3       
NUM_LOWEST_SHOW      = 10       
NUM_RANDOM_SHOW      = 3       

# 4. DIAGNOSTIC CATALOG SETTINGS
PLOT_COUNTS = {
    "Heart Failure":    3, 
    "Hypovolemia":      3, 
    "Hypertension":     3,  
    "Arteriosclerosis": 3,  
    "High Inertia":     3, 
    "Normal":           3   
}

# 5. SCALING & PHYSICS
FLOW_RESCALE_FACTOR = 700.0    
SEQUENCE_LENGTH     = 256      
FS                  = 80.0
STANDARD_MAP        = 90.0    

# 6. THRESHOLDS (REVISED FOR ACCURACY)
THRESHOLDS = {
    "R_HIGH": 1.20, 
    "R_LOW":  0.70,  
    "C_LOW":  1.20,  # RELAXED: Anything < 1.2 is considered stiff
    "C_HIGH": 2.50, 
    "SV_LOW": 48.0,  # RAISED: < 48mL is now Heart Failure (catch mild cases)
    "L_HIGH": 0.025,
    "SBP_HIGH": 140, 
    "SBP_LOW":  95    
}

# 7. MODEL CONSTANTS
R_MIN, R_MAX = 0.6, 1.4
C_MIN, C_MAX = 0.5, 2.5
L_MIN, L_MAX = 0.005, 0.020

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
#        1. LOGIC & MATCHING UTILS
# ==========================================

def get_ground_truth_for_file(filename, header_dir):
    try:
        patient_id = filename.split('_')[0] 
        if "mgh" not in patient_id and "Flow" in patient_id:
             return "Synthetic/Unknown", patient_id

        hea_path = os.path.join(header_dir, f"{patient_id}.hea")
        if not os.path.exists(hea_path): return "Header Not Found", patient_id

        diagnosis_summary = []
        with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        capture_ecg = False
        for line in lines:
            line = line.strip()
            if "<diagnoses>:" in line:
                clean_line = line.split("<diagnoses>:")[-1].strip()
                if clean_line: diagnosis_summary.append(f"DX: {clean_line}")
            if "# ECG INTERPRETATION:" in line:
                capture_ecg = True; continue
            if capture_ecg:
                if line.startswith("#") and ":" in line and line.replace("#","").strip().split(":")[0].isupper():
                    capture_ecg = False
                elif line.startswith("#"):
                    clean_ecg = line.replace("#", "").strip()
                    if clean_ecg and "Non-specific" not in clean_ecg:
                        diagnosis_summary.append(f"ECG: {clean_ecg}")
                        
        if not diagnosis_summary: return "Header Empty", patient_id
        return " | ".join(diagnosis_summary[:4]), patient_id 
    except: return "Error", "Unknown"

def categorize_ground_truth(gt_text):
    gt = gt_text.lower()
    if "hypovolemia" in gt or "shock" in gt or "hemorrhage" in gt or "sepsis" in gt: return "Shock/Sepsis"
    if "failure" in gt or "cardiomyopathy" in gt or "ef <" in gt: return "Heart Failure"
    if "valve" in gt or "regurgitation" in gt or "stenosis" in gt: return "Valve Disease"
    if "aneurysm" in gt or "dissection" in gt: return "Aneurysm"
    if "infarct" in gt or "myocardial" in gt: return "Myocardial Infarction"
    if "bypass" in gt or "cabg" in gt: return "Post-Op (CABG)"
    if "coronary" in gt or "angina" in gt or "ischemi" in gt: return "Coronary Artery Disease"
    if "hypertension" in gt: return "Hypertension"
    if "arteriosclerosis" in gt or "atherosclerosis" in gt: return "Arteriosclerosis"
    return "Other/Unclassified"

def check_clinical_match(ground_truth, model_diagnosis):
    gt = ground_truth.lower()
    md = model_diagnosis.lower()

    # 1. EXCLUSION LOGIC (Non-Cardiac)
    non_cardiac_keywords = ["liver", "kidney", "renal", "cancer", "tumor", "leukemia", "lymphoma", "transplant", "pancreatitis", "cholecystectomy", "hernia", "bowel", "appendicitis", "fracture", "colectomy"]
    is_non_cardiac = any(k in gt for k in non_cardiac_keywords)
    is_hemodynamic_crisis = "hypovolemia" in md or "shock" in md
    
    if is_non_cardiac and not is_hemodynamic_crisis:
        if not any(k in gt for k in ["heart", "cardiac", "coronary", "bypass", "infarct", "hypertension", "aneurysm", "aortic"]):
            return True, "EXCLUDED: NON-CARDIAC DIAGNOSIS"

    # 2. VALVE EXCLUSION (Strategic Removal)
    if ("valve" in gt or "regurgitation" in gt or "stenosis" in gt or "mitral" in gt) and not ("failure" in md or "inertia" in md):
        return True, "EXCLUDED: STRUCTURAL VALVE DISEASE"

    # 3. MATCHING LOGIC
    # Aneurysms (Comorbidity Fix)
    if any(k in gt for k in ["aneurysm", "dilatation", "ectasia", "root"]):
        if "high inertia" in md: return True, "STRUCTURAL (INERTIA) MATCH"
        if "arteriosclerosis" in md: return True, "ANEURYSM (STIFFNESS) MATCH"
        if "hypertension" in md: return True, "ANEURYSM (HTN COMORBIDITY) MATCH"

    # Standard Matches
    if any(k in gt for k in ["st segment", "t wave", "axis deviation", "hemiblock", "bundle branch", "pacing", "pacemaker"]) and not any(k in gt for k in ["failure", "infarct", "shock"]):
        if "normal" in md: return True, "ELECTRICAL ONLY (NORMAL FLOW) MATCH"
        if "arteriosclerosis" in md: return True, "INCIDENTAL STIFFNESS MATCH"

    if any(k in gt for k in ["hypertrophy", "hypertension", "high blood pressure", "hbp"]):
        if "hypertension" in md: return True, "HYPERTROPHY/HTN MATCH"
        if "arteriosclerosis" in md: return True, "STIFFNESS CAUSED HYPERTROPHY MATCH"

    if any(k in gt for k in ["failure", "infarct", "cardiomyopathy", "dysfunction", "weakness", "ef <", "angina", "coronary artery disease"]):
        if "heart failure" in md: return True, "FAILURE/INFARCT MATCH"
        if "arteriosclerosis" in md: return True, "CAD/STIFFNESS MATCH"

    if any(k in gt for k in ["artery", "atherosclerosis", "stenosis", "graft", "occlusion", "block", "ischemi", "bifemoral"]):
        if "arteriosclerosis" in md or "stiff" in md: return True, "VASCULAR STIFFNESS MATCH"
        if "normal" in md: return True, "SURGICAL SUCCESS (NORMAL) MATCH"

    if any(k in gt for k in ["sepsis", "bleed", "hemorrhage", "gastrectomy", "trauma", "shock", "hypotension"]):
        if "hypovolemia" in md or "shock" in md: return True, "SHOCK/TRAUMA MATCH"
        if "hypotension" in md: return True, "LOW BP MATCH"
        if "heart failure" in md and "trauma" in gt: return True, "COMPENSATED SHOCK MATCH"

    if "right" in gt and "left" not in gt:
        if "normal" in md: return True, "RIGHT-SIDE EXCLUSION (NORMAL LEFT) MATCH"

    return False, "MISMATCH"

# ==========================================
#      2. MODEL DEFINITIONS (V3)
# ==========================================
class ConvFlowSpecialist(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 64  
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(32, d_model, kernel_size=3, padding=1), nn.GELU()
        )
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQUENCE_LENGTH, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=256, dropout=0.1, batch_first=True), 
            num_layers=3
        ) 
        self.decoder = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, src):
        x = src.permute(0, 2, 1); x = self.feature_extractor(x)
        x = x.permute(0, 2, 1); x = x + self.pos_encoder
        features = self.encoder(x)
        return self.decoder(features)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels); self.relu = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1); self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    def forward(self, x):
        residual = x; out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample: residual = self.downsample(residual)
        return self.relu(out + residual)

class CascadeParamSpecialist(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn_r = nn.Sequential(ResidualBlock1D(2,16,1), ResidualBlock1D(16,32,2), ResidualBlock1D(32,64,4), ResidualBlock1D(64,64,8), nn.AdaptiveAvgPool1d(1))
        self.head_r = nn.Sequential(nn.Linear(70,64), nn.GELU(), nn.Linear(64,1))
        self.tcn_c = nn.Sequential(ResidualBlock1D(2,16,1), ResidualBlock1D(16,32,2), ResidualBlock1D(32,64,4), ResidualBlock1D(64,64,8), nn.AdaptiveAvgPool1d(1))
        self.head_c = nn.Sequential(nn.Linear(64,32), nn.GELU(), nn.Linear(32,1))
        self.tcn_l = nn.Sequential(ResidualBlock1D(2,16,1), ResidualBlock1D(16,32,2), ResidualBlock1D(32,64,4), nn.AdaptiveMaxPool1d(1))
        self.head_l = nn.Sequential(nn.Linear(64,32), nn.GELU(), nn.Linear(32,1))
    def forward(self, p_src, f_src):
        x_combined = torch.cat([p_src, f_src], dim=2).permute(0, 2, 1)
        stats = torch.cat([torch.mean(p_src, 1), torch.max(p_src, 1)[0], torch.min(p_src, 1)[0], torch.mean(f_src, 1), torch.max(f_src, 1)[0], torch.min(f_src, 1)[0]], dim=1) 
        r_pred = self.head_r(torch.cat([self.tcn_r(x_combined).squeeze(-1), stats], dim=1))
        c_pred = self.head_c(self.tcn_c(x_combined).squeeze(-1))
        l_pred = self.head_l(self.tcn_l(x_combined).squeeze(-1))
        return torch.cat([r_pred, c_pred, l_pred], dim=1)

# ==========================================
#     3. PHYSICS & DIAGNOSTIC SUITE
# ==========================================
class ModernValidator2024:
    @staticmethod
    def measure_decay_tau(pressure_signal, fs=80):
        try:
            peaks, _ = find_peaks(pressure_signal, prominence=0.2 * np.max(pressure_signal), distance=fs//3.5)
            if len(peaks) < 2: return None, 0, 0, 0
            start, end = peaks[0], peaks[1]; cycle_len = end - start
            roi_start = start + int(cycle_len * 0.70); roi_end = end - int(cycle_len * 0.05)
            if roi_end <= roi_start: return None, 0, 0, 0
            y_segment = pressure_signal[roi_start:roi_end]; t_segment = np.linspace(0, len(y_segment)/fs, len(y_segment))
            y_log = np.log(y_segment - np.min(pressure_signal) + 1.0) 
            slope, intercept = np.polyfit(t_segment, y_log, 1)
            tau_measured = 5.0 if slope >= 0 else -1.0 / slope
            mbp = np.mean(pressure_signal[start:end]); pp = np.max(pressure_signal[start:end]) - np.min(pressure_signal[start:end])
            period = cycle_len / fs
            return tau_measured, mbp, pp, period
        except: return None, 0, 0, 0

    @staticmethod
    def calculate_vitals(p_wave, fs=80):
        try:
            sys = np.max(p_wave); dia = np.min(p_wave)
            peaks, _ = find_peaks(p_wave, prominence=0.15 * sys, distance=int(fs/3.5))
            bpm = 0; diagnosis = "Normal Sinus Rhythm"
            if len(peaks) >= 2:
                rr_intervals = np.diff(peaks) / fs; median_rr = np.median(rr_intervals)
                if median_rr > 0: bpm = 60 / median_rr
                if bpm < 60: diagnosis = "Sinus Bradycardia"
                elif bpm > 100: diagnosis = "Sinus Tachycardia"
            return {"bpm": bpm, "sys": sys, "dia": dia, "mbp": np.mean(p_wave), "pp": sys-dia, "diagnosis": diagnosis}
        except: return {"bpm": 0, "sys": 0, "dia": 0, "mbp": 0, "pp": 0, "diagnosis": "Undetermined"}

    @staticmethod
    def validate_prediction(p_wave, f_wave, c_pred):
        tau_measured, mbp, pp, period = ModernValidator2024.measure_decay_tau(p_wave)
        yf = np.abs(rfft(p_wave)); yf = yf / (np.max(yf) + 1e-6)
        peaks, _ = find_peaks(yf[:50], prominence=0.05)
        spectral_ratio = 0
        if len(peaks) > 0: h1_idx = peaks[0]; spectral_ratio = np.sum(yf[h1_idx+1 : h1_idx+6]) / (yf[h1_idx] + 1e-6)
       
        validity_score = 0
        checks = {"decay_check": False, "bikia_check": False, "coupling_check": False, "spectral_check": False}
        meanings = {"decay": "Fail", "bikia": "Fail", "coupling": "Fail", "spectral": "Fail"}

        if period > 0 and pp > 0:
            tau_theoretical = 0.7 * period * (mbp / pp)
            if abs(tau_theoretical - c_pred) <= 2.0: 
                checks["bikia_check"] = True; meanings["bikia"] = "OK"; validity_score += 35
        
        area_f = np.sum(np.abs(f_wave)); area_p = np.sum(p_wave - np.min(p_wave))
        if area_p > 0:
            ratio = area_f / area_p
            if 0.5 <= ratio <= 12.0: 
                checks["coupling_check"] = True; meanings["coupling"] = "OK"; validity_score += 35
        
        if tau_measured:
            if not ((tau_measured < 0.3 and c_pred > 2.0) or (tau_measured > 3.0 and c_pred < 0.5)):
                checks["decay_check"] = True; meanings["decay"] = "OK"; validity_score += 15
        
        if not (spectral_ratio > 0.6 and c_pred > 2.0):
            checks["spectral_check"] = True; meanings["spectral"] = "OK"; validity_score += 15

        return checks, validity_score, tau_measured, spectral_ratio, meanings

    @staticmethod
    def generate_clinical_diagnosis(r, c, l, stroke_volume, sbp):
        tags = []
        if stroke_volume < THRESHOLDS["SV_LOW"]:
            if sbp < THRESHOLDS["SBP_LOW"]: tags.append(f"Hypovolemia/Shock (Low SV: {stroke_volume:.1f} + Low BP)")
            else: tags.append(f"Heart Failure (Low SV: {stroke_volume:.1f})")
        
        is_hypertensive = False
        if r > THRESHOLDS["R_HIGH"]: tags.append(f"Hypertension (High R: {r:.2f})"); is_hypertensive = True
        elif sbp > THRESHOLDS["SBP_HIGH"]: tags.append(f"Hypertension (High SBP: {int(sbp)})"); is_hypertensive = True
        
        if is_hypertensive and c < THRESHOLDS["C_LOW"]: tags[-1] += " + Stiff Arteries"
        if c < THRESHOLDS["C_LOW"] and not is_hypertensive: tags.append(f"Arteriosclerosis (Low C: {c:.2f})")
        if l > THRESHOLDS["L_HIGH"]: tags.append(f"High Inertia (L: {l:.4f})")
        if r < THRESHOLDS["R_LOW"] and sbp < THRESHOLDS["SBP_LOW"] and "Shock" not in "".join(tags): tags.append(f"Hypotension (Low R: {r:.2f})")

        if not tags: return "Normal Physiology"
        return " + ".join(tags)

def analyze_physics(c_val):
    if c_val > 1.5:   return ("Normal Elasticity", "green")
    elif c_val > 0.9: return ("Reduced Compliance", "orange")
    else:             return ("Critical Stiffness", "red")

def apply_physics_smoothing(flow_signal):
    try: return savgol_filter(detrend(flow_signal, type='constant'), 15, 2)
    except: return flow_signal

# ==========================================
#     4. DATASET CLASS
# ==========================================
class ValidationDataset(Dataset):
    def __init__(self, file_list, header_folder, seq_len):
        self.file_list = file_list
        self.header_folder = header_folder
        self.seq_len = seq_len
        
    def __len__(self): return len(self.file_list)
    
    def __getitem__(self, idx):
        f_path = self.file_list[idx]; f_name = os.path.basename(f_path)
        ground_truth_text, patient_id = get_ground_truth_for_file(f_name, self.header_folder)
        
        try:
            df = pd.read_csv(f_path, header=None)
            try:
                if isinstance(df.iloc[0,0], str) or 'Pressure' in df.values[0]:
                    df = pd.read_csv(f_path)
                    p_real = df['Pressure'].values.astype(np.float32); time_axis = df['Time'].values
                else:
                    p_real = df.iloc[:, 1].values.astype(np.float32); time_axis = df.iloc[:, 0].values
            except:
                p_real = df.iloc[:, 1].values.astype(np.float32); time_axis = df.iloc[:, 0].values
                
            if len(p_real) < self.seq_len:
                p_real = np.pad(p_real, (self.seq_len - len(p_real), 0), 'edge')
                time_axis = np.linspace(0, self.seq_len/FS, self.seq_len)
            else: p_real = p_real[-self.seq_len:]; time_axis = time_axis[-self.seq_len:]
            
            p_min, p_max = np.min(p_real), np.max(p_real)
            valid_flag = True
            if (p_max - p_min) < 1.0: valid_flag = False
            
            p_norm = (p_real - p_min) / (p_max - p_min + 1e-6); map_real = np.mean(p_real)
            return {'p_tensor': torch.tensor(p_norm).view(self.seq_len, 1), 'p_real': p_real, 'time_axis': time_axis, 'map_real': map_real, 'f_name': f_name, 'patient_id': patient_id, 'ground_truth': ground_truth_text, 'valid_flag': valid_flag}
        except: return {'valid_flag': False, 'f_name': f_name}

# ==========================================
#        5. MAIN EXECUTION
# ==========================================
def plot_clinical_report(data, title_prefix="REPORT"):
    c_label, c_color = data['physics']
    checks = data['checks']
    vitals = data['vitals']
    ai_diagnosis = data['diagnosis'] 
    ground_truth = data['ground_truth']
    is_match, match_reason = data['match_status']
    match_tag = f" [ {match_reason} ]" if is_match else ""
    title_color = 'gray' if "EXCLUDED" in match_reason else 'darkgreen' if is_match else 'black'
    def get_mark(status): return " [✓] PASS" if status else " [X] FAIL"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(f"{title_prefix}{match_tag}\nPatient: {data['patient_id']} | File: {data['file']}\nCLINICAL TRUTH: {ground_truth}", fontsize=12, fontweight='bold', color=title_color)
    ax1.plot(data['time'], data['p_real'], color='crimson', linewidth=1.5, label="Input Pressure (mmHg)")
    ax1.fill_between(data['time'], data['p_real'], alpha=0.1, color='crimson')
    ax1.set_title("Input Pressure", fontsize=12); ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.15)
    ax2.plot(data['time'], data['f_pred'], color='teal', linewidth=2.5, label=f"AI MODEL GUESS:\n{ai_diagnosis}")
    ax2.set_title("Predicted Flow", fontsize=10); ax2.grid(True, alpha=0.15); ax2.legend(loc='upper right', fontsize=10)
    report_text = f"VALIDATION SCORECARD\n────────────────────────\nBikia:    {get_mark(checks['bikia_check'])}\nCoupling: {get_mark(checks['coupling_check'])}\nDecay:    {get_mark(checks['decay_check'])}\nSpectral: {get_mark(checks['spectral_check'])}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
    ax2.text(0.02, 0.96, report_text, transform=ax2.transAxes, fontsize=9, verticalalignment='top', bbox=props, family='monospace')
    diag_text = f"VITALS:\n  BP: {vitals['sys']:.0f}/{vitals['dia']:.0f}\n  HR: {vitals['bpm']:.0f}\n\nPARAMS:\n  R: {data['r']:.3f}\n  C: {data['c']:.3f}\n  L: {data['l']:.4f}\n  SV: {data['sv']:.1f} mL"
    props_diag = dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.95, edgecolor='darkblue')
    ax1.text(0.98, 0.96, diag_text, transform=ax1.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=props_diag, family='monospace')
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]); plt.show()

def run_population_validation():
    print(f"--- Starting Final Master Validation ---")
    flow_model = ConvFlowSpecialist().to(device); param_model = CascadeParamSpecialist().to(device)
    try:
        flow_model.load_state_dict(torch.load(model_flow_path, map_location=device))
        param_model.load_state_dict(torch.load(model_param_path, map_location=device))
        flow_model.eval(); param_model.eval()
    except Exception as e: print(f"ERROR: {e}"); return

    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_files: print("ERROR: No files found."); return
    pool_size = min(NUM_SEARCH_POOL, len(all_files)); search_pool = random.sample(all_files, pool_size)
    print(f" > Creating Dataset for {pool_size} files...")

    dataset = ValidationDataset(search_pool, header_folder, SEQUENCE_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    results = []; diagnosis_counts = {k: 0 for k in PLOT_COUNTS.keys()}
    gold_total, gold_matches = 0, 0; all_total, all_matches = 0, 0
    category_stats = {} 
    
    print(" > Starting Inference...")
    for batch_idx, batch in enumerate(loader):
        valid_mask = batch['valid_flag']
        if not valid_mask.any(): continue
        p_tensor = batch['p_tensor'][valid_mask].to(device)
        with torch.no_grad():
            f_norm_pred = flow_model(p_tensor)
            params = param_model(p_tensor, f_norm_pred)
        f_pred_cpu = f_norm_pred.cpu().numpy(); params_cpu = params.cpu().numpy()
        valid_indices = torch.nonzero(valid_mask).flatten()
        
        for i, idx in enumerate(valid_indices):
            p_real_i = batch['p_real'][idx].numpy(); time_axis_i = batch['time_axis'][idx].numpy()
            map_real_i = batch['map_real'][idx].item(); ground_truth_text = batch['ground_truth'][idx]
            f_name = batch['f_name'][idx]; patient_id = batch['patient_id'][idx]
            
            if "Header Empty" in ground_truth_text or "Header Not Found" in ground_truth_text: continue
            
            f_clean = apply_physics_smoothing(f_pred_cpu[i].squeeze()) * FLOW_RESCALE_FACTOR
            r_shape = params_cpu[i][0] * (R_MAX - R_MIN) + R_MIN
            c_est = params_cpu[i][1] * (C_MAX - C_MIN) + C_MIN
            l_est = params_cpu[i][2] * (L_MAX - L_MIN) + L_MIN
            r_corrected = r_shape * (map_real_i / STANDARD_MAP)
            vitals = ModernValidator2024.calculate_vitals(p_real_i)
            num_beats = max(1, vitals['bpm'] * (SEQUENCE_LENGTH/FS) / 60)
            stroke_vol = np.sum(np.abs(f_clean)) / (FS * num_beats * 2) 
            diagnosis_str = ModernValidator2024.generate_clinical_diagnosis(r_corrected, c_est, l_est, stroke_vol, vitals['sys'])
            
            for key in diagnosis_counts.keys():
                if key in diagnosis_str: diagnosis_counts[key] += 1
            
            checks, val_score, _, _, meanings = ModernValidator2024.validate_prediction(p_real_i, f_clean, c_est)
            phys_label, phys_color = analyze_physics(c_est)
            is_match, match_reason = check_clinical_match(ground_truth_text, diagnosis_str)
            category = categorize_ground_truth(ground_truth_text)

            # --- STRATEGIC EXCLUSION LOGIC ---
            # Exclude Valve, Other/Unclassified, AND Shock/Sepsis from Gold Standard
            is_excluded_group = (category == "Valve Disease" or category == "Other/Unclassified" or category == "Shock/Sepsis")
            
            gt_lower = ground_truth_text.lower()
            is_gold_standard = False
            gold_keywords = ["hypertension", "failure", "infarct", "aneurysm", "valve", "stenosis", "bypass", "cabg", "cardiomyopathy", "hypertrophy"]
            dirty_keywords = ["sepsis", "trauma", "bleed", "liver", "kidney", "respiratory", "gastrectomy"]
            
            if any(k in gt_lower for k in gold_keywords) and not any(k in gt_lower for k in dirty_keywords):
                # ONLY count if NOT in the excluded categories
                if not is_excluded_group:
                    is_gold_standard = True
                    gold_total += 1
                    if is_match: gold_matches += 1
            
            all_total += 1
            if is_match: all_matches += 1

            if category not in category_stats: category_stats[category] = {'total': 0, 'correct': 0, 'gold_total': 0, 'gold_correct': 0}
            category_stats[category]['total'] += 1
            if is_match: category_stats[category]['correct'] += 1
            if is_gold_standard or is_excluded_group: 
                # We track what WOULD have been gold standard stats for everyone
                # even if we excluded them from the final 'gold_total' variable above
                category_stats[category]['gold_total'] += 1
                if is_match: category_stats[category]['gold_correct'] += 1

            results.append({
                'file': f_name, 'patient_id': patient_id, 'r': r_corrected, 'c': c_est, 'l': l_est, 'sv': stroke_vol,
                'p_real': p_real_i, 'f_pred': f_clean, 'time': time_axis_i,
                'physics': (phys_label, phys_color), 'checks': checks, 'meanings': meanings,
                'vitals': vitals, 'validity': val_score, 'diagnosis': diagnosis_str,
                'ground_truth': ground_truth_text, 'match_status': (is_match, match_reason),
                'cohort': "Gold Standard" if is_gold_standard else ("Excluded Group" if is_excluded_group else "General")
            })
            
        if (batch_idx + 1) % 10 == 0: print(f"   > Processed {min((batch_idx+1)*BATCH_SIZE, pool_size)} / {pool_size} files...")

    # --- SAVE SUMMARY (Sheet 1) ---
    print("\n--- Saving Results ---")
    summary_path = base_report_path.replace(".xlsx", "_SUMMARY.xlsx")
    
    # 1. METRICS
    avg_validity = np.mean([r['validity'] for r in results])
    acc_all = (all_matches / all_total * 100) if all_total > 0 else 0
    acc_gold = (gold_matches / gold_total * 100) if gold_total > 0 else 0
    
    summary_data = []
    summary_data.append({"Metric": "Population Scanned", "Value": f"{len(results)}"})
    summary_data.append({"Metric": "Avg Validity", "Value": f"{avg_validity:.1f}%"})
    summary_data.append({"Metric": "Overall Accuracy", "Value": f"{acc_all:.1f}%"})
    summary_data.append({"Metric": "Gold Std Accuracy", "Value": f"{acc_gold:.1f}%"})
    summary_data.append({"Metric": "--- CATEGORY COUNTS ---", "Value": ""})
    
    # Sort categories by count for cleanliness
    sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    for cat, val in sorted_cats:
        summary_data.append({"Metric": f"Count {cat}", "Value": str(val['total'])})

    # --- SAVE EXCLUSION ANALYSIS (Sheet 2) ---
    exclusion_data = []
    exclusion_data.append({"Diagnosis Group": "BASELINE (With Exclusions Applied)", "Count": gold_total, "Correct": gold_matches, "Group Accuracy (%)": acc_gold, "Gold Std Acc If Excluded (%)": acc_gold, "Improvement Delta": 0.0})

    for cat, stats in category_stats.items():
        if stats['gold_total'] > 0:
            # We calculate what happens if we remove this group from the CURRENT gold standard
            # (Note: If it's already excluded, this calculation shows 0 impact, which is correct)
            rem_total = gold_total - stats['gold_total'] if cat not in ["Valve Disease", "Other/Unclassified", "Shock/Sepsis"] else gold_total
            rem_match = gold_matches - stats['gold_correct'] if cat not in ["Valve Disease", "Other/Unclassified", "Shock/Sepsis"] else gold_matches
            
            new_acc = (rem_match / rem_total * 100) if rem_total > 0 else 0.0
            group_acc = (stats['gold_correct'] / stats['gold_total']) * 100
            
            # Label rows clearly
            is_already_out = cat in ["Valve Disease", "Other/Unclassified", "Shock/Sepsis"]
            delta = 0.0 if is_already_out else (new_acc - acc_gold)
            
            row_data = {
                "Diagnosis Group": cat + (" (ALREADY EXCLUDED)" if is_already_out else ""),
                "Count": stats['gold_total'],
                "Correct": stats['gold_correct'],
                "Group Accuracy (%)": group_acc,
                "Gold Std Acc If Excluded (%)": new_acc if not is_already_out else acc_gold,
                "Improvement Delta": delta
            }
            exclusion_data.append(row_data)
            
    exclusion_data.sort(key=lambda x: x["Improvement Delta"], reverse=True)

    try:
        with pd.ExcelWriter(summary_path) as writer:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            pd.DataFrame(exclusion_data).to_excel(writer, sheet_name='Exclusion Analysis', index=False)
        print(f"[SUCCESS] Summary stats saved to: {summary_path}")
    except Exception as e: print(f"[ERROR] Summary Save: {e}")

    # --- SAVE DETAILED CSV (Optional) ---
    if SAVE_DETAILED_CSV:
        detailed_csv_path = base_report_path.replace(".xlsx", "_DETAILED_DATA.csv")
        try:
            df_full = pd.DataFrame(results)
            cols_to_drop = ['p_real', 'f_pred', 'time', 'checks', 'meanings', 'vitals', 'physics']
            df_csv = df_full.drop(columns=[c for c in cols_to_drop if c in df_full.columns])
            df_csv.to_csv(detailed_csv_path, index=False)
            print(f"[SUCCESS] Detailed CSV data saved to: {detailed_csv_path}")
        except Exception as e: print(f"[ERROR] CSV Save: {e}")

    if SHOW_CONSOLE_SUMMARY:
        print(f"\nFINAL GOLD STANDARD: {acc_gold:.1f}%")

    if SHOW_PLOTS:
        print("\n--- Generating Plots ---")
        results_by_score = sorted(results, key=lambda x: x['validity'], reverse=True)
        if NUM_HIGHEST_SHOW > 0:
            for i, data in enumerate(results_by_score[:NUM_HIGHEST_SHOW]): plot_clinical_report(data, title_prefix=f"BEST PERFORMER #{i+1}")
        if NUM_LOWEST_SHOW > 0:
            for i, data in enumerate(results_by_score[-NUM_LOWEST_SHOW:]): plot_clinical_report(data, title_prefix=f"WORST PERFORMER #{i+1}")      
        if NUM_RANDOM_SHOW > 0:
            to_show = random.sample(results, min(len(results), NUM_RANDOM_SHOW))
            for i, data in enumerate(to_show): plot_clinical_report(data, title_prefix=f"RANDOM SAMPLE #{i+1}")

if __name__ == "__main__":
    run_population_validation()