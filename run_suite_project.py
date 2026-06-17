# run_suite_project.py
import os
import torch
import numpy as np
import pandas as pd
torch.set_default_dtype(torch.float32)
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
import bcs
from physics import setup_physics
from loop import run_simulation

# --- EXECUTION TOGGLES ---
RUN_PHASE_1 = False
RUN_PHASE_2 = True

def run_batched_sweep():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # ==========================================
    # PHASE 1: AMPLIFICATION SWEEP
    # ==========================================
    if RUN_PHASE_1:
        print("\n── PHASE 1: AMP SWEEP ──")
        all_amps = [1, 10, 50, 100, 250, 500, 1000, 1250, 2500, 5000, 7500, 10000, 17500, 25000, 50000, 100000]
        chunk_size = 8
        time_series_data = []

        for i in range(0, len(all_amps), chunk_size):
            amp_factors = all_amps[i:i+chunk_size]
            batch_size = len(amp_factors)
            
            # --- EXPLICIT GRID DEFINITION (PHASE 1) ---
            cfg.batch_size = batch_size
            # cfg.Nx = 176
            # cfg.Ny = 155  

            cfg.snapshot_time = 1.0
            cfg.Total_Time = 600.0
            
            cfg.radius = np.array([1.0] * batch_size, dtype=np.float32)
            cfg.Lx = 20.0 * cfg.radius
            cfg.Ly = 10.0 * cfg.radius
            cfg.dx = cfg.Lx / (cfg.Nx - 1)
            cfg.dy = cfg.Ly / (cfg.Ny - 1)
            cfg.cx = 5.0 * cfg.radius
            cfg.cy = cfg.Ly / 2.0
            cfg.U_bg = 2.2 * (cfg.radius / 1.0) ** 0.56
            
            bcs.inflow.o2  = np.array([10.0] * batch_size, dtype=np.float32).reshape(batch_size, 1)
            bcs.inflow.no3 = np.array([10.0] * batch_size, dtype=np.float32).reshape(batch_size, 1)
            
            device, state = setup_physics(cfg)

            poc_tensor = torch.tensor([500000.0] * batch_size, dtype=torch.float32, device=device).view(batch_size, 1, 1)
            state['poc_initial'] = poc_tensor
            
            # Apply Amplification Factors
            amp_tensor = torch.tensor(amp_factors, dtype=torch.float32, device=device).view(batch_size, 1, 1)
            state['bgc'].krem  *= amp_tensor
            state['bgc'].kAo   *= amp_tensor
            state['bgc'].kNo   *= amp_tensor
            state['bgc'].kDen1 *= amp_tensor
            state['bgc'].kDen2 *= amp_tensor
            state['bgc'].kDen3 *= amp_tensor
            state['bgc'].kAx   *= amp_tensor
            
            print(f"Running Phase 1 Batch {i//chunk_size + 1} / {int(np.ceil(len(all_amps)/chunk_size))}...")
            
            results = run_simulation(state, cfg, device)
            c_snapshots = results[0]      
            p_masks = state['particle_mask'].cpu().numpy()  
            
            for b_idx, amp_val in enumerate(amp_factors):
                core_mask = p_masks[b_idx] > 0
                total_pixels = np.sum(core_mask)
                
                for step_idx, snapshot in enumerate(c_snapshots):
                    min_o2 = np.min(snapshot[b_idx][core_mask])
                    anoxic_pixels = np.sum((snapshot[b_idx] < 1.0) & core_mask)
                    anoxic_fraction = (anoxic_pixels / total_pixels) * 100.0
                    time_min = (step_idx * cfg.snapshot_time) / 60.0
                    
                    time_series_data.append({
                        'Time_min': float(time_min),
                        'Amp_Factor': f"{amp_val}x",
                        'Amp_Numeric': int(amp_val),
                        'Min_Core_O2': float(min_o2),
                        'Anoxic_Volume_Pct': float(anoxic_fraction)
                    })
                    
        df_amp = pd.DataFrame(time_series_data)
        df_amp.to_csv("outputs/Project_Anoxia_Amp_Data.csv", index=False)
        print("Phase 1 complete. Saved Project_Anoxia_Amp_Data.csv.")

        # PHASE 1 PLOTS ---
        # PLOT 1: O2 Time Series
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_amp, x='Time_min', y='Min_Core_O2', hue='Amp_Factor', palette='viridis')
        plt.axhline(1.0, color='red', linestyle='--', label='Anoxic Threshold')
        plt.title("Phase 1: Minimum Core O2 Over Time (500k POC)", fontweight='bold')
        plt.xlabel("Simulation Time (Minutes)")
        plt.ylabel("Minimum Core O2 (mmol/m³)")
        plt.legend(title="Amp Factor", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("outputs/Phase1_O2_TimeSeries.png", dpi=300)
        plt.close()

        # PLOT 2: Anoxic Volume % Over Time
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_amp, x='Time_min', y='Anoxic_Volume_Pct', hue='Amp_Factor', palette='magma')
        plt.title("Phase 1: Anoxic Core Volume % Over Time (500k POC)", fontweight='bold')
        plt.xlabel("Simulation Time (Minutes)")
        plt.ylabel("Anoxic Volume (%)")
        plt.legend(title="Amp Factor", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("outputs/Phase1_Volume_TimeSeries.png", dpi=300)
        plt.close()

    else:
        print("\n── PHASE 1: AMP SWEEP (SKIPPED) ──")

    # ==========================================
    # PHASE 2: POC DENSITY SWEEP
    # ==========================================
    if RUN_PHASE_2:
        target_amp = 1000
        print(f"\n── PHASE 2: POC SWEEP (Using {target_amp}x) ──")
        
        # 16 Points Total: Clustered around the 0.5M - 0.7M inflection point
        # p1 = np.linspace(47000, 450000, 4, endpoint=False)
        # p2 = np.linspace(450000, 750000, 8, endpoint=False)
        # p3 = np.linspace(750000, 1000000, 4)

        # poc_levels = np.concatenate([p1, p2, p3]).tolist()

        p1 = np.linspace(47000, 1400000, 8)
        # p2 = np.linspace(450000, 750000, 8, endpoint=False)
        # p3 = np.linspace(750000, 1000000, 4)

        poc_levels = p1.tolist()
        
        chunk_size = 8
        poc_time_series_data = []
        poc_summary_data = []

        for i in range(0, len(poc_levels), chunk_size):
            current_batch = poc_levels[i:i+chunk_size]
            batch_size = len(current_batch)
            
            # --- EXPLICIT GRID DEFINITION (PHASE 2) ---
            cfg.batch_size = batch_size
            # cfg.Nx = 176
            # cfg.Ny = 155 
            cfg.Nx = 116
            cfg.Ny = 101   
            cfg.snapshot_time = 1.0
            cfg.Total_Time = 600.0
            
            cfg.radius = np.array([1.0] * batch_size, dtype=np.float32)
            cfg.Lx = 20.0 * cfg.radius
            cfg.Ly = 10.0 * cfg.radius
            cfg.dx = cfg.Lx / (cfg.Nx - 1)
            cfg.dy = cfg.Ly / (cfg.Ny - 1)
            cfg.cx = 5.0 * cfg.radius
            cfg.cy = cfg.Ly / 2.0
            cfg.U_bg = 2.2 * (cfg.radius / 1.0) ** 0.56
            
            bcs.inflow.o2  = np.array([10.0] * batch_size, dtype=np.float32).reshape(batch_size, 1)
            bcs.inflow.no3 = np.array([10.0] * batch_size, dtype=np.float32).reshape(batch_size, 1)
            
            device, state = setup_physics(cfg)
            
            # Apply POC Densities (Fixed: using correct poc_initial tensor format)
            poc_tensor = torch.tensor(current_batch, dtype=torch.float32, device=device).view(batch_size, 1, 1)
            state['poc_initial'] = poc_tensor
            
            # Apply 1000x Amplification Factor 
            amp_tensor = torch.tensor([target_amp] * batch_size, dtype=torch.float32, device=device).view(batch_size, 1, 1)
            state['bgc'].krem  *= amp_tensor
            state['bgc'].kAo   *= amp_tensor
            state['bgc'].kNo   *= amp_tensor
            state['bgc'].kDen1 *= amp_tensor
            state['bgc'].kDen2 *= amp_tensor
            state['bgc'].kDen3 *= amp_tensor
            state['bgc'].kAx   *= amp_tensor
            state['bgc'].k_hyd   *= amp_tensor

            print(f"Running Phase 2 Batch {i//chunk_size + 1} / {int(np.ceil(len(poc_levels)/chunk_size))}...")
            
            results = run_simulation(state, cfg, device)
            c_snapshots = results[0]      
            p_masks = state['particle_mask'].cpu().numpy()  
            
            # Process Data
            for b_idx, poc_density in enumerate(current_batch):
                core_mask = p_masks[b_idx] > 0
                total_pixels = np.sum(core_mask)
                
                min_o2_over_time = []
                time_mins = []
                
                for step_idx, snapshot in enumerate(c_snapshots):
                    min_o2 = np.min(snapshot[b_idx][core_mask])
                    anoxic_pixels = np.sum((snapshot[b_idx] < 1.0) & core_mask)
                    anoxic_fraction = (anoxic_pixels / total_pixels) * 100.0
                    time_min = (step_idx * cfg.snapshot_time) / 60.0
                    
                    min_o2_over_time.append(min_o2)
                    time_mins.append(time_min)
                    
                    poc_time_series_data.append({
                        'Time_min': time_min,
                        'Initial_POC': float(poc_density),
                        'Min_Core_O2': float(min_o2),
                        'Anoxic_Volume_Pct': float(anoxic_fraction)
                    })
                
                # Robust Duration Logic
                min_o2_over_time = np.array(min_o2_over_time)
                time_mins = np.array(time_mins)
                anoxic_indices = np.where(min_o2_over_time < 1.0)[0]
                
                if len(anoxic_indices) > 0:
                    first_idx = anoxic_indices[0]
                    last_idx = anoxic_indices[-1]
                    duration = time_mins[last_idx] - time_mins[first_idx] + (cfg.snapshot_time / 60.0)
                else:
                    duration = 0.0
                    
                poc_summary_data.append({
                    'Initial_POC': float(poc_density),
                    'Anoxia_Duration_min': duration
                })

        # Save Fresh Data
        df_ts = pd.DataFrame(poc_time_series_data)
        df_ts.to_csv("outputs/Project_Anoxia_POC_TimeSeries1000x_corrected_khyd.csv", index=False)
        
        df_sum = pd.DataFrame(poc_summary_data)
        df_sum.to_csv("outputs/Project_Anoxia_POC_Summary1000x_corrected_khyd.csv", index=False)

        # PLOT 1: O2 Time Series
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_ts, x='Time_min', y='Min_Core_O2', hue='Initial_POC', palette='viridis')
        plt.axhline(1.0, color='red', linestyle='--', label='Anoxic Threshold')
        plt.title(f"Time to Reach Anoxia (Amp = {int(target_amp)}x)", fontweight='bold')
        plt.xlabel("Simulation Time (Minutes)")
        plt.ylabel("Minimum Core O2 (mmol/m³)")
        plt.legend(title="Initial POC", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("outputs/Phase2_O2_TimeSeries1000x_corrected_khyd.png", dpi=300)
        plt.close()

        # PLOT 2: Duration vs POC
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df_sum, x='Initial_POC', y='Anoxia_Duration_min', marker='o', markersize=8, color='#ff7f0e', linewidth=2.5)
        plt.title("Total Anoxic Duration vs POC Density", fontweight='bold')
        plt.xlabel("Initial POC Density (mmol C / m³)")
        plt.ylabel("Total Anoxic Duration (Minutes)")
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig("outputs/Phase2_Duration_vs_POC1000x_corrected_khyd.png", dpi=300)
        plt.close()

        # PLOT 3: Anoxic Volume % Over Time
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_ts, x='Time_min', y='Anoxic_Volume_Pct', hue='Initial_POC', palette='magma')
        plt.title(f"Anoxic Core Volume % Over Time (Amp = {int(target_amp)}x)", fontweight='bold')
        plt.xlabel("Simulation Time (Minutes)")
        plt.ylabel("Anoxic Volume (%)")
        plt.legend(title="Initial POC", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("outputs/Phase2_Volume_TimeSeries1000x_corrected_khyd.png", dpi=300)
        plt.close()
        
        print("Phase 2 complete. All plots and CSVs generated.")

if __name__ == "__main__":
    run_batched_sweep()