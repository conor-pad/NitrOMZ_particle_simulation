# run_suite.py
import os
import sys
import torch
torch.set_default_dtype(torch.float32)
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

import config as cfg
import bcs
from biopar import BioPar
from sms import nit_sms_omz
from physics import setup_physics
from loop import run_simulation

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

def run_experiment(radius_list, ext_o2_list, ext_no3_list):
    """Runs a batch of simulations and integrates transient fluxes over the particle lifespan."""
    
    cfg.batch_size = 12
    
    # ── 1. Update Geometry & Speed (Batched Arrays) ──
    cfg.radius = np.array(radius_list)
    cfg.U_bg = 2.2 * (cfg.radius / 1.0)**0.56
    
    cfg.Lx = 20.0 * cfg.radius
    cfg.Ly = 10.0 * cfg.radius
    cfg.cx = 5.0 * cfg.radius
    cfg.cy = cfg.Ly / 2.0
    cfg.dx = cfg.Lx / (cfg.Nx - 1)
    cfg.dy = cfg.Ly / (cfg.Ny - 1)
    
    cfg.K = float(cfg.nu / cfg.Sc_target)
    
    bcs.inflow.o2 = np.array(ext_o2_list).reshape(cfg.batch_size, 1)
    bcs.inflow.no3 = np.array(ext_no3_list).reshape(cfg.batch_size, 1)

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    device, state = setup_physics(cfg)
    results = run_simulation(state, cfg, device)
    sys.stdout.close()
    sys.stdout = original_stdout

    # ── 2. Time Integration Setup ──
    frames_saved = len(results[0])
    time_between_frames = cfg.Total_Time / max(1, frames_saved - 1)
    
    batch_metrics = []
    bgc = BioPar()
    
    # We must pull the initial POC mass we calculated in setup_physics
    poc_initial_b = state['poc_initial'].cpu().numpy().flatten()
    
    # ── 3. Integrate Lifecycle over all Frames ──
    for b in range(cfg.batch_size):
        dV_m3 = (cfg.dx[b] * cfg.dy[b] * 1.0) * 1e-9  
        particle_mask = state['particle_mask'][b]
        plume_mask = 1.0 - particle_mask
        
        # Initialize Lifespan Accumulators (Absolute Mass in mmol)
        tot_n2o_flux = 0.0
        tot_n2_flux = 0.0
        tot_c_consumed = 0.0
        
        tot_n2o_internal = 0.0
        tot_n2o_plume = 0.0
        tot_n2o_ammox_internal = 0.0
        tot_n2o_denit_internal = 0.0
        tot_n2o_ammox_plume = 0.0
        tot_n2o_denit_plume = 0.0
        
        tot_oxic_core = 0.0
        tot_den1_core, tot_den2_core, tot_den3_core = 0.0, 0.0, 0.0
        tot_oxic_plume = 0.0
        tot_den1_plume, tot_den2_plume, tot_den3_plume = 0.0, 0.0, 0.0
        
        tot_o2_flux_in = 0.0
        tot_no3_flux_in = 0.0
        tot_doc_flux_out = 0.0
        
        avg_anoxic_core_frac = 0.0
        
        # Loop through every snapshot to build the Riemann Sum integral
        for f in range(frames_saved):
            ft_b = {
                'o2':  torch.tensor(results[0][f][b], device=device),
                'n2o': torch.tensor(results[1][f][b], device=device),
                'no3': torch.tensor(results[2][f][b], device=device),
                'no2': torch.tensor(results[3][f][b], device=device),
                'n2':  torch.tensor(results[4][f][b], device=device),
                'doc': torch.tensor(results[5][f][b], device=device),
                'nh4': torch.tensor(results[6][f][b], device=device),
                'po4': torch.zeros_like(torch.tensor(results[0][f][b], device=device))
            }
            
            # Calculate instantaneous SMS rates (mmol/m3/d)
            ddt, diags = nit_sms_omz(ft_b, bgc)
            
            # Reconstruct the exact DOC flux happening at this specific frame
            current_time = f * time_between_frames
            current_poc = poc_initial_b[b] * np.exp(-bgc.k_hyd * current_time)
            frame_doc_flux = bgc.k_hyd * current_poc

            # --- Riemann Sum Multiplier ---
            # Rate (mmol/m3/d) * Volume (m3) * Time Delta (d) = Absolute Mass (mmol)
            integration_factor = dV_m3 * time_between_frames
            
            tot_n2o_flux += torch.sum(ddt['n2o']).item() * integration_factor
            tot_n2_flux += torch.sum(ddt['n2']).item() * integration_factor
            
            tot_c_consumed += torch.sum(diags['RemOx_C'] + diags['RemDen1_C'] + 
                                        diags['RemDen2_C'] + diags['RemDen3_C']).item() * integration_factor
                                        
            # Core vs Plume N2O
            tot_n2o_internal += torch.sum(ddt['n2o'] * particle_mask).item() * integration_factor
            tot_n2o_plume += torch.sum(ddt['n2o'] * plume_mask).item() * integration_factor
            
            tot_n2o_ammox_internal += torch.sum(ddt['n2o_ammox'] * particle_mask).item() * integration_factor
            tot_n2o_denit_internal += torch.sum(ddt['n2o_denit'] * particle_mask).item() * integration_factor
            tot_n2o_ammox_plume += torch.sum(ddt['n2o_ammox'] * plume_mask).item() * integration_factor
            tot_n2o_denit_plume += torch.sum(ddt['n2o_denit'] * plume_mask).item() * integration_factor
            
            # Core vs Plume Carbon Metabolism
            tot_oxic_core += torch.sum(diags['RemOx_C'] * particle_mask).item() * integration_factor
            tot_den1_core += torch.sum(diags['RemDen1_C'] * particle_mask).item() * integration_factor
            tot_den2_core += torch.sum(diags['RemDen2_C'] * particle_mask).item() * integration_factor
            tot_den3_core += torch.sum(diags['RemDen3_C'] * particle_mask).item() * integration_factor
            
            tot_oxic_plume += torch.sum(diags['RemOx_C'] * plume_mask).item() * integration_factor
            tot_den1_plume += torch.sum(diags['RemDen1_C'] * plume_mask).item() * integration_factor
            tot_den2_plume += torch.sum(diags['RemDen2_C'] * plume_mask).item() * integration_factor
            tot_den3_plume += torch.sum(diags['RemDen3_C'] * plume_mask).item() * integration_factor
            
            # Anoxic Core Tracking
            oxic_threshold = 1.0 
            anoxic_core_vol = torch.sum((ft_b['o2'] < oxic_threshold) * particle_mask).item()
            total_core_vol = torch.sum(particle_mask).item()
            if total_core_vol > 0:
                avg_anoxic_core_frac += (anoxic_core_vol / total_core_vol) / frames_saved
                
            # Fluxes
            o2_consumption = torch.where(ddt['o2'] < 0, ddt['o2'], torch.tensor(0.0, device=device))
            no3_consumption = torch.where(ddt['no3'] < 0, ddt['no3'], torch.tensor(0.0, device=device))
            tot_o2_flux_in += torch.sum(torch.abs(o2_consumption) * particle_mask).item() * integration_factor
            tot_no3_flux_in += torch.sum(torch.abs(no3_consumption) * particle_mask).item() * integration_factor
            
            # DOC Leakage (Produced - Consumed internally)
            doc_produced_internal = torch.sum(torch.tensor(frame_doc_flux, device=device) * particle_mask).item() * integration_factor
            doc_consumed_internal = torch.sum(torch.abs(ddt['doc']) * particle_mask).item() * integration_factor
            tot_doc_flux_out += max(0.0, doc_produced_internal - doc_consumed_internal)

        # ── 4. Normalization (Yields and Fractions) ──
        n2o_yield_efficiency = tot_n2o_flux / tot_c_consumed if tot_c_consumed > 0 else 0.0

        core_c_consumed = tot_oxic_core + tot_den1_core + tot_den2_core + tot_den3_core
        plume_c_consumed = tot_oxic_plume + tot_den1_plume + tot_den2_plume + tot_den3_plume

        frac_oxic_core = tot_oxic_core / core_c_consumed if core_c_consumed > 0 else 0
        frac_den1_core = tot_den1_core / core_c_consumed if core_c_consumed > 0 else 0
        frac_den2_core = tot_den2_core / core_c_consumed if core_c_consumed > 0 else 0
        frac_den3_core = tot_den3_core / core_c_consumed if core_c_consumed > 0 else 0

        frac_oxic_plume = tot_oxic_plume / plume_c_consumed if plume_c_consumed > 0 else 0
        frac_den1_plume = tot_den1_plume / plume_c_consumed if plume_c_consumed > 0 else 0
        frac_den2_plume = tot_den2_plume / plume_c_consumed if plume_c_consumed > 0 else 0
        frac_den3_plume = tot_den3_plume / plume_c_consumed if plume_c_consumed > 0 else 0

        metrics = {
            'Radius_mm': cfg.radius[b],
            'Speed_mms': cfg.U_bg[b],
            'Ext_O2': ext_o2_list[b],
            'Ext_NO3': ext_no3_list[b],
            'Lifespan_Days': cfg.Total_Time,
            'Initial_POC_mmol': poc_initial_b[b] * (torch.sum(particle_mask).item() * dV_m3),
            'N2O_Total_mmol_lifetime': tot_n2o_flux,
            'N2_Total_mmol_lifetime': tot_n2_flux,
            'Avg_Anoxic_Core_Frac': avg_anoxic_core_frac,
            'N2O_Internal_mmol_lifetime': tot_n2o_internal,
            'N2O_Plume_mmol_lifetime': tot_n2o_plume,
            'N2O_Yield_per_C': n2o_yield_efficiency,
            'Frac_Oxic_Core': frac_oxic_core,
            'Frac_Den1_Core': frac_den1_core,
            'Frac_Den2_Core': frac_den2_core,
            'Frac_Den3_Core': frac_den3_core,
            'Frac_Oxic_Plume': frac_oxic_plume,
            'Frac_Den1_Plume': frac_den1_plume,
            'Frac_Den2_Plume': frac_den2_plume,
            'Frac_Den3_Plume': frac_den3_plume,
            'Plume_N2O_Ammox': tot_n2o_ammox_plume,
            'Plume_N2O_Denit': tot_n2o_denit_plume,
            'Internal_N2O_Ammox': tot_n2o_ammox_internal,  
            'Internal_N2O_Denit': tot_n2o_denit_internal, 
            'O2_Flux_In_mmol_lifetime': tot_o2_flux_in,
            'NO3_Flux_In_mmol_lifetime': tot_no3_flux_in,
            'DOC_Leakage_mmol_lifetime': tot_doc_flux_out
        }
        batch_metrics.append(metrics)
        
    return batch_metrics

def generate_all_plots(csv_filename):
    df = pd.read_csv(csv_filename)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})

    available_no3 = df['Ext_NO3'].unique()
    available_o2 = df['Ext_O2'].unique()
    
    base_no3 = available_no3[0] if len(available_no3) == 1 else (10.0 if 10.0 in available_no3 else available_no3[0])
    base_o2 = available_o2[0] if len(available_o2) == 1 else (6.0 if 6.0 in available_o2 else available_o2[0])
    
    print(f"\n📊 Plotting Baselines Auto-Selected -> O2: {base_o2}, NO3: {base_no3}")

    # ── PLOT 1: Normalized N2O Yield Contour ──
    plt.figure(figsize=(9, 6))
    df_p1 = df[df['Ext_NO3'] == base_no3]
    pivot1 = df_p1.pivot(index='Ext_O2', columns='Radius_mm', values='N2O_Yield_per_C')
    X1, Y1 = np.meshgrid(pivot1.columns, pivot1.index)
    Z1 = pivot1.values

    cf1 = plt.contourf(X1, Y1, Z1, levels=20, cmap='viridis')
    cbar1 = plt.colorbar(cf1)
    cbar1.set_label('Total Lifespan N2O Yield\n(mmol N2O / mmol Total C metabolized)')
    plt.scatter(X1, Y1, color='black', s=15, alpha=0.5, zorder=5)
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Ambient O2 (mmol/m3)")
    plt.title(f"N2O Production Efficiency (Core + Plume) | NO3 = {base_no3}", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("outputs/Plot1_N2O_Yield.png", dpi=300, bbox_inches='tight')

    # ── PLOTS 2 & 3: Metabolic Architecture ──
    df_base = df[(df['Ext_O2'] == base_o2) & (df['Ext_NO3'] == base_no3)].copy()

    def get_melted_fractions(df_in, zone_suffix, x_var):
        cols = [f'Frac_Oxic_{zone_suffix}', f'Frac_Den1_{zone_suffix}', f'Frac_Den2_{zone_suffix}', f'Frac_Den3_{zone_suffix}']
        melted = df_in.melt(id_vars=[x_var], value_vars=cols, var_name='Pathway', value_name='Fraction')
        melted['Pathway'] = melted['Pathway'].str.replace('Frac_', '').str.replace(f'_{zone_suffix}', '')
        return melted

    df_core_rad = get_melted_fractions(df_base, 'Core', 'Radius_mm')
    df_plume_rad = get_melted_fractions(df_base, 'Plume', 'Radius_mm')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.lineplot(data=df_core_rad, x='Radius_mm', y='Fraction', hue='Pathway', marker="s", linewidth=2.5, ax=axes[0])
    axes[0].set_title("Inside Particle Only (Core Metabolism)", fontweight='bold')
    axes[0].set_xlabel("Particle Radius (mm)")
    axes[0].set_ylabel("Fraction of DOC Consumed in Zone (Lifespan Avg)")

    sns.lineplot(data=df_plume_rad, x='Radius_mm', y='Fraction', hue='Pathway', marker="s", linewidth=2.5, ax=axes[1])
    axes[1].set_title("Outside Particle Only (Plume/Wake Metabolism)", fontweight='bold')
    axes[1].set_xlabel("Particle Radius (mm)")

    plt.suptitle(f"Spatial Metabolic Partitioning vs. Particle Size (O2={base_o2}, NO3={base_no3})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot2_Metabolism_vs_Radius.png", dpi=300, bbox_inches='tight')

    # Plot 3
    df_core_vel = get_melted_fractions(df_base, 'Core', 'Speed_mms')
    df_plume_vel = get_melted_fractions(df_base, 'Plume', 'Speed_mms')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.lineplot(data=df_core_vel, x='Speed_mms', y='Fraction', hue='Pathway', marker="^", linewidth=2.5, ax=axes[0])
    axes[0].set_title("Inside Particle Only (Core Metabolism)", fontweight='bold')
    axes[0].set_xlabel("Sinking Velocity (mm/s)")
    axes[0].set_ylabel("Fraction of DOC Consumed in Zone (Lifespan Avg)")

    sns.lineplot(data=df_plume_vel, x='Speed_mms', y='Fraction', hue='Pathway', marker="^", linewidth=2.5, ax=axes[1])
    axes[1].set_title("Outside Particle Only (Plume/Wake Metabolism)", fontweight='bold')
    axes[1].set_xlabel("Sinking Velocity (mm/s)")

    plt.suptitle(f"Spatial Metabolic Partitioning vs. Sinking Velocity (O2={base_o2}, NO3={base_no3})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot3_Metabolism_vs_Velocity.png", dpi=300, bbox_inches='tight')

    # (PLOT 4 Residence Time Removed - Invalid for transient lifecycle)

    # ── PLOT 5: N2O Source Apportionment (Inside vs Outside) ──
    df_plume = df_base.melt(id_vars=['Radius_mm', 'Speed_mms'], 
                            value_vars=['Plume_N2O_Ammox', 'Plume_N2O_Denit'],
                            var_name='Pathway', value_name='N2O Produced (mmol)')
    df_plume['Pathway'] = df_plume['Pathway'].str.replace('Plume_N2O_', '')

    df_internal = df_base.melt(id_vars=['Radius_mm', 'Speed_mms'], 
                               value_vars=['Internal_N2O_Ammox', 'Internal_N2O_Denit'],
                               var_name='Pathway', value_name='N2O Produced (mmol)')
    df_internal['Pathway'] = df_internal['Pathway'].str.replace('Internal_N2O_', '')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.lineplot(data=df_plume, x='Radius_mm', y='N2O Produced (mmol)', 
                 hue='Pathway', marker="X", linewidth=2.5, palette=['teal', 'darkred'], ax=axes[0])
    axes[0].set_title("Outside Particle (Plume/Wake N2O Sources)", fontweight='bold')
    axes[0].set_xlabel("Particle Radius (mm)")
    axes[0].set_ylabel("Total Lifespan N2O Production (mmol)")
    axes[0].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=df_internal, x='Radius_mm', y='N2O Produced (mmol)', 
                 hue='Pathway', marker="o", linewidth=2.5, palette=['teal', 'darkred'], ax=axes[1])
    axes[1].set_title("Inside Particle (Core N2O Sources)", fontweight='bold')
    axes[1].set_xlabel("Particle Radius (mm)")
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(f"Total N2O Source Apportionment vs. Particle Size (O2={base_o2}, NO3={base_no3})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot4_N2O_Sources_Dual.png", dpi=300, bbox_inches='tight')

    # ── PLOT 6: Boundary Layer Exchange Fluxes ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.lineplot(data=df, x='Ext_O2', y='O2_Flux_In_mmol_lifetime', hue='Radius_mm', 
                 palette='viridis', marker='o', linewidth=2.5, ax=axes[0])
    axes[0].set_title('Total O2 Consumed from Ambient', fontweight='bold')
    axes[0].set_xlabel('Ambient O2 (mmol/m3)')
    axes[0].set_ylabel('Total O2 Influx (mmol)')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=df, x='Ext_NO3', y='NO3_Flux_In_mmol_lifetime', hue='Radius_mm', 
                 palette='flare', marker='s', linewidth=2.5, ax=axes[1])
    axes[1].set_title('Total NO3 Consumed from Ambient', fontweight='bold')
    axes[1].set_xlabel('Ambient NO3 (mmol/m3)')
    axes[1].set_ylabel('Total NO3 Influx (mmol)')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=df, x='Ext_O2', y='DOC_Leakage_mmol_lifetime', hue='Radius_mm', 
                 palette='copper', marker='^', linewidth=2.5, ax=axes[2])
    axes[2].set_title('Total DOC Lost to Plume', fontweight='bold')
    axes[2].set_xlabel('Ambient O2 (mmol/m3)')
    axes[2].set_ylabel('Total DOC Leakage (mmol)')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("outputs/Plot5_Boundary_Fluxes.png", dpi=300, bbox_inches='tight')

   # ── PLOT 7: Absolute N2O Flux Contour (Inside Core Only) ──
    plt.figure(figsize=(9, 6))
    
    df_p7 = df[df['Ext_NO3'] == base_no3]
    pivot7 = df_p7.pivot(index='Ext_O2', columns='Radius_mm', values='N2O_Internal_mmol_lifetime')
    X7, Y7 = np.meshgrid(pivot7.columns, pivot7.index)
    Z7 = pivot7.values

    max_abs = max(abs(np.nanmax(Z7)), abs(np.nanmin(Z7)))
    if max_abs == 0: max_abs = 1e-10 
    
    cf7 = plt.contourf(X7, Y7, Z7, levels=30, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    
    cbar7 = plt.colorbar(cf7)
    cbar7.set_label('Total Internal Core N2O (mmol)\n<-- Net Consumption (Blue)   |   Net Production (Red) -->')

    plt.contour(X7, Y7, Z7, levels=[0], colors='black', linewidths=2)
    plt.scatter(X7, Y7, color='black', s=15, alpha=0.5, zorder=5)

    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Ambient O2 (mmol/m3)")
    plt.title(f"Total Internal N2O Generated (Inside Particle Core Only) | NO3 = {base_no3}", fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig("outputs/Plot6_Internal_Flux_Only.png", dpi=300, bbox_inches='tight')

    # ── PLOT 8: Anoxic "Dead Core" Fraction ──
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='Radius_mm', y='Avg_Anoxic_Core_Frac', hue='Ext_O2', 
                 marker="s", linewidth=2.5, palette="crest")
    
    plt.title("Time-Averaged Anoxic Core Fraction", fontweight='bold')
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Lifespan Avg Core Volume (< 1.0 mmol/m3 O2)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Ambient O2 (mmol/m3)")
    plt.savefig("outputs/Plot7_Anoxic_Core.png", dpi=300, bbox_inches='tight')

    # ── PLOT 9: Terminal Ratio ──
    df['Terminal_Ratio'] = df['N2_Total_mmol_lifetime'] / (df['N2O_Total_mmol_lifetime'] + 1e-12)

    plt.figure(figsize=(8, 6))
    df_ratio = df[df['Ext_NO3'] == base_no3]

    sns.lineplot(data=df_ratio, x='Radius_mm', y='Terminal_Ratio', hue='Ext_O2', 
                 marker="D", linewidth=2.5, palette="rocket")
    
    plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, zorder=0) 
    plt.yscale('log') 
    
    plt.title(f"Total Lifespan Terminal N2/N2O Ratio (Core + Plume, NO3={base_no3})", fontweight='bold')
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Total System N2 / N2O Ratio (Log Scale)\n<-- System is N2O Source  |  System is N2 Source -->")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Ambient O2 (mmol/m3)")
    plt.savefig("outputs/Plot8_Terminal_Ratio.png", dpi=300, bbox_inches='tight')

    # ── PLOT 10: Metabolic pathway vs Ambient O2 ──
    available_radii = df['Radius_mm'].unique()
    base_radius = 1.0 if 1.0 in available_radii else available_radii[0]

    df_base10 = df[(df['Radius_mm'] == base_radius) & (df['Ext_NO3'] == base_no3)].copy()

    df_core_o2 = get_melted_fractions(df_base10, 'Core', 'Ext_O2')
    df_plume_o2 = get_melted_fractions(df_base10, 'Plume', 'Ext_O2')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.lineplot(data=df_core_o2, x='Ext_O2', y='Fraction', hue='Pathway', marker="o", linewidth=2.5, ax=axes[0])
    axes[0].set_title("Inside Particle Only (Core Metabolism)", fontweight='bold')
    axes[0].set_xlabel("Ambient O2 (mmol/m3)")
    axes[0].set_ylabel("Fraction of DOC Consumed")

    sns.lineplot(data=df_plume_o2, x='Ext_O2', y='Fraction', hue='Pathway', marker="o", linewidth=2.5, ax=axes[1])
    axes[1].set_title("Outside Particle Only (Plume/Wake Metabolism)", fontweight='bold')
    axes[1].set_xlabel("Ambient O2 (mmol/m3)")

    plt.suptitle(f"Metabolic Partitioning vs. Oxygen Levels (Radius={base_radius}mm, NO3={base_no3})", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot9_Metabolism_vs_O2.png", dpi=300, bbox_inches='tight')
    print("✅ Plots generated successfully!")

    # ── PLOT 11: Absolute N2O Produced (Entire Domain) ──
    plt.figure(figsize=(8, 6))
    # Note: Replace 'Domain_Net_N2O' with your actual CSV column name for total N2O
    pivot_abs_n2o = df.pivot(index='Ext_O2', columns='Radius_mm', values='N2O_Total_mmol_lifetime')    

    sns.heatmap(pivot_abs_n2o, cmap="magma", annot=True, fmt=".2e", cbar_kws={'label': 'Absolute N2O (mmols)'})
    plt.title("Total Absolute N2O Escaped into Domain", fontweight='bold')
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Ambient O2 (mmol/m³)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/Plot11_Absolute_N2O.png", dpi=300)
    plt.close()

def main():
    print("🌊 Starting NitrOMZ Parameter Suite...\n")
    cfg.is_suite = True 

    radii = np.round(np.linspace(1, 3.0, 6), 2).tolist()
    o2_levels = [1, 2.0, 3, 4, 5.0, 7.5, 10.0, 15, 25, 50, 75, 100]
    no3_levels = [10.0]
    
    chunk_size = 12
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    csv_filename = "outputs/NitrOMZ_Suite.csv"

    print(f"🆕 Starting a fresh suite. Saving to '{csv_filename}'...\n")
    results_data = []

    run_configs = [(r, o2, no3) for r in radii for o2 in o2_levels for no3 in no3_levels]
    # ── Sort by radius so similar neccecary time steps are grouped.
    run_configs.sort(key=lambda x: x[0])
    total_configs = len(run_configs)
    
    if total_configs > 0:
        print(f"📦 Batching enabled: Running {chunk_size} parallel experiments per chunk.\n")
    
    with tqdm(total=total_configs, desc="Simulating Parameters", unit="run", bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for i in range(0, total_configs, chunk_size):
            batch = run_configs[i : i + chunk_size]
            
            r_str = ", ".join([f"{b[0]}" for b in batch])
            tqdm.write(f"▶ Simulating Chunk {i//chunk_size + 1}/{(total_configs + chunk_size - 1)//chunk_size} | Radii: [{r_str}] mm")
            
            r_batch = [b[0] for b in batch]
            o2_batch = [b[1] for b in batch]
            no3_batch = [b[2] for b in batch]
            
            batched_results = run_experiment(r_batch, o2_batch, no3_batch)
            results_data.extend(batched_results)
            
            temp_df = pd.DataFrame(results_data)
            temp_df.to_csv(csv_filename, index=False)
            pbar.update(len(batch))

    final_df = pd.DataFrame(results_data)
    final_df = final_df.sort_values(by=['Radius_mm', 'Ext_O2', 'Ext_NO3'])
    final_df.to_csv(csv_filename, index=False)
    
    print(f"\n✅ Suite Complete! Neatly organized results safely saved to {csv_filename}")
    generate_all_plots(csv_filename)

if __name__ == "__main__":
    main()