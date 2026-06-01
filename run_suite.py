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
    """Runs a batch of simulations simultaneously and extracts volume-integrated fluxes."""
    
    cfg.batch_size = len(radius_list)
    
    # ── 1. Update Geometry & Speed (Batched Arrays) ──
    cfg.radius = np.array(radius_list)
    cfg.U_bg = 2.2 * (cfg.radius / 1.0)**0.56
    
    cfg.Lx = 20.0 * cfg.radius
    cfg.Ly = 10.0 * cfg.radius
    cfg.cx = 5.0 * cfg.radius
    cfg.cy = cfg.Ly / 2.0
    cfg.dx = cfg.Lx / (cfg.Nx - 1)
    cfg.dy = cfg.Ly / (cfg.Ny - 1)
    
    # ── 2. Resolve the Time Desync ──
    cfg.Total_Time = float(np.max(5.0 * (cfg.Lx / cfg.U_bg)))
    cfg.K = float(cfg.nu / cfg.Sc_target)
    
    cfg.Re_actual = (cfg.U_bg * (2.0 * cfg.radius)) / cfg.nu
    cfg.Sh = 1 + 0.619 * (cfg.Re_actual ** 0.412) * (cfg.Sc_target ** (1/3))
    
    bcs.inflow.o2 = np.array(ext_o2_list).reshape(cfg.batch_size, 1)
    bcs.inflow.no3 = np.array(ext_no3_list).reshape(cfg.batch_size, 1)

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    device, state = setup_physics(cfg)
    results = run_simulation(state, cfg, device)
    sys.stdout.close()
    sys.stdout = original_stdout

    # ── 1. Reconstruct Final Biological State (Time-Targeted Extraction) ──
    frames_saved = len(results[1])
    batch_metrics = []
    bgc = BioPar()
    
    # Extract each experiment from the batch INDIVIDUALLY based on its specific flush time
    for b in range(cfg.batch_size):
        
        # Calculate EXACTLY how much time THIS specific particle needed for 5 flushes
        required_time = 5.0 * (cfg.Lx[b] / cfg.U_bg[b])
        
        # Find which snapshot frame index matches that exact time
        time_between_frames = cfg.Total_Time / max(1, frames_saved - 1)
        frame_idx = min(frames_saved - 1, int(required_time / time_between_frames))
        prev_idx = max(0, frame_idx - 1)
        
        # Extract THIS particle's data at its EXACT 5-flush completion time
        ft_b = {
            'o2':  torch.tensor(results[0][frame_idx][b], device=device),
            'n2o': torch.tensor(results[1][frame_idx][b], device=device),
            'no3': torch.tensor(results[2][frame_idx][b], device=device),
            'no2': torch.tensor(results[3][frame_idx][b], device=device),
            'n2':  torch.tensor(results[4][frame_idx][b], device=device),
            'doc': torch.tensor(results[5][frame_idx][b], device=device),
            'nh4': torch.tensor(results[6][frame_idx][b], device=device),
            'po4': torch.zeros_like(torch.tensor(results[0][frame_idx][b], device=device))
        }
        
        # Calculate N2O accumulation using the targeted frame timing
        prev_n2o = torch.tensor(results[1][prev_idx][b], device=device)
        n2o_accum_b = (ft_b['n2o'] - prev_n2o) / time_between_frames

        particle_mask = state['particle_mask'][b]
        plume_mask = 1.0 - particle_mask
        
        # Calculate rates (Units are in per day!)
        ddt, diags = nit_sms_omz(ft_b, bgc)

        # ── 2. Volume Integrals for Flux (mmol / day) ──
        dV_m3 = (cfg.dx[b] * cfg.dy[b] * 1.0) * 1e-9  
        
        total_n2o_flux = torch.sum(ddt['n2o']).item() * dV_m3
        total_n2_flux = torch.sum(ddt['n2']).item() * dV_m3

        total_c_consumed = torch.sum(diags['RemOx_C'] + diags['RemDen1_C'] + 
                                     diags['RemDen2_C'] + diags['RemDen3_C']).item() * dV_m3

        # ── 3. Internal vs. Plume Spatial Masking ──
        oxic_threshold = 1.0 
        anoxic_core_vol = torch.sum((ft_b['o2'] < oxic_threshold) * particle_mask).item() * dV_m3
        total_core_vol = torch.sum(particle_mask).item() * dV_m3
        frac_anoxic_core = anoxic_core_vol / total_core_vol if total_core_vol > 0 else 0.0
        
        n2o_production_internal = torch.sum(ddt['n2o'] * particle_mask).item() * dV_m3
        internal_n2o_ammox = torch.sum(ddt['n2o_ammox'] * particle_mask).item() * dV_m3
        internal_n2o_denit = torch.sum(ddt['n2o_denit'] * particle_mask).item() * dV_m3
        
        n2o_accumulation = torch.sum(n2o_accum_b * particle_mask).item() * dV_m3
        true_n2o_leakage_out = n2o_production_internal - n2o_accumulation

        n2o_production_plume = torch.sum(ddt['n2o'] * plume_mask).item() * dV_m3
        plume_n2o_ammox = torch.sum(ddt['n2o_ammox'] * plume_mask).item() * dV_m3
        plume_n2o_denit = torch.sum(ddt['n2o_denit'] * plume_mask).item() * dV_m3

        # ── 4. Normalization (Yields and Fractions) ──
        n2o_yield_efficiency = total_n2o_flux / total_c_consumed if total_c_consumed > 0 else 0.0

        core_c_consumed = torch.sum((diags['RemOx_C'] + diags['RemDen1_C'] + 
                                     diags['RemDen2_C'] + diags['RemDen3_C']) * particle_mask).item() * dV_m3
        plume_c_consumed = torch.sum((diags['RemOx_C'] + diags['RemDen1_C'] + 
                                      diags['RemDen2_C'] + diags['RemDen3_C']) * plume_mask).item() * dV_m3

        frac_oxic_core = (torch.sum(diags['RemOx_C'] * particle_mask).item() * dV_m3) / core_c_consumed if core_c_consumed > 0 else 0
        frac_den1_core = (torch.sum(diags['RemDen1_C'] * particle_mask).item() * dV_m3) / core_c_consumed if core_c_consumed > 0 else 0
        frac_den2_core = (torch.sum(diags['RemDen2_C'] * particle_mask).item() * dV_m3) / core_c_consumed if core_c_consumed > 0 else 0
        frac_den3_core = (torch.sum(diags['RemDen3_C'] * particle_mask).item() * dV_m3) / core_c_consumed if core_c_consumed > 0 else 0

        frac_oxic_plume = (torch.sum(diags['RemOx_C'] * plume_mask).item() * dV_m3) / plume_c_consumed if plume_c_consumed > 0 else 0
        frac_den1_plume = (torch.sum(diags['RemDen1_C'] * plume_mask).item() * dV_m3) / plume_c_consumed if plume_c_consumed > 0 else 0
        frac_den2_plume = (torch.sum(diags['RemDen2_C'] * plume_mask).item() * dV_m3) / plume_c_consumed if plume_c_consumed > 0 else 0
        frac_den3_plume = (torch.sum(diags['RemDen3_C'] * plume_mask).item() * dV_m3) / plume_c_consumed if plume_c_consumed > 0 else 0

        # ── 5. Internal Residence Times ──
        residence_times = {}
        for tracer in ['o2', 'no3', 'n2o', 'n2', 'nh4']:
            stock = torch.sum(ft_b[tracer] * particle_mask).item() * dV_m3
            bio_flux = torch.sum(torch.abs(ddt[tracer]) * particle_mask).item() * dV_m3
            tau = stock / bio_flux if bio_flux > 1e-12 else 0.0
            residence_times[f'Tau_{tracer}'] = tau

        # ── 6. O2 and NO3 Inward Fluxes ──
        o2_consumption = torch.where(ddt['o2'] < 0, ddt['o2'], torch.tensor(0.0, device=device))
        no3_consumption = torch.where(ddt['no3'] < 0, ddt['no3'], torch.tensor(0.0, device=device))
        
        o2_flux_in = torch.sum(torch.abs(o2_consumption) * particle_mask).item() * dV_m3
        no3_flux_in = torch.sum(torch.abs(no3_consumption) * particle_mask).item() * dV_m3

        # ── 7. DOC Outward Flux (Leakage) ──
        doc_produced_internal = torch.sum(torch.tensor(cfg.doc_flux_rate, device=device) * particle_mask).item() * dV_m3
        doc_consumed_internal = torch.sum(torch.abs(ddt['doc']) * particle_mask).item() * dV_m3
        doc_flux_out = max(0.0, doc_produced_internal - doc_consumed_internal)

        metrics = {
            'Radius_mm': cfg.radius[b],
            'Speed_mms': cfg.U_bg[b],
            'Ext_O2': ext_o2_list[b],
            'Ext_NO3': ext_no3_list[b],
            'N2O_Flux_Total_mmol_d': total_n2o_flux,
            'N2_Flux_Total_mmol_d': total_n2_flux,
            'Frac_Anoxic_Core': frac_anoxic_core,
            'N2O_Flux_Internal': n2o_production_internal,
            'N2O_Flux_Plume': n2o_production_plume,
            'N2O_Leakage_Out_mmol_d': true_n2o_leakage_out, 
            'N2O_Yield_per_C': n2o_yield_efficiency,
            'Frac_Oxic_Core': frac_oxic_core,
            'Frac_Den1_Core': frac_den1_core,
            'Frac_Den2_Core': frac_den2_core,
            'Frac_Den3_Core': frac_den3_core,
            'Frac_Oxic_Plume': frac_oxic_plume,
            'Frac_Den1_Plume': frac_den1_plume,
            'Frac_Den2_Plume': frac_den2_plume,
            'Frac_Den3_Plume': frac_den3_plume,
            'Plume_N2O_Ammox': plume_n2o_ammox,
            'Plume_N2O_Denit': plume_n2o_denit,
            'Internal_N2O_Ammox': internal_n2o_ammox,  
            'Internal_N2O_Denit': internal_n2o_denit, 
            'O2_Flux_In_mmol_d': o2_flux_in,
            'NO3_Flux_In_mmol_d': no3_flux_in,
            'DOC_Flux_Out_mmol_d': doc_flux_out
        }
        metrics.update(residence_times)
        batch_metrics.append(metrics)
        
    return batch_metrics

def generate_all_plots(csv_filename):
    df = pd.read_csv(csv_filename)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})

    # ── DYNAMIC BASELINE SELECTION ──
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
    cbar1.set_label('Total Domain N2O Yield\n(mmol N2O / mmol Total C metabolized)')
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
    axes[0].set_ylabel("Fraction of DOC Consumed in Zone")

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
    axes[0].set_ylabel("Fraction of DOC Consumed in Zone")

    sns.lineplot(data=df_plume_vel, x='Speed_mms', y='Fraction', hue='Pathway', marker="^", linewidth=2.5, ax=axes[1])
    axes[1].set_title("Outside Particle Only (Plume/Wake Metabolism)", fontweight='bold')
    axes[1].set_xlabel("Sinking Velocity (mm/s)")

    plt.suptitle(f"Spatial Metabolic Partitioning vs. Sinking Velocity (O2={base_o2}, NO3={base_no3})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot3_Metabolism_vs_Velocity.png", dpi=300, bbox_inches='tight')

    # ── PLOT 4: Residence Times (Inside Core Only) ──
    tau_cols = ['Tau_o2', 'Tau_no3', 'Tau_n2o', 'Tau_n2', 'Tau_nh4']
    df_tau = df.melt(id_vars=['Radius_mm', 'Ext_O2', 'Ext_NO3'],
                     value_vars=tau_cols,
                     var_name='Species', value_name='Residence_Time_s')

    df_tau['Species'] = df_tau['Species'].str.replace('Tau_', '').str.upper()

    g4 = sns.FacetGrid(df_tau, col="Radius_mm", row="Species", hue="Ext_NO3",
                       palette="viridis", height=3.0, aspect=1.5, sharey="row")
    
    g4.map(sns.lineplot, "Ext_O2", "Residence_Time_s", marker="o", linewidth=2.5)
    g4.add_legend(title="Ambient NO3 (mmol/m3)")
    g4.set_axis_labels("Ambient O2 (mmol/m3)", "Core Residence Time (s)")
    g4.set_titles(col_template="Radius: {col_name} mm", row_template="{row_name} Turnover")
    
    plt.subplots_adjust(top=0.92)
    g4.fig.suptitle("Internal Microenvironmental Residence Times vs. Ambient Forcing (Inside Particle Only)", fontsize=16, fontweight='bold')
    plt.savefig(f"outputs/Plot4_Residence_Times.png", dpi=300, bbox_inches='tight')

    # ── PLOT 5: N2O Source Apportionment (Inside vs Outside) ──
    df_plume = df_base.melt(id_vars=['Radius_mm', 'Speed_mms'], 
                            value_vars=['Plume_N2O_Ammox', 'Plume_N2O_Denit'],
                            var_name='Pathway', value_name='N2O Production (mmol/d)')
    df_plume['Pathway'] = df_plume['Pathway'].str.replace('Plume_N2O_', '')

    df_internal = df_base.melt(id_vars=['Radius_mm', 'Speed_mms'], 
                               value_vars=['Internal_N2O_Ammox', 'Internal_N2O_Denit'],
                               var_name='Pathway', value_name='N2O Production (mmol/d)')
    df_internal['Pathway'] = df_internal['Pathway'].str.replace('Internal_N2O_', '')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.lineplot(data=df_plume, x='Radius_mm', y='N2O Production (mmol/d)', 
                 hue='Pathway', marker="X", linewidth=2.5, palette=['teal', 'darkred'], ax=axes[0])
    axes[0].set_title("Outside Particle Only (Plume/Wake N2O Sources)", fontweight='bold')
    axes[0].set_xlabel("Particle Radius (mm)")
    axes[0].set_ylabel("N2O Production Rate (mmol/d)")
    axes[0].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=df_internal, x='Radius_mm', y='N2O Production (mmol/d)', 
                 hue='Pathway', marker="o", linewidth=2.5, palette=['teal', 'darkred'], ax=axes[1])
    axes[1].set_title("Inside Particle Only (Core N2O Sources)", fontweight='bold')
    axes[1].set_xlabel("Particle Radius (mm)")
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(f"N2O Source Apportionment vs. Particle Size (O2={base_o2}, NO3={base_no3})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot5_N2O_Sources_Dual.png", dpi=300, bbox_inches='tight')

    # ── PLOT 6: Boundary Layer Exchange Fluxes ──
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    sns.lineplot(data=df, x='Ext_O2', y='O2_Flux_In_mmol_d', hue='Radius_mm', 
                 palette='viridis', marker='o', linewidth=2.5, ax=axes[0])
    axes[0].set_title('O2 Flux Across Boundary (Into Particle)', fontweight='bold')
    axes[0].set_xlabel('Ambient O2 (mmol/m3)')
    axes[0].set_ylabel('O2 Flux (mmol/d)')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=df, x='Ext_NO3', y='NO3_Flux_In_mmol_d', hue='Radius_mm', 
                 palette='flare', marker='s', linewidth=2.5, ax=axes[1])
    axes[1].set_title('NO3 Flux Across Boundary (Into Particle)', fontweight='bold')
    axes[1].set_xlabel('Ambient NO3 (mmol/m3)')
    axes[1].set_ylabel('NO3 Flux (mmol/d)')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=df, x='Ext_O2', y='DOC_Flux_Out_mmol_d', hue='Radius_mm', 
                 palette='copper', marker='^', linewidth=2.5, ax=axes[2])
    axes[2].set_title('DOC Flux Across Boundary (Out of Particle)', fontweight='bold')
    axes[2].set_xlabel('Ambient O2 (mmol/m3)')
    axes[2].set_ylabel('DOC Leakage (mmol/d)')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=df, x='Ext_O2', y='N2O_Leakage_Out_mmol_d', hue='Radius_mm', 
                 palette='magma', marker='X', linewidth=2.5, ax=axes[3])
    axes[3].set_title('N2O Flux Across Boundary (Out of Particle)', fontweight='bold')
    axes[3].set_xlabel('Ambient O2 (mmol/m3)')
    axes[3].set_ylabel('N2O Leakage (mmol/d)')
    axes[3].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("outputs/Plot6_Boundary_Fluxes.png", dpi=300, bbox_inches='tight')

   # ── PLOT 7: Absolute N2O Flux Contour (Inside Core Only) ──
    plt.figure(figsize=(9, 6))
    
    df_p7 = df[df['Ext_NO3'] == base_no3]
    pivot7 = df_p7.pivot(index='Ext_O2', columns='Radius_mm', values='N2O_Flux_Internal')
    X7, Y7 = np.meshgrid(pivot7.columns, pivot7.index)
    Z7 = pivot7.values

    max_abs = max(abs(np.nanmax(Z7)), abs(np.nanmin(Z7)))
    if max_abs == 0: max_abs = 1e-10 
    
    cf7 = plt.contourf(X7, Y7, Z7, levels=30, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    
    cbar7 = plt.colorbar(cf7)
    cbar7.set_label('Internal Core N2O Net Rate (mmol/d)\n<-- Net Consumption (Blue)   |   Net Production (Red) -->')

    plt.contour(X7, Y7, Z7, levels=[0], colors='black', linewidths=2)
    plt.scatter(X7, Y7, color='black', s=15, alpha=0.5, zorder=5)

    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Ambient O2 (mmol/m3)")
    plt.title(f"Absolute N2O Net Flux (Inside Particle Core Only) | NO3 = {base_no3}", fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig("outputs/Plot7_Internal_Flux_Only.png", dpi=300, bbox_inches='tight')

    # ── PLOT 8: Anoxic "Dead Core" Fraction (Inside Core Only) ──
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='Radius_mm', y='Frac_Anoxic_Core', hue='Ext_O2', 
                 marker="s", linewidth=2.5, palette="crest")
    
    plt.title("Anoxic Core Fraction (Inside Particle Only)", fontweight='bold')
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Fraction of Core Volume (< 1.0 mmol/m3 O2)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Ambient O2 (mmol/m3)")
    plt.savefig("outputs/Plot8_Anoxic_Core.png", dpi=300, bbox_inches='tight')

    # ── PLOT 9: Terminal Ratio (Whole Domain) ──
    df['Terminal_Ratio'] = df['N2_Flux_Total_mmol_d'] / (df['N2O_Flux_Total_mmol_d'] + 1e-12)

    plt.figure(figsize=(8, 6))
    df_ratio = df[df['Ext_NO3'] == base_no3]

    sns.lineplot(data=df_ratio, x='Radius_mm', y='Terminal_Ratio', hue='Ext_O2', 
                 marker="D", linewidth=2.5, palette="rocket")
    
    plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, zorder=0) 
    plt.yscale('log') 
    
    plt.title(f"Total Domain Terminal N2/N2O Ratio (Core + Plume, NO3={base_no3})", fontweight='bold')
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Total System N2 / N2O Ratio (Log Scale)\n<-- System is N2O Source  |  System is N2 Source -->")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Ambient O2 (mmol/m3)")
    plt.savefig("outputs/Plot9_Terminal_Ratio.png", dpi=300, bbox_inches='tight')

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
    plt.savefig("outputs/Plot10_Metabolism_vs_O2.png", dpi=300, bbox_inches='tight')
    print("✅ Plot 10 saved: outputs/Plot10_Metabolism_vs_O2.png")

    plt.show()

def main():
    print("🌊 Starting NitrOMZ Parameter Suite...\n")
    cfg.is_suite = True 

    radii = np.round(np.linspace(0.1, 3.0, 16), 2).tolist()
    o2_levels = [2.0, 5.0, 10.0, 20.0, 50, 100]
    no3_levels = [10.0]
    
    # Define explicit chunk size (adjust based on VRAM capacity)
    chunk_size = 4
    
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    csv_filename = "outputs/NitrOMZ_Suite.csv"

    # FORCE FRESH START EVERY TIME
    print(f"🆕 Starting a fresh suite. Saving to '{csv_filename}'...\n")
    results_data = []

    run_configs = [(r, o2, no3) for r in radii for o2 in o2_levels for no3 in no3_levels]
    total_configs = len(run_configs)
    
    if total_configs > 0:
        print(f"📦 Batching enabled: Running {chunk_size} parallel experiments per chunk.\n")
    
    with tqdm(total=total_configs, desc="Simulating Parameters", unit="run", bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for i in range(0, total_configs, chunk_size):
            batch = run_configs[i : i + chunk_size]
            
            # Formatted console output showing parallel tasks
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