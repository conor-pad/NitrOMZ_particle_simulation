# run_suite.py
import os
import sys
import torch
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

def run_experiment(radius, ext_o2, ext_no3):
    """Runs simulation and extracts true volume-integrated fluxes and C-yields."""
    
    cfg.radius = radius
    cfg.U_bg = (cfg.Re_target * 1.04) / radius  # Velocity derived from Re=1 and nu=1.04
    cfg.Lx = 20.0 * radius
    cfg.Ly = 10.0 * radius
    cfg.cx = 5.0 * radius
    cfg.cy = cfg.Ly / 2.0
    cfg.dx = cfg.Lx / (cfg.Nx - 1)
    cfg.dy = cfg.Ly / (cfg.Ny - 1)
    cfg.Total_Time = 5.0 * (cfg.Lx / cfg.U_bg)
    cfg.nu = 1.04
    cfg.K = cfg.nu / cfg.Sc_target

    bcs.inflow.o2 = ext_o2
    bcs.inflow.no3 = ext_no3

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    device, state = setup_physics(cfg)
    results = run_simulation(state, cfg, device)
    sys.stdout.close()
    sys.stdout = original_stdout

    # ── 1. Reconstruct Final Biological State ──
    # Extract the final snapshot arrays and push them back to PyTorch
    final_tensors = {
        'o2':  torch.tensor(results[0][-1], device=device),
        'n2o': torch.tensor(results[1][-1], device=device),
        'no3': torch.tensor(results[2][-1], device=device),
        'no2': torch.tensor(results[3][-1], device=device),
        'n2':  torch.tensor(results[4][-1], device=device),
        'doc': torch.tensor(results[5][-1], device=device),
        'nh4': torch.tensor(results[6][-1], device=device)
    }
    
    bgc = BioPar()
    ddt, diags = nit_sms_omz(final_tensors, bgc)

    # ── 2. Gauss's Theorem: Volume Integrals for Flux ──
    # Volume of a single grid cell assuming 1mm slice depth (converted to m^3)
    dV_m3 = (cfg.dx * cfg.dy * 1.0) * 1e-9  
    
    # Total N2O biological production rate (mmol / s)
    total_n2o_flux = torch.sum(ddt['n2o']).item() * dV_m3

    # Total DOC biologically consumed by all respiration pathways (mmol C / s)
    total_c_consumed = torch.sum(diags['RemOx_C'] + diags['RemDen1_C'] + 
                                 diags['RemDen2_C'] + diags['RemDen3_C']).item() * dV_m3

    # ── 3. Internal vs. Plume Spatial Masking ──
    particle_mask = state['particle_mask']
    n2o_production_internal = torch.sum(ddt['n2o'] * particle_mask).item() * dV_m3
    n2o_production_plume = total_n2o_flux - n2o_production_internal

    # ── 4. Normalization (Yields and Fractions) ──
    # Total DOC physically produced by the particle via hydrolysis (mmol C / s)
    total_doc_produced = torch.sum(cfg.doc_flux_rate * particle_mask).item() * dV_m3
    
    # N2O Yield: How much N2O is leaked per unit of Carbon dissolved?
    n2o_yield_efficiency = total_n2o_flux / total_doc_produced if total_doc_produced > 0 else 0.0

    # Metabolic Fractions (What percentage of C is eaten by which bug?)
    frac_oxic = (torch.sum(diags['RemOx_C']).item() * dV_m3) / total_c_consumed if total_c_consumed > 0 else 0
    frac_den1 = (torch.sum(diags['RemDen1_C']).item() * dV_m3) / total_c_consumed if total_c_consumed > 0 else 0
    frac_den2 = (torch.sum(diags['RemDen2_C']).item() * dV_m3) / total_c_consumed if total_c_consumed > 0 else 0
    frac_den3 = (torch.sum(diags['RemDen3_C']).item() * dV_m3) / total_c_consumed if total_c_consumed > 0 else 0

    return {
        'Radius_mm': radius,
        'Speed_mms': cfg.U_bg,
        'Ext_O2': ext_o2,
        'Ext_NO3': ext_no3,
        'N2O_Flux_Total_mmol_s': total_n2o_flux,
        'N2O_Flux_Internal': n2o_production_internal,
        'N2O_Flux_Plume': n2o_production_plume,
        'N2O_Yield_per_C': n2o_yield_efficiency,
        'Frac_Oxic': frac_oxic,
        'Frac_Den1': frac_den1,
        'Frac_Den2': frac_den2,
        'Frac_Den3': frac_den3
    }

def generate_all_plots(csv_filename):
    df = pd.read_csv(csv_filename)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})

    # ── PLOT 1: Normalized N2O Flux ──
    g1 = sns.FacetGrid(df, col="Radius_mm", hue="Ext_NO3", palette="magma", height=4.5, aspect=1.2, sharey=False)
    g1.map(sns.lineplot, "Ext_O2", "N2O_Yield_per_C", marker="o", linewidth=2.5, markersize=8)
    g1.set_axis_labels("Ambient O2 (mmol/m3)", "N2O Yield (mmol N2O / mmol C)")
    g1.set_titles(col_template="Particle Radius: {col_name} mm")
    g1.add_legend(title="Ambient NO3 (mmol/m3)", loc='upper center', bbox_to_anchor=(0.45, -0.15), ncol=3, frameon=True)
    plt.subplots_adjust(top=0.85, bottom=0.25)
    g1.fig.suptitle("N2O Production Efficiency vs. Ambient O2 & NO3", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig("Plot1_N2O_Yield.png", dpi=300, bbox_inches='tight')

    # Prepare data for metabolic fraction plots (Filter for a baseline condition to isolate radius/velocity effects)
    # Using Ambient O2 = 6.0 and Ambient NO3 = 10.0 as the baseline for these plots
    df_base = df[(df['Ext_O2'] == 6.0) & (df['Ext_NO3'] == 10.0)].copy()
    melted_df = df_base.melt(id_vars=['Radius_mm', 'Speed_mms'], 
                             value_vars=['Frac_Oxic', 'Frac_Den1', 'Frac_Den2', 'Frac_Den3'],
                             var_name='Pathway', value_name='Fraction of DOC Consumed')

    # ── PLOT 2: Metabolic Architecture vs. Radius ──
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=melted_df, x='Radius_mm', y='Fraction of DOC Consumed', hue='Pathway', marker="s", linewidth=2.5)
    plt.title("Metabolic Partitioning vs. Particle Size (O2=6.0, NO3=10.0)")
    plt.xlabel("Particle Radius (mm)")
    plt.savefig("Plot2_Metabolism_vs_Radius.png", dpi=300, bbox_inches='tight')

    # ── PLOT 3: Metabolic Architecture vs. Velocity ──
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=melted_df, x='Speed_mms', y='Fraction of DOC Consumed', hue='Pathway', marker="^", linewidth=2.5)
    plt.title("Metabolic Partitioning vs. Sinking Velocity (O2=6.0, NO3=10.0)")
    plt.xlabel("Sinking Velocity (mm/s)")
    plt.savefig("Plot3_Metabolism_vs_Velocity.png", dpi=300, bbox_inches='tight')

    print("✅ All 3 publication plots saved successfully!")
    plt.show()

def main():
    print("🌊 Starting NitrOMZ Parameter Suite (Flux & Carbon Normalization)...\n")
    cfg.is_suite = True 

    radii = [1.0, 2.0, 3.0]
    o2_levels = [2.0, 6.0, 10.0]
    no3_levels = [5.0, 10.0, 15.0]
    csv_filename = "NitrOMZ_Flux_Suite.csv"

    if os.path.exists(csv_filename):
        results_df = pd.read_csv(csv_filename)
        results_df = results_df.round({'Radius_mm': 2, 'Ext_O2': 2, 'Ext_NO3': 2})
        results_df = results_df.drop_duplicates(subset=['Radius_mm', 'Ext_O2', 'Ext_NO3'], keep='last')
        results_df.to_csv(csv_filename, index=False)
        completed_runs = set(zip(results_df['Radius_mm'], results_df['Ext_O2'], results_df['Ext_NO3']))
        results_data = results_df.to_dict('records')
    else:
        completed_runs = set()
        results_data = []

    total_runs = len(radii) * len(o2_levels) * len(no3_levels)
    
    with tqdm(total=total_runs, desc="Simulating Parameters", unit="run") as pbar:
        pbar.update(len(completed_runs))
        for r in radii:
            for o2 in o2_levels:
                for no3 in no3_levels:
                    run_id = (round(r, 2), round(o2, 2), round(no3, 2))
                    if run_id in completed_runs:
                        continue
                    
                    metrics = run_experiment(radius=r, ext_o2=o2, ext_no3=no3)
                    results_data.append(metrics)
                    pd.DataFrame(results_data).to_csv(csv_filename, index=False)
                    pbar.update(1)

    generate_all_plots(csv_filename)

if __name__ == "__main__":
    main()