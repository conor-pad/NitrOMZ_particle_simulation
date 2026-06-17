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
from physics import setup_physics
from loop import run_simulation

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)


def run_experiment(radius_list, ext_o2_list, ext_no3_list):
    """
    Runs a batch of simulations.
    Lifespan integrals are accumulated on-the-fly inside loop.py.
    This function just unpacks them and computes normalised yields.
    """
    cfg.batch_size = 12

    # ── 1. Update Geometry & Speed (Batched Arrays) ──────────────────────────
    cfg.radius = np.array(radius_list)
    cfg.U_bg   = 2.2 * (cfg.radius / 1.0) ** 0.56

    cfg.Lx = 20.0 * cfg.radius
    cfg.Ly = 10.0 * cfg.radius
    cfg.cx = 5.0  * cfg.radius
    cfg.cy = cfg.Ly / 2.0
    cfg.dx = cfg.Lx / (cfg.Nx - 1)
    cfg.dy = cfg.Ly / (cfg.Ny - 1)
    cfg.K  = float(cfg.nu / cfg.Sc_target)

    bcs.inflow.o2  = np.array(ext_o2_list).reshape(cfg.batch_size, 1)
    bcs.inflow.no3 = np.array(ext_no3_list).reshape(cfg.batch_size, 1)

    # Silence per-run console output
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    device, state = setup_physics(cfg)
    results = run_simulation(state, cfg, device)
    sys.stdout.close()
    sys.stdout = original_stdout

    # ── 2. Unpack results ────────────────────────────────────────────────────
    # run_simulation returns a 14-tuple; last element is the integrals dict
    integrals = results[-1]   # dict of lists, each length == batch_size

    # POC initial mass per batch item [bs] — shape [bs, 1, 1] on GPU, flatten here
    poc_initial_np = state['poc_initial'].cpu().numpy()   # [bs, 1, 1]
    particle_mask  = state['particle_mask'].cpu().numpy() # [bs, Nx, Ny]

    batch_metrics = []

    for b in range(cfg.batch_size):
        # dV for this batch item (dx, dy are [bs,1,1] arrays)
        dV_m3 = float(cfg.dx[b] * cfg.dy[b] * 1.0) * 1e-9

        # Initial POC mass (mmol)
        p_mask_b        = particle_mask[b]
        poc_density_b   = float(poc_initial_np[b].flat[0])   # mmol/m³
        initial_poc_mmol = poc_density_b * float(p_mask_b.sum()) * dV_m3

        # ── Pull pre-integrated lifespan totals ───────────────────────────────
        tot_n2o_flux    = integrals['n2o'][b]
        tot_n2_flux     = integrals['n2'][b]
        tot_c_consumed  = integrals['c_total'][b]

        tot_n2o_internal        = integrals['n2o_internal'][b]
        tot_n2o_plume           = integrals['n2o_plume'][b]
        tot_n2o_ammox_internal  = integrals['n2o_ammox_internal'][b]
        tot_n2o_denit_internal  = integrals['n2o_denit_internal'][b]
        tot_n2o_ammox_plume     = integrals['n2o_ammox_plume'][b]
        tot_n2o_denit_plume     = integrals['n2o_denit_plume'][b]

        tot_oxic_core   = integrals['oxic_core'][b]
        tot_den1_core   = integrals['den1_core'][b]
        tot_den2_core   = integrals['den2_core'][b]
        tot_den3_core   = integrals['den3_core'][b]

        tot_oxic_plume  = integrals['oxic_plume'][b]
        tot_den1_plume  = integrals['den1_plume'][b]
        tot_den2_plume  = integrals['den2_plume'][b]
        tot_den3_plume  = integrals['den3_plume'][b]

        tot_o2_flux_in  = integrals['o2_flux_in'][b]
        tot_no3_flux_in = integrals['no3_flux_in'][b]
        tot_doc_flux_out= integrals['doc_flux_out'][b]
        avg_anoxic_frac = integrals['anoxic_core_frac_sum'][b]  # already averaged in loop.py

        # ── 3. Normalised Yields ──────────────────────────────────────────────
        n2o_yield_efficiency = tot_n2o_flux / tot_c_consumed if tot_c_consumed > 0 else 0.0

        core_c_consumed  = tot_oxic_core  + tot_den1_core  + tot_den2_core  + tot_den3_core
        plume_c_consumed = tot_oxic_plume + tot_den1_plume + tot_den2_plume + tot_den3_plume

        def safe_frac(num, denom):
            return num / denom if denom > 0 else 0.0

        frac_oxic_core  = safe_frac(tot_oxic_core,  core_c_consumed)
        frac_den1_core  = safe_frac(tot_den1_core,  core_c_consumed)
        frac_den2_core  = safe_frac(tot_den2_core,  core_c_consumed)
        frac_den3_core  = safe_frac(tot_den3_core,  core_c_consumed)

        frac_oxic_plume = safe_frac(tot_oxic_plume,  plume_c_consumed)
        frac_den1_plume = safe_frac(tot_den1_plume,  plume_c_consumed)
        frac_den2_plume = safe_frac(tot_den2_plume,  plume_c_consumed)
        frac_den3_plume = safe_frac(tot_den3_plume,  plume_c_consumed)

        metrics = {
            'Radius_mm':                cfg.radius[b],
            'Speed_mms':                cfg.U_bg[b],
            'Ext_O2':                   ext_o2_list[b],
            'Ext_NO3':                  ext_no3_list[b],
            'Lifespan_Days':            cfg.Total_Time,
            'Initial_POC_mmol':         initial_poc_mmol,
            'N2O_Total_mmol_lifetime':  tot_n2o_flux,
            'N2_Total_mmol_lifetime':   tot_n2_flux,
            'Avg_Anoxic_Core_Frac':     avg_anoxic_frac,
            'N2O_Internal_mmol_lifetime': tot_n2o_internal,
            'N2O_Plume_mmol_lifetime':    tot_n2o_plume,
            'N2O_Yield_per_C':          n2o_yield_efficiency,
            'Frac_Oxic_Core':           frac_oxic_core,
            'Frac_Den1_Core':           frac_den1_core,
            'Frac_Den2_Core':           frac_den2_core,
            'Frac_Den3_Core':           frac_den3_core,
            'Frac_Oxic_Plume':          frac_oxic_plume,
            'Frac_Den1_Plume':          frac_den1_plume,
            'Frac_Den2_Plume':          frac_den2_plume,
            'Frac_Den3_Plume':          frac_den3_plume,
            'Plume_N2O_Ammox':          tot_n2o_ammox_plume,
            'Plume_N2O_Denit':          tot_n2o_denit_plume,
            'Internal_N2O_Ammox':       tot_n2o_ammox_internal,
            'Internal_N2O_Denit':       tot_n2o_denit_internal,
            'O2_Flux_In_mmol_lifetime':  tot_o2_flux_in,
            'NO3_Flux_In_mmol_lifetime': tot_no3_flux_in,
            'DOC_Leakage_mmol_lifetime': tot_doc_flux_out,
            'Oxic_Total':               tot_oxic_core + tot_oxic_plume,
            'Denit1_Total':             tot_den1_core + tot_den1_plume,
            'Denit2_Total':             tot_den2_core + tot_den2_plume,
            'Denit3_Total':             tot_den3_core + tot_den3_plume,
        }
        batch_metrics.append(metrics)

    return batch_metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

def generate_all_plots(csv_filename):
    df = pd.read_csv(csv_filename)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})

    available_no3 = df['Ext_NO3'].unique()
    available_o2  = df['Ext_O2'].unique()

    base_no3 = available_no3[0] if len(available_no3) == 1 else (10.0 if 10.0 in available_no3 else available_no3[0])
    base_o2  = available_o2[0]  if len(available_o2)  == 1 else (6.0  if 6.0  in available_o2  else available_o2[0])

    print(f"\n📊 Plotting Baselines -> O2: {base_o2}, NO3: {base_no3}")

    # ── PLOT 1: Normalised N2O Yield Contour ─────────────────────────────────
    plt.figure(figsize=(9, 6))
    df_p1  = df[df['Ext_NO3'] == base_no3]
    pivot1 = df_p1.pivot(index='Ext_O2', columns='Radius_mm', values='N2O_Yield_per_C')
    X1, Y1 = np.meshgrid(pivot1.columns, pivot1.index)
    Z1     = pivot1.values

    cf1  = plt.contourf(X1, Y1, Z1, levels=20, cmap='viridis')
    cbar1 = plt.colorbar(cf1)
    cbar1.set_label('Total Lifespan N2O Yield\n(mmol N2O / mmol Total C metabolized)')
    plt.scatter(X1, Y1, color='black', s=15, alpha=0.5, zorder=5)
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Ambient O2 (mmol/m3)")
    plt.title(f"N2O Production Efficiency (Core + Plume) | NO3 = {base_no3}",
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("outputs/Plot1_N2O_Yield.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Calculate Total Fractions ──────────────────────────────────────────────
    if 'Frac_Oxic_Total' not in df.columns:
        tot = df['Oxic_Total'] + df['Denit1_Total'] + df['Denit2_Total'] + df['Denit3_Total']
        df['Frac_Oxic_Total'] = df['Oxic_Total'] / tot
        df['Frac_Den1_Total'] = df['Denit1_Total'] / tot
        df['Frac_Den2_Total'] = df['Denit2_Total'] / tot
        df['Frac_Den3_Total'] = df['Denit3_Total'] / tot

    def get_melted_fractions(df_in, zone_suffix, x_var):
        cols = [f'Frac_Oxic_{zone_suffix}', f'Frac_Den1_{zone_suffix}', 
                f'Frac_Den2_{zone_suffix}', f'Frac_Den3_{zone_suffix}']
        melted = df_in.melt(id_vars=[x_var], value_vars=cols, 
                            var_name='Pathway', value_name='Fraction')
        melted['Pathway'] = melted['Pathway'].str.replace('Frac_', '').str.replace(f'_{zone_suffix}', '')
        return melted

    # ── PLOT 2: Fraction vs Radius (Fixed O2) ────────────────────────────────
    base_o2 = 1.0
    base_no3 = 10.0
    df_p2 = df[(df['Ext_O2'] == base_o2) & (df['Ext_NO3'] == base_no3)].copy()

    df_core2 = get_melted_fractions(df_p2, 'Core', 'Radius_mm')
    df_tot2  = get_melted_fractions(df_p2, 'Total', 'Radius_mm')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.lineplot(data=df_core2, x='Radius_mm', y='Fraction', hue='Pathway', marker="s", linewidth=2.5, ax=axes[0])
    axes[0].set_title("Core Only (Inside Particle)", fontweight='bold')
    axes[0].set_xlabel("Particle Radius (mm)")
    axes[0].set_ylabel("Fraction of DOC Consumed")

    sns.lineplot(data=df_tot2, x='Radius_mm', y='Fraction', hue='Pathway', marker="s", linewidth=2.5, ax=axes[1])
    axes[1].set_title("Total Domain (Core + Plume)", fontweight='bold')
    axes[1].set_xlabel("Particle Radius (mm)")

    plt.suptitle(f"Metabolic Partitioning vs. Radius (O2={base_o2}, NO3={base_no3})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot2_Metabolism_vs_Radius.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── PLOT 3: Fraction vs Ambient O2 (Fixed Radius) ────────────────────────
    base_radius = 1.0
    df_p3 = df[(df['Radius_mm'] == base_radius) & (df['Ext_NO3'] == base_no3)].copy()

    df_core3 = get_melted_fractions(df_p3, 'Core', 'Ext_O2')
    df_tot3  = get_melted_fractions(df_p3, 'Total', 'Ext_O2')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.lineplot(data=df_core3, x='Ext_O2', y='Fraction', hue='Pathway', marker="o", linewidth=2.5, ax=axes[0])
    axes[0].set_title("Core Only (Inside Particle)", fontweight='bold')
    axes[0].set_xlabel("Ambient O2 (mmol/m³)")
    axes[0].set_ylabel("Fraction of DOC Consumed")

    sns.lineplot(data=df_tot3, x='Ext_O2', y='Fraction', hue='Pathway', marker="o", linewidth=2.5, ax=axes[1])
    axes[1].set_title("Total Domain (Core + Plume)", fontweight='bold')
    axes[1].set_xlabel("Ambient O2 (mmol/m³)")

    plt.suptitle(f"Metabolic Partitioning vs. Ambient O2 (Radius={base_radius}mm, NO3={base_no3})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot3_Metabolism_vs_O2.png", dpi=300, bbox_inches='tight')
    plt.close()
    # ── PLOT 4 (Residence Time removed — invalid for transient lifecycle) ──

    # ── PLOT 4/5 REPLACEMENT: N2O Source Apportionment Contours ──────────────
    plt.figure(figsize=(14, 10))

    sources = ['Internal_N2O_Ammox', 'Internal_N2O_Denit',
               'Plume_N2O_Ammox', 'Plume_N2O_Denit']
    titles = ['Inside Particle (Core Ammox)', 'Inside Particle (Core Denitrification)',
              'Outside Particle (Plume Ammox)', 'Outside Particle (Plume Denitrification)']

    for i, (col, title) in enumerate(zip(sources, titles), 1):
        plt.subplot(2, 2, i)

        pivot_data = df.pivot_table(index='Ext_O2', columns='Radius_mm', values=col)
        X, Y = np.meshgrid(pivot_data.columns, pivot_data.index)
        Z = pivot_data.values

        cf = plt.contourf(X, Y, Z, levels=20, cmap='inferno')
        cbar = plt.colorbar(cf, format="%.1e")
        cbar.set_label('Total Lifespan N2O (mmol)')

        plt.title(title, fontweight='bold')
        plt.xlabel("Particle Radius (mm)")
        plt.ylabel("Ambient O2 (mmol/m³)")
        plt.gca().invert_yaxis()

    plt.suptitle("N2O Source Apportionment across all O2 Levels", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot4_N2O_Sources_Contours.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── PLOT 6: Boundary Layer Exchange Fluxes ────────────────────────────────
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
    plt.close()

    # ── PLOT 7: Absolute N2O Flux — Internal Core Only ────────────────────────
    plt.figure(figsize=(9, 6))
    df_p7  = df[df['Ext_NO3'] == base_no3]
    pivot7 = df_p7.pivot(index='Ext_O2', columns='Radius_mm', values='N2O_Internal_mmol_lifetime')
    X7, Y7 = np.meshgrid(pivot7.columns, pivot7.index)
    Z7     = pivot7.values

    max_abs = max(abs(np.nanmax(Z7)), abs(np.nanmin(Z7)))
    if max_abs == 0:
        max_abs = 1e-10

    cf7  = plt.contourf(X7, Y7, Z7, levels=30, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    cbar7 = plt.colorbar(cf7)
    cbar7.set_label('Total Internal Core N2O (mmol)\n<-- Net Consumption   |   Net Production -->')
    plt.contour(X7, Y7, Z7, levels=[0], colors='black', linewidths=2)
    plt.scatter(X7, Y7, color='black', s=15, alpha=0.5, zorder=5)
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Ambient O2 (mmol/m3)")
    plt.title(f"Total Internal N2O Generated (Inside Particle Core Only) | NO3 = {base_no3}",
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("outputs/Plot6_Internal_Flux_Only.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── PLOT 8: Anoxic "Dead Core" Fraction ──────────────────────────────────
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='Radius_mm', y='Avg_Anoxic_Core_Frac', hue='Ext_O2',
                 marker="s", linewidth=2.5, palette="crest")
    plt.title("Time-Averaged Anoxic Core Fraction", fontweight='bold')
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Lifespan Avg Core Volume (< 1.0 mmol/m3 O2)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Ambient O2 (mmol/m3)")
    plt.tight_layout()
    plt.savefig("outputs/Plot7_Anoxic_Core.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── PLOT 9: Terminal N2/N2O Ratio ─────────────────────────────────────────
    df['Terminal_Ratio'] = df['N2_Total_mmol_lifetime'] / (df['N2O_Total_mmol_lifetime'] + 1e-12)
    df_ratio = df[df['Ext_NO3'] == base_no3]

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_ratio, x='Radius_mm', y='Terminal_Ratio', hue='Ext_O2',
                 marker="D", linewidth=2.5, palette="rocket")
    plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, zorder=0)
    plt.yscale('log')
    plt.title(f"Total Lifespan N2/N2O Ratio (Core + Plume, NO3={base_no3})", fontweight='bold')
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Total System N2 / N2O Ratio (Log Scale)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Ambient O2 (mmol/m3)")
    plt.tight_layout()
    plt.savefig("outputs/Plot8_Terminal_Ratio.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── PLOT 10 REPLACEMENT: Dominant Metabolic Regime Map ───────────────────
    plt.figure(figsize=(10, 8))
    
    # Define the pathways we are comparing
    path_cols = ['Oxic_Total', 'Denit1_Total', 'Denit2_Total', 'Denit3_Total']
    
    # Find the index of the maximum pathway for each row (0=Oxic, 1=Den1, 2=Den2, 3=Den3)
    df['Dominant_Pathway_Idx'] = df[path_cols].values.argmax(axis=1)
    
    # Pivot into a 2D grid for the contour
    pivot_dom = df.pivot_table(index='Ext_O2', columns='Radius_mm', values='Dominant_Pathway_Idx')
    X, Y = np.meshgrid(pivot_dom.columns, pivot_dom.index)
    
    # Custom discrete colormap
    from matplotlib.colors import ListedColormap
    cmap_custom = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']) # Blue, Orange, Green, Red
    
    cf = plt.contourf(X, Y, pivot_dom.values, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cmap_custom)
    
    cbar = plt.colorbar(cf, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Oxic', 'Denit 1 (NO3->NO2)', 'Denit 2 (NO2->N2O)', 'Denit 3 (N2O->N2)'])
    cbar.set_label('Dominant Carbon Sink')
    
    plt.title("Dominant Metabolic Regime (Total Domain)", fontweight='bold')
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Ambient O2 (mmol/m³)")
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("outputs/Plot10_Regime_Map.png", dpi=300)
    plt.close()

    # ── PLOT 11: Absolute N2O Produced (Entire Domain) ───────────────────────
    plt.figure(figsize=(10, 8))
    pivot_abs = df.pivot_table(index='Ext_O2', columns='Radius_mm',
                               values='N2O_Total_mmol_lifetime')
    
    # annot_kws overrides the global bold/large font for the grid text
    sns.heatmap(pivot_abs, cmap="magma", annot=True, fmt=".1e",
                annot_kws={"size": 7, "weight": "normal"}, 
                cbar_kws={'label': 'Absolute N2O (mmols)'})
    
    plt.title("Total Absolute N2O Escaped into Domain", fontweight='bold')
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Ambient O2 (mmol/m³)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/Plot11_Absolute_N2O.png", dpi=300)
    plt.close()
    print("✅ All plots generated successfully!")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    print("🌊 Starting NitrOMZ Parameter Suite...\n")
    cfg.terminal_snapshot_only = True

    radii      = np.round(np.linspace(0.1, 1.5, 10), 2).tolist()
    o2_levels  = [1, 2.0, 3, 4, 5.0, 7.5, 10.0, 12.5, 15, 22.5, 37.5, 50]
    no3_levels = [10.0]

    chunk_size   = 12
    out_dir      = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    csv_filename = "outputs/NitrOMZ_Suite.csv"

    print(f"🆕 Starting a fresh suite. Saving to '{csv_filename}'...\n")
    results_data = []

    run_configs = [(r, o2, no3) for r in radii for o2 in o2_levels for no3 in no3_levels]
    run_configs.sort(key=lambda x: x[0])   # group similar timestep sizes
    total_configs = len(run_configs)

    print(f"📦 Batching enabled: {chunk_size} parallel experiments per chunk.\n")

    with tqdm(total=total_configs, desc="Simulating Parameters", unit="run",
              bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

        for i in range(0, total_configs, chunk_size):
            batch = run_configs[i: i + chunk_size]

            r_str = ", ".join([f"{b[0]}" for b in batch])
            tqdm.write(f"▶ Chunk {i//chunk_size + 1}/{(total_configs + chunk_size - 1)//chunk_size} | Radii: [{r_str}] mm")

            r_batch   = [b[0] for b in batch]
            o2_batch  = [b[1] for b in batch]
            no3_batch = [b[2] for b in batch]

            batched_results = run_experiment(r_batch, o2_batch, no3_batch)
            results_data.extend(batched_results)

            temp_df = pd.DataFrame(results_data)
            temp_df.to_csv(csv_filename, index=False)
            pbar.update(len(batch))

    final_df = pd.DataFrame(results_data)
    final_df = final_df.sort_values(by=['Radius_mm', 'Ext_O2', 'Ext_NO3'])
    final_df.to_csv(csv_filename, index=False)

    print(f"\n✅ Suite Complete! Results saved to {csv_filename}")
    generate_all_plots(csv_filename)


if __name__ == "__main__":
    main()