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
from matplotlib.colors import ListedColormap

import config as cfg
import bcs
from biopar import BioPar
from physics import setup_physics
from loop import run_simulation

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)


# ═════════════════════════════════════════════════════════════════════════════
# ── SWEEP MODE SWITCH ────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
#
#   'poc_o2'    — Initial POC density (mmol C/m³)  ×  Ambient O₂ (mmol/m³)
#   'no3_o2'    — Ambient NO₃ (mmol/m³)            ×  Ambient O₂ (mmol/m³)
#   'radius_poc'— Particle radius (mm)              ×  Initial POC density (mmol C/m³)
#
SWEEP_MODE = 'no3_o2'


# ═════════════════════════════════════════════════════════════════════════════
# ── SWEEP AXIS DEFINITIONS ───────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
#   Each mode defines N_AXIS1 × N_AXIS2 experiments on a regular grid.
#   Anything not being swept is pulled from the FIXED PARAMETERS block below.

# ── poc_o2 ───────────────────────────────────────────────────────────────────
N_POC         = 8
N_O2          = 8
POC_LEVELS    = np.linspace(50_000, 800_000, N_POC).tolist()   # mmol C m⁻³
O2_LEVELS     = np.linspace(1.0, 25.0, N_O2).tolist()          # mmol O₂ m⁻³

# ── no3_o2 ───────────────────────────────────────────────────────────────────
N_NO3         = 8
# N_O2 reused from above (keep grids the same size by default)
NO3_SWEEP_LEVELS = np.linspace(10, 50.0, N_NO3).tolist()      # mmol NO₃ m⁻³
# O2_LEVELS reused

# ── radius_poc ───────────────────────────────────────────────────────────────
N_RADIUS      = 8
# N_POC reused from above
RADIUS_LEVELS = np.linspace(0.5, 1.25, N_RADIUS).tolist()       # mm
# POC_LEVELS reused


# ═════════════════════════════════════════════════════════════════════════════
# ── FIXED PARAMETERS  (values used when a variable is NOT being swept) ───────
# ═════════════════════════════════════════════════════════════════════════════
RADIUS_FIXED  = 1.0     # mm         — fixed for poc_o2 and no3_o2 sweeps
NO3_FIXED     = 10.0    # mmol m⁻³   — fixed for poc_o2 and radius_poc sweeps
O2_FIXED      = 6.0     # mmol m⁻³   — fixed for radius_poc sweep
POC_FIXED     = 200_000 # mmol C m⁻³ — fixed for no3_o2 sweep

# ── Biology & Chemistry Amplifiers (locked to optimised point) ───────────────
BIO_AMP_LOCKED  = 150
KHYD_AMP_LOCKED = 50


# ═════════════════════════════════════════════════════════════════════════════
# ── SWEEP METADATA HELPER ────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def get_sweep_meta():
    """
    Returns a dict describing the active sweep so that run_experiment() and
    generate_all_plots() can adapt without any mode-specific if/else trees.

    Keys
    ----
    axis1_col   : DataFrame column name for the x-axis (outer loop / columns)
    axis2_col   : DataFrame column name for the y-axis (inner loop / rows)
    axis1_vals  : list of values for axis 1
    axis2_vals  : list of values for axis 2
    axis1_label : human-readable axis label  (x-axis on contour plots)
    axis2_label : human-readable axis label  (y-axis on contour plots)
    csv_name    : output CSV filename
    chunk_size  : batch size (reduce for radius_poc — grid rebuilds each chunk)
    """
    if SWEEP_MODE == 'poc_o2':
        return dict(
            axis1_col    = 'Initial_POC_Density',
            axis2_col    = 'Ext_O2',
            axis1_vals   = POC_LEVELS,
            axis2_vals   = O2_LEVELS,
            axis1_label  = 'Initial POC Density (mmol C m⁻³)',
            axis2_label  = 'Ambient O₂ (mmol O₂ m⁻³)',
            csv_name     = 'outputs/NitrOMZ_POC_O2_Sweep.csv',
            chunk_size   = 8,
        )
    elif SWEEP_MODE == 'no3_o2':
        return dict(
            axis1_col    = 'Ext_NO3',
            axis2_col    = 'Ext_O2',
            axis1_vals   = NO3_SWEEP_LEVELS,
            axis2_vals   = O2_LEVELS,
            axis1_label  = 'Ambient NO₃ (mmol NO₃ m⁻³)',
            axis2_label  = 'Ambient O₂ (mmol O₂ m⁻³)',
            csv_name     = 'outputs/NitrOMZ_NO3_O2_Sweep.csv',
            chunk_size   = 8,
        )
    elif SWEEP_MODE == 'radius_poc':
        return dict(
            axis1_col    = 'Radius_mm',
            axis2_col    = 'Initial_POC_Density',
            axis1_vals   = RADIUS_LEVELS,
            axis2_vals   = POC_LEVELS,
            axis1_label  = 'Particle Radius (mm)',
            axis2_label  = 'Initial POC Density (mmol C m⁻³)',
            csv_name     = 'outputs/NitrOMZ_Radius_POC_Sweep.csv',
            chunk_size   = 8,   # radius changes rebuild the physics grid; keep batches small
        )
    else:
        raise ValueError(f"Unknown SWEEP_MODE: '{SWEEP_MODE}'. "
                         f"Choose 'poc_o2', 'no3_o2', or 'radius_poc'.")


# ═════════════════════════════════════════════════════════════════════════════
# ── EXPERIMENT RUNNER ────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(axis1_vals, axis2_vals):
    """
    Runs one batched chunk.  axis1/axis2 correspond to get_sweep_meta() keys.
    The function reads SWEEP_MODE to decide which config fields to patch.
    """
    bs = len(axis1_vals)
    cfg.batch_size = bs

    # ── Force Suite / Early-exit Mode ────────────────────────────────────────
    cfg.is_suite                 = True
    cfg.extrapolate_steady_state = True
    cfg.terminal_snapshot_only   = True
    cfg.use_klawonn_density      = False
    cfg.doc_initial_core         = 0.0

    # ── Resolve per-experiment radius, O2, NO3, POC based on active mode ─────
    if SWEEP_MODE == 'poc_o2':
        radii   = np.array([RADIUS_FIXED] * bs)
        o2_arr  = np.array(axis2_vals, dtype=np.float32)
        no3_arr = np.array([NO3_FIXED]  * bs, dtype=np.float32)
        poc_arr = np.array(axis1_vals,  dtype=np.float32)

    elif SWEEP_MODE == 'no3_o2':
        radii   = np.array([RADIUS_FIXED] * bs)
        o2_arr  = np.array(axis2_vals, dtype=np.float32)
        no3_arr = np.array(axis1_vals, dtype=np.float32)
        poc_arr = np.array([POC_FIXED]  * bs, dtype=np.float32)

    elif SWEEP_MODE == 'radius_poc':
        radii   = np.array(axis1_vals,  dtype=np.float32)
        o2_arr  = np.array([O2_FIXED]   * bs, dtype=np.float32)
        no3_arr = np.array([NO3_FIXED]  * bs, dtype=np.float32)
        poc_arr = np.array(axis2_vals,  dtype=np.float32)

    # ── Apply geometry (radius-dependent, so must come before setup_physics) ─
    cfg.radius = radii
    cfg.U_bg   = 2.2 * (cfg.radius / 1.0) ** 0.56
    cfg.Lx     = 20.0 * cfg.radius
    cfg.Ly     = 10.0 * cfg.radius
    cfg.cx     = 5.0  * cfg.radius
    cfg.cy     = cfg.Ly / 2.0
    cfg.dx     = cfg.Lx / (cfg.Nx - 1)
    cfg.dy     = cfg.Ly / (cfg.Ny - 1)
    cfg.K      = float(cfg.nu / cfg.Sc_target)

    # ── Apply boundary conditions ─────────────────────────────────────────────
    bcs.inflow.o2  = o2_arr .reshape(bs, 1)
    bcs.inflow.no3 = no3_arr.reshape(bs, 1)

    # ── Silence per-run console output ────────────────────────────────────────
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    device_, state = setup_physics(cfg)

    # ── Apply POC initial density ─────────────────────────────────────────────
    poc_t = torch.tensor(poc_arr.reshape(bs, 1, 1), dtype=torch.float32, device=device_)
    state['poc_initial'] = poc_t

    # ── Apply locked amplifiers ───────────────────────────────────────────────
    bio_amp  = torch.tensor([BIO_AMP_LOCKED]  * bs, dtype=torch.float32, device=device_).view(bs, 1, 1)
    khyd_amp = torch.tensor([KHYD_AMP_LOCKED] * bs, dtype=torch.float32, device=device_).view(bs, 1, 1)

    state['bgc'].krem  *= bio_amp
    state['bgc'].kAo   *= bio_amp
    state['bgc'].kNo   *= bio_amp
    state['bgc'].kDen1 *= bio_amp
    state['bgc'].kDen2 *= bio_amp
    state['bgc'].kDen3 *= bio_amp
    state['bgc'].kAx   *= bio_amp
    state['k_hyd']      = state['bgc'].k_hyd * khyd_amp

    # ── Run to steady state & extrapolate ─────────────────────────────────────
    durations, steady_rates = run_simulation(state, cfg, device_)

    sys.stdout.close()
    sys.stdout = original_stdout

    # ── Assemble per-member metrics ───────────────────────────────────────────
    batch_metrics = []
    for b in range(bs):
        dur_hr = durations[b]

        # Extrapolated totals
        tot_n2o          = steady_rates.get('n2o_flux_out',      [0]*bs)[b] * dur_hr
        tot_n2o_internal = steady_rates.get('n2o_internal',      [0]*bs)[b] * dur_hr
        tot_c_consumed   = steady_rates.get('c_consumed',        [0]*bs)[b] * dur_hr
        tot_n2           = steady_rates.get('n2_flux_out',       [0]*bs)[b] * dur_hr

        # Core-only carbon pathways
        tot_oxic = steady_rates.get('oxic_c', [0]*bs)[b] * dur_hr
        tot_den1 = steady_rates.get('den1_c', [0]*bs)[b] * dur_hr
        tot_den2 = steady_rates.get('den2_c', [0]*bs)[b] * dur_hr
        tot_den3 = steady_rates.get('den3_c', [0]*bs)[b] * dur_hr

        # Full-domain carbon pathways
        tot_oxic_domain = steady_rates.get('oxic_c_domain',  [0]*bs)[b] * dur_hr
        tot_den1_domain = steady_rates.get('den1_c_domain',  [0]*bs)[b] * dur_hr
        tot_den2_domain = steady_rates.get('den2_c_domain',  [0]*bs)[b] * dur_hr
        tot_den3_domain = steady_rates.get('den3_c_domain',  [0]*bs)[b] * dur_hr
        tot_c_domain    = steady_rates.get('c_consumed_domain', [0]*bs)[b] * dur_hr

        # N2O yield
        n2o_yield_efficiency = tot_n2o / tot_c_domain if tot_c_domain > 0 else 0.0

        # Core-only metabolic fractions
        sum_c_core = tot_oxic + tot_den1 + tot_den2 + tot_den3
        f_oxic = tot_oxic / sum_c_core if sum_c_core > 0 else 1.0
        f_den1 = tot_den1 / sum_c_core if sum_c_core > 0 else 0.0
        f_den2 = tot_den2 / sum_c_core if sum_c_core > 0 else 0.0
        f_den3 = tot_den3 / sum_c_core if sum_c_core > 0 else 0.0

        # Full-domain metabolic fractions
        sum_c_dom     = tot_oxic_domain + tot_den1_domain + tot_den2_domain + tot_den3_domain
        f_oxic_domain = tot_oxic_domain / sum_c_dom if sum_c_dom > 0 else 1.0
        f_den1_domain = tot_den1_domain / sum_c_dom if sum_c_dom > 0 else 0.0
        f_den2_domain = tot_den2_domain / sum_c_dom if sum_c_dom > 0 else 0.0
        f_den3_domain = tot_den3_domain / sum_c_dom if sum_c_dom > 0 else 0.0

        metrics = {
            # ── Always-present identifier columns ────────────────────────────
            'Initial_POC_Density': poc_arr[b],
            'Ext_O2':              float(o2_arr[b]),
            'Radius_mm':           float(radii[b]),
            'Ext_NO3':             float(no3_arr[b]),
            # ── Core results ─────────────────────────────────────────────────
            'Anoxia_Duration_hr':          dur_hr,
            'Resp_Rate_nmol_C_mm3_hr':     steady_rates.get('resp_rate', [0]*bs)[b],
            # ── N2O budget ───────────────────────────────────────────────────
            'N2O_Net_Domain_SMS_mmol':     tot_n2o,
            'N2O_Net_Core_SMS_mmol':       tot_n2o_internal,
            'N2O_Yield_molN2O_per_molC':   n2o_yield_efficiency,
            'N2_Net_Domain_SMS_mmol':      tot_n2,
            # ── Carbon consumed ──────────────────────────────────────────────
            'C_Consumed_Core_mmol':        tot_c_consumed,
            # ── Core-only pathway fractions and totals ───────────────────────
            'Frac_Oxic_Core':  f_oxic,
            'Frac_Den1_Core':  f_den1,
            'Frac_Den2_Core':  f_den2,
            'Frac_Den3_Core':  f_den3,
            'Oxic_C_Core_mmol': tot_oxic,
            'Den1_C_Core_mmol': tot_den1,
            'Den2_C_Core_mmol': tot_den2,
            'Den3_C_Core_mmol': tot_den3,
            # ── Full-domain pathway fractions and totals ─────────────────────
            'Frac_Oxic_Domain':  f_oxic_domain,
            'Frac_Den1_Domain':  f_den1_domain,
            'Frac_Den2_Domain':  f_den2_domain,
            'Frac_Den3_Domain':  f_den3_domain,
            'Oxic_C_Domain_mmol': tot_oxic_domain,
            'Den1_C_Domain_mmol': tot_den1_domain,
            'Den2_C_Domain_mmol': tot_den2_domain,
            'Den3_C_Domain_mmol': tot_den3_domain,
            'C_Consumed_Domain_mmol': tot_c_domain,
            # ── Boundary fluxes ──────────────────────────────────────────────
            'O2_Consumed_Core_mmol':  steady_rates.get('o2_flux_in',   [0]*bs)[b] * dur_hr,
            'NO3_Consumed_Core_mmol': steady_rates.get('no3_flux_in',  [0]*bs)[b] * dur_hr,
            'DOC_Leakage_Core_mmol':  steady_rates.get('doc_leakage',  [0]*bs)[b] * dur_hr,
        }
        batch_metrics.append(metrics)

    return batch_metrics


# ═════════════════════════════════════════════════════════════════════════════
# ── PLOTTING ─────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def generate_all_plots(csv_filename):
    df   = pd.read_csv(csv_filename)
    meta = get_sweep_meta()

    ax1_col   = meta['axis1_col']    # x-axis on 2-D contour plots
    ax2_col   = meta['axis2_col']    # y-axis on 2-D contour plots
    ax1_label = meta['axis1_label']
    ax2_label = meta['axis2_label']
    ax1_vals  = meta['axis1_vals']
    ax2_vals  = meta['axis2_vals']

    # Blank resp-rate for non-anoxic rows (undefined there)
    dead_zone = df['Anoxia_Duration_hr'] == 0.0
    if 'Resp_Rate_nmol_C_mm3_hr' in df.columns:
        df.loc[dead_zone, 'Resp_Rate_nmol_C_mm3_hr'] = np.nan

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})

    print(f"\n📊 Generating {SWEEP_MODE} sweep plots…")

    # ── Shared helpers ────────────────────────────────────────────────────────

    def plot_contour(values_col, title, cbar_label, filename, cmap='viridis'):
        """
        Generic 2-D filled-contour plot.  Pivot uses ax2 as the row index
        (y-axis) and ax1 as the column index (x-axis) — matching the live
        axis label assignments above.
        """
        plt.figure(figsize=(9, 6))
        pivot = df.pivot_table(index=ax2_col, columns=ax1_col,
                               values=values_col, dropna=False)
        X, Y = np.meshgrid(pivot.columns, pivot.index)
        Z    = pivot.values

        min_z = np.nanmin(Z)
        max_z = np.nanmax(Z)

        if min_z <= 0.0 and max_z > 0.0:
            eps    = max_z * 1e-5
            levels = [0.0, eps] + list(np.linspace(eps, max_z, 20))[1:]
            cf     = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
            c_line = plt.contour(X, Y, Z, levels=[eps],
                                 colors='cyan', linewidths=2, linestyles='dashed')
            plt.clabel(c_line, inline=True, fontsize=10, fmt='Zero Boundary')
        else:
            cf = plt.contourf(X, Y, Z, levels=20, cmap=cmap)

        cbar = plt.colorbar(cf)
        cbar.set_label(cbar_label)
        plt.scatter(X, Y, color='white', edgecolor='black', s=20, alpha=0.8, zorder=5)
        plt.xlabel(ax1_label)
        plt.ylabel(ax2_label)
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_fractions_vs_axis(df_slice, x_col, frac_cols, pathway_labels,
                               title, xlabel, filename, marker):
        """Line plot of four metabolic-pathway fractions against one axis."""
        melted           = df_slice.melt(id_vars=[x_col], value_vars=frac_cols,
                                         var_name='Pathway', value_name='Fraction')
        melted['Pathway'] = melted['Pathway'].map(dict(zip(frac_cols, pathway_labels)))
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=melted, x=x_col, y='Fraction',
                     hue='Pathway', marker=marker, linewidth=2.5)
        plt.title(title, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel('Fraction of Total Carbon Consumed')
        plt.tight_layout()
        plt.savefig(f'outputs/{filename}', dpi=300)
        plt.close()

    # ── Mid-point slices for line plots (Dynamic from CSV) ─────────────
    unique_ax1 = np.sort(df[ax1_col].dropna().unique())
    unique_ax2 = np.sort(df[ax2_col].dropna().unique())
    
    # Find the closest actual value in the dataset to the true center
    median_ax1 = unique_ax1[len(unique_ax1) // 2]
    median_ax2 = unique_ax2[len(unique_ax2) // 2]   

    frac_domain_cols = ['Frac_Oxic_Domain', 'Frac_Den1_Domain',
                        'Frac_Den2_Domain', 'Frac_Den3_Domain']
    frac_core_cols   = ['Frac_Oxic_Core',   'Frac_Den1_Core',
                        'Frac_Den2_Core',   'Frac_Den3_Core']
    pathway_labels   = ['Oxic (O₂)', 'Den1 (NO₃→NO₂)',
                        'Den2 (NO₂→N₂O)', 'Den3 (N₂O→N₂)']

    # ── Plot 1: N₂O yield, full domain ───────────────────────────────────────
    plot_contour(
        'N2O_Yield_molN2O_per_molC',
        f'Plot 1: Net N₂O Production per Carbon Consumed\n'
        f'(Full Domain: core + plume; Den3 consumption subtracted)',
        'mol N₂O mol C⁻¹', 'Plot1_N2O_Yield_Norm.png', 'viridis')

    # ── Plot 2: Absolute net domain N₂O ──────────────────────────────────────
    plot_contour(
        'N2O_Net_Domain_SMS_mmol',
        f'Plot 2: Net Domain-Integrated N₂O SMS Over Particle Lifetime\n'
        f'(Full Domain: core + plume; production minus Den3 consumption; mmol N₂O)',
        'mmol N₂O', 'Plot2_N2O_Domain_Net.png', 'magma')

    # ── Plot 3: Anoxia duration ───────────────────────────────────────────────
    plot_contour(
        'Anoxia_Duration_hr',
        f'Plot 3: Predicted Particle Anoxic Core Duration\n'
        f'(hr; onset = min core O₂ < 0.3 mmol m⁻³; end extrapolated from fuel burn rate)',
        'Duration (hr)', 'Plot3_Anoxia_Duration.png', 'inferno')

    # ── Plot 4: Full-domain fractions vs axis 2 (at median axis 1) ───────────
    df_p4 = df[df[ax1_col] == median_ax1].copy()
    plot_fractions_vs_axis(
        df_p4, ax2_col, frac_domain_cols, pathway_labels,
        f'Plot 4: Full-Domain Carbon Pathway Fractions vs. {ax2_label.split("(")[0].strip()}\n'
        f'(Core + plume; fixed {ax1_label.split("(")[0].strip()} = {median_ax1:.4g})',
        ax2_label,
        'Plot4_Domain_Fractions_vs_Axis2.png', 'o')

    # ── Plot 5: Full-domain fractions vs axis 1 (at median axis 2) ───────────
    df_p5 = df[df[ax2_col] == median_ax2].copy()
    plot_fractions_vs_axis(
        df_p5, ax1_col, frac_domain_cols, pathway_labels,
        f'Plot 5: Full-Domain Carbon Pathway Fractions vs. {ax1_label.split("(")[0].strip()}\n'
        f'(Core + plume; fixed {ax2_label.split("(")[0].strip()} = {median_ax2:.4g})',
        ax1_label,
        'Plot5_Domain_Fractions_vs_Axis1.png', 's')

    # ── Plot 6: Core fractions vs axis 2 (at median axis 1) ──────────────────
    df_p6 = df[df[ax1_col] == median_ax1].copy()
    plot_fractions_vs_axis(
        df_p6, ax2_col, frac_core_cols, pathway_labels,
        f'Plot 6: Particle Core Carbon Pathway Fractions vs. {ax2_label.split("(")[0].strip()}\n'
        f'(Inside particle mask; fixed {ax1_label.split("(")[0].strip()} = {median_ax1:.4g})',
        ax2_label,
        'Plot6_Core_Fractions_vs_Axis2.png', 'o')

    # ── Plot 7: Core fractions vs axis 1 (at median axis 2) ──────────────────
    df_p7 = df[df[ax2_col] == median_ax2].copy()
    plot_fractions_vs_axis(
        df_p7, ax1_col, frac_core_cols, pathway_labels,
        f'Plot 7: Particle Core Carbon Pathway Fractions vs. {ax1_label.split("(")[0].strip()}\n'
        f'(Inside particle mask; fixed {ax2_label.split("(")[0].strip()} = {median_ax2:.4g})',
        ax1_label,
        'Plot7_Core_Fractions_vs_Axis1.png', 's')

    # ── Plot 8: Net N₂O SMS inside particle core ──────────────────────────────
    plt.figure(figsize=(9, 6))
    pivot8 = df.pivot_table(index=ax2_col, columns=ax1_col,
                            values='N2O_Net_Core_SMS_mmol', dropna=False)
    X8, Y8 = np.meshgrid(pivot8.columns, pivot8.index)
    Z8     = pivot8.values
    vmax8  = np.nanmax(np.abs(Z8)) if np.any(np.isfinite(Z8)) else 1.0
    cf8    = plt.contourf(X8, Y8, Z8, levels=20,
                          cmap='RdBu_r', vmin=-vmax8, vmax=vmax8)
    cbar8  = plt.colorbar(cf8)
    cbar8.set_label('mmol N₂O  (+ = net source  ▲,  − = net sink  ▼)')
    plt.scatter(X8, Y8, color='black', s=10, alpha=0.3, zorder=5)
    plt.xlabel(ax1_label)
    plt.ylabel(ax2_label)
    plt.title('Plot 8: Net N₂O SMS Inside Particle Core Over Particle Lifetime\n'
              '(Particle mask only; N₂O production [ammox + Den2] minus '
              'N₂O consumption [Den3]; mmol N₂O)',
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('outputs/Plot8_Core_N2O_Net_SMS.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 9: Steady-state respiration rate ─────────────────────────────────
    plot_contour(
        'Resp_Rate_nmol_C_mm3_hr',
        'Plot 9: Steady-State Microbial Respiration Rate Inside Particle Core\n'
        '(Particle mask only; ΣRemOx + ΣDen1 + ΣDen2 + ΣDen3 at extrapolation '
        'snapshot; nmol C mm⁻³ hr⁻¹)',
        'nmol C mm⁻³ hr⁻¹', 'Plot9_Core_Resp_Rate.png', 'cividis')

    # ── Plot 10: Dominant pathway regime map ──────────────────────────────────
    plt.figure(figsize=(10, 8))
    core_path_cols = ['Oxic_C_Core_mmol', 'Den1_C_Core_mmol',
                      'Den2_C_Core_mmol', 'Den3_C_Core_mmol']
    df['Dominant_Core_Pathway_Idx'] = df[core_path_cols].values.argmax(axis=1)
    pivot10 = df.pivot_table(index=ax2_col, columns=ax1_col,
                             values='Dominant_Core_Pathway_Idx', dropna=False)
    X10, Y10    = np.meshgrid(pivot10.columns, pivot10.index)
    cmap_custom = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    cf10        = plt.contourf(X10, Y10, pivot10.values,
                               levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cmap_custom)
    cbar10      = plt.colorbar(cf10, ticks=[0, 1, 2, 3])
    cbar10.ax.set_yticklabels(['Oxic (O₂)', 'Den1 (NO₃→NO₂)',
                               'Den2 (NO₂→N₂O)', 'Den3 (N₂O→N₂)'])
    cbar10.set_label('Dominant Carbon-Consuming Pathway')
    plt.title('Plot 10: Dominant Carbon-Consuming Pathway Inside Particle Core\n'
              '(argmax of lifetime-integrated C consumed per pathway, '
              'particle mask only)',
              fontweight='bold')
    plt.xlabel(ax1_label)
    plt.ylabel(ax2_label)
    plt.tight_layout()
    plt.savefig('outputs/Plot10_Core_Dominant_Pathway.png', dpi=300)
    plt.close()

    print('✅ All plots generated successfully!')


# ═════════════════════════════════════════════════════════════════════════════
# ── ENTRY POINT ──────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def main():
    meta         = get_sweep_meta()
    ax1_col      = meta['axis1_col']
    ax2_col      = meta['axis2_col']
    ax1_vals     = meta['axis1_vals']
    ax2_vals     = meta['axis2_vals']
    csv_filename = meta['csv_name']
    chunk_size   = meta['chunk_size']

    n_total = len(ax1_vals) * len(ax2_vals)
    print(f"🌊 NitrOMZ Suite  |  mode: {SWEEP_MODE}  |  {len(ax1_vals)} × {len(ax2_vals)} = {n_total} experiments\n")

    os.makedirs('outputs', exist_ok=True)
    print(f"🆕 Starting fresh suite → '{csv_filename}'\n")
    results_data = []

    # Build the full grid: outer = axis1 (columns), inner = axis2 (rows)
    run_configs = [(a1, a2) for a1 in ax1_vals for a2 in ax2_vals]

    print(f"📦 Batch size: {chunk_size} parallel experiments per chunk.\n")

    with tqdm(total=n_total, desc='Simulating Parameters', unit='run',
              bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

        for i in range(0, n_total, chunk_size):
            batch      = run_configs[i: i + chunk_size]
            ax1_batch  = [b[0] for b in batch]
            ax2_batch  = [b[1] for b in batch]

            batched_results = run_experiment(ax1_batch, ax2_batch)
            results_data.extend(batched_results)

            temp_df = pd.DataFrame(results_data)
            temp_df.to_csv(csv_filename, index=False)
            pbar.update(len(batch))

    final_df = pd.DataFrame(results_data)
    final_df = final_df.sort_values(by=[ax1_col, ax2_col])
    final_df.to_csv(csv_filename, index=False)

    print(f'\n✅ Suite complete!  Results → {csv_filename}')
    generate_all_plots(csv_filename)


if __name__ == '__main__':
    main()