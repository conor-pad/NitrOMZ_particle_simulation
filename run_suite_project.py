# run_suite_project.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
torch.set_default_dtype(torch.float32)
import warnings
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore", category=UserWarning)

import time as _time

import config as cfg
import bcs
from physics import setup_physics
from loop import run_simulation

# ── EXECUTION TOGGLES ─────────────────────────────────────────────────────────
RUN_PHASE_1 = True
RUN_PHASE_2 = False

# ── SHARED GRID SETTINGS ──────────────────────────────────────────────────────
RADIUS     = 1.0      # mm — fixed for both phases
NX, NY     = 116, 101
AMBIENT_O2 = 10.0     # mmol/m³
AMBIENT_NO3= 10.0
TOTAL_TIME = 600.0    # s — generous upper bound; extrapolation exits early

# ── PHASE 1 SWEEP PARAMETERS ─────────────────────────────────────────────────
FIXED_POC      = 775000.0          # mmol/m³
N_GRID         = 8         


###################
# ── PHASE 1 OVERNIGHT FOCUS PARAMETERS ───────────────────────────────────────
FIXED_POC  = 775000.0          
N_GRID     = 13  # 16x16 grid = 256 overnight experiments

# The Goldilocks Ridge: High resolution offset sweep to interleave new data
BIO_AMPS_FOCUS  = np.linspace(25, 145, N_GRID)
KHYD_AMPS_FOCUS = np.linspace(1, 125, N_GRID)

SWEEP_BLOCKS = [
    (BIO_AMPS_FOCUS, KHYD_AMPS_FOCUS, "Overnight Ridge Focus (Bio 25-145 x k_hyd 1-125)")
]

###################

# ── PHASE 2 SWEEP PARAMETERS ─────────────────────────────────────────────────
CHUNK_SIZE     = 8                 
MIN_ANOXIA_HRS = 1.5               
# CHANGED: 300k to 500k linearly spaced
POC_LEVELS     = np.linspace(300000, 500000, 16).tolist()   


def _setup_batch(batch_size, poc_density_list, bio_amp_list, khyd_amp_list, device):
    bs = batch_size

    cfg.batch_size = bs
    cfg.Nx, cfg.Ny = NX, NY
    cfg.snapshot_time = 1.0
    cfg.Total_Time    = TOTAL_TIME
    cfg.is_suite      = True
    cfg.use_symmetry  = True
    cfg.terminal_snapshot_only = True

    r = np.array([RADIUS] * bs, dtype=np.float32)
    cfg.radius = r
    cfg.Lx = 20.0 * r;  cfg.Ly = 10.0 * r
    cfg.dx = cfg.Lx / (NX - 1)
    cfg.dy = cfg.Ly / (NY - 1)
    cfg.cx = 5.0 * r
    cfg.cy = cfg.Ly / 2.0
    cfg.U_bg = 2.2 * (r / 1.0) ** 0.56

    bcs.inflow.o2  = np.array([AMBIENT_O2]  * bs, dtype=np.float32).reshape(bs, 1)
    bcs.inflow.no3 = np.array([AMBIENT_NO3] * bs, dtype=np.float32).reshape(bs, 1)

                                                #### BELOWS REPALCE W JUST BIOAMPLS
    if cfg.is_suite and len(poc_density_list) == len(BIO_AMPS_FOCUS) and poc_density_list[0] == FIXED_POC:
        cfg.doc_initial_core = 100.0
    else:
        cfg.doc_initial_core = 0.0
    cfg.use_klawonn_density = False   

    device_, state = setup_physics(cfg)

    poc_arr = np.array(poc_density_list, dtype=np.float32).reshape(bs, 1, 1)
    state['poc_initial'] = torch.tensor(poc_arr, dtype=torch.float32, device=device_)

    bio_amp = torch.tensor(bio_amp_list, dtype=torch.float32, device=device_).view(bs, 1, 1)
    state['bgc'].krem  *= bio_amp
    state['bgc'].kAo   *= bio_amp
    state['bgc'].kNo   *= bio_amp
    state['bgc'].kDen1 *= bio_amp
    state['bgc'].kDen2 *= bio_amp
    state['bgc'].kDen3 *= bio_amp
    state['bgc'].kAx   *= bio_amp

    khyd_amp = torch.tensor(khyd_amp_list, dtype=torch.float32, device=device_).view(bs, 1, 1)
    base_k_hyd = state['bgc'].k_hyd   
    amplified_k_hyd = base_k_hyd * khyd_amp   
    state['k_hyd'] = amplified_k_hyd           

    return device_, state


def run_batched_sweep():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("plots",   exist_ok=True)

    if RUN_PHASE_1:
        n_total_p1 = sum(len(b[0]) * len(b[1]) for b in SWEEP_BLOCKS)
        print(f"\n{'═'*60}")
        print(f"  PHASE 1: PATCHING MISSING GRID BLOCKS  ({n_total_p1} experiments)")
        print(f"{'═'*60}")
        
        for bio, khyd, name in SWEEP_BLOCKS:
            print(f"  {name}:")
            print(f"    Bio amps   : {bio[0]:.0f} → {bio[-1]:.0f}  ({len(bio)} pts)")
            print(f"    k_hyd amps : {khyd[0]:.0f} → {khyd[-1]:.0f}  ({len(khyd)} pts)")
            
        print(f"  Fixed POC  : {FIXED_POC:.0f} mmol/m³")
        print(f"  Radius     : {RADIUS} mm  |  Grid: {NX}×{NY}")
        print(f"{'─'*60}\n")

        p1_rows = []
        p1_start = _time.perf_counter()
        total_batches = len(SWEEP_BLOCKS) * N_GRID
        current_batch = 0

        for bio_amps, khyd_amps, block_name in SWEEP_BLOCKS:
            print(f"\n  ► Starting {block_name}")
            bs = len(bio_amps)

            for i_k, khyd_amp in enumerate(khyd_amps):
                current_batch += 1
                print(f"  ┌─ Batch {current_batch}/{total_batches}  (k_hyd={khyd_amp:.0f}x)")

                t0 = _time.perf_counter()
                device_, state = _setup_batch(
                    batch_size      = bs,
                    poc_density_list= [FIXED_POC] * bs,
                    bio_amp_list    = bio_amps,
                    khyd_amp_list   = [khyd_amp] * bs,
                    device          = None,
                )

                durations, resp_rates = run_simulation(state, cfg, device_)
                elapsed = _time.perf_counter() - t0

                print(f"  │  Results (wall={elapsed:.1f}s):")
                for b, bio_amp in enumerate(bio_amps):
                    d_hr   = durations[b]
                    r_nmol = resp_rates[b]
                    
                    anoxia_str = f"{d_hr:.3f} hr" if d_hr > 0 else "no anoxia  "
                    print(f"  │   bio={bio_amp:>8.0f}x  →  dur={anoxia_str:<12}  rate={r_nmol:>7.2f} nmol/mm³/hr")

                    p1_rows.append({
                        'bio_amp':     float(bio_amp),
                        'khyd_amp':    float(khyd_amp),
                        'duration_hr': d_hr,
                        'resp_rate':   r_nmol,
                    })
                
                mean_s = (_time.perf_counter() - p1_start) / current_batch
                eta_s  = mean_s * (total_batches - current_batch)
                eta_str = f"{eta_s/60:.1f} min" if eta_s > 90 else f"{eta_s:.0f} s"
                print(f"  └─ ETA={eta_str}\n")

        total_p1 = _time.perf_counter() - p1_start
        print(f"{'─'*60}")
        print(f"  Phase 1 complete in {total_p1/60:.1f} min.  Saving data...")

        # APPENDING LOGIC FOR PHASE 1
        new_df_p1 = pd.DataFrame(p1_rows)
        csv_p1_path = "outputs/Phase1_ContourData.csv"
        
        if os.path.exists(csv_p1_path):
            existing_df_p1 = pd.read_csv(csv_p1_path)
            combined_df_p1 = pd.concat([existing_df_p1, new_df_p1]).drop_duplicates(subset=['bio_amp', 'khyd_amp'], keep='last')
            combined_df_p1.to_csv(csv_p1_path, index=False)
            df_p1 = combined_df_p1
            print("  Appended to existing Phase 1 CSV.")
        else:
            new_df_p1.to_csv(csv_p1_path, index=False)
            df_p1 = new_df_p1
            print("  Created new Phase 1 CSV.")

        # ── INTERPOLATE AND PLOT THE ENTIRE CSV DATASET ──


        print("  Generating combined regime maps from full CSV...")
        
        X_pts = df_p1['bio_amp'].values
        Y_pts = df_p1['khyd_amp'].values
        DUR_pts = df_p1['duration_hr'].values
        RATE_pts = df_p1['resp_rate'].values

        # Create dense uniform grid spanning the absolute min/max of all runs
        X = np.linspace(X_pts.min(), X_pts.max(), 100)
        Y = np.linspace(Y_pts.min(), Y_pts.max(), 100)
        XX, YY = np.meshgrid(X, Y)

        # Interpolate scattered data onto smooth grid, force un-run corners to 0.0
        DUR = griddata((X_pts, Y_pts), DUR_pts, (XX, YY), method='linear', fill_value=0.0)
        RATE = griddata((X_pts, Y_pts), RATE_pts, (XX, YY), method='linear', fill_value=0.0)

        # ── Plot 1: Duration contour ──
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor(plt.cm.inferno(0)) # Force background to black for zeroes
        lvls = np.linspace(0.0, max(np.nanmax(DUR), 0.01), 30)
        
        cf = ax.contourf(XX, YY, DUR, levels=lvls, cmap='inferno')
        ax.contour(XX, YY, DUR, levels=[1.5, 2.0], colors=['white', 'cyan'], linewidths=1.5, linestyles='--')
        plt.colorbar(cf, ax=ax, label='Anoxia Duration (hr)')
        ax.set_xlabel('Bio Rate Amplification Factor')
        ax.set_ylabel('k_hyd Amplification Factor')
        ax.set_title(f'Phase 1: Anoxia Duration (Full Map)\n(POC={FIXED_POC:.0f} mmol/m³, R={RADIUS}mm)')
        plt.tight_layout()
        plt.savefig("plots/Phase1_Duration_Contour.png", dpi=300)
        plt.close()

        # ── Plot 2: Resp rate contour ──
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor(plt.cm.viridis(0)) # Force background to dark purple for zeroes
        
        rate_safe = np.where(RATE > 0, RATE, np.nan)
        pos_vals  = rate_safe[np.isfinite(rate_safe) & (rate_safe > 0)]
        
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color=cmap(0)) # Paint any NaNs dark purple
        
        if pos_vals.size == 0:
            cf2 = ax.contourf(XX, YY, RATE, levels=20, cmap=cmap)
        else:
            r_min, r_max = pos_vals.min(), np.nanmax(rate_safe)
            if r_max - r_min < 1.0:
                norm = mcolors.Normalize(vmin=r_min - 10, vmax=r_max + 10)
                cf2 = ax.contourf(XX, YY, rate_safe, levels=20, cmap=cmap, norm=norm)
            else:
                norm = mcolors.LogNorm(vmin=r_min, vmax=r_max)
                cf2 = ax.contourf(XX, YY, rate_safe, levels=20, cmap=cmap, norm=norm)
        
        plt.colorbar(cf2, ax=ax, label='Steady-State Resp. Rate (nmol/mm³/hr)')
        ax.set_xlabel('Bio Rate Amplification Factor')
        ax.set_ylabel('k_hyd Amplification Factor')
        ax.set_title(f'Phase 1: Respiration Rate (Full Map)\n(POC={FIXED_POC:.0f} mmol/m³, R={RADIUS}mm)')
        plt.tight_layout()
        plt.savefig("plots/Phase1_RespRate_Contour.png", dpi=300)
        plt.close()
        print("  Saved: Full Regime Maps.")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: POC SWEEP
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_PHASE_2:
        n_batches_p2 = int(np.ceil(len(POC_LEVELS) / CHUNK_SIZE))
        print(f"\n{'═'*60}")
        print(f"  PHASE 2: POC SWEEP  ({len(POC_LEVELS)} experiments, {n_batches_p2} batch(es) of {CHUNK_SIZE})")
        print(f"{'═'*60}")

        df_p1 = pd.read_csv("outputs/Phase1_ContourData.csv")
        TARGET_HOURS = 2.0
        valid = df_p1[df_p1['duration_hr'] >= TARGET_HOURS].copy()

        if valid.empty:
            print(f"  🚨 No Phase 1 configs achieved ≥ {TARGET_HOURS} hr anoxia. Aborting Phase 2.")
            return

        lowest_khyd = valid['khyd_amp'].min()
        best_khyd_subset = valid[valid['khyd_amp'] == lowest_khyd]
        best = best_khyd_subset.loc[best_khyd_subset['bio_amp'].idxmin()]
        
        best_bio_amp  = float(best['bio_amp'])
        best_khyd_amp = float(best['khyd_amp'])

        print(f"  Amp selection  : bio={best_bio_amp:.0f}x  k_hyd={best_khyd_amp:.0f}x")
        print(f"  (Phase 1 result: dur={best['duration_hr']:.2f} hr  rate={best['resp_rate']:.2f} nmol/mm³/hr)")
        print(f"  POC range      : {POC_LEVELS[0]:.0f} → {POC_LEVELS[-1]:.0f} mmol/m³")
        print(f"{'─'*60}\n")

        p2_rows = []
        p2_start = _time.perf_counter()

        for i in range(0, len(POC_LEVELS), CHUNK_SIZE):
            batch_pocs = POC_LEVELS[i:i + CHUNK_SIZE]
            bs         = len(batch_pocs)
            batch_num  = i // CHUNK_SIZE + 1

            print(f"  ┌─ Batch {batch_num}/{n_batches_p2}")
            t0 = _time.perf_counter()
            device_, state = _setup_batch(
                batch_size      = bs,
                poc_density_list= batch_pocs,
                bio_amp_list    = [best_bio_amp]  * bs,
                khyd_amp_list   = [best_khyd_amp] * bs,
                device          = None,
            )

            durations, resp_rates = run_simulation(state, cfg, device_)
            elapsed = _time.perf_counter() - t0

            print(f"  │  Results (wall={elapsed:.1f}s):")
            for b, poc_val in enumerate(batch_pocs):
                d_hr   = durations[b]
                r_nmol = resp_rates[b]
                anoxia_str = f"{d_hr:.3f} hr" if d_hr > 0 else "no anoxia  "
                print(f"  │   POC={poc_val:>10.0f}  →  dur={anoxia_str:<12}  rate={r_nmol:>7.2f} nmol/mm³/hr")
                p2_rows.append({
                    'poc':         poc_val,
                    'duration_hr': d_hr,
                    'resp_rate':   r_nmol,
                })
            print(f"  └{'─'*57}\n")

        # APPENDING LOGIC FOR PHASE 2
        new_df_p2 = pd.DataFrame(p2_rows)
        csv_p2_path = "outputs/Phase2_POCSweep.csv"

        if os.path.exists(csv_p2_path):
            existing_df_p2 = pd.read_csv(csv_p2_path)
            # Combine and sort by POC density so the plot lines up correctly
            combined_df_p2 = pd.concat([existing_df_p2, new_df_p2]).drop_duplicates(subset=['poc'], keep='last').sort_values('poc')
            combined_df_p2.to_csv(csv_p2_path, index=False)
            df_p2 = combined_df_p2
            print("  Appended to existing Phase 2 CSV.")
        else:
            new_df_p2 = new_df_p2.sort_values('poc')
            new_df_p2.to_csv(csv_p2_path, index=False)
            df_p2 = new_df_p2
            print("  Created new Phase 2 CSV.")

        # Plot 1: Duration vs POC (Plots the COMBINED dataset)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_p2['poc'], df_p2['duration_hr'], 'o-', color='darkorange', linewidth=2, markersize=6)
        ax.axhline(MIN_ANOXIA_HRS, color='red', linestyle='--', alpha=0.7, label=f'{MIN_ANOXIA_HRS} hr threshold')
        ax.set_xlabel('Initial POC Density (mmol C / m³)')
        ax.set_ylabel('Anoxia Duration (hr)') 
        ax.set_title(f'Phase 2: Anoxia Duration vs POC\n(bio={best_bio_amp:.0f}x, k_hyd={best_khyd_amp:.0f}x, R={RADIUS}mm)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("plots/Phase2_Duration_vs_POC.png", dpi=300)
        plt.close()

        # Plot 2: Resp rate vs POC
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_p2['poc'], df_p2['resp_rate'], 's-', color='steelblue', linewidth=2, markersize=6)
        ax.set_xlabel('Initial POC Density (mmol C / m³)')
        ax.set_ylabel('Steady-State Resp. Rate (nmol / mm³ / hr)')
        ax.set_title(f'Phase 2: Respiration Rate vs POC\n(bio={best_bio_amp:.0f}x, k_hyd={best_khyd_amp:.0f}x, R={RADIUS}mm)')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("plots/Phase2_RespRate_vs_POC.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    run_batched_sweep()