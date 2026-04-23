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

# Import your simulation modules
import config as cfg
import bcs
from physics import setup_physics
from loop import run_simulation

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore", message=".*resized since it had shape.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def run_experiment(radius, ext_o2, ext_no3):
    """Runs a single simulation and extracts N2O metrics."""
    
    # ── 1. Apply Sinking Speed ────────────────────────────────────────────────
    # k_stokes = 1.5
    cfg.radius = radius
    # cfg.U_bg = k_stokes * (radius ** 2)  # Commented out Stokes' Law
    cfg.U_bg = 6.0                         # Fixed sinking speed

    # ── 2. Scale the Domain ───────────────────────────────────────────────────
    cfg.Lx = 20.0 * radius
    cfg.Ly = 10.0 * radius
    cfg.cx = 5.0 * radius
    cfg.cy = cfg.Ly / 2.0
    cfg.dx = cfg.Lx / (cfg.Nx - 1)
    cfg.dy = cfg.Ly / (cfg.Ny - 1)

    # ── 3. Dynamic Time Scaling ───────────────────────────────────────────────
    # Time to flush the domain once = Lx / U_bg. 
    # Increased to 5.0 flushes to give DOC flux time to build up anoxic core
    cfg.Total_Time = 5.0 * (cfg.Lx / cfg.U_bg)

    # ── 4. Dynamically Calculate Physical Constants ───────────────────────────
    # Using the dimensionless targets from config.py to calculate nu dynamically
    # nu_constant = 12.0 
    # cfg.nu = nu_constant
    cfg.nu = (cfg.U_bg * cfg.radius) / cfg.Re_target
    cfg.K = cfg.nu / cfg.Sc_target

    # ── 5. Dynamically Update Boundary Conditions ─────────────────────────────
    bcs.inflow.o2 = ext_o2
    bcs.inflow.no3 = ext_no3

    # ── 6. Setup Physics for the New Grid ─────────────────────────────────────
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    device, state = setup_physics(cfg)
    results = run_simulation(state, cfg, device)
    
    sys.stdout.close()
    sys.stdout = original_stdout

    # ── 7. Extract N2O Metrics ────────────────────────────────────────────────
    n2o_snapshots = results[1]      # N2O is index 1 in the returned tuple
    final_n2o = n2o_snapshots[-1]   # The 2D field at the final time step
    
    # A. Total N2O Efflux (Everything in the domain)
    total_n2o_mass = np.sum(final_n2o) * cfg.dx * cfg.dy
    
    # B. Internal N2O (Only N2O trapped inside the solid particle)
    particle_mask_cpu = state['particle_mask'].cpu().numpy()
    internal_n2o_mass = np.sum(final_n2o * particle_mask_cpu) * cfg.dx * cfg.dy
    
    # C. Plume N2O (N2O that has successfully leaked into the wake)
    plume_n2o_mass = total_n2o_mass - internal_n2o_mass

    return {
        'Radius_mm': radius,
        'Speed_mms': cfg.U_bg,
        'Reynolds': cfg.Re_target,
        'Ext_O2': ext_o2,
        'Ext_NO3': ext_no3,
        'N2O_Internal': internal_n2o_mass,
        'N2O_Plume': plume_n2o_mass,
        'N2O_Total_Efflux': total_n2o_mass
    }
def plot_results(csv_filename):
    """Generates a publication-ready plot from the CSV data."""
    print("\n📊 Generating plots...")
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Could not find {csv_filename}!")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})

    g = sns.FacetGrid(df, col="Radius_mm", hue="Ext_NO3", 
                      palette="magma", height=4.5, aspect=1.2, sharey=False)

    g.map(sns.lineplot, "Ext_O2", "N2O_Total_Efflux", marker="o", linewidth=2.5, markersize=8)

    g.set_axis_labels("Ambient O2 Concentration (mmol/m3)", "Total N2O Efflux (mmol)")
    g.set_titles(col_template="Particle Radius: {col_name} mm", weight='bold')

    # ── THE LEGEND FIX ──
    # Move the legend to the bottom center, laid out horizontally in 3 columns
    g.add_legend(title="Ambient NO3 (mmol/m3)", loc='upper center', 
                 bbox_to_anchor=(0.45, -0.15), ncol=3, frameon=True)

    # Adjust layout so there is plenty of room for the title on top and legend on bottom
    plt.subplots_adjust(top=0.85, bottom=0.25)
    g.fig.suptitle("Nitrous Oxide Efflux vs. Ambient Oxygen & Nitrate", 
                   fontsize=16, fontweight='bold', y=0.98)

    output_filename = "N2O_Parameter_Sweep.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved successfully as {output_filename}")
    
    plt.show()
def main():
    print("🌊 Starting NitrOMZ Parameter Suite...\n")
    
    # Tell the physics loop NOT to save video frames, preventing the RAM crash!
    cfg.is_suite = True 

    # Define your parameter space here
    radii = [1.0, 2.0, 3.0]             # Small, Medium, Large marine snow
    o2_levels = [2.0, 6.0, 10.0]        # Severe, Moderate, Light hypoxia
    no3_levels = [5.0, 10.0, 15.0]      # Nitrate availability
    

    csv_filename = "NitrOMZ_Suite_Results.csv"

    # ── ADVISOR'S FIX: RESUME FROM CRASH & AUTO-CLEAN ──
    if os.path.exists(csv_filename):
        print(f"📂 Found existing results! Cleaning and loading {csv_filename}...")
        results_df = pd.read_csv(csv_filename)
        
        # 1. Round the parameters so Python stops getting confused by 6.00000001
        results_df = results_df.round({'Radius_mm': 2, 'Ext_O2': 2, 'Ext_NO3': 2})
        
        # 2. Drop any accidental duplicates created by the bug (keeps the most recent one)
        results_df = results_df.drop_duplicates(subset=['Radius_mm', 'Ext_O2', 'Ext_NO3'], keep='last')
        
        # 3. Save the newly cleaned data right back to the CSV
        results_df.to_csv(csv_filename, index=False)
        
        # 4. Build the completed list
        completed_runs = set(zip(results_df['Radius_mm'], results_df['Ext_O2'], results_df['Ext_NO3']))
        results_data = results_df.to_dict('records')
    else:
        completed_runs = set()
        results_data = []

    total_runs = len(radii) * len(o2_levels) * len(no3_levels)
    runs_left = total_runs - len(completed_runs)
    print(f"Total simulations: {total_runs} | Remaining: {runs_left}")

    # Run the suite
    with tqdm(total=total_runs, desc="Overall Progress", unit="sim") as pbar:
        pbar.update(len(completed_runs))
        
        for r in radii:
            for o2 in o2_levels:
                for no3 in no3_levels:
                    # FIX: Round the run_id to perfectly match the cleaned set
                    run_id = (round(r, 2), round(o2, 2), round(no3, 2))
                    if run_id in completed_runs:
                        continue
                    
                    print(f"\n▶ Running: R={r} | O2={o2} | NO3={no3}")
                    metrics = run_experiment(radius=r, ext_o2=o2, ext_no3=no3)
                    results_data.append(metrics)
                    
                    pd.DataFrame(results_data).to_csv(csv_filename, index=False)
                    pbar.update(1)

    print(f"\n✅ Suite Complete! All results safely saved to {csv_filename}")
    
    # Generate the plot
    plot_results(csv_filename)

if __name__ == "__main__":
    main()