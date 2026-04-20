# plotting.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def generate_plots(c_snapshots, n2o_snapshots, no3_snapshots, no2_snapshots, n2_snapshots, doc_snapshots, nh4_snapshots, w_snapshots, u_snapshots, v_snapshots, snapshot_times, cfg):
    x = np.linspace(0, cfg.Lx, cfg.Nx)
    y = np.linspace(0, cfg.Ly, cfg.Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # ── Safe parameter extraction (Fallbacks if not added to cfg) ───────────────
    dt_val = getattr(cfg, 'dt', 0.0)
    dt_str = f"{dt_val:.6f}" if dt_val > 0 else "Unknown"
    drag_coeff = getattr(cfg, 'drag_max', 240.0)

    # ── DYNAMIC BOUNDS HELPER ───────────────────────────────────────────────────
    def get_bounds(data_list):
        val_min = np.min(data_list)
        val_max = np.max(data_list)
        if val_max - val_min < 1e-5: 
            val_max = val_min + 0.01
        return val_min, val_max
    
    def add_perfect_colorbar(im, ax, label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label=label)

    o2_min, o2_max   = get_bounds(c_snapshots)
    no3_min, no3_max = get_bounds(no3_snapshots)
    no2_min, no2_max = get_bounds(no2_snapshots)
    n2o_min, n2o_max = get_bounds(n2o_snapshots)
    n2_min, n2_max   = get_bounds(n2_snapshots)
    doc_min, doc_max = get_bounds(doc_snapshots)
    nh4_min, nh4_max = get_bounds(nh4_snapshots)

# ── 2D Animation (4 rows, 2 columns) ────────────────────────────────────────
    plt.rcParams['animation.embed_limit'] = 250
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 8.75), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    ax_vel, ax_o2, ax_no3, ax_no2, ax_n2o, ax_n2, ax_doc, ax_nh4 = axes_flat

    # CALCULATE ZOOM
    zoom_y_min = max(0, cfg.cy - 2.5 * cfg.radius)
    zoom_y_max = min(cfg.Ly, cfg.cy + 2.5 * cfg.radius)
    zoom_x_min = 0.0
    zoom_x_max = cfg.Lx

    for ax in axes_flat:
        circle = Circle((cfg.cx, cfg.cy), cfg.radius, color='white', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.axhline(cfg.cy, color='black', linestyle='--', alpha=0.6)
        ax.axvline(cfg.cx, color='black', linestyle='--', alpha=0.6)

    skip = 8

    # Velocity Plot
    speed = np.sqrt(u_snapshots[0]**2 + v_snapshots[0]**2)
    im_vel = ax_vel.imshow(speed.T, origin='lower', extent=[0, cfg.Lx, 0, cfg.Ly], cmap='jet', vmin=0, vmax=cfg.U_bg * 1.5)
    add_perfect_colorbar(im_vel, ax_vel, 'Speed (mm/s)')
    Q = ax_vel.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                      u_snapshots[0][::skip, ::skip], v_snapshots[0][::skip, ::skip],
                      color='white', scale=cfg.U_bg * 40, alpha=0.8) # scale was 70 before.
    ax_vel.set_title("Velocity", fontweight='bold')

    # O2 Plot
    im_o2 = ax_o2.imshow(c_snapshots[0].T, origin='lower', extent=[0, cfg.Lx, 0, cfg.Ly], cmap='magma', vmin=o2_min, vmax=o2_max)
    add_perfect_colorbar(im_o2, ax_o2, 'O2 Concentration')
    ax_o2.set_title("O2", fontweight='bold')

    # NO3 Plot 
    im_no3 = ax_no3.imshow(no3_snapshots[0].T, origin='lower', extent=[0, cfg.Lx, 0, cfg.Ly], cmap='magma', vmin=no3_min, vmax=no3_max)
    add_perfect_colorbar(im_no3, ax_no3, 'NO3 Concentration')
    ax_no3.set_title("NO3", fontweight='bold')

    # 4. NO2 Plot 
    im_no2 = ax_no2.imshow(no2_snapshots[0].T, origin='lower', extent=[0, cfg.Lx, 0, cfg.Ly], cmap='plasma', vmin=no2_min, vmax=no2_max)
    add_perfect_colorbar(im_no2, ax_no2, 'NO2 Concentration')
    ax_no2.set_title("NO2", fontweight='bold')

    # 5. N2O Plot 
    im_n2o = ax_n2o.imshow(n2o_snapshots[0].T, origin='lower', extent=[0, cfg.Lx, 0, cfg.Ly], cmap='magma', vmin=n2o_min, vmax=n2o_max)
    add_perfect_colorbar(im_n2o, ax_n2o, 'N2O Concentration')
    ax_n2o.set_title("N2O", fontweight='bold')

    # 6. N2 Plot 
    im_n2 = ax_n2.imshow(n2_snapshots[0].T, origin='lower', extent=[0, cfg.Lx, 0, cfg.Ly], cmap='magma', vmin=n2_min, vmax=n2_max)
    add_perfect_colorbar(im_n2, ax_n2, 'N2 Concentration')
    ax_n2.set_title("N2", fontweight='bold')

    # 7. DOC Plot 
    im_doc = ax_doc.imshow(doc_snapshots[0].T, origin='lower', extent=[0, cfg.Lx, 0, cfg.Ly], cmap='YlGn', vmin=doc_min, vmax=doc_max)
    add_perfect_colorbar(im_doc, ax_doc, 'DOC Concentration')
    ax_doc.set_title("DOC", fontweight='bold')

    # 8. NH4 Plot 
    im_nh4 = ax_nh4.imshow(nh4_snapshots[0].T, origin='lower', extent=[0, cfg.Lx, 0, cfg.Ly], cmap='magma', vmin=nh4_min, vmax=nh4_max)
    add_perfect_colorbar(im_nh4, ax_nh4, 'NH4 Concentration')
    ax_nh4.set_title("NH4", fontweight='bold')

# ── Titles and Parameter Printout ───────────────────────────────────────────
    global_title = fig.suptitle("Time: 0.00", fontsize=18, fontweight='bold', y=0.98)
    
    # Safely extract dimensionless numbers (in case variable names slightly differ in config)
    Re_val = getattr(cfg, 'Re_target', getattr(cfg, 'Re', 0.0))
    Sc_val = getattr(cfg, 'Sc_target', getattr(cfg, 'Sc_calc', getattr(cfg, 'Sc', 0.0)))
    Pe_val = getattr(cfg, 'Pe_calc', getattr(cfg, 'Pe_target', getattr(cfg, 'Pe', 0.0)))

    # Added a new top row for Re, Sc, and Pe!
    param_str = (f"Re: {Re_val:.2f} | Sc: {Sc_val:.1f} | Pe: {Pe_val:.2f}\n"
                 f"U: {cfg.U_bg} | Radius: {cfg.radius} | $\\nu$: {cfg.nu:.2f} | $K$: {cfg.K:.5f}\n"
                 f"$dx$: {cfg.dx:.3f} | $dy$: {cfg.dy:.3f} | $dt$: {dt_str}")
    
    # Moved the box up slightly (0.94) to make room for the 3rd line of text
    fig.text(0.5, 0.94, param_str, ha='center', va='top', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # ENFORCE ZOOM LAST! (Forces the view box after imshow draws)
    for ax in axes_flat:
        ax.set_xlim(zoom_x_min, zoom_x_max)
        ax.set_ylim(zoom_y_min, zoom_y_max)

    # Force the vertical space to be tiny and leave room at the top for the bigger box
    plt.subplots_adjust(top=0.84, bottom=0.05, left=0.05, right=0.95, hspace=0.15, wspace=0.1)
    def update(frame_idx):
        speed = np.sqrt(u_snapshots[frame_idx]**2 + v_snapshots[frame_idx]**2)
        im_vel.set_data(speed.T)
        Q.set_UVC(u_snapshots[frame_idx][::skip, ::skip], v_snapshots[frame_idx][::skip, ::skip])

        im_o2.set_data(c_snapshots[frame_idx].T)
        im_no3.set_data(no3_snapshots[frame_idx].T)
        im_no2.set_data(no2_snapshots[frame_idx].T)
        im_n2o.set_data(n2o_snapshots[frame_idx].T)
        im_n2.set_data(n2_snapshots[frame_idx].T)
        im_doc.set_data(doc_snapshots[frame_idx].T)
        im_nh4.set_data(nh4_snapshots[frame_idx].T)

        global_title.set_text(f"Time: {snapshot_times[frame_idx]:.2f}")
        return [im_vel, Q, im_o2, im_no3, im_no2, im_n2o, im_n2, im_doc, im_nh4] 

    anim = animation.FuncAnimation(fig, update, frames=len(c_snapshots), interval=15, blit=False)

    # ── Build Dynamic Filename ──────────────────────────────────────────────────
    base_filename = f"U{cfg.U_bg}_R{cfg.radius}_nu{cfg.nu}_K{cfg.K:.5f}_dt{dt_str}_dx{cfg.dx:.3f}_dy{cfg.dy:.3f}_drag{drag_coeff}"
    filename_2d = f"2D_Zoomed_{base_filename}.mp4"

    print(f"\nSaving 2D Animation to {filename_2d}...")
    
    pbar = tqdm(total=len(c_snapshots), desc="Rendering Video", ascii="⡀⡄⡆⡇▞▚░▒▓", unit="frames", bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    def update_progress(current_frame, total_frames):
        pbar.update(1)

    anim.save(filename_2d, writer='ffmpeg', fps=60, progress_callback=update_progress)
              
    pbar.close()
    print(f"Saved {filename_2d} successfully!")

    # plt.show()

    #############

    # ── 1D Cross-Section Animation ────────────────────────────────────────────────
    y_slice = cfg.cy;  idx_y = int(y_slice / cfg.dy)

    fig_1d, axes_1d = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    axes_1d_flat = axes_1d.flatten()

    titles = ["Velocity Speed", "O2", "NO3", "NO2", "N2O", "N2", "DOC", "NH4"]
    colors = ['black', 'blue', 'green', 'orange', 'purple', 'red', 'darkgreen', 'magenta']

    speed_0 = np.sqrt(u_snapshots[0]**2 + v_snapshots[0]**2)

    lines = []
    line_vel, = axes_1d_flat[0].plot(x, speed_0[:, idx_y], color=colors[0], lw=2)
    line_o2,  = axes_1d_flat[1].plot(x, c_snapshots[0][:, idx_y], color=colors[1], lw=2)
    line_no3, = axes_1d_flat[2].plot(x, no3_snapshots[0][:, idx_y], color=colors[2], lw=2)
    line_no2, = axes_1d_flat[3].plot(x, no2_snapshots[0][:, idx_y], color=colors[3], lw=2)
    line_n2o, = axes_1d_flat[4].plot(x, n2o_snapshots[0][:, idx_y], color=colors[4], lw=2)
    line_n2,  = axes_1d_flat[5].plot(x, n2_snapshots[0][:, idx_y], color=colors[5], lw=2)
    line_doc, = axes_1d_flat[6].plot(x, doc_snapshots[0][:, idx_y], color=colors[6], lw=2)
    line_nh4, = axes_1d_flat[7].plot(x, nh4_snapshots[0][:, idx_y], color=colors[7], lw=2)
    lines = [line_vel, line_o2, line_no3, line_no2, line_n2o, line_n2, line_doc, line_nh4]

    y_limits = [
        (0, cfg.U_bg * 3), 
        (max(0, o2_min - 0.05), o2_max + 0.05),
        (max(0, no3_min - 1.0), no3_max + 1.0),
        (max(0, no2_min - 0.001), no2_max + 0.001),
        (max(0, n2o_min - 0.001), n2o_max + 0.001),
        (max(0, n2_min - 0.001), n2_max + 0.001),
        (max(0, doc_min - 1.0), doc_max + 1.0),
        (max(0, nh4_min - 0.001), nh4_max + 0.001)
    ]

    for i, ax in enumerate(axes_1d_flat):
        ax.axvline(cfg.cx - cfg.radius, color='k', linestyle='--', alpha=0.5, label='Particle Bound')
        ax.axvline(cfg.cx + cfg.radius, color='k', linestyle='--', alpha=0.5)
        ax.set_title(titles[i], fontweight='bold')
        ax.set_xlim(0, cfg.Lx)
        ax.set_ylim(y_limits[i])
        ax.set_ylabel("Concentration")
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.set_ylabel("Speed (mm/s)")
            ax.legend(loc='upper right', fontsize=8)
        if i >= 6:
            ax.set_xlabel("X coordinate")

    title_1d = fig_1d.suptitle("Horizontal Cross-Section — Time: 0.00", fontsize=16, fontweight='bold', y=0.98)
    
    # Also print the parameters on the 1D cross-section plot
    fig_1d.text(0.5, 0.94, param_str, ha='center', va='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

    def update_1d(frame_idx):
        speed = np.sqrt(u_snapshots[frame_idx]**2 + v_snapshots[frame_idx]**2)
        lines[0].set_ydata(speed[:, idx_y])
        lines[1].set_ydata(c_snapshots[frame_idx][:, idx_y])
        lines[2].set_ydata(no3_snapshots[frame_idx][:, idx_y])
        lines[3].set_ydata(no2_snapshots[frame_idx][:, idx_y])
        lines[4].set_ydata(n2o_snapshots[frame_idx][:, idx_y])
        lines[5].set_ydata(n2_snapshots[frame_idx][:, idx_y])
        lines[6].set_ydata(doc_snapshots[frame_idx][:, idx_y])
        lines[7].set_ydata(nh4_snapshots[frame_idx][:, idx_y])
        
        title_1d.set_text(f"Horizontal Cross-Section — Time: {snapshot_times[frame_idx]:.2f}")
        return lines

    anim_1d = animation.FuncAnimation(fig_1d, update_1d, frames=len(c_snapshots), interval=15, blit=False)

    filename_1d = f"1D_{base_filename}.mp4"
    print(f"\nSaving 1D Animation to {filename_1d}...")
    
    pbar = tqdm(total=len(c_snapshots), desc="Rendering Video", ascii="⡀⡄⡆⡇▞▚░▒▓", unit="frames", bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    def update_progress(current_frame, total_frames):
        pbar.update(1)

    anim_1d.save(filename_1d, writer='ffmpeg', fps=60, progress_callback=update_progress)
              
    pbar.close()
    print(f"Saved {filename_1d} successfully!")

    plt.show()