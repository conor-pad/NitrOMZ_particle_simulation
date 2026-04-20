import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings
warnings.filterwarnings("ignore", message=".*resized since it had shape.*")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
from IPython.display import HTML, display
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time as _time
from tqdm import tqdm

# ── Disable Autograd for CFD ──────────────────────────────────────────────────
torch.set_grad_enabled(False)

# ── Apple Silicon (M2) Setup ──────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple Silicon GPU (MPS) detected! Running on Metal.")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not found, falling back to CPU.")

# ── Domain setup ──────────────────────────────────────────────────────────────
PLOT_MODE = "velocity"  # Options: "o2", "vorticity", "velocity"

Lx, Ly = 40.0, 12.5
# Nx, Ny = 215, 145  # faster
Nx, Ny = 125, 75

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
Total_Time = 25

# Physics parameters
Sc        = 500
nu        = 1e-6
K         = 0.1025
U_bg      = 10
R         = 0.05
R_denit   = 2.0
O2_thresh = 0.04

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

psi_bg_np = U_bg * Y

# Circular Particle (Drag Mask)
drag_mask_np = np.zeros_like(X)
resp_mask_np = np.zeros_like(X)
cx, cy  = 10.0, Ly / 2.0
radius  = 2.5
particle_idx               = (X - cx)**2 + (Y - cy)**2 <= radius**2
drag_mask_np[particle_idx] = 240.0
resp_mask_np[particle_idx] = 1.0

# ── OPTIMISATION 3: Raised CFL target ────────────────────────────────────────
target_CFL = 0.20
dt_adv  = target_CFL * min(dx, dy) / U_bg
max_alpha = np.max(drag_mask_np)
dt_drag = 0.5 / max_alpha if max_alpha > 0 else float('inf')
dt_diff = 0.25 * min(dx, dy)**2 / max(nu, K)
dt = min(dt_adv, dt_drag, dt_diff)
print(f"Time step: {dt:.6f}  (was {0.05 * min(dx,dy) / U_bg:.6f} at CFL=0.05)")

drag_mask_np = gaussian_filter(drag_mask_np, sigma=1.0)
da_dx_np, da_dy_np = np.gradient(drag_mask_np, dx, dy)

# ── Push Arrays to GPU ────────────────────────────────────────────────────────
psi_bg    = torch.tensor(psi_bg_np,   dtype=torch.float32, device=device)
drag_mask = torch.tensor(drag_mask_np, dtype=torch.float32, device=device)
resp_mask = torch.tensor(resp_mask_np, dtype=torch.float32, device=device)
da_dx     = torch.tensor(da_dx_np,    dtype=torch.float32, device=device)
da_dy     = torch.tensor(da_dy_np,    dtype=torch.float32, device=device)

w          = torch.zeros((Nx, Ny), dtype=torch.float32, device=device)
tracer     = torch.zeros((Nx, Ny), dtype=torch.float32, device=device)
tracer[0, :] = 1.0
tracer_n2o = torch.zeros((Nx, Ny), dtype=torch.float32, device=device)

# Pre-allocate full velocity tensors (reused every step)
u_full = torch.zeros((Nx, Ny), dtype=torch.float32, device=device)
v_full = torch.zeros((Nx, Ny), dtype=torch.float32, device=device)

# ── OPTIMISATION 5: Precompute reciprocal constants ───────────────────────────
inv_4dxdy = 1.0 / (4.0 * dx * dy)
# FIX #2: separate dx² and dy² reciprocals for correct anisotropic Laplacian
inv_dx2   = 1.0 / dx**2
inv_dy2   = 1.0 / dy**2
inv_2dy   = 1.0 / (2.0 * dy)
inv_2dx   = 1.0 / (2.0 * dx)

# ── Precompute Poisson eigenvalues ────────────────────────────────────────────
Ni, Nj = Nx - 2, Ny - 2
ii    = torch.arange(1, Ni + 1, dtype=torch.float32, device=device)
jj    = torch.arange(1, Nj + 1, dtype=torch.float32, device=device)
lam_x = (2 / dx**2) * (torch.cos(np.pi * ii / (Ni + 1)) - 1)
lam_y = (2 / dy**2) * (torch.cos(np.pi * jj / (Nj + 1)) - 1)
Lambda = lam_x[:, None] + lam_y[None, :]

# ── Pre-allocated DST-I buffers (FIX #7: now actually wired in) ───────────────
_dst_buf_axis0 = torch.zeros((2 * (Ni + 1), Nj), dtype=torch.float32, device=device)
_dst_buf_axis1 = torch.zeros((Ni, 2 * (Nj + 1)), dtype=torch.float32, device=device)

def dstn1(x_in):
    n0, n1 = x_in.shape

    # FIX #7: reuse pre-allocated buffers instead of allocating on every call
    _dst_buf_axis0.zero_()
    _dst_buf_axis0[1:n0 + 1, :] = x_in
    _dst_buf_axis0[n0 + 2:,  :] = -torch.flip(x_in, dims=[0])

    y0  = torch.fft.fft(_dst_buf_axis0, dim=0)
    mid = -y0[1:n0 + 1].imag / float(np.sqrt(2 * (n0 + 1)))

    _dst_buf_axis1.zero_()
    _dst_buf_axis1[:, 1:n1 + 1] = mid
    _dst_buf_axis1[:, n1 + 2:]  = -torch.flip(mid, dims=[1])

    y1 = torch.fft.fft(_dst_buf_axis1, dim=1)
    return -y1[:, 1:n1 + 1].imag / float(np.sqrt(2 * (n1 + 1)))

idstn1 = dstn1

def get_psi_pert(w_field):
    rhs       = -w_field[1:-1, 1:-1]
    rhs_hat   = dstn1(rhs)
    psi_inner = idstn1(rhs_hat / Lambda)
    psi       = torch.zeros_like(w_field)
    psi[1:-1, 1:-1] = psi_inner
    return psi

# ── Pre-allocated RHS output buffers ─────────────────────────────────────────
# NOTE: these are overwritten each RK3 stage. Safe because the stage result
# (e.g. w_n + dt/3 * rhs_w) is materialised as a new tensor before the next
# get_rhs_batched call. Don't restructure without keeping this invariant.
_rhs_buf_w   = torch.zeros((Nx, Ny), dtype=torch.float32, device=device)
_rhs_buf_c   = torch.zeros((Nx, Ny), dtype=torch.float32, device=device)
_rhs_buf_n2o = torch.zeros((Nx, Ny), dtype=torch.float32, device=device)

# ── OPTIMISATION 1: Batched RHS — one pass for all 3 fields ──────────────────
def get_rhs_batched(w_f, c_o2, c_n2o, streamfunction, denit_ramp=1.0):
    """
    Compute RHS for vorticity, O2, and N2O in one pass.
    The expensive streamfunction stencil (9 slices) is computed once
    instead of three times.
    """
    p = streamfunction

    # ── Streamfunction stencils (computed ONCE for all fields) ────────────
    p_e  = p[2:,   1:-1];  p_w  = p[:-2,  1:-1]
    p_n  = p[1:-1, 2:];    p_s  = p[1:-1, :-2]
    p_ne = p[2:,   2:];    p_sw = p[:-2,  :-2]
    p_se = p[2:,   :-2];   p_nw = p[:-2,  2:]

    dp_ew    = p_e  - p_w
    dp_ns    = p_n  - p_s
    dp_ne_se = p_ne - p_se
    dp_nw_sw = p_nw - p_sw
    dp_ne_nw = p_ne - p_nw
    dp_se_sw = p_se - p_sw

    # ── Stack all 3 field interiors → [3, Ni, Nj] for batched Jacobian ───
    f_c  = torch.stack([w_f[1:-1, 1:-1], c_o2[1:-1, 1:-1], c_n2o[1:-1, 1:-1]])
    f_e  = torch.stack([w_f[2:,   1:-1], c_o2[2:,   1:-1], c_n2o[2:,   1:-1]])
    f_w  = torch.stack([w_f[:-2,  1:-1], c_o2[:-2,  1:-1], c_n2o[:-2,  1:-1]])
    f_n  = torch.stack([w_f[1:-1, 2:],   c_o2[1:-1, 2:],   c_n2o[1:-1, 2:]])
    f_s  = torch.stack([w_f[1:-1, :-2],  c_o2[1:-1, :-2],  c_n2o[1:-1, :-2]])
    f_ne = torch.stack([w_f[2:,   2:],   c_o2[2:,   2:],   c_n2o[2:,   2:]])
    f_sw = torch.stack([w_f[:-2,  :-2],  c_o2[:-2,  :-2],  c_n2o[:-2,  :-2]])
    f_se = torch.stack([w_f[2:,   :-2],  c_o2[2:,   :-2],  c_n2o[2:,   :-2]])
    f_nw = torch.stack([w_f[:-2,  2:],   c_o2[:-2,  2:],   c_n2o[:-2,  2:]])

    # ── Arakawa Jacobian — all 3 fields in one batch ──────────────────────
    J_std   = (dp_ew * (f_n - f_s) - dp_ns * (f_e - f_w)) * inv_4dxdy
    J_hat   = (p_e  * (f_ne - f_se) - p_w  * (f_nw - f_sw)
             - p_n  * (f_ne - f_nw) + p_s  * (f_se - f_sw)) * inv_4dxdy
    J_tilde = (f_n  * dp_ne_nw - f_s  * dp_se_sw
             - f_e  * dp_ne_se + f_w  * dp_nw_sw) * inv_4dxdy
    J_avg   = (J_std + J_hat + J_tilde) / 3.0  # [3, Ni, Nj]

    # FIX #2: correct anisotropic Laplacian (dx ≠ dy, so they need separate weights)
    lap = (f_e + f_w - 2.0 * f_c) * inv_dx2 + (f_n + f_s - 2.0 * f_c) * inv_dy2

    # ── Cached interior masks ─────────────────────────────────────────────
    rm  = resp_mask[1:-1, 1:-1]
    dm  = drag_mask[1:-1, 1:-1]
    dax = da_dx[1:-1, 1:-1]
    day = da_dy[1:-1, 1:-1]

    # ── Vorticity RHS (field 0) ───────────────────────────────────────────
    u_c   = dp_ns * inv_2dy
    v_c   = -dp_ew * inv_2dx
    drag  = dm * f_c[0] - (day * u_c - dax * v_c)
    rhs_w = J_avg[0] + nu * lap[0] - drag

    # ── O2 RHS (field 1) ─────────────────────────────────────────────────
    rhs_o2 = J_avg[1] + K * lap[1] - R * rm * f_c[1]

    # ── N2O RHS (field 2) ────────────────────────────────────────────────
    denit_src = (R_denit * torch.clamp(O2_thresh - f_c[1], min=0.0)
                 / O2_thresh * rm)
    rhs_n2o   = J_avg[2] + K * lap[2] + denit_src * denit_ramp

    # ── Write to pre-allocated buffers ────────────────────────────────────
    _rhs_buf_w[1:-1, 1:-1]   = rhs_w
    _rhs_buf_c[1:-1, 1:-1]   = rhs_o2
    _rhs_buf_n2o[1:-1, 1:-1] = rhs_n2o

    return _rhs_buf_w, _rhs_buf_c, _rhs_buf_n2o


def apply_bcs(w_f, c_o2, c_n2o):
    w_f[:, 0]  = 0;   w_f[:, -1]  = 0
    w_f[0, :]  = 0;   w_f[-1, :]  = w_f[-2, :]
    c_o2[0, :]   = 1.0
    c_o2[-1, :]  = c_o2[-2, :]
    c_o2[:, 0]   = c_o2[:, 1];   c_o2[:, -1]  = c_o2[:, -2]
    c_n2o[0, :]  = 0.0
    c_n2o[-1, :] = c_n2o[-2, :]
    c_n2o[:, 0]  = c_n2o[:, 1];  c_n2o[:, -1] = c_n2o[:, -2]
    return w_f, c_o2, c_n2o


# ── OPTIMISATION 2: GPU bilinear interpolation ────────────────────────────────
def bilinear_interp_gpu(field, px_t, py_t):
    ix = torch.clamp((px_t / dx).long(), 0, Nx - 2)
    iy = torch.clamp((py_t / dy).long(), 0, Ny - 2)
    fx = px_t / dx - ix.float()
    fy = py_t / dy - iy.float()
    return (field[ix,     iy    ] * (1 - fx) * (1 - fy)
          + field[ix + 1, iy    ] * fx        * (1 - fy)
          + field[ix,     iy + 1] * (1 - fx)  * fy
          + field[ix + 1, iy + 1] * fx         * fy)


# ── GPU particle arrays ───────────────────────────────────────────────────────
n_particles = 250
px_t = torch.empty(n_particles, dtype=torch.float32, device=device).uniform_(0.0,  0.2)
py_t = torch.empty(n_particles, dtype=torch.float32, device=device).uniform_(0.1,  Ly - 0.1)

# ── OPTIMISATION 4: torch.compile on hot functions (PyTorch ≥ 2.0) ───────────
# FIX #3: use default "inductor" backend — aot_eager is a debug backend and
# is often slower than no compilation at all.
if hasattr(torch, 'compile'):
    try:
        get_rhs_batched = torch.compile(get_rhs_batched)  # defaults to "inductor"
        print("✅ torch.compile enabled for RHS physics (inductor backend)")
    except Exception as e:
        print(f"⚠️  torch.compile skipped: {e}")

# ── Main SSP-RK3 Loop ─────────────────────────────────────────────────────────
# FIX #5: replaced the original 2nd-order scheme with proper 3rd-order
# SSP-RK3 (Shu-Osher). Eliminates 3 full-grid .clone() calls per step.
#   Stage 1: u1      = un + dt * L(un)
#   Stage 2: u2      = 0.75*un + 0.25*(u1 + dt*L(u1))
#   Stage 3: u_{n+1} = (1/3)*un + (2/3)*(u2 + dt*L(u2))
w_snapshots, c_snapshots, p_snapshots, n2o_snapshots = [], [], [], []
u_snapshots, v_snapshots = [], []
snapshot_times = []
n_steps = int(Total_Time / dt)
snapshot_interval = max(1, int(0.015 / dt))

print(f"Total steps: {n_steps}  |  Snapshot every {snapshot_interval} steps")
print("Starting SSP-RK3 Loop on M2 GPU...")
_loop_start = _time.perf_counter()

for n in tqdm(range(n_steps),
              desc="Simulating..",
              ascii="⡀⡄⡆⡇▞▚░▒▓",
              unit="steps",
              bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):

    current_time = n * dt
    denit_ramp = 0.0 if current_time < 5.0 else min(1.0, (current_time - 5.0) / 10.0)

    # ── STAGE 1:  u1 = un + dt * L(un) ───────────────────────────────────
    psi_pert = get_psi_pert(w)
    psi_tot  = psi_pert + psi_bg
    rhs_w, rhs_c, rhs_n2o = get_rhs_batched(w, tracer, tracer_n2o, psi_tot, denit_ramp)
    w1, c1, n2o1 = apply_bcs(
        w          + dt * rhs_w,
        tracer     + dt * rhs_c,
        tracer_n2o + dt * rhs_n2o)

    # ── STAGE 2:  u2 = 0.75*un + 0.25*(u1 + dt*L(u1)) ───────────────────
    psi_pert = get_psi_pert(w1)
    psi_tot  = psi_pert + psi_bg
    rhs_w, rhs_c, rhs_n2o = get_rhs_batched(w1, c1, n2o1, psi_tot, denit_ramp)
    w2, c2, n2o2 = apply_bcs(
        0.75 * w          + 0.25 * (w1    + dt * rhs_w),
        0.75 * tracer     + 0.25 * (c1    + dt * rhs_c),
        0.75 * tracer_n2o + 0.25 * (n2o1  + dt * rhs_n2o))

    # ── STAGE 3:  u_{n+1} = (1/3)*un + (2/3)*(u2 + dt*L(u2)) ────────────
    psi_pert = get_psi_pert(w2)
    psi_tot  = psi_pert + psi_bg
    rhs_w, rhs_c, rhs_n2o = get_rhs_batched(w2, c2, n2o2, psi_tot, denit_ramp)
    w, tracer, tracer_n2o = apply_bcs(
        (1/3) * w          + (2/3) * (w2    + dt * rhs_w),
        (1/3) * tracer     + (2/3) * (c2    + dt * rhs_c),
        (1/3) * tracer_n2o + (2/3) * (n2o2  + dt * rhs_n2o))

    # ── Velocity field (reuse pre-allocated tensors) ──────────────────────
    u_full.zero_()
    v_full.zero_()
    u_full[:, 1:-1] = (psi_tot[:, 2:] - psi_tot[:, :-2]) * inv_2dy
    v_full[1:-1, :] = -(psi_tot[2:, :] - psi_tot[:-2, :]) * inv_2dx

    # ── FIX #1: sample BOTH velocities at the current position before
    #    updating either coordinate (was using updated px_t for v) ─────────
    u_interp = bilinear_interp_gpu(u_full, px_t, py_t)
    v_interp = bilinear_interp_gpu(v_full, px_t, py_t)
    px_t.add_(u_interp, alpha=dt)
    py_t.add_(v_interp, alpha=dt)

    mask  = px_t >= Lx
    n_out = int(mask.sum().item())
    if n_out:
        px_t[mask] = 0.0
        py_t[mask] = torch.empty(n_out, dtype=torch.float32,
                                 device=device).uniform_(0.1, Ly - 0.1)

    # ── Snapshots: pull to CPU only at save intervals ─────────────────────
    if n % snapshot_interval == 0:
        c_snapshots.append(tracer.cpu().numpy().astype(np.float32))
        n2o_snapshots.append(tracer_n2o.cpu().numpy().astype(np.float32))
        w_snapshots.append(w.cpu().numpy().astype(np.float32))
        u_snapshots.append(u_full.cpu().numpy().astype(np.float32))
        v_snapshots.append(v_full.cpu().numpy().astype(np.float32))
        p_snapshots.append(np.vstack([px_t.cpu().numpy(), py_t.cpu().numpy()]).astype(np.float32))
        snapshot_times.append(current_time)

        # Crucial for M2 stability
        torch.mps.empty_cache()

total_elapsed = _time.perf_counter() - _loop_start
print(f"Simulation complete in {total_elapsed:.1f}s. Generating animations...")

# ── 2D Animation ──────────────────────────────────────────────────────────────
plt.rcParams['animation.embed_limit'] = 250
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)

skip = 8
Q    = None

if PLOT_MODE == "o2":
    c_plot = ax.imshow(c_snapshots[0].T, origin='lower',
                       extent=[0, Lx, 0, Ly], cmap='rainbow', vmin=0, vmax=1.0)
    plt.colorbar(c_plot, label='O2 Concentration')
elif PLOT_MODE == "vorticity":
    vmax = np.max(np.abs(w_snapshots[-1])) * 0.5
    if vmax == 0: vmax = 0.1
    c_plot = ax.imshow(w_snapshots[0].T, origin='lower',
                       extent=[0, Lx, 0, Ly], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(c_plot, label='Vorticity (1/s)')
elif PLOT_MODE == "velocity":
    speed  = np.sqrt(u_snapshots[0]**2 + v_snapshots[0]**2)
    c_plot = ax.imshow(speed.T, origin='lower',
                       extent=[0, Lx, 0, Ly], cmap='jet', vmin=0, vmax=U_bg * 1.5)
    plt.colorbar(c_plot, label='Speed (mm/s)')
    Q = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  u_snapshots[0][::skip, ::skip], v_snapshots[0][::skip, ::skip],
                  color='white', scale=U_bg * 60, alpha=0.8)

p_plot, = ax.plot(p_snapshots[0][0], p_snapshots[0][1], 'ko', markersize=2, alpha=0.5)

circle = Circle((cx, cy), radius, color='white', fill=False, linewidth=2)
ax.add_patch(circle)
ax.axhline(cy, color='black', linestyle='--', alpha=0.6)
ax.axvline(cx, color='black', linestyle='--', alpha=0.6)
ax.plot(cx - (radius / 2.0), cy, 'go', markersize=6, markeredgecolor='white', label='Left')
ax.plot(cx,                   cy, 'ro', markersize=6, markeredgecolor='white', label='Center')
ax.plot(cx + (radius / 2.0), cy, 'bo', markersize=6, markeredgecolor='white', label='Right')
title = ax.set_title("Time: 0.00")
ax.legend(loc='upper right')
plt.close()

def update(frame_idx):
    if PLOT_MODE == "o2":
        c_plot.set_data(c_snapshots[frame_idx].T)
    elif PLOT_MODE == "vorticity":
        c_plot.set_data(w_snapshots[frame_idx].T)
    elif PLOT_MODE == "velocity":
        speed = np.sqrt(u_snapshots[frame_idx]**2 + v_snapshots[frame_idx]**2)
        c_plot.set_data(speed.T)
        Q.set_UVC(u_snapshots[frame_idx][::skip, ::skip],
                  v_snapshots[frame_idx][::skip, ::skip])
    p_plot.set_data(p_snapshots[frame_idx][0], p_snapshots[frame_idx][1])
    title.set_text(f"Time: {snapshot_times[frame_idx]:.2f}")
    if PLOT_MODE == "velocity":
        return c_plot, Q, p_plot, title
    return c_plot, p_plot, title

anim = animation.FuncAnimation(fig, update, frames=len(c_snapshots),
                                interval=100, blit=False)
anim.save('main_sim.mp4', writer='ffmpeg', fps=50)
print("Saved main_sim.mp4")

# ── 1D Cross-Section Animation ────────────────────────────────────────────────
y_slice = cy;  idx_y = int(y_slice / dy)
x_slice = cx;  idx_x = int(x_slice / dx)

fig_1d, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(14, 4))
line_h,     = ax_h.plot(x, c_snapshots[0][:, idx_y],   'b-',  lw=2, label='O2')
line_h_n2o, = ax_h.plot(x, n2o_snapshots[0][:, idx_y], 'm--', lw=2, label='N2O')
ax_h.axvline(cx - radius, color='k', linestyle='--', alpha=0.5, label='Particle Bound')
ax_h.axvline(cx + radius, color='k', linestyle='--', alpha=0.5)
ax_h.set_title("Horizontal Slice"); ax_h.set_xlabel("X coordinate")
ax_h.set_ylabel("Concentration")
ax_h.set_ylim(-0.1, 1.1); ax_h.set_xlim(0, Lx)
ax_h.grid(True, linestyle='--', alpha=0.6); ax_h.legend(loc='upper right')

line_v,     = ax_v.plot(y, c_snapshots[0][idx_x, :],   'r-',  lw=2, label='O2')
line_v_n2o, = ax_v.plot(y, n2o_snapshots[0][idx_x, :], 'm--', lw=2, label='N2O')
ax_v.axvline(cy - radius, color='k', linestyle='--', alpha=0.5, label='Particle Bound')
ax_v.axvline(cy + radius, color='k', linestyle='--', alpha=0.5)
ax_v.set_title("Vertical Slice"); ax_v.set_xlabel("Y coordinate")
ax_v.set_ylabel("Concentration")
ax_v.set_ylim(-0.1, 1.1); ax_v.set_xlim(0, Ly)
ax_v.grid(True, linestyle='--', alpha=0.6); ax_v.legend(loc='upper right')
plt.tight_layout(); plt.close()

def update_1d(frame_idx):
    line_h.set_ydata(c_snapshots[frame_idx][:, idx_y])
    line_h_n2o.set_ydata(n2o_snapshots[frame_idx][:, idx_y])
    line_v.set_ydata(c_snapshots[frame_idx][idx_x, :])
    line_v_n2o.set_ydata(n2o_snapshots[frame_idx][idx_x, :])
    fig_1d.suptitle(f"Time: {snapshot_times[frame_idx]:.2f}", fontsize=14)
    return line_h, line_h_n2o, line_v, line_v_n2o

anim_1d = animation.FuncAnimation(fig_1d, update_1d, frames=len(c_snapshots),
                                   interval=100, blit=False)
# anim_1d.save('1D_Cross_Sections.mp4', writer='ffmpeg', fps=50)
print("Saved 1D_Cross_Sections.mp4")

# ── Time-Series Plot ──────────────────────────────────────────────────────────
x_left, x_center, x_right = cx - (radius / 2.0), cx, cx + (radius / 2.0)
idx_y        = int(cy / dy)
idx_x_left   = int(x_left   / dx)
idx_x_center = int(x_center / dx)
idx_x_right  = int(x_right  / dx)

o2_left    = [snap[idx_x_left,   idx_y] for snap in c_snapshots]
o2_center  = [snap[idx_x_center, idx_y] for snap in c_snapshots]
o2_right   = [snap[idx_x_right,  idx_y] for snap in c_snapshots]
n2o_left   = [snap[idx_x_left,   idx_y] for snap in n2o_snapshots]
n2o_center = [snap[idx_x_center, idx_y] for snap in n2o_snapshots]
n2o_right  = [snap[idx_x_right,  idx_y] for snap in n2o_snapshots]

fig_time, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
fig_time.text(0.04, 0.5, 'O2 Concentration',  va='center', rotation='vertical',
              fontsize=12, fontweight='bold', color='blue')
fig_time.text(0.96, 0.5, 'N2O Concentration', va='center', rotation='vertical',
              fontsize=12, fontweight='bold', color='magenta')

for ax, o2_dat, n2o_dat, loc_title, color in zip(
        [ax1, ax2, ax3],
        [o2_left, o2_center, o2_right],
        [n2o_left, n2o_center, n2o_right],
        ["Left", "Center", "Right"],
        ['green', 'red', 'blue']):
    ax.plot(snapshot_times, o2_dat, '-', color=color, lw=2.5, label='O2 (Solid)')
    ax.set_title(f"{loc_title} Point (at x={x_left:.3f})", color=color,
                 fontweight='bold', fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylim((0.0, 1.1))
    ax.set_xlim((0, Total_Time))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax_n2o = ax.twinx()
    ax_n2o.plot(snapshot_times, n2o_dat, '--', color=color, lw=2.5, label='N2O (Dashed)')
    ax_n2o.set_ylim((0.0, 25.0))
    if ax == ax3:
        lines, labels   = ax.get_legend_handles_labels()
        lines2, labels2 = ax_n2o.get_legend_handles_labels()
        ax_n2o.legend(lines + lines2, labels + labels2, loc='center right', fontsize=10)

plt.suptitle("Coupled O2 and N2O Dynamics inside the Sinking Particle",
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0.05, 0, 0.95, 0.95])
# fig_time.savefig('O2_N2O_Final_TimeSeries.png', dpi=300, bbox_inches='tight')
print("Saved O2_N2O_Final_TimeSeries.png")