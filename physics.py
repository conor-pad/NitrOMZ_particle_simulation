# physics.py
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

# Import our new modular components
from bcs import inflow
from biopar import BioPar
from sms import nit_sms_omz

def setup_physics(cfg):
    """Initializes the grid, calculates dt, and pre-allocates all GPU tensors."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple Silicon GPU (MPS) detected! Running on Metal.")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS not found, falling back to CPU.")

    x = np.linspace(0, cfg.Lx, cfg.Nx)
    y = np.linspace(0, cfg.Ly, cfg.Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    psi_bg_np = cfg.U_bg * Y

    # Circular Particle (Drag Mask)
    drag_mask_np = np.zeros_like(X)
    particle_mask_np = np.zeros_like(X)
    particle_idx = (X - cfg.cx)**2 + (Y - cfg.cy)**2 <= cfg.radius**2
    drag_mask_np[particle_idx] = 500
    particle_mask_np[particle_idx] = 1.0

    max_alpha = np.max(drag_mask_np)
    # 1. Advective limit (How fast the fluid moves)
    dt_adv  = cfg.target_CFL * min(cfg.dx, cfg.dy) / cfg.U_bg
    
    # 2. Drag limit (How fast the particle stops the fluid)
    dt_drag = 0.5 / max_alpha if max_alpha > 0 else float('inf')
    
    # 3. Scalar diffusion limit (How fast the chemicals spread)
    dt_diff = 0.25 * min(cfg.dx, cfg.dy)**2 / cfg.K
    
    # 4. Momentum diffusion limit (How fast the fluid shears/sticks)
    dt_visc = 0.25 * min(cfg.dx, cfg.dy)**2 / cfg.nu

    # Take the absolute smallest required time step!
    dt, min_name = min((dt_adv, "dt_adv"), (dt_drag, "dt_drag"), (dt_diff, "dt_diff"), (dt_visc, "dt_visc"))

    print(f"Time step: {dt:.6f} (Source: {min_name})")
    # store so it can be put onto the plot
    cfg.dt = dt          
    cfg.drag_max = 1500

    

    drag_mask_np = gaussian_filter(drag_mask_np, sigma=1.0)
    da_dx_np, da_dy_np = np.gradient(drag_mask_np, cfg.dx, cfg.dy)

    # Dictionary to hold all dynamic tensor states
    state = {}
    state['dt'] = dt
    
    # Instantiate Biological Parameters
    state['bgc'] = BioPar()
    
    # Tracer names for easy looping
    # tracer_names = ['o2', 'no3', 'doc', 'po4', 'n2o', 'nh4', 'no2', 'n2']
    tracer_names = ['o2', 'no3', 'doc', 'po4', 'n2o', 'n2o_ammox', 'n2o_denit', 'nh4', 'no2', 'n2']
    
    state['tracer_names'] = tracer_names
    
    # Push Static Arrays to GPU
    state['psi_bg']         = torch.tensor(psi_bg_np, dtype=torch.float32, device=device)
    state['drag_mask']      = torch.tensor(drag_mask_np, dtype=torch.float32, device=device)
    state['da_dx']          = torch.tensor(da_dx_np, dtype=torch.float32, device=device)
    state['da_dy']          = torch.tensor(da_dy_np, dtype=torch.float32, device=device)
    state['particle_mask']  = torch.tensor(particle_mask_np, dtype=torch.float32, device=device)

    # Allocate Vorticity
    state['w']          = torch.zeros((cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)
    state['_rhs_buf_w'] = torch.zeros((cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)

    # ── Allocate Tracers ───────────────────────────────────────────────────────
    state['tracers'] = {}
    state['_rhs_tracers'] = {}
    
    for name in tracer_names:
        init_val = getattr(inflow, name)
        
        if name == 'doc':
            # Ambient water has 0 DOC
            starting_doc = getattr(cfg, 'doc_initial_core', 30.0)
            doc_initial = np.zeros((cfg.Nx, cfg.Ny), dtype=np.float32)
            doc_initial[particle_mask_np > 0] = starting_doc
            state['tracers'][name] = torch.tensor(doc_initial, device=device)
        else:
            # All other tracers (O2, NO3, etc.) fill the domain normally
            state['tracers'][name] = torch.full((cfg.Nx, cfg.Ny), init_val, dtype=torch.float32, device=device)
            
        state['_rhs_tracers'][name] = torch.zeros((cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)
    # ───────────────────────────────────────────────────────────────────────────

    # Pre-allocate full velocity tensors
    state['u_full'] = torch.zeros((cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)
    state['v_full'] = torch.zeros((cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)

    # Precompute reciprocal constants
    state['inv_4dxdy'] = 1.0 / (4.0 * cfg.dx * cfg.dy)
    state['inv_dx2']   = 1.0 / cfg.dx**2
    state['inv_dy2']   = 1.0 / cfg.dy**2
    state['inv_2dy']   = 1.0 / (2.0 * cfg.dy)
    state['inv_2dx']   = 1.0 / (2.0 * cfg.dx)

    # Precompute Poisson eigenvalues
    Ni, Nj = cfg.Nx - 2, cfg.Ny - 2
    state['Ni'], state['Nj'] = Ni, Nj
    
    ii = torch.arange(1, Ni + 1, dtype=torch.float32, device=device)
    jj = torch.arange(1, Nj + 1, dtype=torch.float32, device=device)
    lam_x = (2 / cfg.dx**2) * (torch.cos(np.pi * ii / (Ni + 1)) - 1)
    lam_y = (2 / cfg.dy**2) * (torch.cos(np.pi * jj / (Nj + 1)) - 1)
    state['Lambda'] = lam_x[:, None] + lam_y[None, :]

    # Pre-allocated DST-I buffers 
    state['_dst_buf_axis0'] = torch.zeros((2 * (Ni + 1), Nj), dtype=torch.float32, device=device)
    state['_dst_buf_axis1'] = torch.zeros((Ni, 2 * (Nj + 1)), dtype=torch.float32, device=device)

    return device, state


def dstn1(x_in, state):
    n0, n1 = x_in.shape
    buf0 = state['_dst_buf_axis0']
    buf1 = state['_dst_buf_axis1']

    buf0.zero_()
    buf0[1:n0 + 1, :] = x_in
    buf0[n0 + 2:,  :] = -torch.flip(x_in, dims=[0])

    y0  = torch.fft.fft(buf0, dim=0)
    mid = -y0[1:n0 + 1].imag / float(np.sqrt(2 * (n0 + 1)))

    buf1.zero_()
    buf1[:, 1:n1 + 1] = mid
    buf1[:, n1 + 2:]  = -torch.flip(mid, dims=[1])

    y1 = torch.fft.fft(buf1, dim=1)
    return -y1[:, 1:n1 + 1].imag / float(np.sqrt(2 * (n1 + 1)))


def get_psi_pert(w_field, state):
    rhs       = -w_field[1:-1, 1:-1]
    rhs_hat   = dstn1(rhs, state)
    psi_inner = dstn1(rhs_hat / state['Lambda'], state)
    psi       = torch.zeros_like(w_field)
    psi[1:-1, 1:-1] = psi_inner
    return psi


def get_rhs_batched(w_f, tracers_dict, streamfunction, state, cfg):
    """Batched RHS for vorticity and all 8 tracers simultaneously."""
    p = streamfunction

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

    tracer_names = state['tracer_names']

    # Stack 9 total fields (Index 0 = Vorticity, Index 1-8 = Tracers)
    f_c_list  = [w_f[1:-1, 1:-1]] + [tracers_dict[name][1:-1, 1:-1] for name in tracer_names]
    f_e_list  = [w_f[2:,   1:-1]] + [tracers_dict[name][2:,   1:-1] for name in tracer_names]
    f_w_list  = [w_f[:-2,  1:-1]] + [tracers_dict[name][:-2,  1:-1] for name in tracer_names]
    f_n_list  = [w_f[1:-1, 2:  ]] + [tracers_dict[name][1:-1, 2:  ] for name in tracer_names]
    f_s_list  = [w_f[1:-1, :-2 ]] + [tracers_dict[name][1:-1, :-2 ] for name in tracer_names]
    f_ne_list = [w_f[2:,   2:  ]] + [tracers_dict[name][2:,   2:  ] for name in tracer_names]
    f_sw_list = [w_f[:-2,  :-2 ]] + [tracers_dict[name][:-2,  :-2 ] for name in tracer_names]
    f_se_list = [w_f[2:,   :-2 ]] + [tracers_dict[name][2:,   :-2 ] for name in tracer_names]
    f_nw_list = [w_f[:-2,  2:  ]] + [tracers_dict[name][:-2,  2:  ] for name in tracer_names]

    f_c  = torch.stack(f_c_list)
    f_e  = torch.stack(f_e_list)
    f_w  = torch.stack(f_w_list)
    f_n  = torch.stack(f_n_list)
    f_s  = torch.stack(f_s_list)
    f_ne = torch.stack(f_ne_list)
    f_sw = torch.stack(f_sw_list)
    f_se = torch.stack(f_se_list)
    f_nw = torch.stack(f_nw_list)

    # ── Arakawa Jacobian & Laplacian ──
    J_std   = (dp_ew * (f_n - f_s) - dp_ns * (f_e - f_w)) * state['inv_4dxdy']
    J_hat   = (p_e  * (f_ne - f_se) - p_w  * (f_nw - f_sw)
             - p_n  * (f_ne - f_nw) + p_s  * (f_se - f_sw)) * state['inv_4dxdy']
    J_tilde = (f_n  * dp_ne_nw - f_s  * dp_se_sw
             - f_e  * dp_ne_se + f_w  * dp_nw_sw) * state['inv_4dxdy']
    
    J_avg = (J_std + J_hat + J_tilde) / 3.0 
    lap   = (f_e + f_w - 2.0 * f_c) * state['inv_dx2'] + (f_n + f_s - 2.0 * f_c) * state['inv_dy2']

    # ── Masks ──
    dm  = state['drag_mask'][1:-1, 1:-1]
    dax = state['da_dx'][1:-1, 1:-1]
    day = state['da_dy'][1:-1, 1:-1]

    # ── Vorticity RHS ──
    u_c   = dp_ns * state['inv_2dy']
    v_c   = -dp_ew * state['inv_2dx']
    drag  = dm * f_c[0] - (day * u_c - dax * v_c)
    
    rhs_w = J_avg[0] + cfg.nu * lap[0] - drag
    state['_rhs_buf_w'][1:-1, 1:-1] = rhs_w

    # ── Biogeochemistry SMS Terms ──
    interior_tracers = {name: tracers_dict[name][1:-1, 1:-1] for name in tracer_names}
    
    # Calculate biological source/sink rates (ddt)
    ddt, _ = nit_sms_omz(interior_tracers, state['bgc'])

    # ── Tracers RHS ──
    for i, name in enumerate(tracer_names):
        
        # Base physics: Advection + Diffusion + Microbial Consumption (for ALL tracers)
        rhs_t = J_avg[i+1] + cfg.K * lap[i+1] + ddt[name]
        
        # Add the POC-to-DOC flux!
        if name == 'doc':
            flux_rate = getattr(cfg, 'doc_flux_rate', 0.0) 
            
            # Multiply by the mask so the flux ONLY happens inside the white circle
            rhs_t += flux_rate * state['particle_mask'][1:-1, 1:-1]
            
        # 3. Save to the GPU buffer
        state['_rhs_tracers'][name][1:-1, 1:-1] = rhs_t

    return state['_rhs_buf_w'], state['_rhs_tracers']


def bilinear_interp_gpu(field, px_t, py_t, cfg):
    ix = torch.clamp((px_t / cfg.dx).long(), 0, cfg.Nx - 2)
    iy = torch.clamp((py_t / cfg.dy).long(), 0, cfg.Ny - 2)
    fx = px_t / cfg.dx - ix.float()
    fy = py_t / cfg.dy - iy.float()
    return (field[ix,     iy    ] * (1 - fx) * (1 - fy)
          + field[ix + 1, iy    ] * fx        * (1 - fy)
          + field[ix,     iy + 1] * (1 - fx)  * fy
          + field[ix + 1, iy + 1] * fx         * fy)

# Apply Torch compile if available
if hasattr(torch, 'compile'):
    try:
        get_rhs_batched = torch.compile(get_rhs_batched) 
        print("✅ torch.compile enabled for RHS physics (inductor backend)")
    except Exception as e:
        print(f"⚠️  torch.compile skipped: {e}")

        