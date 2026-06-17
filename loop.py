# loop.py
import torch
import numpy as np
import time as _time
from tqdm import tqdm

from bcs import apply_bcs, enforce_symmetry
from physics import get_psi_pert, get_rhs_batched, apply_implicit_visc
from sms import nit_sms_omz


def run_simulation(state, cfg, device):
    dt         = state['dt']
    w          = state['w']
    tracers    = state['tracers']
    psi_bg     = state['psi_bg']
    u_full     = state['u_full']
    v_full     = state['v_full']
    tracer_names = state['tracer_names']
    bgc        = state['bgc']

    impl_drag  = {'s1': state['impl_drag_s1'], 's2': state['impl_drag_s2'], 's3': state['impl_drag_s3']}
    helm_denom = {'s1': state['helm_denom_s1'], 's2': state['helm_denom_s2'], 's3': state['helm_denom_s3']}

    # ── Snapshot lists (used by main.py for animation; skipped in suite mode) ──
    w_snapshots, c_snapshots                   = [], []
    n2o_snapshots, no3_snapshots               = [], []
    no2_snapshots, n2_snapshots                = [], []
    nh4_snapshots, doc_snapshots               = [], []
    u_snapshots, v_snapshots                   = [], []
    n2o_ammox_snapshots, n2o_denit_snapshots   = [], []
    snapshot_times                             = []

    n_steps           = int(cfg.Total_Time / dt)
    snapshot_interval = max(1, int(getattr(cfg, 'snapshot_time', 0.03) / dt))

    # ── dV per cell (m³), shape [bs, 1, 1] so it broadcasts over the grid ──
    # dx/dy are stored as [bs,1,1] numpy arrays in cfg
    dx_np = np.atleast_1d(cfg.dx).reshape(-1, 1, 1)   # [bs,1,1]
    dy_np = np.atleast_1d(cfg.dy).reshape(-1, 1, 1)
    dV_t  = torch.tensor(dx_np * dy_np * 1.0 * 1e-9,   # mm² → m²; ×1mm depth → m³
                         dtype=torch.float32, device=device)   # [bs,1,1]

    bs = w.shape[0]

    # ── On-the-fly Riemann accumulators — shape [bs] each ─────────────────────
    # All units: mmol (sum over space × dV × dt, then divided later for yield)
    acc = {key: torch.zeros(bs, device=device) for key in [
        'n2o', 'n2',
        'n2o_internal', 'n2o_plume',
        'n2o_ammox_internal', 'n2o_denit_internal',
        'n2o_ammox_plume',    'n2o_denit_plume',
        'c_total',
        'oxic_core',   'den1_core',   'den2_core',   'den3_core',
        'oxic_plume',  'den1_plume',  'den2_plume',  'den3_plume',
        'o2_flux_in',  'no3_flux_in', 'doc_flux_out',
        'anoxic_core_frac_sum',   # divided by n_steps at the end
    ]}

    particle_mask = state['particle_mask']   # [bs, Nx, Ny]
    plume_mask    = 1.0 - particle_mask

    oxic_threshold    = 1.0    # mmol/m³ — same as run_suite used before
    acc_steps         = 0      # counts how many steps contributed to anoxic avg

    print(f"Total steps: {n_steps}  |  Snapshot every {snapshot_interval} steps")
    is_suite = getattr(cfg, 'is_suite', False)
    if is_suite:
        print("Suite mode: accumulating integrals on-the-fly, saving only terminal snapshot.")
    print("Starting SSP-RK3 Loop...")
    _loop_start = _time.perf_counter()

    # ── Main Loop ─────────────────────────────────────────────────────────────
    for n in tqdm(range(n_steps),
                  desc="Simulating..",
                  ascii="⡀⡄⡆⡇▞▚░▒▓",
                  unit="steps",
                  bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):

        current_time = n * dt

        # ── 1. Transient DOC Flux from exponential POC decay ──────────────────
        current_poc  = state['poc_initial'] * torch.exp(-state['k_hyd'] * current_time)
        doc_flux_t   = state['k_hyd'] * current_poc

        # ── SSP-RK3 Stage 1 ───────────────────────────────────────────────────
        psi_pert = get_psi_pert(w, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w, tracers, psi_tot, state, cfg, doc_flux_t)

        w1_temp = (w + dt * rhs_w) * impl_drag['s1']
        w1_temp = apply_implicit_visc(w1_temp, helm_denom['s1'], state)
        t1_temp = {name: tracers[name] + dt * rhs_tracers[name] for name in tracer_names}
        w1, t1  = apply_bcs(w1_temp, t1_temp)

        # ── SSP-RK3 Stage 2 ───────────────────────────────────────────────────
        psi_pert = get_psi_pert(w1, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w1, t1, psi_tot, state, cfg, doc_flux_t)

        w2_temp = (0.75 * w + 0.25 * (w1 + dt * rhs_w)) * impl_drag['s2']
        w2_temp = apply_implicit_visc(w2_temp, helm_denom['s2'], state)
        t2_temp = {name: 0.75 * tracers[name] + 0.25 * (t1[name] + dt * rhs_tracers[name]) for name in tracer_names}
        w2, t2  = apply_bcs(w2_temp, t2_temp)

        # ── SSP-RK3 Stage 3 ───────────────────────────────────────────────────
        psi_pert = get_psi_pert(w2, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w2, t2, psi_tot, state, cfg, doc_flux_t)

        w_temp = ((1.0/3.0) * w + (2.0/3.0) * (w2 + dt * rhs_w)) * impl_drag['s3']
        w_temp = apply_implicit_visc(w_temp, helm_denom['s3'], state)
        t_temp = {name: (1.0/3.0) * tracers[name] + (2.0/3.0) * (t2[name] + dt * rhs_tracers[name]) for name in tracer_names}
        w, tracers = apply_bcs(w_temp, t_temp)

        if getattr(cfg, 'use_symmetry', True):
            w, tracers = enforce_symmetry(w, tracers, tracer_names)

        # ── Velocity field (needed for snapshots; cheap) ───────────────────────
        u_full.zero_()
        v_full.zero_()
        u_full[..., 1:-1] = (psi_tot[..., 2:] - psi_tot[..., :-2]) * state['inv_2dy']
        v_full[..., 1:-1, :] = -(psi_tot[..., 2:, :] - psi_tot[..., :-2, :]) * state['inv_2dx']

        # ══════════════════════════════════════════════════════════════════════
        # ON-THE-FLY RIEMANN ACCUMULATION
        # We call nit_sms_omz on the *full-domain* tracers (not just interior)
        # so that spatial sums over particle_mask / plume_mask are consistent.
        # The SMS function internally clamps negatives, so this is safe.
        # ══════════════════════════════════════════════════════════════════════
        with torch.no_grad():
            ddt, diags = nit_sms_omz(tracers, bgc)

            # integration_factor [bs,1,1]: converts (mmol/m³/s) → mmol per step
            IF = dV_t * dt   # [bs,1,1] — broadcasts over [bs,Nx,Ny]

            def spa(field, mask=None):
                """Spatial sum over domain (or masked sub-domain) → shape [bs]."""
                if mask is not None:
                    return (field * mask * IF).sum(dim=(-2, -1))
                return (field * IF).sum(dim=(-2, -1))

            acc['n2o']  += spa(ddt['n2o'])
            acc['n2']   += spa(ddt['n2'])

            # Carbon fluxes (all denominator quantities)
            c_rate = diags['RemOx_C'] + diags['RemDen1_C'] + diags['RemDen2_C'] + diags['RemDen3_C']
            acc['c_total'] += spa(c_rate)

            # Spatial split: core vs plume
            acc['n2o_internal']        += spa(ddt['n2o'],        particle_mask)
            acc['n2o_plume']           += spa(ddt['n2o'],        plume_mask)
            acc['n2o_ammox_internal']  += spa(ddt['n2o_ammox'],  particle_mask)
            acc['n2o_denit_internal']  += spa(ddt['n2o_denit'],  particle_mask)
            acc['n2o_ammox_plume']     += spa(ddt['n2o_ammox'],  plume_mask)
            acc['n2o_denit_plume']     += spa(ddt['n2o_denit'],  plume_mask)

            acc['oxic_core']   += spa(diags['RemOx_C'],   particle_mask)
            acc['den1_core']   += spa(diags['RemDen1_C'], particle_mask)
            acc['den2_core']   += spa(diags['RemDen2_C'], particle_mask)
            acc['den3_core']   += spa(diags['RemDen3_C'], particle_mask)

            acc['oxic_plume']  += spa(diags['RemOx_C'],   plume_mask)
            acc['den1_plume']  += spa(diags['RemDen1_C'], plume_mask)
            acc['den2_plume']  += spa(diags['RemDen2_C'], plume_mask)
            acc['den3_plume']  += spa(diags['RemDen3_C'], plume_mask)

            # O2 / NO3 flux in — only count consumption (negative ddt = consumption)
            o2_cons  = torch.where(ddt['o2']  < 0, ddt['o2'],  torch.zeros_like(ddt['o2']))
            no3_cons = torch.where(ddt['no3'] < 0, ddt['no3'], torch.zeros_like(ddt['no3']))
            acc['o2_flux_in']  += spa(torch.abs(o2_cons),  particle_mask)
            acc['no3_flux_in'] += spa(torch.abs(no3_cons), particle_mask)

            # DOC leakage: POC flux produced inside particle minus DOC consumed inside
            poc_flux_this_step = doc_flux_t   # [bs,1,1] — already broadcast-ready
            doc_produced_core  = (poc_flux_this_step * particle_mask * IF).sum(dim=(-2,-1))
            doc_consumed_core  = spa(torch.abs(ddt['doc']), particle_mask)
            acc['doc_flux_out'] += torch.clamp(doc_produced_core - doc_consumed_core, min=0.0)

            # Anoxic core fraction (time-averaged)
            anoxic_vox   = ((tracers['o2'] < oxic_threshold) * particle_mask).sum(dim=(-2,-1)).float()
            total_core   = particle_mask.sum(dim=(-2,-1))
            frac         = torch.where(total_core > 0, anoxic_vox / total_core, torch.zeros_like(anoxic_vox))
            acc['anoxic_core_frac_sum'] += frac
            acc_steps += 1

        if getattr(cfg, 'terminal_snapshot_only', False):
            take_snapshot = (n == n_steps - 1)
        else:
            take_snapshot = (n % snapshot_interval == 0)

        if take_snapshot:
            c_snapshots.append(tracers['o2'].cpu().numpy().astype(np.float32))
            n2o_snapshots.append(tracers['n2o'].cpu().numpy().astype(np.float32))
            n2o_ammox_snapshots.append(tracers['n2o_ammox'].cpu().numpy().astype(np.float32))
            n2o_denit_snapshots.append(tracers['n2o_denit'].cpu().numpy().astype(np.float32))
            no3_snapshots.append(tracers['no3'].cpu().numpy().astype(np.float32))
            no2_snapshots.append(tracers['no2'].cpu().numpy().astype(np.float32))
            n2_snapshots.append(tracers['n2'].cpu().numpy().astype(np.float32))
            nh4_snapshots.append(tracers['nh4'].cpu().numpy().astype(np.float32))
            doc_snapshots.append(tracers['doc'].cpu().numpy().astype(np.float32))
            w_snapshots.append(w.cpu().numpy().astype(np.float32))
            u_snapshots.append(u_full.cpu().numpy().astype(np.float32))
            v_snapshots.append(v_full.cpu().numpy().astype(np.float32))
            snapshot_times.append(current_time)

        # ── Periodic maintenance ───────────────────────────────────────────────
        if n % 1000 == 0:
            torch.mps.empty_cache()

            if not torch.isfinite(w).all().item():
                print(f"\n🚨 FATAL: NaN/Inf in vorticity at step {n} (t={current_time:.3f}s)")
                break

            tracer_crashed = False
            for name, tensor in tracers.items():
                if not torch.isfinite(tensor).all().item():
                    print(f"\n🚨 FATAL: NaN/Inf in tracer '{name}' at step {n} (t={current_time:.3f}s)")
                    tracer_crashed = True
                    break
            if tracer_crashed:
                break


    total_elapsed = _time.perf_counter() - _loop_start
    print(f"\nSimulation complete in {total_elapsed:.1f}s.")

    # ── Finalise accumulators → Python scalars, then package into 'integrals' ─
    # Normalise the anoxic fraction by actual number of steps taken
    if acc_steps > 0:
        acc['anoxic_core_frac_sum'] /= acc_steps

    # Convert every accumulator tensor to a [bs]-length list of Python floats
    integrals = {key: acc[key].cpu().tolist() for key in acc}

    if not is_suite:
        print("Generating animations...")

    return (
        c_snapshots, n2o_snapshots, no3_snapshots, no2_snapshots,
        n2_snapshots, doc_snapshots, nh4_snapshots,
        w_snapshots, u_snapshots, v_snapshots,
        snapshot_times,
        n2o_ammox_snapshots, n2o_denit_snapshots,
        integrals,   # ← NEW: dict of pre-integrated lifespan totals, shape [bs] each
    )