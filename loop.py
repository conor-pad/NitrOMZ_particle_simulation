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

    is_suite = getattr(cfg, 'is_suite', False)

    # ── Snapshot lists ────────────────────────────────────────────────────────
    w_snapshots, c_snapshots                   = [], []
    n2o_snapshots, no3_snapshots               = [], []
    no2_snapshots, n2_snapshots                = [], []
    nh4_snapshots, doc_snapshots               = [], []
    u_snapshots, v_snapshots                   = [], []
    n2o_ammox_snapshots, n2o_denit_snapshots   = [], []
    snapshot_times                             = []

    n_steps           = int(cfg.Total_Time / dt)
    snapshot_interval = max(1, int(getattr(cfg, 'snapshot_time', 0.03) / dt))

    # ── dV per cell [bs,1,1] mm² → m² × 1mm depth → m³ ──────────────────────
    dx_np = np.atleast_1d(cfg.dx).reshape(-1, 1, 1)
    dy_np = np.atleast_1d(cfg.dy).reshape(-1, 1, 1)
    dV_t  = torch.tensor(dx_np * dy_np * 1.0 * 1e-9,
                         dtype=torch.float32, device=device)   # [bs,1,1]

    bs = w.shape[0]

    # ── On-the-fly Riemann accumulators ──────────────────────────────────────
    acc = {key: torch.zeros(bs, device=device) for key in [
        'n2o', 'n2',
        'n2o_internal', 'n2o_plume',
        'n2o_ammox_internal', 'n2o_denit_internal',
        'n2o_ammox_plume',    'n2o_denit_plume',
        'c_total',
        'oxic_core',   'den1_core',   'den2_core',   'den3_core',
        'oxic_plume',  'den1_plume',  'den2_plume',  'den3_plume',
        'o2_flux_in',  'no3_flux_in', 'doc_flux_out',
        'anoxic_core_frac_sum',
    ]}

    particle_mask = state['particle_mask']   # [bs, Nx, Ny]
    plume_mask    = 1.0 - particle_mask
    oxic_threshold = 1.0
    acc_steps      = 0

    # ── Suite-mode extrapolation state ───────────────────────────────────────
    equilibration_time = 20.0          # physical seconds before we start checking

    # Check every ~30 physical seconds regardless of dt.
    # With a 5-sample window this means the first extrapolation attempt fires
    # ~150s after equilibration — well within a 600s run at any grid resolution.
    extrap_check_interval = max(1, int(30.0 / dt))
    
    if is_suite:
        is_anoxic_b       = torch.zeros(bs, dtype=torch.bool, device=device)
        anoxia_onset_b    = torch.zeros(bs, dtype=torch.float32, device=device)
        is_extrapolated_b = torch.zeros(bs, dtype=torch.bool, device=device)
        final_durations_b = torch.zeros(bs, dtype=torch.float32, device=device)
        final_rates_b     = torch.zeros(bs, dtype=torch.float32, device=device)
        fuel_history_b    = [] 

    # Pre-compute particle cell count for resp-rate normalisation
    n_core_cells = particle_mask.sum(dim=(-2, -1))  # [bs]

    print(f"Total steps: {n_steps}  |  Snapshot every {snapshot_interval} steps")
    if is_suite:
        print("Suite mode: vectorized extrapolation early-exit enabled.")
    print("Starting SSP-RK3 Loop...")
    _loop_start = _time.perf_counter()

    for n in tqdm(range(n_steps),
                  desc="Simulating..",
                  ascii="⡀⡄⡆⡇▞▚░▒▓",
                  unit="steps",
                  bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):

        current_time = n * dt

        # ── 1. POC→DOC hydrolysis flux ────────────────────────────────────────
        current_poc = state['poc_initial'] * torch.exp(-state['k_hyd'] * current_time)
        doc_flux_t  = state['k_hyd'] * current_poc   # [bs,1,1]  mmol/m³/s

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

        # ── Velocity field ─────────────────────────────────────────────────────
        u_full.zero_()
        v_full.zero_()
        u_full[..., 1:-1]    = (psi_tot[..., 2:]    - psi_tot[..., :-2])    * state['inv_2dy']
        v_full[..., 1:-1, :] = -(psi_tot[..., 2:, :] - psi_tot[..., :-2, :]) * state['inv_2dx']

        # ══════════════════════════════════════════════════════════════════════
        # ON-THE-FLY RIEMANN ACCUMULATION
        # ══════════════════════════════════════════════════════════════════════
        with torch.no_grad():
            ddt, diags = nit_sms_omz(tracers, bgc)
            IF = dV_t * dt   # [bs,1,1]

            def spa(field, mask=None):
                if mask is not None:
                    return (field * mask * IF).sum(dim=(-2, -1))
                return (field * IF).sum(dim=(-2, -1))

            acc['n2o']  += spa(ddt['n2o'])
            acc['n2']   += spa(ddt['n2'])

            c_rate = diags['RemOx_C'] + diags['RemDen1_C'] + diags['RemDen2_C'] + diags['RemDen3_C']
            acc['c_total'] += spa(c_rate)

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

            o2_cons  = torch.where(ddt['o2']  < 0, ddt['o2'],  torch.zeros_like(ddt['o2']))
            no3_cons = torch.where(ddt['no3'] < 0, ddt['no3'], torch.zeros_like(ddt['no3']))
            acc['o2_flux_in']  += spa(torch.abs(o2_cons),  particle_mask)
            acc['no3_flux_in'] += spa(torch.abs(no3_cons), particle_mask)

            poc_flux_this_step = doc_flux_t
            doc_produced_core  = (poc_flux_this_step * particle_mask * IF).sum(dim=(-2,-1))
            doc_consumed_core  = spa(torch.abs(ddt['doc']), particle_mask)
            acc['doc_flux_out'] += torch.clamp(doc_produced_core - doc_consumed_core, min=0.0)

            anoxic_vox = ((tracers['o2'] < oxic_threshold) * particle_mask).sum(dim=(-2,-1)).float()
            total_core = particle_mask.sum(dim=(-2,-1))
            frac       = torch.where(total_core > 0,
                                     anoxic_vox / total_core,
                                     torch.zeros_like(anoxic_vox))
            acc['anoxic_core_frac_sum'] += frac
            acc_steps += 1

        # ══════════════════════════════════════════════════════════════════════
        # SUITE MODE: VECTORIZED STEADY-STATE EXTRAPOLATION
        # ══════════════════════════════════════════════════════════════════════
        if is_suite and current_time > equilibration_time and n % extrap_check_interval == 0:
            with torch.no_grad():
                core_o2 = (tracers['o2'] * particle_mask)
                core_o2_min = (core_o2 + (1 - particle_mask) * 1e6).min(dim=-1).values.min(dim=-1).values  # [bs]

                # ── One-time chemistry diagnostic ──────────────────────────────
                if not getattr(run_simulation, '_diag_printed', False):
                    run_simulation._diag_printed = True
                    core_doc_max = (tracers['doc'] * particle_mask).max().item()
                    print(f"\n  [Chem diag @ t={current_time:.1f}s]"
                          f"  core_o2_min={core_o2_min.cpu().numpy()}"
                          f"  max_core_DOC={core_doc_max:.2f} mmol/m³"
                          f"  extrap_interval={extrap_check_interval} steps ({30.0:.0f}s)")

                # Independent anoxia tracking
                currently_anoxic = (core_o2_min < oxic_threshold)
                newly_anoxic = currently_anoxic & ~is_anoxic_b
                if newly_anoxic.any():
                    anoxia_onset_b[newly_anoxic] = float(current_time)
                    is_anoxic_b = is_anoxic_b | newly_anoxic

                # Exact original fuel math (no unit division)
                k_hyd_val = state['k_hyd']   
                poc_now   = state['poc_initial'] * torch.exp(-k_hyd_val * current_time) 
                poc_remaining_density = poc_now 
                
                core_cells_t = particle_mask  
                poc_remaining = (poc_remaining_density * core_cells_t * dV_t).sum(dim=(-2,-1))  
                doc_in_core = (tracers['doc'] * core_cells_t * dV_t).sum(dim=(-2,-1))   
                total_fuel  = poc_remaining + doc_in_core   # [bs] tensor

            fuel_history_b.append((current_time, total_fuel))

            if len(fuel_history_b) > 5:
                fuel_history_b.pop(0)

            # Independent Extrapolation Logic
            if len(fuel_history_b) == 5:
                t0, f0 = fuel_history_b[0]
                t1_val, f1 = fuel_history_b[-1]
                dt_window = t1_val - t0

                burn_rates = (f0 - f1) / dt_window

                # Condition: Anoxic AND Not Extrapolated AND Positive Burn
                ready = is_anoxic_b & ~is_extrapolated_b & (burn_rates > 0.0)

                if ready.any():
                    ready_idx = torch.where(ready)[0]
                    for idx in ready_idx:
                        rem_fuel = f1[idx]
                        b_rate   = burn_rates[idx]

                        time_to_empty = rem_fuel / b_rate
                        total_anoxia_sec = (current_time - anoxia_onset_b[idx]) + time_to_empty
                        
                        final_durations_b[idx] = total_anoxia_sec / 3600.0

                        # Resp rate calculation matched to exact original math
                        core_vol_m3 = n_core_cells[idx].item() * float(dx_np[idx if dx_np.shape[0]>1 else 0, 0, 0]) \
                                      * float(dy_np[idx if dy_np.shape[0]>1 else 0, 0, 0]) * 1e-9
                        rate_mmol_m3_s = b_rate.item() / core_vol_m3
                        rate_nmol_mm3_hr = rate_mmol_m3_s * 3600.0 / 1000.0
                        final_rates_b[idx] = rate_nmol_mm3_hr

                    is_extrapolated_b[ready] = True
                    print(f"\n  ↳ Extrapolated {len(ready_idx)} particle(s) at t={current_time:.1f}s")

            # Exit strictly when ALL members have successfully extrapolated
            if is_extrapolated_b.all():
                print(f"\n  ↳ All batch members completed at t={current_time:.1f}s")
                break

        # ── Snapshots ─────────────────────────────────────────────────────────
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

    # ── Suite mode: fallback for incomplete ───────────────────────────────────
    if is_suite:
        # Stop reporting 0.0 for stranded particles. Report actual physical time.
        stranded = is_anoxic_b & ~is_extrapolated_b
        if stranded.any():
            stranded_idx = torch.where(stranded)[0]
            for idx in stranded_idx:
                phys_sec = current_time - anoxia_onset_b[idx]
                final_durations_b[idx] = phys_sec / 3600.0
                
                # If they had a valid burn rate window, assign it, else 0.0
                if len(fuel_history_b) == 5:
                    t0, f0 = fuel_history_b[0]
                    t1_val, f1 = fuel_history_b[-1]
                    b_rate = (f0[idx] - f1[idx]) / (t1_val - t0)
                    core_vol_m3 = n_core_cells[idx].item() * float(dx_np[idx if dx_np.shape[0]>1 else 0, 0, 0]) \
                                  * float(dy_np[idx if dy_np.shape[0]>1 else 0, 0, 0]) * 1e-9
                    rate_mmol_m3_s = b_rate.item() / core_vol_m3
                    final_rates_b[idx] = rate_mmol_m3_s * 3600.0 / 1000.0
                else:
                    final_rates_b[idx] = 0.0

        return final_durations_b.cpu().numpy().tolist(), final_rates_b.cpu().numpy().tolist()

    # ── Normal run: finalise and return snapshots ─────────────────────────────
    if acc_steps > 0:
        acc['anoxic_core_frac_sum'] /= acc_steps
    integrals = {key: acc[key].cpu().tolist() for key in acc}

    print("Generating animations...")
    return (
        c_snapshots, n2o_snapshots, no3_snapshots, no2_snapshots,
        n2_snapshots, doc_snapshots, nh4_snapshots,
        w_snapshots, u_snapshots, v_snapshots,
        snapshot_times,
        n2o_ammox_snapshots, n2o_denit_snapshots,
        integrals,
    )