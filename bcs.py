# bcs.py
import torch
from dataclasses import dataclass

@dataclass
class InflowBCs:
    """
    Boundary and initial conditions for biological variables.
    Converted from hab_initialize_omz.m
    Values represent concentrations in mmol/m3
    """
    o2:  float = 6
    no3: float = 10.0
    doc: float = 0 # 17.2   # 10 * 1.72 (approx conversion to OM)

    # DOC: treat as it's STATIC inside the particle, not flowing in.

    po4: float = 0.0
    n2o: float = 0.0
    nh4: float = 0.0
    no2: float = 0.0
    n2:  float = 0.0

    poc: float = 0.0               # Ambient water has no solid POC
    poc_core: float = 1000000.0    # Solid core density (mmol C / m^3). Used to calculate 2D Area.

# Instantiate the inflow values so other files can import and use them
inflow = InflowBCs()

def apply_bcs(w_f, tracers_dict):
    """
    Applies boundary conditions to the vorticity field and all tracers.
    
    w_f: 2D PyTorch tensor for vorticity
    tracers_dict: Dictionary of 2D PyTorch tensors {'o2': tensor, 'no3': tensor, ...}
    """
    # ── Vorticity Boundary Conditions ──
    w_f[:, 0]  = 0.0
    w_f[:, -1] = 0.0
    w_f[0, :]  = 0.0
    w_f[-1, :] = w_f[-2, :]
    
    # ── Tracer Boundary Conditions ──
    for name, tensor in tracers_dict.items():
        # Grab the specific inflow value for this tracer from the dataclass
        inflow_value = getattr(inflow, name)
        
        # Left boundary: Constant inflow
        tensor[0, :] = inflow_value
        
        # Right boundary: Zero gradient (outflow)
        tensor[-1, :] = tensor[-2, :]
        
        # Top and Bottom boundaries: Free slip / zero gradient
        tensor[:, 0]  = tensor[:, 1]
        tensor[:, -1] = tensor[:, -2]
        
    return w_f, tracers_dict

