# config.py
# ── Particle Parameters ───────────────────────────────────────────────────────
radius = 2.0

# ── Domain (Scaled by radius) ─────────────────────────────────────────────────
Lx = 20.0 * radius  
Ly = 10 * radius  
Nx, Ny = int(331), int(287)
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
Total_Time = 20
cx = 5.0 * radius   
cy = Ly / 2.0       

# ── DOC ───────────────────────────────────────────────────────────────────────
doc_flux_rate = 1.0       # Hydrolysis rate: solid POC to dissolved DOC (mmol/m^3/s)
doc_initial_core = 0.0    # Initial DOC concentration inside the particle

# ── Target Dimensionless Numbers ──────────────────────────────────────────────
Re_target = 1.0   
Sc_target = 660.0   
Pe_calc = Re_target * Sc_target  

# ── Derived Physics Parameters ────────────────────────────────────────────────
nu = 9.04  # Physically realistic kinematic viscosity for seawater (mm^2/s)

# Calculate required background velocity to hit the target Reynolds number
# Formula: Re = (U * radius) / nu  --->  U = (Re * nu) / radius
U_bg = (Re_target * nu) / radius

# K = nu / Sc_target
K_ideal = nu / Sc_target

# 2. Calculate the absolute minimum diffusivity required to stop the grid from exploding
# Formula derived from forcing Pe_grid = (U_bg * dx) / K <= 2.0
K_stable = (U_bg * dx) / 2.0  

# 3. Use whichever is larger!
K = max(K_ideal, K_stable)


Sh = 1 + 0.619 * Re_target ** 0.412 * Sc_target**(1/3)

print(f"\n── Simulation Physics ──")
print(f"Targets  | Re: {Re_target}  |  Sc: {Sc_target}  |  Pe: {Pe_calc}")
print(f"Derived  | U_bg: {U_bg:.3f} mm/s | nu: {nu:.2f} |  K:  {K:.4f}")
print(f"────────────────────────\n")

# ── Time Stepping ─────────────────────────────────────────────────────────────
target_CFL = 0.2