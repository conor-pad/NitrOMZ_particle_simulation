# ── Particle Parameters (Define this FIRST) ───────────────────────────────────
radius = 2.0

# ── Domain (Scaled by radius) ─────────────────────────────────────────────────
# Lx is 20x the radius (leaves exactly 15 radii of space downstream for the wake)
# Ly is 17.5x the radius (leaves ~8.75 radii of space above and below)
Lx = 20.0 * radius  
Ly = 10 * radius  

Nx, Ny = int(331), int(287)

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
Total_Time = 25

# Particle Center
cx = 5.0 * radius   # Positioned 5 radii from the left wall
cy = Ly / 2.0       # perfectly centered vertically

# ── DOC ───────────────────────────────────────────────────────────────────────
doc_flux_rate = 1       # Hydrolysis rate: solid POC to dissolved DOC (mmol/m^3/s)
doc_initial_core = 0 # 30.0   # Initial DOC concentration inside the particle

# ── Target Dimensionless Numbers ──────────────────────────────────────────────
Re_target = 1   # Reynolds number
Sc_target = 660   # Schmidt number

# Pe = Re * Sc
Pe_calc = Re_target * Sc_target  

# ── Derived Physics Parameters ────────────────────────────────────────────────
U_bg = 6.0  

# Calculate required Kinematic Viscosity
# Formula: Re = (U * radius) / nu  --->  nu = (U * radius) / Re
nu = (U_bg * radius) / Re_target

# Calculate required Scalar Diffusivity (K)
# Formula: Sc = nu / K        --->  K = nu / Sc
K = nu / Sc_target

# Sherwood number
Sh = 1 + 0.619 * Re_target * 0.412 * Sc_target**(1/3)

print(f"\n── Simulation Physics ──")
print(f"Targets  | Re: {Re_target}  |  Sc: {Sc_target}  |  Pe: {Pe_calc}")
print(f"Derived  | Sh: {Sh:.2f}  |  nu: {nu:.2f} |  K:  {K:.4f}")
print(f"────────────────────────\n")

# ── Time Stepping ─────────────────────────────────────────────────────────────
target_CFL = 0.2

