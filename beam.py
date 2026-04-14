import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ── 1. Beam properties ──────────────────────────────────────────────────────
L = 1.0          # length of beam in meters
E = 200e9        # Young's modulus (steel), in Pascals
I = 8.33e-10     # second moment of area, m^4
rho = 7850       # density of steel, kg/m^3
A = 0.001        # cross-sectional area, m^2

# ── 2. First three natural frequencies ──────────────────────────────────────
beta_L = [1.8751, 4.6941, 7.8548]
omega = [(b**2) * np.sqrt((E * I) / (rho * A * L**4)) for b in beta_L]

print("Analytical natural frequencies (Hz):")
for i, w in enumerate(omega):
    print(f"  Mode {i+1}: {w / (2 * np.pi):.2f} Hz")

# ── 3. Three-mode equation of motion ────────────────────────────────────────
# Each mode behaves like an independent damped oscillator.
# State vector: [x1, v1, x2, v2, x3, v3]
# where x1, x2, x3 are modal displacements and v1, v2, v3 are velocities.
zeta = 0.02   # 2% damping for all modes

def beam_odes(t, y):
    x1, v1, x2, v2, x3, v3 = y
    dx1dt = v1
    dv1dt = -2 * zeta * omega[0] * v1 - omega[0]**2 * x1
    dx2dt = v2
    dv2dt = -2 * zeta * omega[1] * v2 - omega[1]**2 * x2
    dx3dt = v3
    dv3dt = -2 * zeta * omega[2] * v3 - omega[2]**2 * x3
    return [dx1dt, dv1dt, dx2dt, dv2dt, dx3dt, dv3dt]

# ── 4. Initial conditions ────────────────────────────────────────────────────
# Give each mode an initial velocity to excite all three simultaneously.
# Mode participation factors scale how much each mode is excited by the impulse.
# Higher modes are excited less strongly, so we scale them down.
x0 = [0.0, 1.0,    # Mode 1: no displacement, full velocity
       0.0, 0.5,    # Mode 2: no displacement, half velocity
       0.0, 0.8]    # Mode 3: no displacement, small velocity

# ── 5. Solve the ODE ────────────────────────────────────────────────────────
t_start = 0
t_end = 10.0
dt = 0.001
t_span = (t_start, t_end)
t_eval = np.arange(t_start, t_end, dt)

solution = solve_ivp(beam_odes, t_span, x0, t_eval=t_eval, method='RK45')

t = solution.t

# ── 6. Total tip deflection = sum of all three modal contributions ───────────
x1 = solution.y[0]
x2 = solution.y[2]
x3 = solution.y[4]
x = x1 + x2 + x3   # superposition of all three modes

# ── 7. Plot ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(t, x, color='steelblue')
plt.xlabel("Time (s)")
plt.ylabel("Tip Deflection (m)")
plt.title("Cantilever Beam Tip Deflection — Three Modes")
plt.grid(True)
plt.tight_layout()
plt.show()