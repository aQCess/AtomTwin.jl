"""
Rabi oscillation benchmark — QuTiP

Companion to rabi_benchmark.jl. Identical system parameters, analytical
reference, and accuracy metric (max|err| over last Rabi period).

System: |g⟩ ↔ |e⟩, resonant drive Ω/2π = 1 MHz, decay Γ/2π = 5 kHz.
Duration: 1000 Rabi periods (1 ms).
Time step: dt = T / 100_000 (100 steps per Rabi period), passed as tlist
  so all three solvers output on the same fixed grid.  QuTiP uses adaptive
  ODE integrators internally; tlist only controls the output times.

Parallelism: mcsolve uses all available CPUs by default (ntraj distributed
  across workers).  This matches the Julia benchmark where MCWF uses 
  Threads.@threads and SE/ME are serial.

Run with:  python benchmark/rabi_benchmark_qutip.py
Requires:  qutip >= 5, numpy
"""

import time
import numpy as np
import qutip as qt

# ── Parameters (matching rabi_benchmark.jl) ─────────────────────────────────

Omega = 2 * np.pi * 1.0e6          # Rabi frequency (rad/s)
Gamma = 2 * np.pi * 0.5e3          # Spontaneous decay rate |e⟩ → |g⟩ (rad/s)
T     = 1000 / (Omega / (2*np.pi)) # 1000 Rabi periods (s)
dt    = T / 100_000                # 100 fixed steps per Rabi period
SHOTS = 100                        # Monte Carlo trajectories
SAMPLES = 10

N_total = round(T / dt)
tlist   = np.linspace(0.0, T, N_total + 1)   # N+1 points: [0, dt, …, T]

# ── Analytical reference (same expressions as rabi_benchmark.jl) ──────────────
#
# Schrödinger (unitary, resonant drive, |g⟩ initial state):
#   P_e(t) = sin²(Ωt/2)    [exact]
#
# Master equation (resonant Lindblad with L = √Γ |g⟩⟨e|):
#   Exact closed-form via Bloch equations.
#   Bloch matrix A₂ = [[-Γ/2, Ω], [-Ω, -Γ]] has eigenvalues
#   λ = -3Γ/4 ± iΩ_R,  Ω_R = √(Ω² - Γ²/16).
#   Steady state: z_ss = -Γ²/(Γ²+2Ω²).
#   Transient: δz(t) = exp(A₂t)·δz₀  (closed-form via 2×2 matrix exp).

def analytical_me(t, gamma=Gamma):
    denom   = gamma**2 + 2*Omega**2
    z_ss    = -gamma**2 / denom
    dy0     =  2*Omega*gamma / denom    # -y_ss
    dz0     = -2*Omega**2 / denom       # -1 - z_ss
    Omega_R = np.sqrt(Omega**2 - gamma**2 / 16)
    e  = np.exp(-3*gamma * t / 4)
    cs = np.cos(Omega_R * t)
    sn = np.sin(Omega_R * t) / Omega_R
    dz = e * (-Omega * sn * dy0 + (cs - (gamma/4) * sn) * dz0)
    return (1 + z_ss + dz) / 2

# Reference arrays for the last Rabi period.
# tlist includes t=0, so tlist[-n_last:] covers times [(N-n_last+1)·dt … T],
# matching the Julia end-indexed slice.
n_last   = round((1 / (Omega / (2*np.pi))) / dt)   # = 100
t_last   = np.array([(N_total - n_last + i) * dt for i in range(1, n_last + 1)])
P_se_ref = analytical_me(t_last, gamma=0)
P_me_ref = analytical_me(t_last)

def maxerr(sim, ref): return np.max(np.abs(sim[-n_last:] - ref))

# ── Helpers ───────────────────────────────────────────────────────────────────

def report_time(label, t_ms, n_repeats):
    print(f"  {label:<46}  {t_ms:7.1f} ms   (min of {n_repeats} runs)")

def report_acc(label, val):
    print(f"  {label:<46}  max|err| = {val:.2e}")

def timeit(fn, repeats):
    """Return (min_time_ms, last_result)."""
    result, times = None, []
    for _ in range(repeats):
        t0     = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3, result

# ── Operators ─────────────────────────────────────────────────────────────────

g      = qt.basis(2, 0)
e      = qt.basis(2, 1)
psi0   = g
H      = (Omega / 2) * (e * g.dag() + g * e.dag())
c_ops  = [np.sqrt(Gamma) * g * e.dag()]   # L = √Γ |g⟩⟨e|
proj_e = [e * e.dag()]                     # |e⟩⟨e|

# ── Header ────────────────────────────────────────────────────────────────────

print("=" * 72)
print(f"Rabi benchmark (QuTiP {qt.__version__}) — "
      f"{round(T*1e6, 1)} μs, {dt*1e9:.2f} ns step, {SHOTS} shots")
print(f"Ω/2π = {Omega/2/np.pi/1e6:.1f} MHz,  "
      f"Γ/2π = {Gamma/2/np.pi/1e3:.1f} kHz,  "
      f"{round(T * Omega / (2*np.pi))} Rabi periods")
print("=" * 72)
print("\n── QuTiP ─────────────────────────────────────────────────────────────")

# ── Accuracy ──────────────────────────────────────────────────────────────────

# Tight tolerances for accuracy comparison — default rtol=1e-6 accumulates
# significant phase error over 1000 Rabi cycles with an adaptive integrator.
tight = {"atol": 1e-10, "rtol": 1e-8}

print("  Accuracy (max|err| last Rabi period, analytical reference):")

r_se_def = qt.sesolve(H, psi0, tlist, e_ops=proj_e, progress_bar=False)
report_acc("Schrödinger (default tol)", maxerr(r_se_def.expect[0], P_se_ref))

#r_se = qt.sesolve(H, psi0, tlist, e_ops=proj_e, options=tight, progress_bar=False)
#report_acc("Schrödinger (tight tol)", maxerr(r_se.expect[0], P_se_ref))

r_me_def = qt.mesolve(H, psi0, tlist, c_ops, e_ops=proj_e, progress_bar=False)
report_acc("master equation (default tol)", maxerr(r_me_def.expect[0], P_me_ref))

#r_me = qt.mesolve(H, psi0, tlist, c_ops, e_ops=proj_e, options=tight, progress_bar=False)
#report_acc("master equation (tight tol)", maxerr(r_me.expect[0], P_me_ref))

r_mc = qt.mcsolve(H, psi0, tlist, c_ops, e_ops=proj_e, ntraj=SHOTS, options=tight, progress_bar=False)
report_acc(f"MCWF ({SHOTS} shots avg, tight tol)", maxerr(r_mc.expect[0], P_me_ref))

print()

# ── Timing ────────────────────────────────────────────────────────────────────

print("  Timing (minimum over samples):")

import multiprocessing
n_cpus = multiprocessing.cpu_count()

t_se, _ = timeit(lambda: qt.sesolve(H, psi0, tlist, e_ops=proj_e, progress_bar=False), repeats=SAMPLES)
t_me, _ = timeit(lambda: qt.mesolve(H, psi0, tlist, c_ops, e_ops=proj_e, progress_bar=False), repeats=SAMPLES)
t_mc1,_ = timeit(lambda: qt.mcsolve(H, psi0, tlist, c_ops, e_ops=proj_e, ntraj=SHOTS,
                                     options={"map": "serial"}, progress_bar=False), repeats=SAMPLES)
t_mcN,_ = timeit(lambda: qt.mcsolve(H, psi0, tlist, c_ops, e_ops=proj_e, ntraj=SHOTS,
                                     options={"map": "parallel"}, progress_bar=False), repeats=SAMPLES)

report_time("Schrödinger (unitary)",                          t_se,  SAMPLES)
report_time("master equation",                                t_me,  SAMPLES)
report_time(f"MCWF ({SHOTS} shots, sequential)",              t_mc1, SAMPLES)
report_time(f"MCWF ({SHOTS} shots, parallel, {n_cpus} CPUs)", t_mcN, SAMPLES)

print("\n" + "=" * 72)
