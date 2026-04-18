"""
Rydberg blockade benchmark — QuTiP
Companion to rydberg_benchmark.jl. Identical parameters, same output cadence.
Reports N-scaling timing for Schrödinger, master equation, and MCWF solvers.

Run with:  python benchmark/rydberg_benchmark_qutip.py
Requires:  qutip >= 5, numpy
"""

import time
import multiprocessing
import numpy as np
import qutip as qt

# ── Parameters (matching rydberg_benchmark.jl exactly) ───────────────────────

Omega  = 2 * np.pi * 1.0e6        # Rabi frequency (rad/s)
V      = 2 * np.pi * 100.0e6      # blockade interaction (rad/s); V/Ω = 100
gamma  = 2 * np.pi * 250.0e3      # dephasing rate on |r⟩ (rad/s)
T      = 1.0 / (Omega / (2*np.pi))  # 1 bare Rabi period (s)
SHOTS      = 100
BENCH_MCWF = False

def dt_n(n):
    return 1.0 / (25 * np.sqrt(n) * V / (2*np.pi))

def tlist_n(n):
    dt = dt_n(n)
    return np.arange(0.0, T + dt/2, dt)

# ── Builder ───────────────────────────────────────────────────────────────────

def build_qutip(n):
    I   = qt.qeye(2)
    n_r = qt.num(2)          # |r⟩⟨r|

    def embed(op, i):
        ops = [I] * n
        ops[i] = op
        return qt.tensor(*ops)

    H       = sum((Omega/2) * embed(qt.sigmax(), i) for i in range(n))
    H      += sum(V * embed(n_r, i) * embed(n_r, j)
                  for i in range(n) for j in range(i+1, n))
    J       = [np.sqrt(gamma) * embed(n_r, i) for i in range(n)]
    n_r_tot = sum(embed(n_r, i) for i in range(n))
    psi0    = qt.tensor(*[qt.basis(2, 0) for _ in range(n)])
    return H, J, psi0, n_r_tot

# ── Helpers ───────────────────────────────────────────────────────────────────

def timeit(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return min(times)

# ── Header ────────────────────────────────────────────────────────────────────

print("=" * 93)
print(f"Rydberg blockade (QuTiP): Ω/2π={Omega/2/np.pi/1e6:.0f} MHz  "
      f"V/Ω={round(V/Omega)}  γ/2π={gamma/2/np.pi/1e3:.0f} kHz  "
      f"T={T*1e6:.2f} μs  SHOTS={SHOTS}  CPUs={multiprocessing.cpu_count()}")
print("Adaptive ODE integrator (default tolerances); tlist matches AtomTwin output cadence")
print("=" * 93)
header = f"\n  {'N':>2} {'d':>5} │ {'SE (ms)':>9} │ {'ME (ms)':>9}"
if BENCH_MCWF: header += f" │ {'MCWF (ms)':>9}"
print(header)
print("  " + "-"*(40 if BENCH_MCWF else 29))

# ── N-scaling benchmark ───────────────────────────────────────────────────────

for n in range(2, 9):
    H, J, psi0, n_r_tot = build_qutip(n)
    tl = tlist_n(n)

    t_se = timeit(lambda H=H, psi=psi0, tl=tl:
                      qt.sesolve(H, psi, tl, progress_bar=False))
    t_me = timeit(lambda H=H, J=J, psi=psi0, tl=tl:
                      qt.mesolve(H, psi, tl, J, progress_bar=False))

    row = f"  {n:>2} {2**n:>5} │ {t_se:>9.1f} │ {t_me:>9.1f}"
    if BENCH_MCWF:
        t_mc = timeit(lambda H=H, J=J, psi=psi0, tl=tl:
                          qt.mcsolve(H, psi, tl, J, ntraj=SHOTS,
                                     options={"map": "parallel"}, progress_bar=False))
        row += f" │ {t_mc:>9.1f}"
    print(row)

print("\n" + "=" * 93)
