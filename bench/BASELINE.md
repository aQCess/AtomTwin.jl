# Recorded baseline

Numbers to diff against when changing the solvers. Re-run all four benchmarks
and compare.

```
AtomTwin 934b5b2 (0.2.0 release candidate)   julia 1.12.7, warm session, 10 threads
recorded 2026-10-18
```

Timings are a warm daemon on an idle machine; treat sub-millisecond rows as
resolution-limited.

| benchmark | case | best[s] | alloc | accuracy |
|---|---|---|---|---|
| 1 — small | TDSE  d=4 (no dissipation) | 0.0003 | 59.9 KiB | 5.52e-14 vs analytic |
| 1 — small | QME   d=4 (decay) | 0.020 | 29.8 KiB | 7.29e-05 vs Bloch RK4 |
| 1 — small | WFMC  d=4 (400 shots) | 0.005 | 18.6 MiB | 7.04e-02 vs Bloch (sampling noise) |
| 2 — large | TDSE  d=4096 (no dissipation) | 0.011 | 12.7 MiB | 1.43e-14 vs analytic |
| 2 — large | WFMC  d=4096 (decay, 1 shot) | 0.015 | 13.9 MiB | 2.36e-02 vs undamped |
| 3 — stiff | TDSE  d=512 (V = 200x) | 0.036 | 1002 KiB | 2.45e-11 vs 8x finer grid |
| 3 — stiff | QME   d=128 (dephasing) | 0.87 | 1.8 MiB | 1.24e-08 vs tol=1e-10 |
| 4 — motion | WFMC imaging pulses, planar, 200 shots | 0.20 | 128.5 MiB | 1.42 E_r/photon (ref ≈ 1.28) |
| 4 — motion | WFMC same drive, GlobalCoupling | 0.063 | 126.9 MiB | 0.31 E_r/photon (emission only ≈ 0.33) |
| 4 — motion | TDSE 20 π pulses +k/−k | 0.0009 | 909.5 KiB | 20.00 ħk (ref 20) |
| 4 — motion | QME CW 1 µs, s = 40 | 0.012 | 30.5 KiB | 1.01 (ref ≈ 1) |

## How to read a diff

**Deterministic accuracy figures must match exactly**: the TDSE and QME rows, and
the seeded WFMC rows as long as the jump sequence is unchanged. A change in any of
them means the numerics changed.

**Monte Carlo rows are noisy.** Benchmark 1's WFMC figure is dominated by sampling
noise (0.02–0.07 across seeds). Benchmark 4's imaging row scatters by about ±10 %
between seeds even at 1000 shots (1.18–1.43, mean 1.32); treat a change beyond that,
or a drift of the GlobalCoupling control, as the signal.

**Allocation must not grow.** The figures are dominated by per-shot and
per-instruction setup, not by the step loop; the Chebyshev and dissipator hot loops
are pinned allocation-free by `test/unit/test_solver_order.jl`.

**Timing is the loose one.** Treat anything within ~20 % as unchanged. These are
warm-session timings: self-consistent for before/after comparison in one session, not
comparable with a fresh process.
