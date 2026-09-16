# Recorded baseline

Numbers to diff against when refactoring `src/Dynamiq/solvers.jl`. Re-run all
three benchmarks and compare; the refactor's contract is that none of these
move beyond the noted tolerances.

```
AtomTwin b019d21 (refactor-solvers)   julia 1.12.7 via atwin daemon
recorded 2026-09-16 (superseding the 0783b16 run of 2026-09-15)
```

Re-recorded after the solvers.jl split. Every accuracy figure was verified
identical at the branch point (41fed4e) and after the refactor, so the split
itself changed nothing. The numbers below differ from the previous 0783b16
table because several numerics commits landed in between -- the interval
spectral bound (dad1938) and the `force()` wavelength fix (769a4d0) -- and
because that table was recorded under julia 1.11.9, which is not comparable.

Timings are a warm `atwin` daemon on an idle machine; treat sub-millisecond
rows as resolution-limited.

| benchmark | case | best[s] | alloc | accuracy |
|---|---|---|---|---|
| 1 — small | TDSE  d=4 (no dissipation) | 0.0002–0.0003 | 50.3 KiB | 5.52e-14 vs analytic |
| 1 — small | QME   d=4 (decay) | 0.020–0.021 | 28.2 KiB | 7.29e-05 vs Bloch RK4 |
| 1 — small | WFMC  d=4 (400 shots) | 0.014–0.015 | 16.1 MiB | 1.16e-01 vs Bloch (sampling noise) |
| 2 — large | TDSE  d=4096 (no dissipation) | 0.012–0.013 | 12.2 MiB | 1.43e-14 vs analytic |
| 2 — large | WFMC  d=4096 (decay, 1 shot) | 0.017 | 30.3 MiB | 2.36e-02 vs undamped |
| 3 — stiff | TDSE  d=512 (V = 200x) | 0.036 | 933.3 KiB | 2.45e-11 vs 8x finer grid |
| 3 — stiff | QME   d=128 (dephasing) | 0.88–0.90 | 1.7 MiB | 1.24e-08 vs tol=1e-10 |

## How to read a diff

**Accuracy must match exactly**, with one exception. Six of the seven figures
are deterministic and should reproduce digit for digit; a change in any of them
means the refactor altered the numerics. The exception is benchmark 2's WFMC
row, which is a single trajectory whose value depends on how many jumps fired —
it is seeded, so it too should reproduce exactly, but if it moves, check whether
the jump *count* changed before concluding the flow is wrong.

**Allocation must not grow.** The figures above are dominated by per-shot and
per-instruction setup, not by the step loop. Chebyshev's hot loop is
allocation-free and there are `@allocated == 0` assertions in
`test/unit/test_solver_order.jl` pinning that; this table is the coarser,
whole-run check.

**Timing is the loose one.** Ranges above are spreads observed across repeated
runs on an otherwise idle machine. Treat anything inside the range as unchanged
and anything beyond ~20% as worth investigating. `bin/atwin-examples` is the
finer timing instrument; these benchmarks exist to localise a regression to a
solver path once that suite flags one.

## Timing caveat

These were recorded through the `atwin` daemon, which carries warm state. They
are self-consistent and fine for before/after comparison in the same session,
but they are **not** comparable with `testlogs/` numbers, which are measured in
a fresh process under julia 1.12 with `-t10`.
