# Solver benchmarks

Three small benchmarks covering the performance and accuracy envelope of the
Dynamiq solvers. They exist to gate the `solvers.jl` refactor: the contract is
that **time, allocations and accuracy all stay put**, so each benchmark prints
all three as a flat, diffable table.

```
atwin run bench/bench1_small.jl     # d = 4     per-step overhead
atwin run bench/bench2_large.jl     # d = 4096  matvec throughput
atwin run bench/bench3_stiff.jl     # d = 512 / 128, stiff
```

Each runs in well under ten seconds. Run all three before and after a change and
diff the tables.

## What each one is for

| | dimension | what dominates | what it catches |
|---|---|---|---|
| **1 — small** | d = 4 | per-step overhead | a refactor that adds indirection, dispatch or setup to the hot loop; invisible at large d |
| **2 — large** | d = 4096 | `apply!` over COO triples | a perturbed matvec kernel, Chebyshev recurrence or buffer rotation; also makes a stray per-step allocation obvious |
| **3 — stiff** | d = 512 / 128 | Chebyshev degree, adaptive controller | high-order `besselj_series!`; a broken `_strang_adapt` feedback loop, as either a runtime blow-up or a silent accuracy loss |

Together they cover TDSE, WFMC and QME, with and without dissipation.

## Accuracy references

Benchmarks 1 and 2 check against **independent** references — the analytic Rabi
solution, and for the damped case a Bloch-equation RK4 that shares no code with
AtomTwin. Those are true error figures.

Benchmark 3 has no closed form (an interacting chain), so it self-consistency
checks against a more finely resolved run. That catches a refactor that changes
the answer, which is what it is for, but would not catch an error already
present in the reference.

## Two traps worth knowing

**`play` has no `seed` keyword.** The parameter is `rng`. Passing `seed = 1234`
is silently swallowed by `kwargs...` and does nothing — and it looks like a
working seed right up until the number drifts between runs. Use
`rng = MersenneTwister(1234)`. Measured: with `seed =`, three identical 400-shot
calls gave 0.4197 / 0.4181 / 0.3961; with `rng =` they agree bit-for-bit.

**`tol` does not affect the TDSE path.** `dt` is the output grid, the Chebyshev
step is spectrally exact, and `play` clamps the propagator tolerance to 1e-12
regardless — so tol = 1e-4 and tol = 1e-10 give bit-identical trajectories.
A tol-based reference on a wavefunction path compares a run against itself and
reports a meaningless `0.0`. Vary the output grid instead. Only the QME path,
where `tol` drives the sub-step controller, responds to it.

## Reading the numbers

Timing is best-of-N after a warm-up call. If a single case looks anomalous,
re-run it alone before believing it: the first call in a cold daemon absorbs
compilation and reads exactly like a real pathology.

Allocation is from one measured run. The wavefunction paths allocate per shot;
what matters is that the figure does not *grow*.

Benchmark 1's WFMC row is a smoke test, not an accuracy bound: at 400 shots the
statistic is dominated by sampling noise, measured at 0.019–0.051 across eight
seeds (mean 0.031). Its value is that the seed is fixed, so it reproduces
exactly — treat a *changed* number as the signal, not a large one.
