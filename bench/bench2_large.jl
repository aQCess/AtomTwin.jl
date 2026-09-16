# Benchmark 2 — large Hilbert space, d = 4096, short duration.
#
# Twelve two-level atoms, resonantly driven: d = 2^12 = 4096. The duration is
# deliberately short (a fifth of a Rabi period) so the case stays inside its time
# budget while still taking enough steps to measure.
#
# This case is the OPPOSITE of benchmark 1: per-step overhead is negligible and
# essentially all the time is in `apply!` walking the COO triples. It is what
# catches a refactor that perturbs the matvec kernel, the Chebyshev recurrence,
# or the buffer rotation -- and, because the state vector is 64 KiB and the
# workspace three times that, it is also where an accidental per-step allocation
# becomes impossible to miss.
#
# WHY NO QME HERE. The master equation propagates a d x d density matrix: at
# d = 4096 that is 268 MB per matrix and the solver needs several. Measured
# scaling on this machine is 0.21 s (d=64) -> 0.60 s (d=128) -> 7.8 s (d=256),
# so d = 4096 is out of reach by orders of magnitude, not by a factor. The
# dissipative case at this size is therefore WFMC, which propagates a state
# vector and is the method one would actually use here. QME's large-d behaviour
# is covered by benchmark 3 at a dimension it can reach.
#
#   atwin run bench/bench2_large.jl

using AtomTwin
using Random: MersenneTwister
include(joinpath(@__DIR__, "harness.jl"))

const N        = 12              # 2^12 = 4096
const Ω        = 2π * 1.0e6
const γ        = 2π * 0.05e6
const DURATION = 2e-7            # ~1/5 of a Rabi period
const DT       = 2π / (20Ω)

function build(; decay::Bool = false)
    g, e  = Level("g"), Level("e")
    atoms = [Atom(; levels = [g, e]) for _ in 1:N]
    system = System(atoms)
    cs = [add_coupling!(system, a, g => e, Ω; active = false) for a in atoms]
    if decay
        for a in atoms
            add_decay!(system, a, e => g, γ)
        end
    end
    add_detector!(system, PopulationDetectorSpec(atoms[1], e; name = "P_e"))
    return system, cs, g, e
end

# The atoms are independent and identically driven, so despite d = 4096 the
# single-atom observable is exactly the two-level Rabi solution. That is what
# makes an analytic check possible at this size at all -- and it is a real check
# on the many-body machinery: an error in the tensor-product structure, the
# basis indexing or the population read-out breaks it immediately.
exact_rabi(t) = sin(Ω * t / 2)^2

function main()
    rows  = Tuple{String,Float64,Int,Union{Nothing,Float64},String}[]
    total = 0.0

    # ── TDSE, d = 4096, no dissipation ───────────────────────────────────────
    sys, cs, g, _ = build()
    seq = Sequence(DT; tol = 1e-8)
    @sequence seq begin
        Pulse(cs, DURATION)
    end
    r = bench(() -> play(sys, seq; initial_state = fill(g, N)))
    err = relerr(r.value.detectors["P_e"], exact_rabi.(r.value.times))
    push!(rows, ("TDSE  d=4096  (no dissipation)", r.best, r.bytes, err, "vs analytic"))
    total += r.best

    # ── WFMC, d = 4096, with decay ───────────────────────────────────────────
    # A single trajectory, so this measures the deterministic non-Hermitian flow
    # plus the jump test -- the cost structure of the path -- rather than a
    # converged expectation value. The accuracy column compares against the
    # undamped analytic curve, which over this short a window and at this decay
    # rate is close but NOT equal: expect ~1e-2, and treat it as a smoke test
    # that the trajectory is on the right curve, not as an error bound.
    sysd, csd, gd, _ = build(; decay = true)
    seqd = Sequence(DT; tol = 1e-8)
    @sequence seqd begin
        Pulse(csd, DURATION)
    end
    # `rng`, not `seed` -- `play` has no `seed` keyword; see bench1 for the trap.
    r = bench(() -> play(sysd, seqd; initial_state = fill(gd, N), shots = 1,
                         rng = MersenneTwister(7)))
    err = relerr(r.value.detectors["P_e"], exact_rabi.(r.value.times))
    push!(rows, ("WFMC  d=4096  (decay, 1 shot)", r.best, r.bytes, err, "vs undamped, ~1e-2 expected"))
    total += r.best

    report("Benchmark 2 — large Hilbert space (d = 4096): matvec throughput", rows)
    checkbudget(total)
    return nothing
end

main()
