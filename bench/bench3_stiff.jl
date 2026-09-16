# Benchmark 3 — stiff system, intermediate Hilbert space (d = 512 / 128).
#
# A chain of two-level atoms under a STRONG nearest-neighbour Rydberg blockade:
# V/2pi = 200 MHz against a drive of Omega/2pi = 1 MHz, a 200x separation of
# timescales. The interaction sets a fast phase rotation the propagator must
# resolve while the physics of interest evolves on the slow Rabi timescale --
# which is what "stiff" means here.
#
# This is the case the recent solver work exists for, and the one most sensitive
# to a refactor:
#
#   * The Chebyshev degree is set by dE*dt, so a stiff H drives the expansion
#     deep -- this is the only benchmark that exercises `besselj_series!` at
#     high order and the three-term recurrence over many terms. Note the step is
#     ~2000x past the Taylor-4 stability limit; Chebyshev has no such limit and
#     absorbs it by raising its degree, which is precisely the property under
#     test. `warn_if_step_too_large` fires on the Taylor bound regardless, so the
#     warning below is expected and is not a failure.
#   * The QME case drives the ADAPTIVE STEP CONTROLLER hard: stiffness is what
#     makes the Strang splitting error large enough that `k` must climb. A
#     refactor that breaks `_strang_adapt`'s feedback shows up here first, either
#     as a runtime blow-up (k climbing without bound) or as a silent accuracy
#     loss.
#
# Dimensions differ between the two cases by necessity: the master equation
# propagates a d x d matrix and the cost grows far faster than d. Measured on
# this system at tol = 1e-8: d = 64 -> 0.45 s, d = 128 -> 5.4 s, d = 256 ->
# 66.9 s. d = 128 is therefore the largest stiff QME that fits the budget, and
# d = 512 is out of reach for the master equation by two orders of magnitude.
#
# ACCURACY REFERENCE. There is no closed form for an interacting chain, so each
# case is checked against a more finely resolved run of the same solver. That is
# a self-consistency check, not independent validation: it catches a refactor
# that changes the answer, which is what it is for, but it would not catch an
# error already present in the reference. Benchmarks 1 and 2 carry the
# independent references.
#
# The two cases need DIFFERENT references, and the reason is worth stating.
# Varying `tol` does nothing on the TDSE path: `dt` is the output grid, the
# Chebyshev step is spectrally exact, and `play` clamps the propagator tolerance
# to 1e-12 regardless -- measured, tol = 1e-4 and tol = 1e-10 give bit-identical
# trajectories over the same 20 steps. A tol-based reference there would compare
# a run against itself and report a meaningless 0.0. The TDSE case therefore
# references a run on an 8x finer OUTPUT grid (which does change the stepping),
# measured to agree to 6.2e-11. Only the QME case, where `tol` genuinely drives
# the sub-step controller, uses a tolerance-based reference.
#
# Taylor was tried as a structurally independent cross-check and does not work
# here: this step is ~2000x past its stability limit, so it diverges to Inf.
# That is the documented behaviour, and it is the reason the case is stiff.
#
#   atwin run bench/bench3_stiff.jl

using AtomTwin
include(joinpath(@__DIR__, "harness.jl"))

const Ω        = 2π * 1.0e6       # drive
const V        = 2π * 200.0e6     # blockade -- 200x the drive: this is the stiffness
const γ        = 2π * 0.1e6       # dephasing on |r>
const DT       = 2π / (20Ω)

"""
Chain of `n` atoms with nearest-neighbour blockade. `dephasing` adds a |r>
dephasing channel, which is what turns the TDSE case into a QME one.
"""
function build(n::Int; dephasing::Bool = false)
    g, r  = Level("g"), Level("r")
    atoms = [Atom(; levels = [g, r]) for _ in 1:n]
    system = System(atoms)
    cs = [add_coupling!(system, a, g => r, Ω; active = false) for a in atoms]
    for i in 1:(n - 1)
        add_interaction!(system, (atoms[i], atoms[i + 1]), (r, r) => (r, r), V)
    end
    if dephasing
        for a in atoms
            add_dephasing!(system, a, r, γ)
        end
    end
    add_detector!(system, PopulationDetectorSpec(atoms[1], r; name = "P_r"))
    return system, cs, g, atoms
end

function main()
    rows  = Tuple{String,Float64,Int,Union{Nothing,Float64},String}[]
    total = 0.0

    # ── Stiff TDSE, d = 512 (9 atoms), no dissipation ────────────────────────
    # The step-size warning below is expected: see the header.
    sys, cs, g, _ = build(9)
    mkseq_dt = (dt, tol) -> begin
        s = Sequence(dt; tol = tol)
        @sequence s begin
            Pulse(cs, 1e-6)
        end
        s
    end
    mkseq = tol -> mkseq_dt(DT, tol)
    # Reference on an 8x finer output grid, subsampled back onto the coarse one.
    # See the header for why this is not a `tol` sweep.
    fine = play(sys, mkseq_dt(DT / 8, 1e-8); initial_state = fill(g, 9)).detectors["P_r"]
    ref  = fine[8:8:end]
    r = bench(() -> play(sys, mkseq(1e-8); initial_state = fill(g, 9)))
    err = relerr(r.value.detectors["P_r"], ref)
    push!(rows, ("TDSE  d=512  (stiff, V=200x)", r.best, r.bytes, err, "vs 8x finer grid"))
    total += r.best

    # ── Stiff QME, d = 128 (7 atoms), with dephasing ─────────────────────────
    # This is the adaptive controller's case. See the header for why d differs.
    sysd, csd, gd, _ = build(7; dephasing = true)
    mkseqd = tol -> begin
        s = Sequence(DT; tol = tol)
        @sequence s begin
            Pulse(csd, 2e-7)
        end
        s
    end
    refd = play(sysd, mkseqd(1e-10); initial_state = fill(gd, 7),
                density_matrix = true).detectors["P_r"]
    r = bench(() -> play(sysd, mkseqd(1e-8); initial_state = fill(gd, 7),
                         density_matrix = true); reps = 2)
    err = relerr(r.value.detectors["P_r"], refd)
    push!(rows, ("QME   d=128  (stiff, dephasing)", r.best, r.bytes, err, "vs tol=1e-10"))
    total += r.best

    report("Benchmark 3 — stiff system: Chebyshev degree + adaptive controller", rows)
    checkbudget(total)
    return nothing
end

main()
