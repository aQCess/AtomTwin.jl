# Benchmark 1 — small Hilbert space, d = 4.
#
# Two two-level atoms, resonantly driven, no interaction: d = 2^2 = 4. The atoms
# are independent and each undergoes plain Rabi flopping, so the exact answer is
# known in closed form and the accuracy figure is a true error, not a comparison
# against another run of the same solver.
#
# This case is where per-step OVERHEAD dominates: the matrices are tiny, so any
# cost that does not scale with `d` -- plan setup, the controller, dispatch,
# detector writes -- shows up here and nowhere else. It is the benchmark most
# likely to catch a refactor that adds a layer of indirection to the hot loop.
#
# Covered: TDSE (no dissipation) and QME (with dissipation), plus WFMC to
# exercise the jump path.
#
#   atwin run bench/bench1_small.jl

using AtomTwin
using Random: MersenneTwister
using Statistics: mean
include(joinpath(@__DIR__, "harness.jl"))

const Ω        = 2π * 1.0e6      # Rabi frequency (rad/s)
const γ        = 2π * 0.1e6      # decay rate on |e> (rad/s)
const DURATION = 4e-6            # 4 Rabi periods
const DT       = 2π / (20Ω)      # output grid: 20 points per period

"""
Build a system of `n` independent resonantly-driven two-level atoms.
Returns `(system, couplings, levels, atoms)`.
"""
function build(n::Int; decay::Bool = false)
    g, e  = Level("g"), Level("e")
    atoms = [Atom(; levels = [g, e]) for _ in 1:n]
    system = System(atoms)
    cs = [add_coupling!(system, a, g => e, Ω; active = false) for a in atoms]
    if decay
        for a in atoms
            add_decay!(system, a, e => g, γ)
        end
    end
    add_detector!(system, PopulationDetectorSpec(atoms[1], e; name = "P_e"))
    return system, cs, (g, e), atoms
end

# ── Exact solutions ──────────────────────────────────────────────────────────
# Resonant, undamped: P_e(t) = sin²(Ωt/2).
exact_rabi(t) = sin(Ω * t / 2)^2

# Resonant with decay on the driven transition, in the Lindblad master equation.
# The Bloch equations for a resonantly driven, decaying two-level atom give
#   P_e(t) = (Ω²/(Ω² + γ²/2)) * ... -- rather than transcribe the closed form and
# risk a typo in the very thing we are validating against, integrate the 3-vector
# Bloch system with a fine RK4. It is a 3-ODE problem, so "fine" is free, and it
# is an INDEPENDENT integrator: nothing in AtomTwin is involved.
function exact_damped(times::AbstractVector{Float64})
    # d/dt (u, v, w) with u = 2Re ρ_ge, v = 2Im ρ_ge, w = ρ_ee - ρ_gg
    f = function (s)
        u, v, w = s
        return (-γ/2 * u,
                -γ/2 * v - Ω * w,
                 Ω * v - γ * (w + 1))
    end
    s = (0.0, 0.0, -1.0)                      # start in |g>
    out = Float64[]
    h  = 1e-11                                 # far finer than anything measured
    t  = 0.0
    for tt in times
        while t < tt - 1e-18
            step = min(h, tt - t)
            k1 = f(s)
            k2 = f(s .+ (step/2) .* k1)
            k3 = f(s .+ (step/2) .* k2)
            k4 = f(s .+ step .* k3)
            s  = s .+ (step/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
            t += step
        end
        push!(out, (s[3] + 1) / 2)             # P_e = (w+1)/2
    end
    return out
end

function main()
    rows  = Tuple{String,Float64,Int,Union{Nothing,Float64},String}[]
    total = 0.0

    # ── TDSE: no dissipation, d = 4 ──────────────────────────────────────────
    sys, cs, (g, e), _ = build(2)
    seq = Sequence(DT; tol = 1e-8)
    @sequence seq begin
        Pulse(cs, DURATION)
    end
    r = bench(() -> play(sys, seq; initial_state = [g, g]))
    P = r.value.detectors["P_e"]
    err = relerr(P, exact_rabi.(r.value.times))
    push!(rows, ("TDSE  d=4  (no dissipation)", r.best, r.bytes, err, "vs analytic"))
    total += r.best

    # ── QME: with decay, d = 4 ───────────────────────────────────────────────
    sysd, csd, (gd, ed), _ = build(2; decay = true)
    seqd = Sequence(DT; tol = 1e-8)
    @sequence seqd begin
        Pulse(csd, DURATION)
    end
    # The measured error here is ~8e-6 at tol = 1e-8, NOT 1e-8: `tol` bounds the
    # error of one step and this run takes 80 of them, so the accumulated global
    # error is the larger number. Verified to be genuine solver error and not a
    # bad reference -- tightening tol by a decade buys a decade, converging on
    # P_e(end) = 0.4220190. It is a stable figure, which is what makes it useful
    # as a regression check.
    r = bench(() -> play(sysd, seqd; initial_state = [gd, gd], density_matrix = true))
    P = r.value.detectors["P_e"]
    err = relerr(P, exact_damped(collect(r.value.times)))
    push!(rows, ("QME   d=4  (decay, master eq)", r.best, r.bytes, err, "vs Bloch RK4"))
    total += r.best

    # ── WFMC: same system, trajectories. Exercises the jump path. ────────────
    # This column is a SMOKE TEST, not an accuracy bound. The statistic is
    # max|trajectory mean - master equation| over the whole trace, and at 400
    # shots it is dominated by sampling noise: measured across 8 seeds it spans
    # 0.019 to 0.051, mean 0.031. (Per-point 1-sigma is ~0.025 = 0.5/sqrt(400),
    # and taking a max over ~80 points inflates that by 2.5-3x.) So ~3e-2 is
    # normal and 5e-2 is an ordinary tail, NOT a regression.
    #
    # What makes it useful is that the seed is fixed, so the number is
    # deterministic: it should reproduce EXACTLY run to run, and a refactor that
    # breaks the jump path moves it far beyond the seed-to-seed spread.
    # `rng`, NOT `seed`: `play` has no `seed` keyword, and passing one is
    # silently absorbed by `kwargs...` and ignored -- which looks exactly like a
    # working seed until the number moves between runs. Measured: with `seed =`
    # three identical calls gave 0.4197 / 0.4181 / 0.3961; with `rng =` they
    # agree bit-for-bit.
    r = bench(() -> play(sysd, seqd; initial_state = [gd, gd], shots = 400,
                         rng = MersenneTwister(1234)); reps = 1)
    Pm  = vec(mean(r.value.detectors["P_e"]; dims = 2))
    err = relerr(Pm, exact_damped(collect(r.value.times)))
    push!(rows, ("WFMC  d=4  (400 shots)", r.best, r.bytes, err, "vs Bloch; ~3e-2 is sampling noise"))
    total += r.best

    report("Benchmark 1 — small Hilbert space (d = 4): per-step overhead", rows)
    checkbudget(total)
    return nothing
end

main()
