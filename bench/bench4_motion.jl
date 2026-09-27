# Benchmark 4 — semiclassical motion with plane-wave drives.
#
# One Yb-174 two-level atom in free space, driven on a 29 MHz line by
# plane-wave couplings (`PlanarBeam`) — the configuration of fast fluorescence
# imaging, where the radiation pressure of the drives matters. Covers the three
# semiclassical solvers (WFMC, TDSE, QME) and, as a control, WFMC with the same
# drive as a uniform `GlobalCoupling` (no radiation force to compute), which
# isolates the cost of the force itself.
#
# Accuracy figures are physics references, not solver-vs-solver:
#   - WFMC pulses: y kinetic energy per photon vs 1 + Q + 1/3 ≈ 1.26 E_r
#     (quantised absorption + isotropic emission; Monte Carlo, ±~5 %).
#   - TDSE π pulses +k,−k,…: momentum transferred vs exactly N ħk.
#   - QME CW: momentum after 1 µs vs ħk·R·t from the steady-state rate
#     (mean force; transient ≲ 1 %).
#
#   atwin run bench/bench4_motion.jl

using AtomTwin
using Random: MersenneTwister
using Statistics: mean
include(joinpath(@__DIR__, "harness.jl"))

const Γ    = 2π * 29.1e6
const λ    = 398.9e-9
const m    = 174 * AtomTwin.Units.amu
const vrec = AtomTwin.Units.hbar * (2π / λ) / m

g = Level("g"; term = l"1S0")
e = Level("e"; term = l"1P1")

function build(; planar::Bool, Ω, decay::Bool, clicks::Bool = true)
    yb  = Ytterbium174Atom(; levels = [g, e], x_init = [0.0, 0.0, 0.0],
                           v_init = [0.0, 0.0, 0.0])
    sys = System(yb)
    if planar
        cp = add_coupling!(sys, yb, g => e, Ω; active = false,
                           beam = PlanarBeam(λ, 1.0, [0.0,  1.0, 0.0], [1.0, 0, 0]))
        cm = add_coupling!(sys, yb, g => e, Ω; active = false,
                           beam = PlanarBeam(λ, 1.0, [0.0, -1.0, 0.0], [1.0, 0, 0]))
    else
        cp = add_coupling!(sys, yb, g => e, Ω; active = false)
        cm = add_coupling!(sys, yb, g => e, Ω; active = false)
    end
    pd = clicks ? PhotoDetectorSpec(name = "clicks") : nothing
    clicks && add_detector!(sys, pd)
    decay && add_decay!(sys, yb, e => g, Γ; clicks = pd, λ = λ)
    add_detector!(sys, MotionDetectorSpec(yb; dims = [1, 2, 3], name = "x"))
    sys, cp, cm
end

vy(out; s = 1) = let x = out.detectors["x"], t = out.times
    ndims(x) == 3 ? (x[end, 2, s] - x[end-1, 2, s]) / (t[end] - t[end-1]) / vrec :
                    (x[end, 2] - x[end-1, 2]) / (t[end] - t[end-1]) / vrec
end

# 1. WFMC, alternating 400 ns pulses with half-length ends (imaging), 200 shots.
function pulses(planar)
    sys, cp, cm = build(; planar = planar, Ω = Γ * sqrt(20), decay = true)
    seq = Sequence(5e-9)
    for k in 1:6
        d = (k == 1 || k == 6) ? 200e-9 : 400e-9
        push!(seq, Pulse(isodd(k) ? cp : cm, d; downsample = round(Int, d / 5e-9)))
        push!(seq, Wait(450e-9 - d; downsample = round(Int, (450e-9 - d) / 5e-9)))
    end
    push!(seq, Wait(60e-9; downsample = 12)); push!(seq, Wait(2e-9; dt = 1e-9, downsample = 1))
    () -> begin
        o = play(sys, seq; initial_state = g, shots = 200, rng = MersenneTwister(4))
        N = mean(sum(o.detectors["clicks"], dims = 1))
        mean(vy(o; s = s)^2 for s in 1:200) / N
    end
end

# 2. TDSE, 20 alternating π pulses, no decay: exactly 20 ħk.
function pipulses()
    Ω = 2π * 50e6
    sys, cp, cm = build(; planar = true, Ω = Ω, decay = false)
    seq = Sequence(1e-9)
    for k in 1:20
        push!(seq, Pulse(isodd(k) ? cp : cm, π / Ω; dt = π / Ω / 20, downsample = 20))
    end
    push!(seq, Wait(2e-9; dt = 1e-9, downsample = 1))
    () -> vy(play(sys, seq; initial_state = g))
end

# 3. QME, CW on one beam for 1 µs: mean force ħk R, R = (Γ/2) s/(1+s).
function cwdm()
    s = 40.0
    sys, cp, cm = build(; planar = true, Ω = Γ * sqrt(s / 2), decay = true, clicks = false)
    seq = Sequence(5e-9)
    push!(seq, Pulse(cp, 1e-6; downsample = 200)); push!(seq, Wait(2e-9; dt = 1e-9, downsample = 1))
    R = Γ / 2 * s / (1 + s)
    () -> vy(play(sys, seq; initial_state = g, density_matrix = true)) / (R * 1e-6)
end

r1 = bench(pulses(true))
r0 = bench(pulses(false))
r2 = bench(pipulses())
r3 = bench(cwdm())
report("4 — motion (semiclassical, plane-wave drives)", [
    ("WFMC  imaging pulses, planar, 200 shots", r1.best, r1.bytes, r1.value, "E_r/photon (ref ≈1.26)"),
    ("WFMC  same, GlobalCoupling (control)",    r0.best, r0.bytes, r0.value, "E_r/photon (emission only ≈0.33)"),
    ("TDSE  20 π pulses +k/−k",                 r2.best, r2.bytes, r2.value, "p/ħk (ref 20)"),
    ("QME   CW 1 µs, s = 40",                   r3.best, r3.bytes, r3.value, "p/(ħk R t) (ref ≈1)"),
])
