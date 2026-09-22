# # Rabi oscillations with atomic motion
#
# In this example we simulate Rabi oscillations of a single atom trapped in a
# tweezer, including its thermal motion in the trap.
#
# The atom is initialized with a Maxwell–Boltzmann velocity distribution, and
# we monitor both the internal excited–state population and the motional
# degrees of freedom during a resonant Rabi pulse.


using AtomTwin
# Explicit import: a blanket `using AtomTwin.Units` puts e, g and G into Main,
# and runtests.jl includes every file into the SAME Main — which then breaks
# every example using the `g, e = Level(...)` idiom. Everything except those
# three names is safe to bring in.
using AtomTwin.Units: MHz, kHz, GHz, Hz, s, ms, µs, ns, m, cm, mm, µm, nm, mW, µW, W, µK, mK, nK, K, hbar, kb, c, a0, amu
using Statistics
using Plots

# ## Parameters

Ω              = 2π * 0.05MHz   # Rabi frequency (rad/s)
temperature    = 5µK           # Initial temperature (K)

pulse_duration = 200µs          # Pulse duration (s)


# ## System construction
#
# We construct a simple two–level atom (|g⟩, |e⟩) and load it into a single
# tweezer site, with motional degrees of freedom determined by the trap
# parameters and the thermal velocity distribution.

# Two-level atom with thermal velocity distribution
g, e = Level("1S0"), Level("3P0")
atom = Ytterbium171Atom(;
    levels = [g, e],
    x_init = [0.0, 0.0, 0.0],
    v_init = maxwellboltzmann(T = temperature),
)

display(atom)

# Single-site tweezer array with specified geometry and powers.
#
# 759.40 nm is where the shipped Yb-171 models cross, i.e. the magic wavelength
# for ¹S₀–³P₀: both states take the same shift and the transition frequency does
# not move with the trap. The conventional round number 759 nm is 0.4 nm off that
# crossing, which leaves a 69 kHz differential shift at this depth -- larger than
# the 50 kHz Rabi frequency here, so it would visibly detune the oscillation.
# That sensitivity is the point of a magic trap, and it only became observable
# once a trapping beam started shifting the levels it traps.
tweezer = GaussianBeam(
    λ    = 759.40nm,
    w0   = 1.0µm,
    P   = 50mW
)

# Resonant planar beam driving the |g⟩ ↔ |e⟩ transition
beam = PlanarBeam(578e-9, 1.0, [1.0, 0.0, 0.0], [0, 1, 0])

# Build the full system and add a coherent coupling between |g⟩ and |e⟩
system   = System(atom, tweezer)
coupling = add_coupling!(system, atom, g => e, Ω; beam = beam, active = false)

# ## Build Sequence
#
# We measure the excited–state population as well as the atomic motion in the
# transverse directions, while applying a single resonant pulse.

add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))
add_detector!(system, MotionDetectorSpec(atom; dims = [1, 2], name = "atom"))

# `dt` is the OUTPUT grid, not the accuracy knob -- the solver picks its own
# sub-steps from `tol`. Twenty points per Rabi period resolves the
# oscillation cleanly; without a `dt` a sequence records one sample per
# instruction.
seq = Sequence(2π / (20Ω); tol = 1e-4)
@sequence seq begin
    Pulse(coupling, pulse_duration)
end

# ## Run simulations
out = play(system, seq; initial_state = g, shots = 100)

# ## Plot results
#
# The plot shows the individual quantum trajectories (red, faint) and their
# mean excited–state population as a function of time (black line).
tlist = out.times
plt = Plots.plot(
    tlist .* 1e6,
    out.detectors["P_e"],
    label     = "",
    xlabel    = "Time (μs)",
    ylabel    = "Population",
    title     = "Rabi oscillations with atomic motion",
    linewidth = 0.2,
    alpha     = 0.1,
    color     = :red,
    legend    = :none,
)

Plots.plot!(
    plt,
    tlist .* 1e6,
    mean(out.detectors["P_e"], dims = 2),
    color     = :black,
    linewidth = 3,
)

plt
