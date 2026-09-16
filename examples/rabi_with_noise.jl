# # Rabi oscillations with laser phase noise
#
# In this example we simulate Rabi oscillations of a single two–level atom
# driven by a resonant field subject to laser phase noise.
#
# In addition to the coherent drive, a frequency–dependent phase noise
# spectrum is applied via `LaserPhaseNoiseModel`. The example shows how
# this noise damps the Rabi oscillations compared to the ideal coherent case.


using AtomTwin
using Statistics  
using Plots      

# ## Parameters and Noise
# 
# For the noise model we take a common model consisting of white noise
# plus a servo bump

Ω              = 2π * 1e6      # Rabi frequency (rad/s)
pulse_duration = 10e-6         # Total pulse duration (s)

dt             = 1e-9          # Time step (s)
shots          = 100           # Number of quantum trajectories


noise_model = LaserPhaseNoiseModel(
    bump_ampl      = 2.0e5,
    bump_center    = 1.0e6,
    bump_width     = 2e5,
    powerlaw_ampl  = 0.25e5,
)

noise_spectrum = Plots.plot(noise_model; fmin = 0.05e6, fmax = 4e6)

# ## System construction
#
# Simple two-level atom
g, e = Level("g"), Level("e")
atom = Atom(; levels = [g, e])

system = System(atom)

# Add a resonant coupling and attach the noise model
coupling = add_coupling!(
    system, atom, g => e, Ω;
    noise  = noise_model,
    active = false,
)

# ## Build Sequence
# Add detectors for the population and the field amplitude

add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))
add_detector!(system, FieldDetectorSpec(coupling; name = "coupling"))

# Build the sequence and play the simulation using quantum trajectories
# `dt` is the OUTPUT grid, not the accuracy knob -- the solver picks its own
# sub-steps from `tol`. Twenty points per Rabi period resolves the
# oscillation cleanly; without a `dt` a sequence records one sample per
# instruction.
seq = Sequence(2π / (20Ω); tol = 1e-4)
@sequence seq begin
    Pulse(coupling, pulse_duration)
end


# ## Run simulations
out = play(system, seq; initial_state = g, shots = shots)

# ## Plot results
#
# We can compare against an analytical model for the damping rate from
# X. Jiang et al., PRA 107, 042611 (2023)
tlist      = out.times                 # Time grid (s)
t_us       = tlist .* 1e6              # Convert to microseconds
P_excited  = out.detectors["P_e"]      # Trajectory-resolved populations

h0    = noise_model.powerlaw_ampl
hg    = noise_model.bump_ampl
fg    = noise_model.bump_center
sigma = noise_model.bump_width

Omega_eff = Ω

Gamma = 0.5 * (
    h0 +
    hg * exp(-(Ω/2π - fg)^2 / (2 * sigma^2)) +
    hg * exp(-(Ω/2π + fg)^2 / (2 * sigma^2))
)

# Create a plot showing:
#   - Individual noisy trajectories (red, faint)
#   - Trajectory-averaged population (black)
#   - Simple analytical damping model (blue)

plt = Plots.plot(
        t_us, P_excited;
        label      = "",
        xlabel     = "Time (μs)",
        ylabel     = "Excited-state population",
        title      = "Rabi oscillations with laser phase noise",
        linewidth  = 0.2,
        alpha      = 0.1,
        color      = :red,
        legend     = :none,
     )

Plots.plot!(plt, t_us, mean(P_excited, dims = 2);
        color = :black,
        linewidth = 3,
        label = "Quantum trajectories (mean)")

# P_e(t) = ½(1 − cos(Ωt)·e^{−Γt}) for an atom starting in |g⟩: the phase noise
# damps the oscillation toward ½ rather than decaying a population from 1.
Plots.plot!(plt, t_us, 0.5 .* (1 .- cos.(Omega_eff .* tlist) .* exp.(-Gamma .* tlist));
        color = :blue,
        linewidth = 3,
        label = "Analytical damping model")

plt
