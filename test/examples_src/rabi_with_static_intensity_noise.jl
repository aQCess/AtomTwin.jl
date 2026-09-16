# # Rabi oscillations with static intensity noise
#
# In this example we simulate Rabi oscillations of a single two-level atom
# driven by a resonant field whose Rabi frequency fluctuates between shots
# due to static (shot-to-shot) intensity noise.
#
# The Rabi frequency is modeled as a random parameter with a specified mean
# value and standard deviation, and we monitor the excited-state population
# over many trajectories.

# Internal notes for test scripts (not included in docs/examples):         #src
# - Files in `test/examples_src/` are run as tests and also used to        #src
#   generate docs and runnable examples.                                   #src
# - Lines containing `#src` are removed by `make.jl` when generating       #src
#   docs/examples, but are present when running tests.                     #src
# - To omit a line from tests but keep it in docs/examples, wrap it in     #src
#   an `if false ... end` block that is itself tagged with `#src`:         #src
#                                                                          #src
#       if false            #src                                           #src
#           using Plots     # omitted in tests, kept in docs/examples      #src
#       end                 #src                                           #src

using AtomTwin
using Statistics
if false    #src
using Plots
end         #src

# ## Parameters

Ω = Parameter(:Ω, 2π * 0.5e6; std = 2π * 0.05e6)   # Rabi frequency (rad/s)

pulse_duration = 10e-6   # Total pulse duration (s)

descriptor = "Rabi with static intensity noise: Ω/2π = $(Ω.default/2π/1e6) MHz, std = $(Ω.std/2π/1e6) MHz" #src

# ## System construction
#
# We construct a simple two-level atom with states |g⟩ and |e⟩ and add a
# resonant drive with a shot-to-shot fluctuating Rabi frequency.

g, e = Level("g"), Level("e")
atom = Atom(; levels = [g, e])

if false #src
display(atom)
end #src

system   = System(atom)
coupling = add_coupling!(system, atom, g => e, Ω; active = true)

# ## Build sequence
#
# We record the excited-state population and also register a motion detector
# (useful when extending the example to include motional degrees of freedom).

add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))
add_detector!(system, MotionDetectorSpec(atom; dims = [1, 2], name = "atom"))

# `dt` is the OUTPUT grid, not the accuracy knob -- the solver picks its own
# sub-steps from `tol`. Twenty points per Rabi period resolves the
# oscillation cleanly; without a `dt` a sequence records one sample per
# instruction.
seq = Sequence(2π / (20 * Ω.default); tol = 1e-4)
@sequence seq begin
    Pulse(coupling, pulse_duration)
end

# ## Run simulations
runtime = @elapsed begin                                    #src
out = play(system, seq; initial_state = g, shots = 200)
end                                                         #src
checksum_data = out.detectors["P_e"][:,1]                   #src

# ## Plot results
#
# The plot shows individual trajectories (red, faint) and the shot-averaged
# excited-state population (black), illustrating the damping due to static
# intensity fluctuations.
if false                                                    #src
t_us = out.times .* 1e6

plt = Plots.plot(
    t_us,
    out.detectors["P_e"],
    label     = "",
    xlabel    = "Time (μs)",
    ylabel    = "Excited-state population",
    title     = "Rabi oscillations with static intensity noise",
    linewidth = 0.2,
    alpha     = 0.2,
    color     = :red,
    legend    = :none,
)

Plots.plot!(
    plt,
    t_us,
    mean(out.detectors["P_e"], dims = 2);
    color     = :black,
    linewidth = 3,
)

plt
end #src
