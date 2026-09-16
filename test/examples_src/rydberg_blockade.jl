# # Rydberg blockade
#
# In this example we simulate Rabi oscillations of two atoms under
# Rydberg blockade conditions.
#
# Two two-level atoms are driven resonantly on the ground–Rydberg transition,
# with a strong Rydberg–Rydberg interaction that shifts the doubly excited
# state out of resonance, suppressing its population.

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
if false    #src
using Plots 
end         #src

# ## Physical parameters

Ω     = 2π * 1.0e6                # Rabi frequency (rad/s)
V     = Parameter(:V, 2π * 4.0e6) # Blockade interaction strength
gamma = 2π * 0.3e6                # Dephasing rate (rad/s)

# Simulation parameters
pulse_duration = 5e-6             # Total pulse duration (s)
dt             = 1e-9             # Time step (s)

descriptor = "Rydberg blockade: Ω/2π = $(Ω/2π/1e6) MHz, V = $(V.default/2π/1e6) MHz" #src

# ## System definition
#
# We construct two identical two-level atoms with ground state |g⟩ and
# Rydberg state |r⟩, add resonant drives on each atom, and include a
# Rydberg–Rydberg interaction plus dephasing on the Rydberg level.

g, r = Level("g"), Level("r")
atom1 = Atom(; levels = [g, r])
atom2 = Atom(; levels = [g, r])

system = System([atom1, atom2])

# Coherent couplings on each atom
coupling1 = add_coupling!(system, atom1, g => r, Ω; active = false)
coupling2 = add_coupling!(system, atom2, g => r, Ω; active = false)

# Rydberg–Rydberg interaction between the two atoms
interaction = add_interaction!(system, (atom1, atom2), (r, r) => (r, r), V)

# Dephasing on the Rydberg state of each atom
add_dephasing!(system, atom1, r, gamma)
add_dephasing!(system, atom2, r, gamma)

# ## Build sequence
#
# We monitor the Rydberg population of each atom while applying a single
# resonant pulse to both atoms simultaneously.

add_detector!(system, PopulationDetectorSpec(atom1, r; name = "P_r1"))
add_detector!(system, PopulationDetectorSpec(atom2, r; name = "P_r2"))

# `dt` is the OUTPUT grid, not the accuracy knob -- the solver picks its own
# sub-steps from `tol`. Twenty points per oscillation resolves it cleanly;
# without a `dt` a sequence records one sample per instruction.
#
# Under blockade the pair oscillates at `√2 Ω`, not `Ω` -- that enhancement is
# the effect this example exists to show -- so the grid is set from the faster
# rate. Sizing it from `Ω` alone gives 14 points per period instead of 20.
seq = Sequence(2π / (20 * √2 * Ω); tol = 1e-4)
@sequence seq begin
    Pulse([coupling1, coupling2], pulse_duration)
end

# ## Run simulations
#
# We compare the blockaded case to a reference with the interaction
# turned off (V = 0), both simulated at the density-matrix level.
runtime = @elapsed begin                                    #src
out1 = play(system, seq; initial_state = [g, g], density_matrix = true)
out2 = play(system, seq; initial_state = [g, g], V = 0.0, density_matrix = true)
end                                                         #src
checksum_data = out1.detectors["P_r1"]                      #src
                                                            #src
## Validate physical correctness                           #src
# 1. All populations bounded in [0, 1]                    #src
@assert all(x -> 0.0 - 1e-6 ≤ x ≤ 1.0 + 1e-6, out1.detectors["P_r1"]) "Blockaded population out of [0,1]" #src
@assert all(x -> 0.0 - 1e-6 ≤ x ≤ 1.0 + 1e-6, out2.detectors["P_r1"]) "Unblockaded population out of [0,1]" #src
# 2. Under strong blockade (V >> Ω) the total Rydberg     #src
#    excitation is suppressed relative to the free case   #src
P_r_blockaded    = out1.detectors["P_r1"] .+ out1.detectors["P_r2"]  #src
P_r_free         = out2.detectors["P_r1"] .+ out2.detectors["P_r2"]  #src
@assert maximum(P_r_blockaded) < maximum(P_r_free) "Blockade should suppress total Rydberg excitation" #src

if false #src
tlist = out1.times
P_r  = out1.detectors["P_r1"] .+ out1.detectors["P_r2"]      # total Rydberg population
P_r0 = out2.detectors["P_r1"] .+ out2.detectors["P_r2"]      # non-blockaded reference

# ## Plot results
#
# The plot shows the total Rydberg excitation probability in the blockaded
# case (blue line) and half the non-interacting excitation (red), which
# illustrates the suppression of double excitations and collective enhancement of the
# Rabi frequency due to blockade.

plt = Plots.plot(
    tlist .* 1e6,
    [P_r, P_r0 ./ 2];
    label     = ["Blockaded" "Non-blockaded/2"],
    xlabel    = "Time (μs)",
    ylabel    = "Excitation probability",
    title     = "Rydberg blockade dynamics",
    linewidth = 2.0,
    alpha     = 1.0,
)

plt
end #src
