# # K-39 Qubit Initialization Example
#
# This example demonstrates the initialization sequence of a potassium-39 neutral atom qubit
# in the magnetically insensitive states of the hyperfine ground state. 

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

# ## Parameters

Ω = 2π * 5.0e6                # Rabi frequency (rad/s)
Ωmw = 2π * 1.0e6              # Rabi frequency (rad/s)
Δ = 2π * -6.7e6

Gamma = 2π * 6.035e6          # Decay rate
B = 2.0G

pulse_duration = 0.5e-6        # seconds
dt = 1e-9

descriptor = "K-39 State Preparation: Ω/2π = $(Ω/2π/1e6) MHz, B = $(B*1e4) G" #src

# ## System definition
#
# We include all hyperfine states of the D2 transition
ground1 = HyperfineManifold(1, 1//2; label="4S_1/2, F=1", g_F=-0.5006)
ground2 = HyperfineManifold(2, 1//2; label="4S_1/2, F=2", g_F=0.5006)

excited0 = HyperfineManifold(0, 3//2; label="4P_3/2, F'=0", g_F=0.6671)
excited1 = HyperfineManifold(1, 3//2; label="4P_3/2, F'=1", g_F=0.6671)
excited2 = HyperfineManifold(2, 3//2; label="4P_3/2, F'=2", g_F=0.6671)
excited3 = HyperfineManifold(3, 3//2; label="4P_3/2, F'=3", g_F=0.6671)

atom = Potassium39Atom(; levels=[ground1..., ground2..., 
                           excited0..., excited1..., 
                           excited2..., excited3...])
if false            #src
display(atom)
end                 #src
system = System(atom)

# Add laser couplings (random polarization)
coupling23 = add_coupling!(system, atom, ground2=>excited3, Ω, Ω/2, Ω/2; active=false)
coupling22 = add_coupling!(system, atom, ground2=>excited2, Ω, Ω/2, Ω/2; active=false)
coupling21 = add_coupling!(system, atom, ground2=>excited1, Ω, Ω/2, Ω/2; active=false)
coupling12 = add_coupling!(system, atom, ground1=>excited2, Ω, Ω/2, Ω/2; active=false)
coupling11 = add_coupling!(system, atom, ground1=>excited1, Ω, Ω/2, Ω/2; active=false)
coupling10 = add_coupling!(system, atom, ground1=>excited0, Ω, Ω/2, Ω/2; active=false)

# Add microwave coupling (assuming only m_F = 0 -> m_F' = 0)
couplingmw = add_coupling!(system, atom, ground2[0]=>ground1[0], Ωmw; active=false)

# Include Zeeman shifts
add_zeeman_detunings!(system, atom, excited0; B=B, delta=2π*19.4e6+Δ)
add_zeeman_detunings!(system, atom, excited1; B=B, delta=2π*16.1e6+Δ)
add_zeeman_detunings!(system, atom, excited2; B=B, delta=2π*6.7e6+Δ)
add_zeeman_detunings!(system, atom, excited3; B=B, delta=-2π*14.4e6+Δ)

add_zeeman_detunings!(system, atom, ground1; B=B)
add_zeeman_detunings!(system, atom, ground2; B=B)

# Add spontaneous emission
add_decay!(system, atom, excited3=>ground2, Gamma)
add_decay!(system, atom, excited2=>ground1, Gamma/2)
add_decay!(system, atom, excited2=>ground2, Gamma/2)
add_decay!(system, atom, excited1=>ground2, Gamma/2)
add_decay!(system, atom, excited1=>ground1, Gamma/2)
add_decay!(system, atom, excited0=>ground1, Gamma)

# ## Build Sequence
# Monitor populations in all five F=2 Zeeman states of the ground manifold
for (i, s) in enumerate(ground2)
    add_detector!(system, PopulationDetectorSpec(atom, s; name="P_$i"))
end
  
# Here we use a repeated sequence of 20 pulses alternating between laser and mw couplings
seq = Sequence(dt)
@sequence seq begin
    for _ in 1:20
        Pulse(vcat(coupling22,coupling23), pulse_duration)
        Pulse(couplingmw, π/Ωmw) # pi pulse
        Pulse(vcat(coupling12,coupling11), pulse_duration)
        Pulse(couplingmw, π/Ωmw) # pi pulse
    end
    Pulse(couplingmw, π/Ωmw) # pi pulse
    Wait(0.5e-6)
end

# ## Run simulations
runtime = @elapsed begin                                    #src
out = play(system, seq; initial_state = ground2[-2], density_matrix=true)    #master equation
end                                                         #src
checksum_data = out.detectors["P_3"]                        #src

# ## Plot results
if false                                                    #src
times = 1e6*out.times

plt = Plots.plot(xlabel = "time (µs)", ylabel = "population")
for (k,v) in out.detectors
    Plots.plot!(plt, times, v; label=k)
end

plt
end                                                         #src
