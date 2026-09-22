# Yb171 Lightshift
# 
# This is a temporary example script to demonstrate how to 
# apply the AC Stark shift Hamiltonian

using AtomTwin
using AtomTwin.Units
using Printf
using LinearAlgebra
using Plots 

# ## Parameters

Ω              = 2π * 1MHz      # Rabi frequency (rad/s)
temperature    = 5µK            # Initial temperature (K)
# For tweezer at 767nm, 40.92° is magic
p_angle        =  90.           # Angle of linear polarization axis w.r.t B (°)

pulse_duration = 10µs           # Pulse duration (s)
dt             = 1ns            # Time step (s)


B_vec = [-1.0, 0.0, 0.0]

k_tweezer = [0, 1.0, 0]         # Tweezer propagation axis
λ_tweezer = 767nm               # Tweezer wavelength (m)
waist_pos = [0., 0., 0.]        # Waist position
waist_tweezer = 1.5μm           # Waist size
P_tweezer = 5mW                 # Tweezer total power
# Polarization vector for tweezer
pol_tweezer = [-cos(p_angle * π/180), 0, sin(p_angle * π/180)]

# Green Mot transition 1S0 -> 3P1
# Has tensor light-shift
Γ_3P1 = 2π * 182.4kHz
ω_3P1 = 2π * 539.388THz



# ## Define atom
g = HyperfineLevel(0, 0, 0, 0.0, "1S0")
e = HyperfineLevel(1, 1, 0, 0.0, "3P1")

atom = Ytterbium174Atom(;
    levels = [g, e],
    x_init = [0.0, 0.0, 0.0],
    v_init = maxwellboltzmann(T = temperature),
)


# Important to define tweezer as a GeneralGaussianBeam
# GaussianBeam does not have polarisation property
tweezer = GeneralGaussianBeam(
    λ_tweezer,
    waist_tweezer,
    waist_tweezer, 
    P_tweezer, 
    k_tweezer,
    pol_tweezer;
    r0 = waist_pos
)


# ##  Build the full system
system = System(atom, tweezer)
coupling = add_coupling!(system, atom, g => e, Ω; active = false)

# Currently we need to initialize the system in order to compute the polarizability values
# This is not what we want long term
initialize!(atom; beams=[tweezer])
lightshifts = add_lightshifts!(system, active = true; q_axis = B_vec)



# ## Build Sequence
#
# We measure the excited–state population
add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))

seq = Sequence(dt)
@sequence seq begin
    Pulse(coupling, pulse_duration)
end

# ## Run simulations
out = play(system, seq; initial_state = g, shots = 100)

# ## Plot results
# The closer we are to a magic angle for 767nm, 
# the more "full" the Rabi oscillations will appear
tlist = out.times

# Expected Rabi oscillation for zero detuning
Pe_theoretical = 0.5.*(1.0 .- cos.(Ω .* tlist))

plt = plot(
    tlist .* 1e6,
    out.detectors["P_e"],
    label     = "",
    xlabel    = "Time (μs)",
    ylabel    = "Population",
    title     = "Rabi oscillations with tweezer induced lightshifts",
    linewidth = 0.2,
    alpha     = 0.1,
    color     = :red,
)

plot!(plt, tlist .* 1e6, Pe_theoretical; c="blue", alpha=0.5, label="Zero detuning")

display(plt)

# Away from the magic angle, the Rabi rate is reduced due to the effective detuning created by the lightshift
# We also observe dephasing due to the inhomogenous differential lightshift felt by the atom as it moves around the tweezer