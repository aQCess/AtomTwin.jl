# # Rabi oscillations with dissipation
#
# In this example we simulate driven Rabi oscillations in a two-level atom,
# including spontaneous decay and dephasing, and monitor the excited-state population.
#
# We compare two simulation methods:
#
# - Master equation (Lindblad) evolution of the density matrix.
# - Quantum trajectories (Monte Carlo wavefunction method) with many shots.


using AtomTwin
using StatsBase
using Plots      

# ## Parameters

Ω     = 2π * 0.5e6          # Rabi frequency (rad/s)
Gamma = 2π * 50e3           # Spontaneous emission rate |e⟩ → |g⟩ (rad/s)
gamma = 2π * 50e3           # Pure dephasing rate of the excited state (rad/s)

pulse_duration = 10e-6      # Total pulse duration (s)
dt             = 5e-9       # Time step (s)
shots          = 400        # Number of quantum trajectories


# ## System definition

# Define a simple two-level atom
g, e = Level("g"), Level("e")
atom = Atom(; levels = [g, e])
system = System(atom) 
 
# Add a coherent resonant drive between |g⟩ and |e⟩ with Rabi frequency Ω,
# as well as decay and dephasing of the excited state (rates Gamma and gamma respectively)
coupling = add_coupling!(system, atom, g => e, Ω; active = false) 
decay = add_decay!(system, atom, e => g, Gamma; active = true) 
deph = add_dephasing!(system, atom, e, gamma; active = true) 


# ## Build sequence
# Register population detector on |e⟩
add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e")) 

seq = Sequence(dt)
@sequence seq begin
    Pulse(coupling, pulse_duration)
end 

# ## Run simulations
out_me = play(system, seq; initial_state = g, density_matrix = true) # master equation
out_qt = play(system, seq; initial_state = g, shots = 400) # quantum trajectories
  

# ## Plot results

# Collect data
tlist = out_qt.times                 # Time grid (s)
t_us  = tlist .* 1e6                 # Convert to microseconds

P_traj = out_qt.detectors["P_e"]     # excited state populations for each trajectory
P_traj_mean = mean(P_traj, dims = 2) # average excited state population
P_me = out_me.detectors["P_e"]       # average from master equation simulation

# Create a plot showing:
#   - Individual trajectories (red, faint)
#   - Trajectory average (black)
#   - Master-equation result (blue)

plt = Plots.plot(
    t_us, P_traj;
    label      = "",
    xlabel     = "Time (μs)",
    ylabel     = "Excited-state population",
    title      = "Rabi oscillations with dissipation",
    linewidth  = 0.2,
    alpha      = 0.2,
    color      = :red,
    legend     = :bottomleft,
)

Plots.plot!(plt, t_us, P_traj_mean;
      label = "Quantum trajectories (mean)",
      color = :black,
      linewidth = 3)

Plots.plot!(plt, t_us, P_me;
      label = "Master equation",
      color = :blue,
      linewidth = 3)

plt
