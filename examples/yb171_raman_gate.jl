# # Yb-171 nuclear-spin Raman gate
#
# In this example we simulate a two-photon Raman coupling between the
# nuclear spin states of the Yb-171 \\(^{3}P_{0}\\) ground state manifold,
# mediated by the \\(^{3}D_{1}\\) hyperfine structure.
#
# A pair of off-resonant Raman beams drives an effective qubit rotation
# between the ground hyperfine states, including spontaneous decay and
# leakage via the excited-state branching ratios. An effective Raman
# Rabi frequency and scattering rate are compared to the full simulation,
# and a process fidelity is extracted from quantum process tomography.


using AtomTwin
using AtomTwin.Units
using Printf
using LinearAlgebra
using Plots 

# ## Parameters

Ω_base = 2π * 50e6     # Base single-photon Rabi frequency (50 MHz)
Ω_π    = Ω_base / √2   # π-polarization strength
Ω_σ    = Ω_base / 2    # σ±-polarization strength

Δ      = -2π * 1.85e9  # One-photon detuning from ³D₁
A_hfs  = 2π * 4.5e9    # Hyperfine splitting between excited manifolds
Γ_3D1  = 2π * 6.9e6    # ³D₁ natural linewidth (~23 ns lifetime)
η_leak = 0.36          # Branching to ³P₁,₂ (leakage fraction)
B      = 0.0G          # Magnetic field in Gauss

dt      = 0.005e-9
T_total = 1.31e-6    # Total evolution time (s)


# ## System definition
#
# The full Yb-171 hyperfine structure is included, with Zeeman shifts,
# off-resonant Raman couplings via both excited manifolds, and decay to
# both the ground manifold and a leakage level.

# Hyperfine manifolds for ground and excited states
excited  = HyperfineManifold(1//2, 1; label = "³D₁", g_F = 0.667)
excited2 = HyperfineManifold(3//2, 1; label = "³D₁", g_F = 0.333)
ground   = HyperfineManifold(1//2, 0; label = "³P₀", g_F = -0.00067875)

leak = Level(; label = "³P₁")  # leakage level (effective reservoir)

# Yb-171 atom including hyperfine manifolds and leakage level
yb = Ytterbium171Atom(; levels = [ground..., excited..., excited2..., leak])
display(yb)

system = System(yb)

# Zeeman detunings for ground and excited manifolds
add_zeeman_detunings!(system, yb, ground;   B = B, delta = 0)
add_zeeman_detunings!(system, yb, excited;  B = B, delta = Δ)
add_zeeman_detunings!(system, yb, excited2; B = B, delta = Δ + A_hfs)

# Raman couplings: ground → excited, ground → excited2
couplings = vcat(
    add_coupling!(system, yb, ground => excited,  Ω_π, Ω_σ, Ω_σ; active = false),
    add_coupling!(system, yb, ground => excited2, Ω_π, Ω_σ, Ω_σ; active = false),
)

# Decay channels from excited manifolds
add_decay!(system, yb, excited  => ground, (1 - η_leak) * Γ_3D1)
add_decay!(system, yb, excited  => leak,   η_leak * Γ_3D1)
add_decay!(system, yb, excited2 => ground, (1 - η_leak) * Γ_3D1)
add_decay!(system, yb, excited2 => leak,   η_leak * Γ_3D1)

println("System Hamiltonian")
display(gethamiltonian(system))

# ## Build sequence
#
# Populations in the ground-manifold Zeeman states are monitored to track
# Raman population transfer and leakage out of the qubit subspace.

for (i, level) in enumerate(ground)
    add_detector!(system, PopulationDetectorSpec(yb, level; name = "pop_$i"))
end

seq = Sequence(dt)
@sequence seq begin
    Pulse(couplings, T_total)   # nominal π/2–like Raman pulse
end

# ## Run simulations
#
# The dynamics are simulated starting from the |↓⟩ state, and an
# effective single-qubit X gate is characterized by process tomography
# on the nuclear-spin qubit subspace.

out = play(system, seq; initial_state = ground[-1//2], density_matrix = true)

# Ideal X-gate Choi matrix (Pauli-transfer representation)
J_X = [
    0 0 0 0;
    0 1 1 0;
    0 1 1 0;
    0 0 0 0
]

d  = 2
pt = process_tomography(system, seq, yb, [ground[-1//2], ground[1//2]])
fid = abs(tr(pt.choi_matrix * J_X) / d^2)

# ## Plot results
#
# We will compare the numerical simulations with an analytical model
tlist = out.times

# Total leakage probability (one minus population in monitored ground states)
P_leak_total = 1 .- sum(p for p in values(out.detectors))

# Effective Raman Rabi frequency (two-path interference via excited manifolds)
function Ω_eff(Δ, A_hfs, Ω_π, Ω_σ)
    coupling_strength = Ω_π * Ω_σ * √(2/3) * √(1/3) * 2
    return coupling_strength / (2 * Δ) - coupling_strength / (2 * (Δ + A_hfs))
end

# Effective scattering rate from Raman beams
function G_eff(Δ, A_hfs, Ω_π, Ω_σ, Γ_3D1)
    rabi_term = Ω_π^2 + 2 * Ω_σ^2
    return Γ_3D1 * rabi_term / (4 * Δ^2 * 3) +
           2 * Γ_3D1 * rabi_term / (4 * (Δ + A_hfs)^2 * 3)
end

Ω_effective = Ω_eff(Δ, A_hfs, Ω_π, Ω_σ)
G_effective = G_eff(Δ, A_hfs, Ω_π, Ω_σ, Γ_3D1)

P_theory   = @. 0.5 + 0.5 * cos(Ω_effective * tlist)       # ideal Raman oscillation
P_th_leak  = @. η_leak * G_effective * tlist               # approximate leakage growth

# The plot shows the qubit populations, total leakage probability (scaled),
# and the simple effective-theory predictions for coherent oscillations and
# leakage versus time.

plot_data = [
    out.detectors["pop_1"],
    out.detectors["pop_2"],
    100 * P_leak_total,
    P_theory,
    100 * P_th_leak,
]

plot_labels = [
    "Ground |↓⟩",
    "Ground |↑⟩",
    "Leakage ×100",
    "Ideal theory",
    "Leak theory ×100",
]

plt = Plots.plot(
    tlist * 1e6,
    plot_data;
    label     = reshape(plot_labels, 1, :),
    xlabel    = "Time (μs)",
    ylabel    = "Population",
    title     = "Yb-171 nuclear-spin Raman",
    linewidth = 2,
    legend    = :right,
    grid      = true,
)

# ## Performance analysis
#
# From the simulated dynamics we extract the optimal π-pulse time, the
# achieved population transfer, leakage probability, effective Rabi
# frequency, and the process fidelity.

P1_max, i_pi = findmax(out.detectors["pop_2"])
π_time       = 1e6 * tlist[i_pi]
π_leak       = P_leak_total[i_pi]
π_fidelity   = P1_max

println("\nπ-pulse analysis:")
println("  Time:           " * @sprintf("%.2f", π_time) * " μs")
println("  Population:     " * @sprintf("%.4f", π_fidelity))
println("  Leakage:        " * @sprintf("%.2f", 100 * π_leak) * "%")
println("  Eff. Ω/(2π):    " * @sprintf("%.3f", Ω_effective / (2π) / 1e6) * " MHz")
println("  Fidelity:       " * @sprintf("%.4f", fid))

plt
