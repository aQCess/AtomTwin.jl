# # Two-qubit exchange (iSWAP)
#
# Two qubits coupled by a transverse exchange (flip-flop) interaction
#
#     Ĥ_int = (g/2) (σ₁⁺σ₂⁻ + σ₁⁻σ₂⁺),
#
# the resonant part of a capacitive/resonator-mediated qubit–qubit coupling
# (e.g. Krantz *et al.*, Appl. Phys. Rev. 6, 021318 (2019)). Starting from
# |0₁1₂⟩ the single excitation coherently swaps between the qubits,
#
#     P₁(t) = sin²(g t / 2),
#
# reaching full transfer (a √iSWAP → iSWAP) at g t = π.

# Internal notes for test scripts (not included in docs/examples):         #src
# - Files in `test/examples_src/` are run as tests and also used to        #src
#   generate docs and runnable examples.                                   #src
# - Lines containing `#src` are removed by `make.jl` when generating       #src
#   docs/examples, but are present when running tests.                     #src

using AtomTwin
if false    #src
using Plots
end         #src

# ## Physical parameters

g  = 2π * 1.0e6                    # exchange rate (rad/s)
dt = 1e-9                          # time step (s)
T  = 2π / g                        # one full exchange period

descriptor = "Two-qubit exchange (iSWAP): g/2π = $(g/2π/1e6) MHz" #src

# ## System definition
#
# Two identical two-level qubits with computational states |0⟩ and |1⟩. The
# exchange term is added with `add_interaction!` using an *off-diagonal*
# transition pair: `(0,1) => (1,0)` means qubit 1 does 0→1 (σ₁⁺) while qubit 2
# does 1→0 (σ₂⁻); the Hermitian conjugate σ₁⁻σ₂⁺ is included automatically. The
# prefactor `g/2` matches the Hamiltonian above (contrast the *diagonal* form
# `(r,r) => (r,r)`, which builds the Rydberg-blockade projector).

l0, l1 = Level("0"), Level("1")
q1 = Atom(; levels = [l0, l1])
q2 = Atom(; levels = [l0, l1])
system = System([q1, q2])

add_interaction!(system, (q1, q2), (l0, l1) => (l1, l0), g / 2)

add_detector!(system, PopulationDetectorSpec(q1, l1; name = "P1"))
add_detector!(system, PopulationDetectorSpec(q2, l1; name = "P2"))

# ## Build sequence
#
# Free evolution over one exchange period, starting with the excitation on
# qubit 2.

seq = Sequence(dt; downsample = 20)
@sequence seq begin
    Wait(T)
end

# ## Run simulation
runtime = @elapsed begin                                    #src
out = play(system, seq; initial_state = [l0, l1], density_matrix = true)
end                                                         #src
checksum_data = out.detectors["P1"]                         #src

t  = out.times
P1 = out.detectors["P1"]
P2 = out.detectors["P2"]

## Validate physical correctness                            #src
# 1. Populations bounded in [0, 1]                          #src
@assert all(x -> -1e-6 ≤ x ≤ 1.0 + 1e-6, P1) "P1 out of [0,1]" #src
# 2. Full excitation transfer at gt = π (a complete iSWAP)  #src
@assert maximum(P1) > 0.999 "exchange should fully transfer the excitation" #src
# 3. Matches the analytic swap P₁ = sin²(gt/2)              #src
@assert maximum(abs.(P1 .- sin.(g .* t ./ 2).^2)) < 1e-3 "exchange dynamics deviate from sin²(gt/2)" #src
# 4. Single excitation conserved: P1 + P2 ≈ 1              #src
@assert maximum(abs.(P1 .+ P2 .- 1.0)) < 1e-3 "excitation number not conserved" #src

if false #src
plt = Plots.plot(
    t .* 1e6,
    [P1 P2];
    label     = ["Qubit 1" "Qubit 2"],
    xlabel    = "Time (μs)",
    ylabel    = "Excitation probability",
    title     = "Two-qubit exchange (iSWAP)",
    linewidth = 2.0,
)
plt
end #src
