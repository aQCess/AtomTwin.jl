# Two-qubit exchange (iSWAP)

Two qubits coupled by a transverse exchange (flip-flop) interaction

    Ĥ_int = (g/2) (σ₁⁺σ₂⁻ + σ₁⁻σ₂⁺),

the resonant part of a capacitive/resonator-mediated qubit–qubit coupling
(e.g. Krantz *et al.*, Appl. Phys. Rev. 6, 021318 (2019)). Starting from
|0₁1₂⟩ the single excitation coherently swaps between the qubits,

    P₁(t) = sin²(g t / 2),

reaching full transfer (a √iSWAP → iSWAP) at g t = π.

````julia
using AtomTwin
using Plots
````

## Physical parameters

````julia
g  = 2π * 1.0e6                    # exchange rate (rad/s)
dt = 1e-9                          # time step (s)
T  = 2π / g                        # one full exchange period
````

## System definition

Two identical two-level qubits with computational states |0⟩ and |1⟩. The
exchange term is added with `add_interaction!` using an *off-diagonal*
transition pair: `(0,1) => (1,0)` means qubit 1 does 0→1 (σ₁⁺) while qubit 2
does 1→0 (σ₂⁻); the Hermitian conjugate σ₁⁻σ₂⁺ is included automatically. The
prefactor `g/2` matches the Hamiltonian above (contrast the *diagonal* form
`(r,r) => (r,r)`, which builds the Rydberg-blockade projector).

````julia
l0, l1 = Level("0"), Level("1")
q1 = Atom(; levels = [l0, l1])
q2 = Atom(; levels = [l0, l1])
system = System([q1, q2])

add_interaction!(system, (q1, q2), (l0, l1) => (l1, l0), g / 2)

add_detector!(system, PopulationDetectorSpec(q1, l1; name = "P1"))
add_detector!(system, PopulationDetectorSpec(q2, l1; name = "P2"))
````

## Build sequence

Free evolution over one exchange period, starting with the excitation on
qubit 2.

An explicit `dt` here is an OUTPUT-resolution choice, not an accuracy one: the
checks below compare the whole trace against sin²(gt/2), so they need the
trace. Without it a sequence records one sample per instruction -- the final
state -- which is the right default when only `tol` is given.

````julia
seq = Sequence(T / 200; tol = 1e-4)
@sequence seq begin
    Wait(T)
end
````

## Run simulation

````julia
out = play(system, seq; initial_state = [l0, l1], density_matrix = true)

t  = out.times
P1 = out.detectors["P1"]
P2 = out.detectors["P2"]


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
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

