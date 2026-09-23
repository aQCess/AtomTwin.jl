# EIT spectroscopy with dissipation

In this example we simulate electromagnetically induced transparency (EIT)
in a single three-level atom in a ladder configuration.

A weak probe and strong control field couple the ground state |g⟩ to an
excited state |e⟩ and a metastable state |r⟩. By scanning the probe detuning
and including decay and dephasing, we obtain an EIT spectrum. The EIT window
appears as reduced absorption at two-photon resonance.

````julia
using AtomTwin
using Plots
````

## Parameters

````julia
Ωp = Parameter(:Ωp, 2π * 0.1e6)     # Probe Rabi frequency (rad/s)
Ωc = Parameter(:Ωc, 2π * 5e6)       # Control Rabi frequency (rad/s)
Deltap = Parameter(:Deltap, 2π * 6e6)
Deltac = 2π * -12e6                 # Two-photon detuning offset

Gamma = 2π * 6e6    # Decay rate from |e⟩
gamma = 2π * 50e3   # Dephasing rate on |r⟩

pulse_duration = 5e-6   # Pulse duration (s)
dt             = 1e-9   # Time step (s)
````

## System definition

We construct a three-level atom with states |g⟩, |e⟩, and |r⟩, add probe
and control couplings, detunings, and include decay/dephasing channels.

````julia
g, e, r = Level("g"), Level("e"), Level("r")
atom = Atom(; levels = [g, e, r])

display(atom)

system = System(atom)
````

Probe and control couplings

````julia
coupling1 = add_coupling!(system, atom, g => e, Ωp; active = false)
coupling2 = add_coupling!(system, atom, e => r, Ωc; active = false)
````

Detunings on |e⟩ and |r⟩

````julia
add_detuning!(system, atom, e, Deltap)
add_detuning!(system, atom, r, Deltap + Deltac)
````

Decay and dephasing

````julia
add_decay!(system, atom, e => g, Gamma)
add_dephasing!(system, atom, r, gamma)

println("System Hamiltonian")
display(gethamiltonian(system))
````

## Build sequence

We monitor the probe coherence ρ_{eg} after a pulse of both probe and
control fields, for different probe detunings.

````julia
add_detector!(system, CoherenceDetectorSpec(atom, g => e; name = "rho_eg"))
````

`tol` bounds the error of one integration step; the error accumulated over
the run is larger. This value is calibrated so the global error of the
result below is about 1e-4 (measured 9.4e-05 against a tol = 1e-9 reference).

````julia
seq = Sequence(; tol = 3e-4)
@sequence seq begin
    Pulse([coupling1, coupling2], pulse_duration)
end
````

## Run simulations

````julia
delta_span = [-6:0.05:18.0...]          # Probe detuning in MHz
rhoeg  = zeros(ComplexF64, length(delta_span))
rhoeg0 = zeros(ComplexF64, length(delta_span))

for (d, delta) in enumerate(delta_span)
````

With control field (EIT)

````julia
    local out = play(
        system, seq;
        initial_state = g,
        Deltap        = 2π * delta * 1e6,
        Ωc            = 2π * 5e6,
        density_matrix = true,
    )
````

Without control field (reference absorption)

````julia
    local out0 = play(
        system, seq;
        initial_state  = g,
        Deltap         = 2π * delta * 1e6,
        Ωc             = 0.0,
        density_matrix = true,
    )

    rhoeg[d]  = out.detectors["rho_eg"][end]
    rhoeg0[d] = out0.detectors["rho_eg"][end]
end
````

## Plot results

The EIT spectrum is visualized via the imaginary part of the coherence
ρ_{eg}, which is proportional to the probe absorption, with and without
the control field.

````julia
plt = Plots.plot(
    delta_span,
    [imag.(rhoeg) imag.(rhoeg0)];
    label    = ["EIT" "Two-level"],
    xlabel   = "Probe detuning (MHz)",
    ylabel   = "Absorption (arb. units)",
    linewidth = 2,
)

plt
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

