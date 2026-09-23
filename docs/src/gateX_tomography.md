# Gate X process tomography

In this example we simulate a noisy X gate implemented as a π rotation
with amplitude (Rabi-frequency) noise.

Using quantum process tomography, we reconstruct the Choi matrix of
the gate, compute the process fidelity with an ideal X gate, and
visualize the complex Choi matrix as a 3D “cityscape” plot.

````julia
using AtomTwin
using Statistics, LinearAlgebra
using GLMakie, Colors
````

## Parameters

````julia
Ω = Parameter(:Ω, 2π * 1.0e6; std = 2π * 0.2e6)   # Rabi frequency (rad/s)
pulse_duration = 0.5e-6                           # π-pulse duration (s)

dt = 1e-9                                         # Time step (s)
````

## System definition

We construct a simple two-level atom with ground state |g⟩ and excited
state |e⟩ and drive the resonant transition with a noisy Rabi frequency.

````julia
g, e = Level("g"), Level("e")
atom = Atom(; levels = [g, e])

display(atom)

system = System(atom)
coupling = add_coupling!(system, atom, g => e, Ω; active = true)
````

## Build sequence

A single π pulse is applied, and an excited-state population detector
is registered for convenience (not strictly needed for the tomography).

````julia
add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))

seq = Sequence(; tol = 1e-4)
@sequence seq begin
    Pulse(coupling, pulse_duration)
end
````

## Run tomography protocol

We perform process tomography on the single-qubit gate, compute the
Choi matrix, and evaluate the process fidelity with an ideal X gate.

````julia
J_X = [
    0 0 0 0;
    0 1 1 0;
    0 1 1 0;
    0 0 0 0
]

d  = 2
pt = process_tomography(system, seq, atom, [g, e]; shots = 1000)
fid = real(tr(pt.choi_matrix * J_X) / d^2)

@show fid
````

## Plot results

The complex Choi matrix is shown as a 3D bar plot, where the bar height
encodes the magnitude and the color encodes the phase of each element.

````julia
choi = pt.choi_matrix
n, m = size(choi)
bar_width = 0.7

fig = Figure()
ax = Axis3(
    fig[1, 1];
    xlabel = "i",
    ylabel = "j",
    zlabel = "|Element|",
    title  = "Complex Choi matrix city plot",
)

for i in 1:n, j in 1:m
    val    = choi[i, j]
    height = abs(val)
    phase  = angle(val)
    color  = HSV(mod2pi(phase) / (2π), 1.0, 1.0)
    base   = Point3f(i - bar_width / 2, j - bar_width / 2, 0f0)
    box    = Rect3f(base, Vec3f(bar_width, bar_width, Float32(height)))
    mesh!(ax, box, color = color)
end

fig
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

