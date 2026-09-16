# # Gate X process tomography
#
# In this example we simulate a noisy X gate implemented as a π rotation 
# with amplitude (Rabi-frequency) noise.
#
# Using quantum process tomography, we reconstruct the Choi matrix of
# the gate, compute the process fidelity with an ideal X gate, and
# visualize the complex Choi matrix as a 3D “cityscape” plot.

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
using Statistics, LinearAlgebra
if false    #src
using GLMakie, Colors 
end         #src

# ## Parameters

Ω = Parameter(:Ω, 2π * 1.0e6; std = 2π * 0.2e6)   # Rabi frequency (rad/s)
pulse_duration = 0.5e-6                           # π-pulse duration (s)

dt = 1e-9                                         # Time step (s)

descriptor = "GateX Tomography: Ω/2π = $(Ω.default/2π/1e6) MHz, std = $(Ω.std/2π/1e6) MHz" #src

# ## System definition
#
# We construct a simple two-level atom with ground state |g⟩ and excited
# state |e⟩ and drive the resonant transition with a noisy Rabi frequency.

g, e = Level("g"), Level("e")
atom = Atom(; levels = [g, e])

if false                                #src
display(atom)
end                                     #src

system = System(atom)
coupling = add_coupling!(system, atom, g => e, Ω; active = true)

# ## Build sequence 
#
# A single π pulse is applied, and an excited-state population detector
# is registered for convenience (not strictly needed for the tomography).

add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))

seq = Sequence(; tol = 1e-4)
@sequence seq begin
    Pulse(coupling, pulse_duration)
end

# ## Run tomography protocol
#
# We perform process tomography on the single-qubit gate, compute the
# Choi matrix, and evaluate the process fidelity with an ideal X gate.

J_X = [
    0 0 0 0;
    0 1 1 0;
    0 1 1 0;
    0 0 0 0
]

d  = 2
runtime = @elapsed begin                                    #src
pt = process_tomography(system, seq, atom, [g, e]; shots = 1000)
end                                                         #src
checksum_data = pt.choi_matrix[:]                           #src
fid = real(tr(pt.choi_matrix * J_X) / d^2)

if false                                                    #src
@show fid

# ## Plot results
#
# The complex Choi matrix is shown as a 3D bar plot, where the bar height
# encodes the magnitude and the color encodes the phase of each element.

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
end                                                         #src
