# Time-optimal Rydberg Blockade Gate

In this example we simulate a two-qubit Rydberg blockade gate performed using a
shaped laser pulse following Jandura & Pupillo, Quantum 6, 712 (2022).

We use a Pulse instruction with custom defined pulse amplitude

````julia
using AtomTwin
using Printf
using Plots
````

## Parameters

````julia
Ω     = 2π * 1.0e6            # Rabi frequency (rad/s)
V     = 2π * 10.0e6           # Blockade strength

pulse_duration = 7.612e-6/2π  # Total pulse duration (s)
dt             = 1e-9         # Time step (s)
````

## System definition

Define two three-level atoms with states |0⟩, |1⟩ and |r⟩

````julia
zero, one, r = Level("0"), Level("1"), Level("r")
atoms = [Atom(; levels = [zero, one, r]) for _ in 1:2]

system = System(atoms)
````

Add a coherent resonant drive between |1⟩ and |r⟩ with Rabi frequency Ω
this is the coupling that will be modified by the time-optimal pulse shape

````julia
couplings = add_coupling!(system, atoms, one => r, Ω; active = false)
````

Add a Rydberg blockade interaction with strength V

````julia
interaction = add_interaction!(system, (atoms[1],atoms[2]), (r, r) => (r, r), V)
````

## Build sequence

We're going to plot the Rydberg population as a function of time, so lets register population detectors on |r⟩ for each atom

````julia
add_detector!(system, PopulationDetectorSpec(atoms[1], r; name = "P_r1"))
add_detector!(system, PopulationDetectorSpec(atoms[2], r; name = "P_r2"))
````

Time-optimal pulse shape from Jandura & Pupillo, Quantum 6, 712 (2022)

````julia
amplitudes = cis.(-[0,
-0.00520685612112193,
-0.0155559081588863,
-0.0308285428692611,
-0.0510421373004286,
-0.0757448285164276,
-0.104800944811622,
-0.137947539521238,
-0.174889573600685,
-0.215142997613728,
-0.258294623823603,
-0.304178101159189,
-0.352019428558297,
-0.401594848150014,
-0.45228883521041,
-0.503706714556176,
-0.555290018806423,
-0.606558953046575,
-0.656841586100419,
-0.706018505236382,
-0.753191069154603,
-0.798144574015789,
-0.840346172440943,
-0.879455302902513,
-0.915066204060227,
-0.946844618938562,
-0.974328444072949,
-0.997548987218437,
-1.01609870894535,
-1.02964270060675,
-1.03830624739873,
-1.04170068037541,
-1.03991189301145,
-1.03273476275601,
-1.02033617790978,
-1.00268600509327,
-0.979886133680442,
-0.951962901081124,
-0.919103804352448,
-0.881498486677938,
-0.839519038173743,
-0.79329814890767,
-0.743240724378473,
-0.689739845766647,
-0.633156065410953,
-0.573981695867087,
-0.512648277245333,
-0.449730092038307,
-0.385685098302023,
-0.321064299740322,
-0.256469451967428,
-0.19248736102722,
-0.129441078090281,
-0.0683019710232431,
-0.00926042041886888,
0.0475242671082795,
0.100836667323695,
0.151000933428868,
0.197191385073937,
0.239332270886053,
0.276642347994639,
0.309530918583325,
0.337616228882048,
0.360459445871596,
0.378000581352638,
0.39041393288355,
0.397416886561685,
0.399524192258912,
0.395995640837166,
0.387279007216751,
0.373673929127344,
0.35525453644305,
0.332164479176917,
0.304652616274093,
0.272856720847748,
0.237233191283938,
0.198141166542208,
0.156009259564589,
0.110989895357531,
0.0637406719662517,
0.0147896100480912,
-0.0357093980739526,
-0.0869606645257641,
-0.138580326486527,
-0.189935228970358,
-0.240724197891832,
-0.290386349074185,
-0.338279019260814,
-0.384132052601118,
-0.427272497801067,
-0.46755477276085,
-0.504407998449411,
-0.537406687035639,
-0.566619049764934,
-0.591244574420264,
-0.611262863634426,
-0.626683378226236,
-0.636988051044744,
-0.642117952429648
])
````

Build a pulse sequence with time-dependent amplitudes
`dt` is the output grid. Without it a sequence records one sample per
instruction -- 42 points here -- and the two Rydberg peaks land on
different parts of their bins, which made the plotted asymmetry come out
with the wrong sign (second peak 0.0161 HIGHER, against a converged
0.0194 lower). The waveform has 98 holds, so four output points per hold
resolves it.

````julia
seq = Sequence(pulse_duration / 392; tol = 1e-4)
@sequence seq begin
````

`interp = :constant`: this waveform is a list of hold values, exactly what
an AWG emits. Reading it as a smooth curve would round the corners the
optimiser deliberately put there.

````julia
    Pulse(couplings, pulse_duration; amplitudes=amplitudes, interp=:constant)
end
````

## Run simulations

For now we can't do two atom tomography, so we manually construct the truth table for this gate'

````julia
computational_basis = [[zero,zero], [zero,one],[one,zero],[one,one]]

println("Time-optimal Rydberg blockade gate truth table")
println("Input | Output | Probability | ⟨out|U|in⟩")
println("──────┼────────┼─────────────┼──────────────────")

for input in computational_basis
    input_str = join([i.label for i in input])

    result = play(system, seq; initial_state=input, savefinalstate=true)     # Run once per input
    final_state = result.final_states[1]

    output_str = input_str     # For CZ, the output should be the same computational state

    finalstate = getqstate(system, input)
    prob = abs2(finalstate' * final_state) # Probability to return to the input state

    amp = finalstate' * final_state # Compute matrix element |⟨input|U|input⟩|²

    @printf "%4s  |  %4s  |  %.3e  |  %s\n" input_str output_str prob amp
end
````

We can also reproduce figure 1d from the paper by looking at the time-dependent Rydberg population for two initial states

````julia
out01 = play(system, seq; initial_state = [zero,one] )
out11 = play(system, seq; initial_state = [one,one] )
````

## Plot results
Collect data

````julia
tlist = out01.times               # Time grid (s)
t_us  = tlist .* 1e6              # Convert to microseconds

P_0r = out01.detectors["P_r1"] + out01.detectors["P_r2"]
P_W = out11.detectors["P_r1"] + out11.detectors["P_r2"]
````

Create a plot showing:
  - Population in the |0r⟩ state (blue)
  - Population in the |W⟩ = 1/sqrt(2)(|1r⟩ + |r1⟩) state (orange)

````julia
plt = Plots.plot(
    t_us, P_0r;
    label      = "P_0r",
    xlabel     = "Time (μs)",
    ylabel     = "Rydberg-state population",
    title      = "Time-optimal Rydberg blockade gate",
    linewidth  = 2,
    color      = :blue,
)
Plots.plot!(plt, t_us, P_W; label= "P_W", color = :orange, linewidth=2)

plt
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

