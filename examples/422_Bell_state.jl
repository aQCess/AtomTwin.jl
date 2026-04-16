# # [[4,2,2]] Logical Bell State — Encoding and Fidelity
#
# Prepares the logical Bell state (|00_L> + |11_L>)/sqrt(2) of the [[4,2,2]]
# quantum error-detecting code using the full encoding circuit with shuttling.
#
# ## Encoding circuit
#   H(1,2,3,4) → shuttle row 1 → CZ(1,3)·CZ(2,4) → shuttle back → H(3,4)
#   Equivalent: H(1,2) · CX(1,3) · CX(2,4)  with CX = H·CZ·H on target
#
# Input:  |0000>
# Target: (|00_L> + |11_L>)/sqrt(2) = 1/2(|0000> + |0101> + |1010> + |1111>)
#
# ## Analysis
# - Raw fidelity to logical Bell state
# - Z-stabilizer post-selection (even parity)
# - Full syndrome post-selection (code space projection)
# - Atomic trajectory plot

include("422_setup.jl")

const n_shots      = 1_000   # reduce for testing
const results_file = joinpath(@__DIR__, "422_bell_state_results.json")

# ── Build system ──────────────────────────────────────────────────────────────
# Atoms start at the storage separation (ysep) and shuttle to ysep_gate for the gate.

(; sys, ryd, atoms, tweezer) = build_system(y_start = ysep)

# Single-qubit couplings and virtual-Z detunings (not in CZ gate script)
sq  = [add_coupling!(sys, atom, qubit0 => qubit1, Ω_sq; active = false)
       for atom in atoms]
det = [add_detuning!(sys, atom, qubit1, Δ_det; active = false)
       for atom in atoms]

# Motion detectors for trajectory plot
for (i, atom) in enumerate(atoms)
    add_detector!(sys, MotionDetectorSpec(atom; dims = [1, 2], name = "atom_$i"))
end

# ── Gate definitions ──────────────────────────────────────────────────────────
# These correspond directly to paper listing 4.

# Virtual Z-rotation by phi on |1> via timed detuning
RZ(targets, phi) = [Pulse(det[targets], mod(phi, 2π) / Δ_det)]

# Hadamard: H = Rz(π/2)·Rx(π/2)·Rz(π/2) up to global phase
H(targets) = [RZ(targets, π/2)..., Pulse(sq[targets], T_pi/2), RZ(targets, π/2)...]

# X gate: π rotation around x
X(targets) = [Pulse(sq[targets], T_pi)]

# Time-optimal CZ followed by single-atom phase correction θ_cz
CZ() = [Pulse(ryd, T_gate; amplitudes = ryd_amplitudes, interp = :piecewise_constant), RZ(1:4, θ_cz)...]

# ── Encoding sequence ─────────────────────────────────────────────────────────

seq = Sequence(dt; downsample = 1000)
@sequence seq begin
    Wait(0.1µs)                                                           # tweezers settle
    H(1:4)                                                                # global Hadamard
    MoveRow(tweezer, 1,  (ysep - ysep_gate)/µm * MHz, T_move; dt = 5dt)  # shuttle into blockade
    CZ()                                                                  # entangling gate
    X(1:4)                                                                # dynamical decoupling echo
    MoveRow(tweezer, 1, -(ysep - ysep_gate)/µm * MHz, T_move; dt = 5dt)  # return row 1
    X(1:4)                                                                # complete decoupling echo
    H(3:4)                                                                # row-addressed Hadamard
end

# ── Logical basis ─────────────────────────────────────────────────────────────
# [[4,2,2]] logical codewords (arXiv:2412.07670)

ket(lvls...) = getqstate(sys, collect(lvls))

logical_basis = [
    ("00", (ket(qubit0,qubit0,qubit0,qubit0) + ket(qubit1,qubit1,qubit1,qubit1)) / sqrt(2)),
    ("01", (ket(qubit0,qubit0,qubit1,qubit1) + ket(qubit1,qubit1,qubit0,qubit0)) / sqrt(2)),
    ("10", (ket(qubit0,qubit1,qubit1,qubit0) + ket(qubit1,qubit0,qubit0,qubit1)) / sqrt(2)),
    ("11", (ket(qubit0,qubit1,qubit0,qubit1) + ket(qubit1,qubit0,qubit1,qubit0)) / sqrt(2)),
]
logical_bell = (logical_basis[1][2] + logical_basis[4][2]) / sqrt(2)

# ── Simulation (cached) ───────────────────────────────────────────────────────
# Loads cached results from JSON if available; otherwise runs the simulation.

F_raws, P_evens, code_pops, state1, traj_x, traj_y = if isfile(results_file)
    println("Loading results from $results_file")
    data = JSON3.read(read(results_file, String), Dict{String,Any})
    (Float64.(data["F_raws"]), Float64.(data["P_evens"]), Float64.(data["code_pops"]),
     complex.(Float64.(data["state1_re"]), Float64.(data["state1_im"])),
     [Float64.(v) for v in data["traj_x"]],
     [Float64.(v) for v in data["traj_y"]])
else
    println("Running Bell state encoding (n_shots = $n_shots)...")
    print("  ")
    out = play(sys, seq;
               initial_state  = [qubit0, qubit0, qubit0, qubit0],
               savefinalstate = true,
               shots          = n_shots,
               shot_callback  = (s, _) -> s % 10 == 0 && print("."))
    println()

    _F_raws = zeros(n_shots); _P_evens = zeros(n_shots); _code_pops = zeros(n_shots)
    for s in 1:n_shots
        fs            = out.final_states[s]
        _F_raws[s]    = abs2(logical_bell' * fs)
        _code_pops[s] = sum(abs2(cw' * fs) for (_, cw) in logical_basis)
        _P_evens[s]   = sum(abs2(amp)
                             for (elem, amp) in zip(sys.basis.elements, fs)
                             if all(l in (1, 2) for l in elem) && iseven(count(==(2), elem)))
    end

    _traj_x = [out.detectors["atom_$i"][:, 1, 1] for i in 1:length(atoms)]
    _traj_y = [out.detectors["atom_$i"][:, 2, 1] for i in 1:length(atoms)]

    open(results_file, "w") do io
        JSON3.write(io, Dict{String,Any}(
            "n_shots"    => n_shots,
            "F_raws"     => _F_raws,
            "P_evens"    => _P_evens,
            "code_pops"  => _code_pops,
            "state1_re"  => real.(out.final_states[1]),
            "state1_im"  => imag.(out.final_states[1]),
            "traj_x"     => _traj_x,
            "traj_y"     => _traj_y,
        ))
    end
    println("Saved results to $results_file")
    (_F_raws, _P_evens, _code_pops, out.final_states[1], _traj_x, _traj_y)
end

# ── Analysis ──────────────────────────────────────────────────────────────────

println("\nFinal state shot 1 (|amp| > 0.03):")
for (elem, amp) in zip(sys.basis.elements, state1)
    abs(amp) < 0.03 && continue
    @printf "  |%s>  %+.4f%+.4fim  (p = %.4f)\n" join(level_label[b] for b in elem) real(amp) imag(amp) abs2(amp)
end

println("\n[[4,2,2]] codeword overlaps (shot 1):")
for (lbl, codeword) in logical_basis
    @printf "  |%s_L> : %.6f\n" lbl abs2(codeword' * state1)
end
@printf "  Code space total : %.6f\n" code_pops[1]

println("\nFidelity summary (mean over $(length(F_raws)) shots)")
println("-"^45)
@printf "  Raw fidelity F(|Bell_L>)        : %.6f\n" mean(F_raws)
@printf "  P_even  (Z-stabilizer accepted) : %.6f\n" mean(P_evens)
@printf "  Post-selected fidelity          : %.6f\n" mean(F_raws ./ P_evens)
@printf "  Full syndrome fidelity          : %.6f\n" mean(F_raws ./ code_pops)

# ── Atomic trajectory plot ────────────────────────────────────────────────────

gr()
pl = Plots.plot(; xlabel = "x (µm)", ylabel = "y (µm)", aspect_ratio = :equal,
                  legend = false, title = "Atomic trajectories (shot 1)")
for i in 1:length(atoms)
    Plots.scatter!(pl, traj_x[i] * 1e6, traj_y[i] * 1e6;
                   markersize = 2, markerstrokewidth = 0, alpha = 0.5)
end
display(pl)
