"""
Recover effective quantum channels from simulated experiments using process tomography.
"""


"""
    pauli_input_states(levels::Vector{<:AbstractLevel})

Generate the minimal set of input states for single-qubit process tomography:
`|0⟩`, `|1⟩`, `|+⟩ = (|0⟩+|1⟩)/√2`, and `|+i⟩ = (|0⟩+i|1⟩)/√2`.
Returns a Vector of `AbstractLevel` and `Superposition` instances, suitable for use in `getqstate`.
"""
function pauli_input_states(levels::Vector{<:AbstractLevel})
    zero, one = levels
    s2 = 1/sqrt(2)
    [
        zero,                              # |0⟩
        one,                               # |1⟩
        s2*zero + s2*one,                  # |+⟩
        s2*zero + im*s2*one                # |+i⟩
    ]
end

"""
    input_statevectors(system, kets; density_matrix=true)

Generate initial density matrices or statevectors (per `density_matrix`)
for each input state in `kets` by calling `getqstate(system, [ket])` for each.
Returns a Vector of statevectors or density matrices.
"""
function input_statevectors(system, kets; density_matrix=true, kwargs...)
    [getqstate(system, [ket]; density_matrix=density_matrix) for ket in kets]
end

"""
    simulate_process(sys, seq, input_states; density_matrix=nothing, shots=1, kwargs...)

Simulate the quantum process for each input state in `input_states` by calling
`play` for each, properly initializing the system for process tomography analysis.

- If `density_matrix` is not specified:
    - Uses deterministic evolution (`density_matrix = true`) when `shots == 1`.
    - Uses quantum trajectories (`density_matrix = false`) when `shots > 1`.

- If `density_matrix = true`, returns the final density matrix for each input,
  averaging over all shots if `shots > 1`.

- If `density_matrix = false`, runs quantum trajectory simulations. For `shots > 1`,
  returns the average output density matrix constructed from the projectors of each trajectory.
  For `shots == 1`, emits a warning and returns the projector of the single statevector.

Returns a vector of estimated output density matrices (one per input state).

All additional keyword arguments are forwarded to `play`.
"""
function simulate_process(sys, seq, input_states; density_matrix=nothing, shots=1, kwargs...)
    # Set default for density_matrix if not provided
    _density_matrix = isnothing(density_matrix) ? (shots == 1) : density_matrix

    n_states = length(input_states)
    output_states = Vector{Any}(undef, n_states)

    # Save sys.state[] so process_tomography does not leave sys mutated.
    saved_state = sys.state[]

    # Compile once with first state
    job = compile(sys, seq; initial_state=[input_states[1]],
                  density_matrix=_density_matrix, kwargs...)
    # Warn once if needed
    if !_density_matrix && shots == 1
        @warn "Too few shots to construct density matrix" maxlog=1
    end

    for (idx, stvec) in enumerate(input_states)
        sys.state[] = getqstate(sys, [stvec]; density_matrix=_density_matrix)
        recompile!(job, sys; kwargs...)

        result = play(job, sys; savefinalstate=true, shots=shots, kwargs...)
        states = result.final_states

        if _density_matrix
            # Average density matrices
            avg_rho = length(states) == 1 ? states[1] : sum(states) ./ shots
        else
            # Construct density matrix from state vectors
            avg_rho = sum(psi * psi' for psi in states) ./ shots
        end

        output_states[idx] = avg_rho
    end

    # Restore sys.state[] to what it was before simulate_process was called.
    sys.state[] = saved_state

    output_states
end



"""
    build_choi_matrix(input_vecs, outputs, comp_indices)

Construct the unnormalized Choi matrix for the process, restricted to the computational subspace
(indices supplied via `comp_indices`). Reconstructs the action on the operator basis
using outputs for |0⟩, |1⟩, |+⟩, |+i⟩.

Returns a complex `d^2 × d^2` matrix in the standard Choi (swap) convention:
Choi_{ik, jl} = [𝔈(|i⟩⟨j|)]_{k,l}
"""
function build_choi_matrix(input_vecs, outputs, comp_indices)
    function block(mat)
        mat[comp_indices, comp_indices]
    end
    d = 2
    ρ0, ρ1, ρp, ρpi = (block(x) for x in outputs)

    E1_out = ρ0
    E4_out = ρ1
    E2_out = (ρp - 0.5*ρ0 - 0.5*ρ1 + im*(ρpi - 0.5*ρ0 - 0.5*ρ1)) / 2
    E3_out = (ρp - 0.5*ρ0 - 0.5*ρ1 - im*(ρpi - 0.5*ρ0 - 0.5*ρ1)) / 2

    outs = [E1_out, 2*E2_out, 2*E3_out, E4_out]  # <<< Multiply off-diagonals by 2!
    choi = zeros(ComplexF64, d^2, d^2)
    for i in 1:d, j in 1:d
        Eij_out = outs[(i-1)*d + j]
        for k in 1:d, l in 1:d
            choi[(i-1)*d + k, (j-1)*d + l] = Eij_out[k, l]
        end
    end
    return choi
end

function pauli_matrices()
    I = [1.0 0.0; 0.0 1.0]
    X = [0.0 1.0; 1.0 0.0]
    Y = [0.0 -im; im 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    return [I, X, Y, Z]
end

"""
    choi_to_ptm(choi)

Convert a Choi matrix `choi` into the corresponding Pauli transfer matrix (PTM)
for a single-qubit channel, using the `{I, X, Y, Z}` basis.
"""
function choi_to_ptm(choi)
    paulis = pauli_matrices()   # expects length 4 for qubit case
    d = Int(round(sqrt(size(choi, 1))))
    if size(choi,1) != d^2 || size(choi,2) != d^2
        error("choi must be d^2 x d^2")
    end

    # --- reshuffle (Choi -> superoperator S) using permutedims ---
    # reshape choi into 4-index tensor J[i,j,k,l] with i,j,k,l in 1:d.
    J4 = reshape(choi, (d, d, d, d))          # J4[a,b,c,d] corresponds to choi linear indexing
    # Permute to get indices arranged as (i,k,j,l) so that S_{(i,k),(j,l)} = J_{i,j,k,l}
    Jperm = permutedims(J4, (1, 3, 2, 4))     # dims: (i, k, j, l)
    S = reshape(Jperm, (d^2, d^2))            # now S * vec(A) = vec(E(A))

    # --- compute PTM in the Pauli basis ---
    # normalization factor: for Pauli basis used as {I, X, Y, Z} we use 1/d
    PTM = zeros(eltype(choi), length(paulis), length(paulis))
    for j in 1:length(paulis)
        Pj_vec = paulis[j][:]  # vec uses column-major ordering (Julia default)
        Epj_vec = S * Pj_vec
        Epj = reshape(Epj_vec, (d, d))
        for i in 1:length(paulis)
            PTM[i, j] = (1 / d) * tr(paulis[i] * Epj)
        end
    end

    return PTM
end



"""
    extract_kraus_operators(choi)

Extract Kraus operators from a Choi matrix `choi` via its singular-value
decomposition, discarding numerically negligible singular values.
"""
function extract_kraus_operators(choi; tol=1e-12)
    d = trunc(Int, sqrt(size(choi,1)))
    U, S, V = svd(choi)
    [
        reshape(U[:,i], d, d) * sqrt(S[i])
        for i in 1:length(S)
        if S[i] > tol
    ]
end

"""
    process_tomography(
        sys, seq, atom, computational_levels;
        output_levels=computational_levels,
        kwargs...
    )

Perform single-qubit quantum process tomography for `atom` in system `sys`
under the pulse sequence `seq`.

The channel is characterized by simulating its action on the standard set of
input states `|0⟩`, `|1⟩`, `|+⟩`, and `|+i⟩` defined by `computational_levels`.
The dynamics are obtained using `play`, and the output is projected onto the
`output_levels` subspace.

Keyword arguments are forwarded to the underlying simulation calls, e.g. to
control density-matrix vs. trajectory evolution or the number of shots.

Returns a named tuple with fields:

  • `input_states`      – list of prepared input states in the full Hilbert space
  • `output_states`     – list of simulated output density matrices
  • `choi_matrix`       – Choi matrix of the reconstructed quantum channel
  • `ptm_matrix`        – Pauli transfer matrix (PTM) of the channel
  • `kraus_operators`   – vector of Kraus operators

Typical usage:
```@example
res = process_tomography(sys, seq, atom, [g, e]; shots=256)
res.ptm_matrix
res.kraus_operators
```
"""
function process_tomography(
        sys, seq, atom, computational_levels;
        output_levels=computational_levels,
        kwargs...
    )
    # Prepare input states (in the input subspace)
    input_kets = pauli_input_states(computational_levels)
    input_states = input_statevectors(sys, input_kets; kwargs...)

    # Simulate process: outputs projected into output subspace
    output_states = simulate_process(sys, seq, input_kets; kwargs...)

    # Indices for output computational subspace
    out_indices = [per_atom_indices(atom, lev)[1] for lev in output_levels]
    out_indices = [idx for (idx, _) in out_indices]

    # Project output density matrices into computational subspace
    function proj(mat)
        mat[out_indices, out_indices]
    end
    projected_outputs = [proj(out) for out in output_states]

    # Reconstruct output operator basis via linear combinations
    # Ordering: E1 (|0⟩⟨0|), E2 (|0⟩⟨1|), E3 (|1⟩⟨0|), E4 (|1⟩⟨1|)
    E1 = projected_outputs[1]
    E4 = projected_outputs[2]
    E2 = (projected_outputs[3] - 0.5*E1 - 0.5*E4 +
          im*(projected_outputs[4] - 0.5*E1 - 0.5*E4)) / 2
    E3 = (projected_outputs[3] - 0.5*E1 - 0.5*E4 -
          im*(projected_outputs[4] - 0.5*E1 - 0.5*E4)) / 2

    # Pack operator basis outputs (ordering matches Choi convention)
    basis_outputs = [E1, 2*E2, 2*E3, E4]

    # Build Choi matrix
    d = length(out_indices)
    choi = zeros(ComplexF64, d^2, d^2)
    for i in 1:d, j in 1:d
        Eij = basis_outputs[(i-1)*d + j]
        for k in 1:d, l in 1:d
            choi[(i-1)*d + k, (j-1)*d + l] = Eij[k, l]
        end
    end

    # Extract Kraus and PTM matrices
    kraus = extract_kraus_operators(choi)
    ptm = choi_to_ptm(choi)

    return (
        input_states = input_states,
        output_states = output_states,
        choi_matrix = choi,
        ptm_matrix = ptm,
        kraus_operators = kraus
    )
end
