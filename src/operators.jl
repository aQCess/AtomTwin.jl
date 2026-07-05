"""
Materialise symbolic [`Operator`](@ref)s onto a system basis, add them as
Hamiltonian terms, and expose the system's operators (Hamiltonian, jumps,
Liouvillian) for inspection.
"""

#=============================================================================
MATERIALISATION:  symbolic Operator  ->  concrete Op on the system basis
=============================================================================#

"""
    Op(O::Operator, sys::System, atom::AbstractAtom) -> Op

Materialise a symbolic [`Operator`](@ref) into a concrete sparse [`Op`](@ref) on
the basis of `sys`, resolving each `|ket⟩⟨bra|` term to basis indices of `atom`.

Superposition kets/bras expand into the appropriate weighted sum of single-level
outer products. The result is a general (forward-only) operator: no Hermiticity
is assumed, so `O` and `O'` materialise to genuinely different operators.
"""
function Op(O::Operator, sys::System, atom::AbstractAtom)
    dim = sys.basis.dim
    # Accumulate on the dense (index,index)->value map first; sparse at the end.
    acc = Dict{Tuple{Int,Int}, ComplexF64}()
    for ((ketlvl, bralvl), c) in O.terms
        # A term key is (ket-level => bra-level). Either endpoint may itself be a
        # Superposition; expand both into level→coefficient contributions.
        for (lk, ck) in _super(ketlvl).coeffs, (lb, cb) in _super(bralvl).coeffs
            i = _basis_index_of_level(sys, atom, lk)
            j = _basis_index_of_level(sys, atom, lb)
            coeff = c * ck * cb
            for (r, s) in _local_outer_indices(sys, atom, i, j)
                key = (r, s)
                acc[key] = get(acc, key, complex(0.0)) + coeff
            end
        end
    end
    forward = Tuple{Int,Int,ComplexF64}[(r, s, v) for ((r, s), v) in acc if v != 0]
    return Op(forward, dim)
end

# Single-atom level index within `atom`.
function _basis_index_of_level(sys::System, atom::AbstractAtom, lvl::AbstractLevel)
    idx = get(atom.level_indices, lvl, nothing)
    idx === nothing && error("Level $(lvl) not found in atom.level_indices.")
    return idx
end

# Map a single-atom outer product |i⟩⟨j| (local level indices on `atom`) to the
# many-body basis rows/cols where every OTHER atom's level is unchanged — i.e.
# the same |i⟩⟨j| ⊗ I structure the low-level `operator1` builds.
function _local_outer_indices(sys::System, atom::AbstractAtom, i::Int, j::Int)
    b = sys.basis
    atomidx = findfirst(a -> a == atom, b.atoms)
    atomidx === nothing && error("Atom not found in system basis.")
    pairs = Tuple{Int,Int}[]
    for (x, e1) in enumerate(b.elements), (y, e2) in enumerate(b.elements)
        if e1[atomidx] == i && e2[atomidx] == j &&
           Dynamiq.issame(e1, e2, except = [atomidx])
            push!(pairs, (x, y))
        end
    end
    return pairs
end

#=============================================================================
add_hamiltonian!
=============================================================================#

"""
    add_hamiltonian!(sys, atom, O::Operator; active = true) -> Hamiltonian
    add_hamiltonian!(sys, atom, H::AbstractMatrix; active = true) -> Hamiltonian

Add a Hamiltonian term written directly as an operator, rather than as a set of
`add_coupling!`/`add_detuning!` calls.

`O` is a symbolic [`Operator`](@ref) built from the atom's levels (via
[`transition`](@ref)/[`projector`](@ref) or `ket * bra'`); `H` is a dense or
sparse matrix in the atom's level order (`atom.levels[k]` ↔ row/column `k`),
useful when the textbook hands you the matrix directly (e.g. a charge-basis
transmon). Either form is materialised onto the system basis, wrapped in a
[`Hamiltonian`](@ref) node, and enters the dynamics exactly like any other term
— `gethamiltonian(sys)` includes it and the solver integrates it.

The operator must be **Hermitian** (checked here to atol `1e-8` after
materialisation); a non-Hermitian operator would give non-unitary,
trace-violating dynamics. Pass `active = false` to add it switched off (e.g. to
drive it later with a `Pulse`).

# Examples
```julia
# symbolic: one-axis-twisting term χ Ŝz² on a spin ladder
add_hamiltonian!(sys, atom, χ * Sz * Sz)

# matrix: charge-basis transmon 4E_C(n−n_g)² − (E_J/2) hopping
H = Diagonal(4EC .* (ncharge .- ng).^2) + hopping_matrix
add_hamiltonian!(sys, atom, Matrix(H))
```
"""
function add_hamiltonian!(sys::System, atom::AbstractAtom, O::Operator; active=true)
    op = Op(O, sys, atom)
    return _add_hamiltonian_op!(sys, op; active=active)
end

function add_hamiltonian!(sys::System, atom::AbstractAtom, H::AbstractMatrix; active=true)
    n = size(H, 1)
    size(H, 2) == n || error("add_hamiltonian!: matrix must be square, got $(size(H)).")
    n == atom.n ||
        error("add_hamiltonian!: matrix is $(n)×$(n) but atom has $(atom.n) levels. " *
              "Provide the operator in the atom's single-atom level order.")
    # Embed the single-atom matrix on the (possibly many-body) system basis by
    # routing each nonzero H[i,j] through the same |i⟩⟨j| ⊗ I placement.
    dim = sys.basis.dim
    forward = Tuple{Int,Int,ComplexF64}[]
    for i in 1:n, j in 1:n
        v = ComplexF64(H[i, j])
        v == 0 && continue
        for (r, s) in _local_outer_indices(sys, atom, i, j)
            push!(forward, (r, s, v))
        end
    end
    return _add_hamiltonian_op!(sys, Op(forward, dim); active=active)
end

# Shared tail: Hermiticity check + node registration.
function _add_hamiltonian_op!(sys::System, op::Op; active=true)
    A = Dynamiq.sparse(op)
    if maximum(abs.(A - A'); init = 0.0) > 1e-8
        error("add_hamiltonian!: operator is not Hermitian (‖H − H†‖∞ > 1e-8). " *
              "A non-Hermitian Hamiltonian gives non-unitary, trace-violating dynamics; " *
              "symmetrise it (e.g. add the h.c.: `O + O'`) or pass a Hermitian matrix.")
    end
    node = HamiltonianNode(op; active=active)
    build_node!(node, sys.basis)
    push!(sys, node)
    return node._field
end

#=============================================================================
ExpectationDetectorSpec from a symbolic Operator
=============================================================================#

"""
    ExpectationDetectorSpec(sys, atom, O::Operator; name = "") -> DetectorSpec

Detector recording the expectation value ⟨O⟩ of a symbolic [`Operator`](@ref)
over time. `O` is materialised onto the system basis immediately (so `sys` and
`atom` are needed here, unlike the population/coherence specs). Works for both
statevector and density-matrix runs: for a pure state it records
⟨ψ|O|ψ⟩, for a density matrix Tr(Oρ). For a Hermitian `O` the result is real up
to round-off; the stored value is complex in general.

```julia
Sz = 0.5 * (projector(up) - projector(dn))
add_detector!(sys, ExpectationDetectorSpec(sys, atom, Sz; name = "Sz"))
sz = real.(out.detectors["Sz"])
```
"""
ExpectationDetectorSpec(sys::System, atom::AbstractAtom, O::Operator; name::AbstractString = "") =
    Dynamiq.ExpectationDetectorSpec(Op(O, sys, atom); name = name)

#=============================================================================
OPERATOR ACCESSORS:  jumps, effective Hamiltonian, Liouvillian
=============================================================================#

"""
    getjumps(sys::System) -> Vector{Matrix{ComplexF64}}

Return the Lindblad jump (collapse) operators Lₖ of `sys`, one dense matrix per
`add_decay!`/`add_dephasing!` term, in the system basis. Each Lₖ already carries
its rate as `Lₖ = √γₖ |lo⟩⟨hi|` (so the dissipator is `Lₖ ρ Lₖ† − ½{Lₖ†Lₖ, ρ}`
with no extra rate factor). Companion to [`gethamiltonian`](@ref); together they
give the full open-system generator.
"""
function getjumps(sys::System)
    Ls = Matrix{ComplexF64}[]
    for node in sys.nodes
        obj = node_output(node)
        obj isa Jump || continue
        push!(Ls, Matrix(Dynamiq.sparse(obj.J) .* obj._coeff[]))
    end
    return Ls
end

"""
    getheffective(sys::System) -> Matrix{ComplexF64}

Return the non-Hermitian effective Hamiltonian of the wavefunction Monte Carlo
(MCWF) method,

    H_eff = H − (i/2) Σₖ Lₖ† Lₖ,

where `H = gethamiltonian(sys)` and the `Lₖ` are [`getjumps`](@ref). Between
jumps the MCWF trajectory evolves under `H_eff`; this exposes the operator the
solver uses instead of leaving it implicit.
"""
function getheffective(sys::System)
    H = gethamiltonian(sys)
    Heff = ComplexF64.(H)
    for L in getjumps(sys)
        Heff .-= 0.5im .* (L' * L)
    end
    return Heff
end

"""
    getliouvillian(sys::System; max_dim = 4096) -> Matrix{ComplexF64}

Return the Liouvillian superoperator 𝓛 of `sys` as a dense `d²×d²` matrix acting
on `vec(ρ)` (column-stacked), such that `vec(dρ/dt) = 𝓛 vec(ρ)` for the Lindblad
master equation

    dρ/dt = −i[H, ρ] + Σₖ ( Lₖ ρ Lₖ† − ½ {Lₖ†Lₖ, ρ} ),

with `H = gethamiltonian(sys)` and `Lₖ = getjumps(sys)`. Uses the column-stacking
identity `vec(A ρ B) = (Bᵀ ⊗ A) vec(ρ)`.

This is a pedagogy/inspection accessor, not a solver path: the returned matrix is
**dense** and scales as `d²×d²`. To avoid accidentally allocating tens of GB, the
call errors when `d² > max_dim` (default `4096`, i.e. `d ≤ 64`); raise `max_dim`
deliberately if you really want a larger one.
"""
function getliouvillian(sys::System; max_dim::Int = 4096)
    H = ComplexF64.(gethamiltonian(sys))
    d = size(H, 1)
    d2 = d * d
    if d2 > max_dim
        gib = 16 * d2^2 / 2^30   # ComplexF64 dense d²×d² in GiB
        error("getliouvillian: basis dimension d=$d gives a $(d2)×$(d2) dense " *
              "Liouvillian (~$(round(gib, digits=2)) GiB), exceeding max_dim=$max_dim. " *
              "This accessor is for small pedagogical systems; pass a larger " *
              "`max_dim` to override if you are sure.")
    end
    Id = Matrix{ComplexF64}(I, d, d)
    # Coherent part: −i(I⊗H − Hᵀ⊗I)
    Lmat = -1im .* (kron(Id, H) .- kron(transpose(H), Id))
    # Dissipators
    for L in getjumps(sys)
        LdL = L' * L
        Lmat .+= kron(conj(L), L)
        Lmat .-= 0.5 .* kron(Id, LdL)
        Lmat .-= 0.5 .* kron(transpose(LdL), Id)
    end
    return Lmat
end
