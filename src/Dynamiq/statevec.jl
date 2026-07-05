using SparseArrays
using LinearAlgebra # Diagonal

#------------------------------------------------------------------------------
# Basis for multi-atom Hilbert spaces
#------------------------------------------------------------------------------

"""
    Basis{N}

Tensor-product basis for `N` atoms.

Each basis element is an `N`-tuple of internal level indices, one per atom.
The full Hilbert-space dimension is stored in `dim`.
"""
struct Basis{N}
    atoms::NTuple{N, AbstractAtom}
    elements::Vector{NTuple{N, Int}}
    dim::Int

    """
        Basis(atoms::Vector; maxoccupations = [])

    Construct the many-body basis for a list of atoms.

    - `atoms`: vector of `AbstractAtom` objects.
    - `maxoccupations`: optional list of `(levelidx, maxocc)` pairs imposing a
      maximum number of atoms allowed in a given internal level.

    The constructor populates `atom._pidx` for each atom with index lists
    that are reused when building single-atom operators.
    """
    function Basis(atoms::Vector; maxoccupations = [])
        # @assert allunique(atoms)
        elements = [collect(Iterators.product([1:a.n for a in atoms]...))...]

        # Filter elements based on maxoccupations
        for (levelidx, maxocc) in maxoccupations
            elements = filter(n -> count(x -> x == levelidx, n) <= maxocc, elements)
        end

        # Precompute per-atom index lists for each internal level
        for (i, atom) in enumerate(atoms)
            atom._pidx = [findall(v -> v[i] == lvl, elements) for lvl in 1:atom.n]
        end

        new{length(atoms)}(Tuple(atoms), elements, length(elements))
    end

    """
        Basis(atoms::Vector, elements::Vector)

    Construct a basis explicitly from the given `elements` list.

    This variant bypasses automatic enumeration and is useful when a custom
    subspace or symmetry sector is required.
    """
    function Basis(atoms::Vector, elements::Vector)
        new{length(atoms)}(Tuple(atoms), elements, length(elements))
    end
end

#------------------------------------------------------------------------------
# Sparse operator representation
#------------------------------------------------------------------------------

"""
    Op

Sparse quantum operator with optional forward/reverse parts.

Fields:
- `forward`: nonzero entries of the "forward" operator, stored as `(row, col, value)`.
  For Hermitian operators, all diagonal elements must be stored here
  (and not duplicated in `reverse`).
- `reverse`: nonzero entries of the paired "reverse" operator, stored as `(row, col, value)`.
- `dim`: Hilbert-space dimension.

Typical usage:
- For Hermitian Hamiltonians with complex drives, store a base operator H₀ in
  `forward` and its Hermitian partner (e.g. H₀† without the diagonal) in
  `reverse`. The time-dependent Hamiltonian term is then applied as
  `c * forward + conj(c) * reverse`.
- For general non-Hermitian operators (e.g. L, L†L, H_eff), store the full
  matrix in `forward` and leave `reverse` empty, and apply it as `c * forward`.
"""
struct Op
    forward::Vector{Tuple{Int,Int,ComplexF64}}
    reverse::Vector{Tuple{Int,Int,ComplexF64}}
    dim::Int

    # Internal canonical constructor
    function Op(forward::Vector{Tuple{Int,Int,ComplexF64}},
                reverse::Vector{Tuple{Int,Int,ComplexF64}},
                dim::Int)
        new(forward, reverse, dim)
    end

    """
        Op(ijv, dim)

    Construct an operator from precomputed `(row, col, value)` triplets
    stored in `ijv`, treating them as the `forward` part of a general operator.
    """
    function Op(ijv::Vector{Tuple{Int,Int,ComplexF64}}, dim::Int)
        new(ijv, Tuple{Int,Int,ComplexF64}[], dim)
    end

    """
        Op(A::SparseMatrixCSC)

    Construct an `Op` from a sparse matrix `A`, treating `A` as a general
    (forward-only) operator.
    """
    function Op(A::SparseMatrixCSC{<:Complex,Int})
        i, j, v = findnz(A)
        forward = collect(zip(i, j, ComplexF64.(v)))
        new(forward, Tuple{Int,Int,ComplexF64}[], A.n)
    end

    """
        Op(A::AbstractMatrix{<:Complex})

    Construct an `Op` from a dense complex matrix by promoting to a sparse form.
    """
    function Op(A::AbstractMatrix{<:Complex})
        Op(sparse(A))
    end

    """
        Op(A::SparseMatrixCSC{<:Real})

    Construct an `Op` from a real sparse matrix by promoting entries to `ComplexF64`.
    """
    function Op(A::SparseMatrixCSC{<:Real,Int})
        i, j, v = findnz(A)
        forward = collect(zip(i, j, ComplexF64.(v)))
        new(forward, Tuple{Int,Int,ComplexF64}[], A.n)
    end

    """
        Op(A::AbstractMatrix{<:Real})

    Construct an `Op` from a real dense matrix by promoting to complex and
    converting to sparse format.
    """
    function Op(A::AbstractMatrix{<:Real})
        Op(sparse(ComplexF64.(A)))
    end
end

# High-level constructors analogous to the original Op2 ones,
# constructing forward/reverse parts via `operator1`/`operator2`.

"""
    Op(b::Basis, atoms::Vector{<:AbstractAtom}, transition, rate; kwargs...)

Build a single-atom operator for `transition` acting on the atoms in `atoms`
within basis `b` and with prefactor `rate`.

Additional keyword arguments are forwarded to `operator1` (e.g. `jump`,
`blockade`) to control whether the operator is treated as a collapse
operator or a Hamiltonian term.
"""
function Op(b::Basis, atoms::Vector{<:AbstractAtom},
            transition::Pair{Int,Int}, rate::Number; kwargs...)
    forward, reverse = operator1(b, atoms, transition, rate; kwargs...)
    return Op(forward, reverse, b.dim)
end

"""
    Op(b::Basis, atom::AbstractAtom, transition, rate; kwargs...)

Convenience constructor for a single-atom operator acting on `atom`
in basis `b`.
"""
function Op(b::Basis, atom::AbstractAtom,
            transition::Pair{Int,Int}, rate::Number; kwargs...)
    forward, reverse = operator1(b, [atom], transition, rate; kwargs...)
    return Op(forward, reverse, b.dim)
end

"""
    Op(b::Basis, pair, transition1, transition2, rate; kwargs...)

Two-atom operator acting on a pair of atoms `pair` with transitions
`transition1` and `transition2`, and overall prefactor `rate`.

The internal structure is generated by `operator2` and then scaled
by `rate`.
"""
function Op(b::Basis,
            pair::Pair{<:AbstractAtom,<:AbstractAtom},
            transition1::Pair{Int,Int},
            transition2::Pair{Int,Int},
            rate::Number; kwargs...)
    forward, reverse = operator2(b, pair, transition1, transition2; kwargs...)
    # Apply rate as an overall prefactor
    forward = [(i, j, rate * v) for (i, j, v) in forward]
    reverse = [(i, j, rate * u) for (i, j, u) in reverse]
    return Op(forward, reverse, b.dim)
end

"""
    sparse(op::Op)

Reconstruct the sparse matrix representation from an `Op`.

Returns the full operator `A = A_forward + A_reverse`, built from
`op.forward` and `op.reverse` triplets.
"""
function SparseArrays.sparse(op::Op)
    dim = op.dim
    A = spzeros(ComplexF64, dim, dim)

    if !isempty(op.forward)
        i = [t[1] for t in op.forward]
        j = [t[2] for t in op.forward]
        v = [t[3] for t in op.forward]
        A .+= sparse(i, j, v, dim, dim)
    end

    if !isempty(op.reverse)
        i = [t[1] for t in op.reverse]
        j = [t[2] for t in op.reverse]
        v = [t[3] for t in op.reverse]
        A .+= sparse(i, j, v, dim, dim)
    end

    return A
end

Base.Matrix(op::Op) = Matrix(sparse(op))

function Base.show(io::IO, op::Op)
    nforward = length(op.forward)
    nreverse = length(op.reverse)
    println(io,
            "$(op.dim)x$(op.dim) operator with ",
            "$nforward forward and $nreverse reverse stored element(s)")
    Base.print_matrix(io, sparse(op))
end

#------------------------------------------------------------------------------
# Basic algebra on Op
#------------------------------------------------------------------------------

import Base: +
"""
    op1 + op2

Sum of two `Op` objects, implemented by converting both to sparse matrices
and constructing a new `Op`. This is simple and robust, and can be
optimized further if needed.
"""
function +(op1::Op, op2::Op)
    # Combine both parts element-wise; cheap to write, can be optimized later
    return Op(sparse(op1) + sparse(op2))
end

import Base: *
"""
    coeff * op

Scale the full operator (forward + reverse) by a scalar `coeff`.
"""
function *(coeff::Number, op::Op)
    # Scale full operator (forward + reverse)
    return Op(coeff * sparse(op))
end

"""
    op * psi

Apply the full operator (forward + reverse) to a state vector `psi`.
"""
function *(op::Op, psi::Vector{ComplexF64})
    # Apply full operator (forward + reverse) to a state vector.
    psi2 = zero(psi)
    for (i, j, v) in op.forward
        psi2[i] += v * psi[j]
    end
    for (i, j, u) in op.reverse
        psi2[i] += u * psi[j]
    end
    return psi2
end

"""
    mul!(psi_, coeff, op, psi)

In-place application of `op` with complex prefactor `coeff`.

Applies the full operator `A = A_forward + A_reverse` such that
`psi_ = coeff * A * psi`.
"""
function mul!(psi_::Vector{ComplexF64},
              coeff::ComplexF64,
              op::Op,
              psi::Vector{ComplexF64})
    fill!(psi_, 0.0)
    for (i, j, v) in op.forward
        psi_[i] += coeff * v * psi[j]
    end
    for (i, j, u) in op.reverse
        psi_[i] += coeff * u * psi[j]
    end
end

"""
    mul!(op, psi, psi_)

In-place application of `op` to `psi` using a temporary buffer `psi_`.
"""
function mul!(op::Op, psi::Vector{ComplexF64}, psi_::Vector{ComplexF64})
    mul!(psi_, Complex(1.0), op, psi)
    psi .= psi_
end

"""
    expect(op, psi)

Expectation value ⟨psi|Op|psi⟩ for the full operator (forward + reverse).
"""
function expect(op::Op, psi::Vector{ComplexF64})
    a = 0.0 + 0.0im
    for (i, j, v) in op.forward
        a += conj(psi[i]) * v * psi[j]
    end
    for (i, j, u) in op.reverse
        a += conj(psi[i]) * u * psi[j]
    end
    return a
end

#------------------------------------------------------------------------------
# Operator builders on a Basis
#------------------------------------------------------------------------------

"""
    operator1(b, atoms, transition, rate; jump = false, blockade = 0)

Build single-atom operator triplets on basis `b`.

Returns two vectors:
- `forward::Vector{Tuple{Int,Int,ComplexF64}}`
- `reverse::Vector{Tuple{Int,Int,ComplexF64}}`

If `jump == true`, the operator is a one-way collapse operator and
`reverse` is empty. Otherwise, `forward` corresponds to the
\"raising\"/\"lowering\" direction defined by `transition`, and `reverse`
contains the Hermitian-conjugate entries. The `blockade` parameter
optionally suppresses matrix elements when too many atoms occupy a
given level.
"""
function operator1(b::Basis, atoms::Vector{<:AbstractAtom},
                   transition::Pair{Int,Int}, rate;
                   jump::Bool = false, blockade::Int = 0)
    forward = Tuple{Int,Int,ComplexF64}[]
    reverse = Tuple{Int,Int,ComplexF64}[]
    lvl1, lvl2 = transition

    for atom in atoms
        atomidx = findfirst(a -> a == atom, b.atoms)
        i = Int[]
        j = Int[]
        for (x, b1) in enumerate(b.elements), (y, b2) in enumerate(b.elements)
            if b1[atomidx] == lvl2 && b2[atomidx] == lvl1 &&
               issame(b1, b2, except = [atomidx])
                if blockade == 0 || (sum(b1 .== blockade) < 2 && sum(b2 .== blockade) < 2)
                    push!(i, x)
                    push!(j, y)
                end
            end
        end
        v = rate * ones(ComplexF64, length(i))
        if jump
            # Collapse operator L: only one direction in forward
            append!(forward, zip(i, j, v))
        else
            # Hamiltonian H: store explicit forward and reverse parts
            append!(forward, zip(i, j, v)) # H_ij
            if i != j
                append!(reverse, zip(j, i, conj.(v)))
            end
        end
    end

    return forward, reverse
end

"""
    operator2(b, atoms, transition1, transition2; jump = false)

Two-atom operator `Â₁ ⊗ Â₂` constructed from a pair of single-atom transitions,
where `Âₖ` is the raising/lowering operator of `transitionₖ` on atom `k`.

The two single-atom operators act on disjoint sites, so the two-body operator is
their product. Because each atom's forward operator has at most one entry per
column, the product's forward triplets are obtained by chaining `f1`'s columns
into `f2`'s rows. For a Hamiltonian term (`jump = false`) the `reverse` part
carries the Hermitian conjugate of the off-diagonal forward entries; a diagonal
term (e.g. the blockade projector `(r,r) => (r,r)`) therefore has empty
`reverse`, while an off-diagonal exchange term (`(0,1) => (1,0)`, i.e. σ₁⁺σ₂⁻)
gets its `+ h.c.` automatically.

Returns `(forward, reverse)` triplet lists for use in `Op(forward, reverse, b.dim)`.
"""
function operator2(b::Basis,
                   atoms::Pair{<:AbstractAtom,<:AbstractAtom},
                   transition1::Pair{Int,Int},
                   transition2::Pair{Int,Int};
                   jump::Bool = false)
    atom1, atom2 = atoms
    # Forward parts are the bare ladder operators of each transition; compose
    # them (`jump = true` keeps everything in `forward`, so this works for both).
    f1, _ = operator1(b, [atom1], transition1, 1.0, jump = true)
    f2, _ = operator1(b, [atom2], transition2, 1.0, jump = true)

    # Product Â₁·Â₂: match atom1's target column to atom2's source row.
    forward = Tuple{Int,Int,ComplexF64}[]
    for (i, j, v) in f1, (jj, k, w) in f2
        j == jj && push!(forward, (i, k, v * w))
    end

    # Hamiltonian terms carry the Hermitian conjugate; jump operators are one-way.
    reverse = Tuple{Int,Int,ComplexF64}[]
    if !jump
        for (i, j, v) in forward
            i != j && push!(reverse, (j, i, conj(v)))
        end
    end

    return forward, reverse
end

"""
    issame(b1, b2; except = [])

Check whether two basis elements `b1` and `b2` match on all sites
except those in `except`.

Used internally when constructing local operators on a many-body basis.
"""
function issame(b1, b2; except = [])
    @assert length(b1) == length(b2)
    res = true
    for idx in 1:length(b1)
        if !issubset(idx, except)
            res *= b1[idx] == b2[idx]
        end
    end
    return res
end

#------------------------------------------------------------------------------
# Product states
#------------------------------------------------------------------------------

"""
    productstate(b::Basis{N}, s::NTuple{N,Int}; rho = false)

Construct a pure product state in basis `b` with local levels `s`.

If `rho == false`, returns the state vector `|s⟩`. If `rho == true`,
returns the projector `|s⟩⟨s|` as a density matrix.
"""
function productstate(b::Basis{N}, s::NTuple{N,Int}; rho = false) where {N}
    i = findfirst(x -> x == s, b.elements)
    psi = zeros(ComplexF64, b.dim)
    psi[i] = 1
    if !rho
        return psi
    else
        return psi * psi'
    end
end

"""
    productstate(b::Basis, s...; rho = false)

Convenience wrapper for specifying a product state by a list of level
indices instead of an explicit tuple.
"""
productstate(b::Basis, s::Int...; kwargs...) = productstate(b, s; kwargs...)
