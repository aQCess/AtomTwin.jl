"""
Specify internal levels and manifolds for simulations.
"""

# Global physical constants
const BOHR_MAGNETON_RAD_S_TESLA = 2π * 1.39962e10  # ~1.4 MHz/Gauss in rad/s

import Base: +, -, *, /

#=============================================================================
CORE TYPES
=============================================================================#
"""
    FineLevel <: AbstractLevel

Represents a specific fine-structure level with quantum numbers J, mJ, and Landé g-factor.

# Fields
- `J::Rational{Int}`: Total electronic angular momentum quantum number
- `mJ::Rational{Int}`: Magnetic quantum number
- `g_J::Float64`: Landé g-factor for this level
- `label::String`: Human-readable label (e.g., "nS₁/₂")

# Constructor
    FineLevel(J, mJ; g_J=1.0, label="")
    FineLevel(J, mJ, g_J, label)
"""
struct FineLevel <: AbstractLevel
    J::Rational{Int}
    mJ::Rational{Int}
    g_J::Float64
    label::String
end

FineLevel(J, mJ; g_J=1.0, label="") = FineLevel(J, mJ, convert(Float64, g_J), String(label))

"""
    HyperfineLevel <: AbstractLevel

Represents a specific hyperfine level with quantum numbers F, mF and Landé g-factor.

# Fields
- `F::Rational{Int}`: Total angular momentum quantum number
- `J::Rational{Int}`: Orbital angular momentum quantum number
- `mF::Rational{Int}`: Magnetic quantum number
- `g_F::Float64`: Landé g-factor for this level
- `label::String`: Human-readable label (e.g., "³P₀")

# Constructor
    HyperfineLevel(F, J, mF; g_F=1.0, label="")
    HyperfineLevel(F, J, mF, g_F, label)
"""
struct HyperfineLevel <: AbstractLevel
    F::Rational{Int}
    J::Rational{Int}
    mF::Rational{Int}
    g_F::Float64
    label::String
end

HyperfineLevel(F, J, mF; g_F=1.0, label="") = HyperfineLevel(F, J, mF, convert(Float64, g_F), String(label))

"""
    Level <: AbstractLevel

Represents a generic atomic level without hyperfine structure (e.g., leak states).

# Fields
- `label::String`: Human-readable label
"""
struct Level <: AbstractLevel
    label::String
    Level(; label="") = new(label)
    Level(label::AbstractString) = new(label)
end
Base.copy(l::Level) = Level(; label = l.label)


#=============================================================================
COMPOSITE TYPES
=============================================================================#
"""
    Superposition

Sparse linear combination of atomic levels.

`Superposition` stores a dictionary `coeffs` mapping each `AbstractLevel`
to a complex amplitude. It is constructed implicitly using arithmetic on
levels, for example `2 * ℓ1 - ℓ2` or `ℓ1 + ℓ2`.
"""
struct Superposition <: AbstractLevel
    coeffs::Dict{AbstractLevel, ComplexF64}
end

# Simple algebra on levels to build superpositions like ℓ1 + ℓ2, 2ℓ1 - ℓ2, etc.
# Scalar multiplication. A bare level becomes a one-term superposition; a
# Superposition is scaled coefficient-wise. The Superposition method must be more
# specific than the AbstractLevel fallback (Superposition <: AbstractLevel),
# otherwise `a * (ℓ1 + ℓ2)` would nest the superposition as a dict KEY instead of
# scaling it — silently producing a malformed state. See test_levels.jl.
*(a::Number, l::AbstractLevel) = Superposition(Dict(l => complex(a)))
*(a::Number, s::Superposition) =
    Superposition(Dict(l => a * c for (l, c) in s.coeffs))
*(l::AbstractLevel, a::Number) = a * l            # commute (covers Superposition too)
/(l::AbstractLevel, a::Number) = inv(a) * l       # e.g. (ℓ1 + ℓ2)/√2
-(l::AbstractLevel) = -1 * l                      # unary minus

# A bare level promotes to a one-term superposition; `+` then merges coefficients.
# Subtraction is defined as `a + (-1)*b` rather than `mergewith(-, …)`: `mergewith`
# only applies its combiner where keys COLLIDE, so `mergewith(-, …)` silently kept
# the WRONG SIGN on any non-overlapping term (e.g. 2ℓ1 - 0.5ℓ2 gave +0.5ℓ2). The
# `a + (-1)*b` form is correct for every key overlap.  See test_levels.jl.
_super(l::Superposition) = l
_super(l::AbstractLevel)  = Superposition(Dict(l => complex(1.0)))

+(l1::AbstractLevel, l2::AbstractLevel) =
    Superposition(mergewith(+, _super(l1).coeffs, _super(l2).coeffs))
-(l1::AbstractLevel, l2::AbstractLevel) = l1 + (-1) * l2


#=============================================================================
SYMBOLIC OPERATORS
=============================================================================#
"""
    Bra

Adjoint (dual) of a level or superposition, produced by `ℓ'`. A `Bra` only
exists to be paired with a ket via the outer product `ket * bra`, which builds
an [`Operator`](@ref); it carries the same coefficient dictionary as the
underlying `Superposition`.
"""
struct Bra
    coeffs::Dict{AbstractLevel, ComplexF64}
end

Base.adjoint(l::AbstractLevel) = Bra(_super(l).coeffs)
Base.adjoint(b::Bra) = Superposition(Dict(l => conj(c) for (l, c) in b.coeffs))

"""
    Operator

Symbolic, basis-free linear operator on the internal level space.

`Operator` stores a dictionary mapping each ket–bra pair `ket => bra` (both
`AbstractLevel`s) to a complex amplitude, so it represents
`∑ c |ket⟩⟨bra|`. It is a purely symbolic object — the analogue of
[`Superposition`](@ref) for operators — and is materialised into a concrete
sparse [`Op`](@ref) only when attached to a system (via `add_hamiltonian!` or an
`ExpectationDetectorSpec`), at which point levels are resolved to basis indices.

Construct one with [`transition`](@ref) / [`projector`](@ref), or directly from
the outer product of a ket and a bra, e.g. `ℓ2 * ℓ1'` for `|ℓ2⟩⟨ℓ1|`. Operators
compose with `+`, `-`, scalar `*`, and adjoint `'`:

    Sz = 0.5 * (projector(up) - projector(dn))     # ½(|↑⟩⟨↑| − |↓⟩⟨↓|)
    Sx = 0.5 * (up * dn' + dn * up')               # ½(|↑⟩⟨↓| + |↓⟩⟨↑|)
"""
struct Operator
    terms::Dict{Pair{AbstractLevel, AbstractLevel}, ComplexF64}
end

Operator() = Operator(Dict{Pair{AbstractLevel, AbstractLevel}, ComplexF64}())

"""
    projector(ℓ) -> Operator

The projector `|ℓ⟩⟨ℓ|` onto level `ℓ` as a symbolic [`Operator`](@ref).
"""
projector(l::AbstractLevel) = Operator(Dict((l => l) => complex(1.0)))

"""
    transition(ℓfrom => ℓto) -> Operator

The transition operator `|ℓto⟩⟨ℓfrom|` as a symbolic [`Operator`](@ref) — i.e.
it takes population from `ℓfrom` to `ℓto`. Its adjoint is the reverse
transition: `transition(a => b)' == transition(b => a)`.
"""
transition(p::Pair{<:AbstractLevel, <:AbstractLevel}) =
    Operator(Dict((p.second => p.first) => complex(1.0)))

# Outer product ket * bra → Operator. A bare ket is promoted to a one-term
# superposition; the tensor of the ket coeffs and (conjugated-at-construction)
# bra coeffs gives every |ket⟩⟨bra| contribution. Kept consistent with the
# `transition`/`projector` convention (key = ket => bra).
function *(ket::AbstractLevel, bra::Bra)
    terms = Dict{Pair{AbstractLevel, AbstractLevel}, ComplexF64}()
    for (lk, ck) in _super(ket).coeffs, (lb, cb) in bra.coeffs
        key = lk => lb
        terms[key] = get(terms, key, complex(0.0)) + ck * cb
    end
    return Operator(terms)
end

# Operator algebra (mirrors the Superposition rules).
*(a::Number, O::Operator) = Operator(Dict(k => a * c for (k, c) in O.terms))
*(O::Operator, a::Number) = a * O
/(O::Operator, a::Number) = inv(a) * O
-(O::Operator) = -1 * O

+(O1::Operator, O2::Operator) = Operator(mergewith(+, O1.terms, O2.terms))
-(O1::Operator, O2::Operator) = O1 + (-1) * O2

"""
    adjoint(O::Operator) -> Operator

Hermitian conjugate: `(|ket⟩⟨bra|)' = |bra⟩⟨ket|` with conjugated coefficients.
"""
Base.adjoint(O::Operator) =
    Operator(Dict((k.second => k.first) => conj(c) for (k, c) in O.terms))


#=============================================================================
MANIFOLDS
=============================================================================#

"""
    HyperfineManifold <: AbstractManifold

Container for all magnetic sublevels of a hyperfine manifold with quantum number F.

# Fields
- `F::Rational{Int}`: Total angular momentum quantum number
- `J::Rational{Int}`: Orbital angular momentum quantum number
- `label::String`: Manifold label (e.g., "³P₀", "³D₁")
- `g_F::Float64`: Landé g-factor (common for all levels in manifold)
- `Γ::Float64`: Natural linewidth in rad/s (for excited states)
- `levels::Vector{HyperfineLevel}`: All magnetic sublevels mF = -F, -F+1, ..., +F

# Constructor
    HyperfineManifold(F, J; label="", g_F=1.0, Γ=0.0)

Automatically creates all 2F+1 magnetic sublevels.
"""
struct HyperfineManifold <: AbstractManifold
    F::Rational{Int}
    J::Rational{Int}      # <-- add this field!
    label::String
    g_F::Float64
    Γ::Float64
    levels::Vector{HyperfineLevel}

    function HyperfineManifold(F, J; label="", g_F=1.0, Γ=0.0)
        levels = [HyperfineLevel(F, J, mF, g_F, label) for mF in range(-F, F, step=1)]
        new(F, J, label, g_F, Γ, levels)
    end
end


Base.length(h::HyperfineManifold) = length(h.levels)


"""
    FineManifold <: AbstractManifold

Container for all magnetic sublevels of a fine-structure manifold with quantum number J.

# Fields
- `J::Rational{Int}`: Total electronic angular momentum quantum number
- `label::String`: Manifold label (e.g., "nS₁/₂ Rydberg", "P₃/₂")
- `g_J::Float64`: Landé g-factor (common for all levels in manifold)
- `Γ::Float64`: Natural linewidth in rad/s (for excited states)
- `levels::Vector{FineLevel}`: All magnetic sublevels mJ = -J, -J+1, ..., +J

# Constructor
    FineManifold(J; label="", g_J=1.0, Γ=0.0)

Automatically creates all 2J+1 magnetic sublevels.
"""
struct FineManifold <: AbstractManifold
    J::Rational{Int}
    label::String
    g_J::Float64
    Γ::Float64
    levels::Vector{FineLevel}
    
    function FineManifold(J; label="", g_J=1.0, Γ=0.0)
        levels = [FineLevel(J, mJ, g_J, label) for mJ in range(-J, J, step=1)]
        new(J, label, g_J, Γ, levels)
    end
end

Base.length(f::FineManifold) = length(f.levels)


"""
    manifold[mF]

Access specific magnetic sublevel by mF quantum number.
"""
Base.getindex(m::HyperfineManifold, mF::Rational{Int}) = m.levels[findfirst(l -> l.mF == mF, m.levels)] #rational m
Base.getindex(m::HyperfineManifold, mF::Integer) = getindex(m, Rational{Int}(mF)) #integer m
Base.getindex(f::FineManifold, mJ::Rational{Int}) = f.levels[findfirst(l -> l.mJ == mJ, f.levels)]
Base.getindex(f::FineManifold, mJ::Integer) = getindex(f, Rational{Int}(mJ))

"""Iterator interface for manifolds (allows `for level in manifold`)"""
Base.iterate(m::AbstractManifold, state=1) = state > length(m.levels) ? nothing : (m.levels[state], state + 1)
Base.iterate(m::FineManifold, state=1) = state > length(m.levels) ? nothing : (m.levels[state], state + 1)


