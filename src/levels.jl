"""
Specify internal levels and manifolds for simulations.
"""

# Global physical constants
const BOHR_MAGNETON_RAD_S_TESLA = 2π * 1.39962e10  # ~1.4 MHz/Gauss in rad/s

import Base: +, -, *

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
"""
struct FineLevel <: AbstractLevel
    J::Rational{Int}
    mJ::Rational{Int}
    g_J::Float64
    label::String
end

"""
    HyperfineLevel <: AbstractLevel

Represents a specific hyperfine level with quantum numbers F, mF and Landé g-factor.

# Fields
- `F::Rational{Int}`: Total angular momentum quantum number
- `J::Rational{Int}`: Orbital angular momentum quantum number
- `mF::Rational{Int}`: Magnetic quantum number  
- `g_F::Float64`: Landé g-factor for this level
- `label::String`: Human-readable label (e.g., "³P₀")
"""
struct HyperfineLevel <: AbstractLevel
    F::Rational{Int}
    J::Rational{Int}
    mF::Rational{Int}
    g_F::Float64
    label::String
end

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
# Scalar multiplication
*(a::Number, l::AbstractLevel) = Superposition(Dict(l => complex(a)))

# Level + Level returns a LevelSuperposition
+(l1::AbstractLevel, l2::AbstractLevel) =
    Superposition(Dict(l1 => 1.0, l2 => 1.0))
-(l1::AbstractLevel, l2::AbstractLevel) =
    Superposition(Dict(l1 => 1.0, l2 => -1.0))

# Level + LevelSuperposition and vice versa
+(l::AbstractLevel, s::Superposition) = Superposition(mergewith(+, Dict(l => 1.0), s.coeffs))
+(s::Superposition, l::AbstractLevel) = l + s # commutes
-(l::AbstractLevel, s::Superposition) = Superposition(mergewith(-, Dict(l => 1.0), s.coeffs))
-(s::Superposition, l::AbstractLevel) = Superposition(mergewith(-, s.coeffs, Dict(l => 1.0)))

# Addition of two superpositions
+(s1::Superposition, s2::Superposition) =
    Superposition(mergewith(+, s1.coeffs, s2.coeffs))
-(s1::Superposition, s2::Superposition) =
    Superposition(mergewith(-, s1.coeffs, s2.coeffs))


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


