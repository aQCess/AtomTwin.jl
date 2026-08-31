using .Dynamiq.Units

"""
Instantiate concrete atomic species with realistic properties.
"""

#------------------------------------------------------------------------------
# Samplers (internal)
#------------------------------------------------------------------------------

"""
    GaussianPosition

Position sampler that draws from a 3D Gaussian distribution. Each axis standard
deviation may be a `Number`, `Parameter`, or `ParametricExpression`, enabling
parametric position noise.

Implements `_resolve_node_default` (returns `zeros(3)`) and
`_resolve_node_value` (samples position from the distribution).

# Example
```julia
σ = Parameter(:σ_pos, 1e-6; std = 0.1e-6)
atom = Ytterbium171Atom(levels=[g,e], x_init = GaussianPosition(σ, σ, 0))
```
"""
struct GaussianPosition
    σx::Any
    σy::Any
    σz::Any
end

GaussianPosition(σ) = GaussianPosition(σ, σ, σ)

_resolve_node_default(::GaussianPosition) = zeros(3)

function _resolve_node_value(g::GaussianPosition, param_values, rng)
    σx = _resolve_node_value(g.σx, param_values, rng)
    σy = _resolve_node_value(g.σy, param_values, rng)
    σz = _resolve_node_value(g.σz, param_values, rng)
    return [σx, σy, σz] .* randn(rng, 3)
end

"""
    MaxwellBoltzmann

Velocity sampler that draws from a Maxwell-Boltzmann distribution at temperature
`T`. `T` may be a `Number`, `Parameter`, or `ParametricExpression`.

Implements `_resolve_node_default` (returns `zeros(3)`) and is resolved in
`initialize!` using the atom's mass so that `_resolve_node_value` has access
to `m` (required for σ = √(kB T / m)).

# Example
```julia
T = Parameter(:T_mot, 1e-6; std = 0.1e-6)  # motional temperature
atom = Ytterbium171Atom(levels=[g,e], v_init = MaxwellBoltzmann(T))
```
"""
struct MaxwellBoltzmann
    T::Any   # Number, Parameter, or ParametricExpression
end

_resolve_node_default(::MaxwellBoltzmann) = zeros(3)

"""
    gaussian(σx, σy, σz) -> GaussianPosition
    gaussian(σ)          -> GaussianPosition(σ, σ, σ)

Construct a `GaussianPosition` sampler. Arguments may be `Number`,
`Parameter`, or `ParametricExpression`.
"""
gaussian(σx, σy, σz) = GaussianPosition(σx, σy, σz)
gaussian(σ)          = GaussianPosition(σ)

"""
    maxwellboltzmann(; T) -> MaxwellBoltzmann

Construct a `MaxwellBoltzmann` velocity sampler. `T` may be a `Number`,
`Parameter`, or `ParametricExpression`.
"""
maxwellboltzmann(; T) = MaxwellBoltzmann(T)

#------------------------------------------------------------------------------
# Generic wrapper for NLevelAtom
#------------------------------------------------------------------------------

"""
    AtomWrapper{S} <: AbstractAtom

Low-level, species-parametric atom wrapper providing `NLevelAtom`
compatibility. The type parameter `S` is a `Symbol`, e.g.
`AtomWrapper{:Ytterbium171}`.

Users are encouraged to use the convenience aliases `Atom`,
`Ytterbium171Atom`, etc. instead of constructing `AtomWrapper`
directly, unless fine-grained control is needed.
"""
struct AtomWrapper{S} <: AbstractAtom
    inner::NLevelAtom
    x_init::Union{Nothing, GaussianPosition, Vector{Float64}}
    v_init::Union{Nothing, MaxwellBoltzmann, Vector{Float64}}
    levels::Vector{AbstractLevel}
    level_indices::Dict{AbstractLevel, Int}
    I::Rational{Int}
end

base_atom(wrapper::AtomWrapper) = wrapper.inner

getpolarizabilitymodels(::AtomWrapper) = Dict{String, PolarizabilityModel}()

#------------------------------------------------------------------------------
# Constructor
#------------------------------------------------------------------------------

"""
    AtomWrapper{S}(; levels=nothing, x=zeros(3), v=zeros(3),
                    x_init=nothing, v_init=nothing, 
                    mass=nothing, polarizabilities=nothing,
                    I=nothing)

Construct a parametric atomic species wrapper of type `S`, e.g.
`AtomWrapper{:Ytterbium171}()`.

Defaults for `mass`, `polarizabilities`, and `I` are taken from
`ATOM_DEFAULTS` if not provided explicitly.
"""
function AtomWrapper{S}(; levels=nothing, x=zeros(3), v=zeros(3), 
                         x_init=nothing, v_init=nothing,
                         mass=nothing, I=nothing) where S
    
    defaults = get(ATOM_DEFAULTS, S, (mass=87amu, I=1//2))
    
    m = mass === nothing ? defaults.mass : mass
    I_used = I === nothing ? defaults.I : I
    
    nlevels = levels === nothing ? 1 : length(levels)

    inner = NLevelAtom(nlevels;
                       x = Vector{Float64}(x),
                       v = Vector{Float64}(v),
                       m = m,
                       alphas  = Dict{Float64, Vector{Float64}}(),  # Empty - filled by initspeciesdata!
                       alphas2 = Dict{Float64, Vector{Float64}}(),
                       lambdas = Dict{Pair{Int,Int},Float64}())

    indices = levels === nothing ? Dict{AbstractLevel,Int}() :
               Dict(level => i for (i, level) in enumerate(levels))
    levels_vec = levels === nothing ? AbstractLevel[] : levels

    return AtomWrapper{S}(inner, x_init, v_init, levels_vec, indices, I_used)
end



############################
# Public atom types
############################

"""
    Atom

Generic neutral atom model used as the default species in AtomTwin.
It wraps an internal `NLevelAtom` with reasonable generic defaults.
"""
const Atom = AtomWrapper{:Generic}

"""
    Rubidium87Atom

Convenience alias for an Rb-87 atom with default species parameters.
"""
const Rubidium87Atom = AtomWrapper{:Rubidium87}

"""
    Strontium88Atom

Convenience alias for an Sr-88 atom with default species parameters.
"""
const Strontium88Atom = AtomWrapper{:Strontium88}

"""
    Potassium39Atom

Convenience alias for a K-39 atom with default species parameters.
"""
const Potassium39Atom = AtomWrapper{:Potassium39}

############################
# Implementation details
############################

#------------------------------------------------------------------------------
# Example species defaults (internal)
#------------------------------------------------------------------------------

"""
    ATOM_DEFAULTS

Internal dictionary of default species parameters keyed by a `Symbol`.
Each entry stores a named tuple `(mass, polarizabilities, I)` used by
`AtomWrapper{S}` when explicit values are not provided.
"""
const ATOM_DEFAULTS = Dict{Symbol, NamedTuple}(
    :Ytterbium171 => (mass = 171amu, I = 1//2),
    :Rubidium87 => (mass = 87amu, I = 3//2),
    :Strontium88 => (mass = 88amu, I = 0//1),
    :Potassium39 => (mass = 39amu, I = 3//2),
    :Generic => (mass = 87amu, I = 1//2)
)

#------------------------------------------------------------------------------
# Property forwarding
#------------------------------------------------------------------------------

Base.getproperty(a::AtomWrapper, s::Symbol) =
    s in fieldnames(typeof(a)) ? getfield(a, s) : getproperty(a.inner, s)

function Base.setproperty!(a::AtomWrapper, s::Symbol, v) 
    if s in fieldnames(typeof(a))
        setfield!(a, s, v)
    else
        setproperty!(a.inner, s, v)
    end
end

Base.convert(::Type{NLevelAtom}, a::AtomWrapper) = a.inner

#------------------------------------------------------------------------------
# Equality and interoperability
#------------------------------------------------------------------------------

Base.:(==)(a::AtomWrapper, b::NLevelAtom) = a.inner == b
Base.:(==)(a::NLevelAtom, b::AtomWrapper) = a == b.inner

Base.isequal(a::AtomWrapper, b::AbstractAtom) = isequal(a.inner, b)
Base.isequal(a::AbstractAtom, b::AtomWrapper) = isequal(a, b.inner)

Base.hash(a::AtomWrapper, h::UInt) = hash(a.inner, h)

#------------------------------------------------------------------------------
# Initialization methods
#------------------------------------------------------------------------------

"""
    initspeciesdata!(a::AtomWrapper{S}, beams) where S

Generic species data initialization. If the species defines a constant
`Symbol(S, "_POLARIZABILITY_MODELS")` as a `Dict{String, PolarizabilityModel}`,
polarizabilities are automatically computed for all beam wavelengths and all levels.

Level labels must match keys in the polarizability dictionary, otherwise the polarizability
is assumed as zero and a warning will be issued.
"""
function _init_species_data!(a::AtomWrapper, inner::NLevelAtom, beams)
    models = getpolarizabilitymodels(a)
    I = a.I
    isempty(models) && return nothing
    wavelengths = unique([getwavelength(b) for b in beams])
    for λ in wavelengths
        @info "Populating α at λ = $(round(λ, sigdigits=3)*1e9)nm" maxlog=1
        α_si = map(a.levels) do l
            if haskey(models, l.label)
                polarizability_si(models[l.label], λ * 1e9)
            else
                @warn "Polarizability model not found for level '$(l.label)'; defaulting to α = 0.0" maxlog=1
                0.0
            end
        end

        # Tensor polarizability
        α2_si = map(a.levels) do l
            if haskey(models, l.label)
                # need to get F and I values
                if l isa HyperfineLevel
                    F = l.F
                    tensor_polarizability_si(models[l.label], λ * 1e9; F=F, I=I)
                else
                    @warn "Tensor polarizability can only be assigned to Hyperfinelevel, '$(l.label)' is of type $(typeof(l)); defaulting to α2 = 0.0" maxlog=1
                    0.0
                end
                
            else
                @warn "Polarizability model not found for level '$(l.label)'; defaulting to α2 = 0.0" maxlog=1
                0.0
            end
        end
        inner.alpha[λ] = α_si
        inner.alpha2[λ] = α2_si
    end
    return nothing
end

"""
    initialize!(atom::AtomWrapper, inner::NLevelAtom; rng=Random.default_rng(), beams=AbstractBeam[])

Initialize an atom's position, velocity, and species-specific data based on its configuration.

# Arguments
- `atom::AtomWrapper`: Wrapper containing initialization configuration (position/velocity samplers or fixed values)
- `inner::NLevelAtom`: The underlying NLevelAtom object to be initialized

# Keyword Arguments
- `rng::AbstractRNG`: Random number generator for stochastic sampling (default: `Random.default_rng()`)
- `beams::Vector{AbstractBeam}`: Beams for computing wavelength-dependent polarizabilities

# Behavior
1. **Position initialization**: If `atom.x_init` is set:
   - If `PositionSampler`: draws random position from configured distribution
   - If `Vector`: sets position to fixed value
2. **Velocity initialization**: If `atom.v_init` is set:
   - If `VelocitySampler`: draws random velocity (e.g., Maxwell-Boltzmann distribution)
   - If `Vector`: sets velocity to fixed value
3. **Polarizability initialization**: For species with polarizability models, computes
   α values for all beam wavelengths and stores in `inner.alpha`

# Returns
- `inner::NLevelAtom`: The initialized atom (modified in-place and returned)

# Examples
```julia
# Initialize with thermal velocity distribution
atom = Ytterbium171Atom(
    levels = [g, e],
    v_init = maxwellboltzmann(T=1e-6)  # 1 μK
)
inner = NLevelAtom(2)
initialize!(atom, inner; beams=[tweezer_beam])

# Initialize with fixed position
atom = Atom(
    levels = [g, e],
    x_init = [0.0, 0.0, 5e-6]  # 5 μm above origin
)
initialize!(atom, inner)
"""
function initialize!(a::AtomWrapper, inner::NLevelAtom;
                     rng          = Random.default_rng(),
                     beams        = AbstractBeam[],
                     param_values = Dict{Symbol,Any}())
    # 1. position — GaussianPosition uses _resolve_node_value; Vector passes through
    if a.x_init !== nothing
        x = _resolve_node_value(a.x_init, param_values, rng)
        @assert length(x) == 3
        @inbounds for i in 1:3; inner.x[i] = Float64(x[i]); end
    end

    # 2. velocity — MaxwellBoltzmann requires mass, handled specially
    if a.v_init !== nothing
        v = if a.v_init isa MaxwellBoltzmann
            T_val = _resolve_node_value(a.v_init.T, param_values, rng)
            σ = sqrt(kb * T_val / inner.m)
            σ .* randn(rng, 3)
        else
            _resolve_node_value(a.v_init, param_values, rng)
        end
        @assert length(v) == 3
        @inbounds for i in 1:3; inner.v[i] = Float64(v[i]); end
    end

    # 3. species-specific atomic physics (fallback does nothing)
    _init_species_data!(a, inner, beams)
    return inner
end

"""
    initalize!(atom::AbstractAtom)

Convenience wrapper that initializes an AbstractAtom in place using its inner NLevelAtom.

This forwards to initialize!(::AtomWrapper, ::NLevelAtom), passing atom.inner
and the supplied beams and rng.
"""
function initialize!(atom::AbstractAtom;
                     beams = AbstractBeam[],  # or atom.beams if you store them
                     rng   = Random.default_rng())
    initialize!(atom, atom.inner; beams = beams, rng = rng)
    return atom
end


#------------------------------------------------------------------------------
# Copying
#------------------------------------------------------------------------------

"""
    copy(a::AtomWrapper)

Create a deep copy of `a`, including the wrapped `NLevelAtom` and
its configured velocity initialization.
"""
Base.copy(a::AtomWrapper{S}) where {S} = AtomWrapper{S}(
    copy(a.inner),
    a.x_init isa Vector{Float64} ? copy(a.x_init) : a.x_init,
    a.v_init isa Vector{Float64} ? copy(a.v_init) : a.v_init,
    copy(a.levels),
    copy(a.level_indices),
    a.I
)

#------------------------------------------------------------------------------
# Display
#------------------------------------------------------------------------------

# Compact 1-line form (for arrays, structs, etc.)
function atom_summary(a::AtomWrapper)
    ty = string(getspecies(a), "Atom")

    labels = String[]
    if hasfield(typeof(a), :levels)
        for lev in getfield(a, :levels)
            if hasfield(typeof(lev), :label)
                push!(labels, String(getfield(lev, :label)))
            end
        end
    end

    return isempty(labels) ? ty : string(ty, "[", join(labels, ","), "]")
end

Base.show(io::IO, a::AtomWrapper) = print(io, atom_summary(a))

function Base.show(io::IO, ::MIME"text/plain", atom::AtomWrapper)
    x = getproperty(atom.inner, :x)
    v = getproperty(atom.inner, :v)

    # Header line in AtomTwin style
    show(io, atom)            # uses the 1-line method above
    println(io)

    # Hierarchical details
    println(io, "├─ Position: ", x)
    println(io, "├─ Velocity: ", v)

    if !isempty(atom.levels)
        println(io, "└─ Levels:")
        for (i, lev) in enumerate(atom.levels)
            if lev isa HyperfineLevel
                println(io, "   $i: |$(lev.F), $(lev.mF)⟩ ($(lev.label), g_F=$(lev.g_F))")
            elseif lev isa FineLevel
                println(io, "   $i: |$(lev.J), $(lev.mJ)⟩ ($(lev.label), g_J=$(lev.g_J))")
            else
                println(io, "   $i: $(lev.label)")
            end
        end
    end
end

#------------------------------------------------------------------------------
# Helpers
#------------------------------------------------------------------------------

getspecies(::AtomWrapper{S}) where {S} = S

