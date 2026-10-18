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
    polarizabilities::Dict{String, PolarizabilityModel}   # per-atom, over the species'
end

base_atom(wrapper::AtomWrapper) = wrapper.inner

"""
    getpolarizabilitymodels(atom) -> Dict{String, PolarizabilityModel}

The polarizability models a species ships, keyed by term (`"1S0"`, `"3P1"`, …).
Empty for a generic `Atom`; a species defines its own method. Models given to an
atom's constructor (`polarizabilities = Dict(term => model)`) override these.
"""
getpolarizabilitymodels(::AtomWrapper) = Dict{String, PolarizabilityModel}()

"""
    _polarizability_models(atom) -> Dict{String, PolarizabilityModel}

The models `atom` actually uses: its species' ([`getpolarizabilitymodels`](@ref)),
overridden and extended by the ones given to its constructor
(`polarizabilities = …`).
"""
_polarizability_models(a::AtomWrapper) =
    isempty(a.polarizabilities) ? getpolarizabilitymodels(a) :
                                  merge(getpolarizabilitymodels(a), a.polarizabilities)
_polarizability_models(a::AbstractAtom) = getpolarizabilitymodels(a)

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

Defaults for `mass` and `I` are taken from `ATOM_DEFAULTS` if not provided
explicitly. `polarizabilities` — a `Dict` from term (`"1P1"` or `l"1P1"`) to
[`PolarizabilityModel`](@ref) — adds models for states the species does not ship
and overrides those it does, for this atom only; levels match them by term as
usual.
"""
function AtomWrapper{S}(; levels=nothing, x=zeros(3), v=zeros(3), 
                         x_init=nothing, v_init=nothing,
                         mass=nothing, I=nothing, polarizabilities=nothing) where S
    
    defaults = get(ATOM_DEFAULTS, S, (mass=87amu, I=1//2))
    
    m = mass === nothing ? defaults.mass : mass
    I_used = I === nothing ? defaults.I : I
    
    nlevels = levels === nothing ? 1 : length(levels)

    inner = NLevelAtom(nlevels;
                       x = Vector{Float64}(x),
                       v = Vector{Float64}(v),
                       m = m,
                       alphas = Dict{Float64, Vector{Float64}}(),  # Empty - filled by initspeciesdata!
                       lambdas = Dict{Pair{Int,Int},Float64}())

    indices = levels === nothing ? Dict{AbstractLevel,Int}() :
               Dict(level => i for (i, level) in enumerate(levels))
    levels_vec = levels === nothing ? AbstractLevel[] : levels

    pol = polarizabilities === nothing ? Dict{String, PolarizabilityModel}() :
          Dict{String, PolarizabilityModel}(termname(k) => v for (k, v) in polarizabilities)
    return AtomWrapper{S}(inner, x_init, v_init, levels_vec, indices, I_used, pol)
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
Each entry stores a named tuple `(mass, I)` used by `AtomWrapper{S}` when explicit
values are not provided.
"""
const ATOM_DEFAULTS = Dict{Symbol, NamedTuple}(
    :Ytterbium171 => (mass = 171amu, I = 1//2),
    :Ytterbium174 => (mass = 174amu, I = 0//1),
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
# Both wrappers: without this the two methods above are ambiguous and throw.
Base.isequal(a::AtomWrapper, b::AtomWrapper) = isequal(a.inner, b.inner)

Base.hash(a::AtomWrapper, h::UInt) = hash(a.inner, h)

#------------------------------------------------------------------------------
# Initialization methods
#------------------------------------------------------------------------------

"""
    _level_term(level) -> String

The polarizability lookup key for a level: its `term` if it has one, else its
`label`. A bare [`Level`](@ref) sets `term` from `label`, so `Level("1S0")` keeps
resolving as it always did.
"""
_level_term(l::AbstractLevel) = hasproperty(l, :term) ? getproperty(l, :term) :
                                hasproperty(l, :label) ? getproperty(l, :label) : ""

"""
    _beam_epsilon_z(beam, q_axis) -> Float64

`ε_z`, the projection of a beam's polarization onto the quantization axis. `1`
when the polarization lies along the axis, `0` when perpendicular. Beams with no
`pol` field are treated as polarised along the axis (`ε_z = 1`), which reproduces
their pre-existing behaviour.
"""
function _beam_epsilon_z(beam, q_axis)
    hasproperty(beam, :pol) || return 1.0
    p = getproperty(beam, :pol)
    n = sqrt(sum(abs2, p))
    n == 0 && return 1.0
    return abs(sum(conj(p[i]) * q_axis[i] for i in 1:3)) / n
end

"""
    _quantization_frame(q_axis) -> (x′, y′, z′)

The orthonormal frame magnetic sublevels are defined in: `z′` along `q_axis`, `x′`
by Gram–Schmidt from `x̂` (from `ŷ` when the axis is near `x̂`). The couplings'
spherical components and the tensor light shift share it, so both mean the same
`|m⟩`, phases included.
"""
function _quantization_frame(q_axis)
    ezp = normalize(collect(Float64, q_axis))
    ref = abs(ezp[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    exp = normalize(ref - dot(ref, ezp) * ezp)
    return exp, cross(ezp, exp), ezp
end

# A beam's unit polarisation in the quantisation frame; along the axis for a beam
# with no `pol`, as in `_beam_epsilon_z`.
function _beam_epsilon(beam, q_axis)
    hasproperty(beam, :pol) || return ComplexF64[0, 0, 1]
    p = ComplexF64.(collect(getproperty(beam, :pol)))
    n = norm(p)
    n == 0 && return ComplexF64[0, 0, 1]
    ex, ey, ez = _quantization_frame(q_axis)
    return ComplexF64[sum(ex .* p), sum(ey .* p), sum(ez .* p)] ./ n
end

const _NO_PAIRS = Tuple{Int,Int,Rational{Int},Rational{Int},Rational{Int}}[]
const _NO_OFFS  = Dict{Float64,Vector{ComplexF64}}()
const _NO_SET   = Set{Int}()

# The sublevel pairs `(i, j)`, `i < j`, the tensor light shift can mix: two levels
# of one manifold (same term and F, or same J for a nuclear-spin-free fine level)
# with 0 < |Δm| ≤ 2. Returned with that manifold's F and the two m.
function _tensor_pairs(a::AtomWrapper)
    key(l::HyperfineLevel) = (_level_term(l), l.F, l.J, l.mF)
    key(l::FineLevel)      = a.I == 0//1 ? (_level_term(l), l.J, l.J, l.mJ) : nothing
    key(l)                 = nothing
    out = _NO_PAIRS
    lv  = a.levels
    for i in eachindex(lv), j in i+1:lastindex(lv)
        ki, kj = key(lv[i]), key(lv[j])
        (ki === nothing || kj === nothing) && continue
        ki[1:3] == kj[1:3] && 0 < abs(ki[4] - kj[4]) <= 2 || continue
        out === _NO_PAIRS && (out = similar(_NO_PAIRS, 0))
        push!(out, (i, j, ki[2], ki[4], kj[4]))
    end
    return out
end

"""
    _init_species_data!(a::AtomWrapper, inner::NLevelAtom, beams; q_axis)

Fill `inner.alpha[λ]` with one polarizability per level, for every wavelength the
`beams` use. Levels are matched by [`_level_term`](@ref) to the atom's models: the
species' ([`getpolarizabilitymodels`](@ref)), overridden by any given to its
constructor (`polarizabilities`).

For a level that carries `F` and `mF`, the **tensor** contribution is folded in
here, using each beam's polarization against `q_axis`. It vanishes identically for
`J ≤ 1/2` or `F ≤ 1/2`, so every ¹S₀/³P₀ state is unaffected. Sublevels of one
manifold can therefore have different `α`.

Where the beam's polarization is not along `q_axis` (nor circular about it) the
tensor part also mixes sublevels, `Δm = ±1, ±2`: those elements go to
`inner.alpha_off[λ]`, for the level pairs `inner._cpairs`. The pairs are fixed when
`tensor_pairs = :fit` (at `compile`: the pairs mixed now) or `:all` (every pair that
can mix, for a quantization axis that varies per shot); the default `:keep`
refreshes the values of the pairs already fixed.

`α` is cached per wavelength, so beams sharing a wavelength must agree on their
polarization relative to `q_axis`; they are checked, and a disagreement is an error
rather than an arbitrary choice.

A level with no matching model gets `α = 0`. That is legitimate — a leakage level
or a Rydberg state has no model here — so it is not an error, but it IS reported
once per atom listing every unmatched term together, because the consequence is
otherwise silent: that level feels no trap.
"""
function _init_species_data!(a::AtomWrapper, inner::NLevelAtom, beams;
                             q_axis = [0.0, 0.0, 1.0], tensor_pairs::Symbol = :keep)
    models = _polarizability_models(a)
    isempty(models) && return nothing
    wavelengths = unique([getwavelength(b) for b in beams])
    # Per shot (`:keep`) with nothing mixed at compile there is nothing to refresh.
    mixing  = !(tensor_pairs === :keep && isempty(inner._cpairs))
    pairs   = mixing ? _tensor_pairs(a) : _NO_PAIRS
    none    = isempty(pairs)
    offs    = none ? _NO_OFFS : Dict{Float64,Vector{ComplexF64}}()  # per λ, per `pairs` entry
    mixable = none ? _NO_SET : Set{Int}()          # pairs with a tensor polarizability

    missing_terms = String[]
    for λ in wavelengths
        # α is stored per wavelength, but the tensor term depends on the beam's
        # own polarization. Beams sharing a wavelength must therefore agree on it —
        # a tweezer array does, being one beam replicated. Rather than silently
        # picking one, say so.
        # This behaviour may be subject to change in the future.
        same_λ = [b for b in beams if getwavelength(b) == λ]
        εs = [_beam_epsilon_z(b, q_axis) for b in same_λ]
        if !all(e -> isapprox(e, first(εs); atol = 1e-12), εs)
            error("beams at λ = $(λ*1e9) nm have different polarizations relative " *
                  "to the quantization axis (ε_z = $(round.(εs; digits=6))). The " *
                  "tensor light shift differs between them, but α is cached per " *
                  "wavelength. Give them a common polarization, or separate them " *
                  "in wavelength.")
        end
        ε_z = first(εs)
        # compute polarizability for every internal level
        α_si = map(a.levels) do l
            key = _level_term(l)
            if !haskey(models, key)
                key in missing_terms || push!(missing_terms, key)
                return 0.0
            end
            model = models[key]
            if l isa HyperfineLevel
                α = polarizability_si(model, λ * 1e9; 
                                        F = l.F, mF = l.mF, I = a.I, ε_z = ε_z)
            elseif a.I == 0//1 && l isa FineLevel
                α = polarizability_si(model, λ * 1e9; 
                                        F = l.J, mF = l.mJ, I = a.I, ε_z = ε_z)
            else
                α = polarizability_si(model, λ * 1e9)
            end
            return α
        end
        inner.alpha[λ] = α_si

        # Sublevel mixing: the tensor operator's off-diagonal elements.
        none && continue
        εs = [_beam_epsilon(b, q_axis) for b in same_λ]
        S(ε) = real(ε * ε')            # what the tensor part depends on
        all(e -> isapprox(S(e), S(first(εs)); atol = 1e-12), εs) || error(
            "beams at λ = $(λ*1e9) nm have different polarizations; the tensor light " *
            "shift differs between them, but α is cached per wavelength. Give them a " *
            "common polarization, or separate them in wavelength.")
        ε = first(εs)
        Tcache = Dict{Tuple{String,Rational{Int}},Tuple{Float64,Matrix{ComplexF64}}}()
        vals = zeros(ComplexF64, length(pairs))
        for (p, (i, j, F, mi, mj)) in enumerate(pairs)
            key = _level_term(a.levels[i])
            haskey(models, key) || continue
            α2, T = get!(Tcache, (key, F)) do
                (_alpha2_si(models[key], λ * 1e9; F = F, I = a.I), _tensor_operator(F, ε))
            end
            α2 == 0.0 && continue
            push!(mixable, p)
            t = T[Int(mi + F) + 1, Int(mj + F) + 1]
            abs(t) > 1e-12 && (vals[p] = α2 * t)      # round-off of an axis-aligned ε is no mixing
        end
        offs[λ] = vals
    end
    mixing && _set_tensor_mixing!(inner, pairs, offs, mixable, tensor_pairs)

    if !isempty(missing_terms)
        @warn """
              No polarizability model for $(length(missing_terms)) level(s) of \
              $(getspecies(a)); each takes α = 0 and feels no dipole force.
              Unmatched: $(join(map(t -> isempty(t) ? "(no term)" : "'$t'", missing_terms), ", "))
              Known: $(join(sort(collect(keys(models))), ", "))
              Set `term=` on the level or manifold if one of these was meant.\
              """
    end
    return nothing
end

# Fix (`tensor_pairs = :fit` or `:all`) or keep (`:keep`) the sublevel pairs the
# tensor light shift mixes, and store their off-diagonal α per wavelength. Fixing
# them drops the basis-state index, which `_tensor_mixing_index!` rebuilds.
function _set_tensor_mixing!(inner::NLevelAtom, pairs, offs, mixable, mode::Symbol)
    mode in (:keep, :fit, :all) || throw(ArgumentError("tensor_pairs must be :keep, :fit or :all"))
    if isempty(pairs)
        empty!(inner._cpairs); empty!(inner._cidx); empty!(inner._C); empty!(inner.alpha_off)
        return inner
    end
    if mode === :keep
        fixed = Set(inner._cpairs)
        sel = [p for p in eachindex(pairs) if (pairs[p][1], pairs[p][2]) in fixed]
        any(v -> any(p -> v[p] != 0 && !((pairs[p][1], pairs[p][2]) in fixed),
                     eachindex(pairs)), values(offs)) &&
            @warn "The tensor light shift now mixes sublevels it did not mix at compile; " *
                  "that mixing is left out. Recompile, or make the quantization axis a " *
                  "Parameter so every mixable pair is kept." maxlog = 1
    else
        sel = [p for p in eachindex(pairs)
               if mode === :all ? p in mixable : any(v -> v[p] != 0, values(offs))]
        inner._cpairs = [(pairs[p][1], pairs[p][2]) for p in sel]
        empty!(inner._cidx)
        empty!(inner._C)
    end
    empty!(inner.alpha_off)
    isempty(sel) && return inner
    for (λ, v) in offs
        inner.alpha_off[λ] = v[sel]
    end
    return inner
end

"""
    _tensor_mixing_index!(inner, basis)

The basis-state pairs `(x, y)` connected by each mixed level pair `(i, j)` of
`inner._cpairs` — `x` with the atom in `i`, `y` the same state with it in `j` —
which `updatepop!` sums into the reduced coherence `ρ_ij`.
"""
function _tensor_mixing_index!(inner::NLevelAtom, basis)
    k = findfirst(a -> a == inner, basis.atoms)          # as `operator1`
    inner._cidx = map(inner._cpairs) do (i, j)
        idx = Tuple{Int,Int}[]
        for (x, e) in enumerate(basis.elements)
            e[k] == i || continue
            y = get(basis.index, Base.setindex(e, j, k), 0)
            y == 0 || push!(idx, (x, y))
        end
        idx
    end
    inner._C = zeros(ComplexF64, length(inner._cpairs))
    return inner
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
                     param_values = Dict{Symbol,Any}(),
                     q_axis       = [0.0, 0.0, 1.0],
                     tensor_pairs = :keep)
    # 0. Drop any force cached by a previous shot. `fclassical!` (velocity Verlet)
    #    reuses the last step's force as this step's F_old, so a stale value from
    #    another shot — or from the position this atom had before re-initialising
    #    below — would corrupt the first step of the trajectory.
    Dynamiq.reset_force!(inner)
    fill!(inner._Eb, 0.0)          # nor a radiation-pressure branch (fresh internal state)

    # 1. position — GaussianPosition uses _resolve_node_value; Vector passes through.
    #    With no initializer the atom starts at the origin, and zeroing it is not
    #    optional: `inner` is reused across shots, so leaving it would start shot n
    #    where shot n-1 ended — an ensemble silently becomes one long trajectory.
    if a.x_init !== nothing
        x = _resolve_node_value(a.x_init, param_values, rng)
        @assert length(x) == 3
        @inbounds for i in 1:3; inner.x[i] = Float64(x[i]); end
    else
        @inbounds for i in 1:3; inner.x[i] = 0.0; end
    end

    # 2. velocity — MaxwellBoltzmann requires mass, handled specially. Same
    #    cross-shot reasoning as the position above.
    if a.v_init === nothing
        @inbounds for i in 1:3; inner.v[i] = 0.0; end
    else
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
    _init_species_data!(a, inner, beams; q_axis = q_axis, tensor_pairs = tensor_pairs)
    return inner
end

"""
    initalize!(atom::AbstractAtom)

Convenience wrapper that initializes an AbstractAtom in place using its inner NLevelAtom.

This forwards to initialize!(::AtomWrapper, ::NLevelAtom), passing atom.inner
and the supplied beams and rng.
"""
function initialize!(atom::AbstractAtom;
                     beams  = AbstractBeam[],  # or atom.beams if you store them
                     rng    = Random.default_rng(),
                     q_axis = [0.0, 0.0, 1.0])
    initialize!(atom, atom.inner; beams = beams, rng = rng, q_axis = q_axis)
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
    a.I,
    copy(a.polarizabilities)
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

"""
    getspecies(atom) -> Symbol

The species tag of an atom: `:Ytterbium171` for a `Ytterbium171Atom`,
`:Generic` for an `Atom`. Used to look up the species' polarizability models.
"""
getspecies(::AtomWrapper{S}) where {S} = S

