"""
AtomTwin-layer constructor overloads for beam types.

When `GaussianBeam` or `GeneralGaussianBeam` is called with `Parameter`,
`ParametricExpression`, or `MixedPolarization` arguments, a `ParametricBeam{T}`
is returned instead of a concrete beam. The parametric beam is resolved to a
concrete instance at `play` time by the DAG node lifecycle.
"""

const _ParametricArg = Union{Parameter, ParametricExpression}

"""
    ParametricBeam{T}

Wrapper that stores the constructor arguments for a beam type `T` when one or
more arguments are `Parameter`, `ParametricExpression`, or `MixedPolarization`
objects. Resolved to a concrete beam at `compile` / `recompile!` time by the
DAG node lifecycle.
"""
struct ParametricBeam{T}
    args::Tuple
    kwargs::NamedTuple
end

"""
    MixedPolarization

A parametric polarization that models a dominant component with a small
admixture of an orthogonal contamination. Used as the `pol` keyword argument
to `GeneralGaussianBeam` to represent realistic polarization impurity.

The mixing amplitude `ε` may be a `Number`, `Parameter`, or
`ParametricExpression`, enabling parametric sweeps or Monte Carlo sampling of
the polarization extinction ratio. At compile time the polarization vector
`pol_main + ε * pol_perp` is computed and normalized.

# Example
```julia
epol = Parameter(:epol, 0.0; std = 1e-3)
pol  = MixedPolarization([0, 1im, 1]/√2, epol; k=[1,0,0])  # pol_perp = k × pol_main*
pol  = MixedPolarization([1, 0, 0], [0, 1, 0], epol)        # explicit pol_perp
beam = GeneralGaussianBeam(λ, w0x, w0y, P, [1,0,0], pol; r0=r0)
```

Future derived-value types (e.g. `BeamACStarkShift` for light-shift detunings)
follow the same pattern: a struct with `_resolve_node_default` and
`_resolve_node_value` methods.
"""
struct MixedPolarization
    pol_main::Vector{ComplexF64}
    pol_perp::Vector{ComplexF64}
    ε::Any   # Number, Parameter, or ParametricExpression
    function MixedPolarization(pol_main, pol_perp, ε)
        return new(Vector{ComplexF64}(pol_main), Vector{ComplexF64}(pol_perp), ε)
    end
end

"""
    MixedPolarization(pol_main, ε; k)

Two-argument constructor: `pol_perp` is computed as `normalize(k × pol_main*)`,
the unique polarization transverse to the beam propagation direction `k` and orthogonal
to `pol_main` (`pol_perp* · pol_main = 0`). Maxwell's equations require `pol ⊥ k`, so
both lie in the transverse plane. The conjugate matters only for complex `pol_main`:
`k × pol_main` is parallel to a circular `pol_main`, while `k × pol_main*` is the
opposite helicity, so σ⁺ is contaminated by σ⁻.

# Example
```julia
epol = Parameter(:epol, 0.0; std = 1e-3)
pol  = MixedPolarization([0, 1im, 1]/√2, epol; k = [1, 0, 0])
beam = GeneralGaussianBeam(λ, w0x, w0y, P, [1,0,0], pol; r0=r0)
```
"""
function MixedPolarization(pol_main, ε; k)
    k_hat = normalize(Float64.(k))
    pol_perp = normalize(cross(k_hat, conj.(Vector{ComplexF64}(pol_main))))
    MixedPolarization(normalize(Vector{ComplexF64}(pol_main)), pol_perp, ε)
end

function _mixed_pol_vec(pol_main, pol_perp, ε_val)
    v = pol_main .+ ComplexF64(ε_val) .* pol_perp
    return v ./ norm(v)
end

function _resolve_node_default(mp::MixedPolarization)
    _mixed_pol_vec(mp.pol_main, mp.pol_perp, _resolve_node_default(mp.ε))
end

function _resolve_node_value(mp::MixedPolarization, param_values, rng)
    _mixed_pol_vec(mp.pol_main, mp.pol_perp, _resolve_node_value(mp.ε, param_values, rng))
end

#=============================================================================
PER ESTIMATION
=============================================================================#

"""
    estimate_PER(mp::MixedPolarization) -> Float64

Estimate the polarization extinction ratio (PER) from a `MixedPolarization`
model, using the default value of the mixing amplitude `ε`.

    PER = power in main polarization / power in contamination
        = 1 / |ε|²

Returns `Inf` for a pure polarization (ε = 0). For a stochastic `ε`
(nonzero `std`), the default value gives the design-point PER; use Monte
Carlo sampling over `play(; epol = ...)` calls to estimate the PER distribution.
"""
function estimate_PER(mp::MixedPolarization)
    ε_val = ComplexF64(_resolve_node_default(mp.ε))
    ε2 = abs2(ε_val)
    ε2 == 0 && return Inf
    return 1.0 / ε2
end

"""
    estimate_PER(pol, pol_main) -> Float64

Estimate the PER from a concrete polarization vector `pol` relative to the
intended polarization direction `pol_main`.

    PER = |⟨pol_main | pol⟩|² / (|pol|² - |⟨pol_main | pol⟩|²)

Returns `Inf` if the polarization is pure (no orthogonal component).
"""
function estimate_PER(pol::AbstractVector, pol_main::AbstractVector)
    n = normalize(ComplexF64.(pol_main))
    p = ComplexF64.(pol)
    power_main = abs2(dot(n, p))
    power_perp = sum(abs2, p) - power_main
    power_perp ≈ 0 && return Inf
    return power_main / power_perp
end

"""
    estimate_PER(beam::ParametricBeam) -> Float64

Extract the `MixedPolarization` from a `ParametricBeam` and return its PER.
"""
function estimate_PER(beam::ParametricBeam{GeneralGaussianBeam})
    # pol is the 6th positional arg in the new positional API
    mp = length(beam.args) >= 6 ? beam.args[6] : nothing
    mp isa MixedPolarization || error("beam does not have a MixedPolarization as pol")
    return estimate_PER(mp)
end

"""
    estimate_PER_dB(args...) -> Float64

Return the polarization extinction ratio in dB: `10 log₁₀(PER)`.
"""
estimate_PER_dB(args...) = 10 * log10(estimate_PER(args...))

#=============================================================================
PARAMETRIC BEAM CONSTRUCTORS
=============================================================================#

# GaussianBeam: w0 or P as Parameter/ParametricExpression
function GaussianBeam(λ::Real, w0::_ParametricArg, P; r0 = [0.0, 0.0, 0.0])
    ParametricBeam{GaussianBeam}((Float64(λ), w0, P), (r0 = Vector{Float64}(r0),))
end

function GaussianBeam(λ::Real, w0::Real, P::_ParametricArg; r0 = [0.0, 0.0, 0.0])
    ParametricBeam{GaussianBeam}((Float64(λ), Float64(w0), P), (r0 = Vector{Float64}(r0),))
end

# GeneralGaussianBeam: k and pol are positional to enable clean dispatch on pol type.
# When at least one of w0x/w0y/P is parametric, or pol is MixedPolarization, a
# ParametricBeam is returned. The fully-concrete case (all Real + Vector{<:Number})
# dispatches to the Dynamiq constructor instead, which is more specific.
function GeneralGaussianBeam(λ::Real,
                             w0x::Union{Real, _ParametricArg},
                             w0y::Union{Real, _ParametricArg},
                             P::Union{Real, _ParametricArg},
                             k,
                             pol::Union{Vector{<:Number}, MixedPolarization};
                             r0 = [0.0, 0.0, 0.0])
    return ParametricBeam{GeneralGaussianBeam}(
        (Float64(λ), w0x, w0y, P, Vector{Float64}(k), pol),
        (r0 = Vector{Float64}(r0),),
    )
end

#=============================================================================
BEAM RESOLUTION HELPERS AND BEAMNODE LIFECYCLE
=============================================================================#

_resolve_beam_default(beam) = beam
function _resolve_beam_default(beam::ParametricBeam{T}) where T
    resolved_args   = map(_resolve_node_default, beam.args)
    resolved_kwargs = NamedTuple{keys(beam.kwargs)}(map(_resolve_node_default, values(beam.kwargs)))
    T(resolved_args...; resolved_kwargs...)
end

_resolve_beam(beam, param_values, rng) = beam
function _resolve_beam(beam::ParametricBeam{T}, param_values, rng) where T
    resolved_args   = map(a -> _resolve_node_value(a, param_values, rng), beam.args)
    resolved_kwargs = NamedTuple{keys(beam.kwargs)}(
        map(v -> _resolve_node_value(v, param_values, rng), values(beam.kwargs)))
    T(resolved_args...; resolved_kwargs...)
end

function build_node!(node::BeamNode)
    node._compiled[] = _resolve_beam_default(node.beam)
end

function compile_node!(node::BeamNode, basis, rng, param_values)
    node._compiled[] = _resolve_beam(node.beam, param_values, rng)
end

recompile_node!(node::BeamNode, ::Any, rng, param_values) =
    compile_node!(node, nothing, rng, param_values)
