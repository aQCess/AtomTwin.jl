"""
    AmplitudeModifier{F} <: AbstractModifier

Modifier that sets the complex amplitude `_coeff[]` of a field or beam from a
sampled envelope, read at arbitrary time.

`vals` is the envelope the user gave, at its own resolution, spanning
`[0, duration]` — it is NOT aligned with the solver's steps. The solver chooses
its step from `tol` and sub-divides further when the error estimator asks, so it
evaluates the envelope wherever it lands via [`sample_at`](@ref).

# Fields

- `field::F`: Target field or beam (must have a `_coeff::Ref{ComplexF64}`).
- `vals::Vector{ComplexF64}`: Envelope samples, uniform over `[0, duration]`.
- `duration::Float64`: Physical span the samples cover.
- `interp::Symbol`: `:constant`, `:linear` or `:cubic` — see [`sample_at`](@ref).

# Constructors

- `AmplitudeModifier(field, vals, duration; interp = :cubic)`
- `AmplitudeModifier(beam, vals, duration; interp = :cubic)`
"""
struct AmplitudeModifier{F} <: AbstractModifier
    field::F
    vals::Vector{ComplexF64}
    duration::Float64
    interp::Symbol

    function AmplitudeModifier(field::Union{AbstractField, AbstractBeam},
                               vals::AbstractVector{<:Number},
                               duration::Real; interp::Symbol = :cubic)
        # Convert once - no allocation if already ComplexF64
        new{typeof(field)}(field, convert(Vector{ComplexF64}, vals),
                           Float64(duration), interp)
    end

    # Inner constructor for Ref-based target (used by GaussianCoupling redirect)
    function AmplitudeModifier(field::Base.RefValue{ComplexF64},
                               vals::AbstractVector{<:Number},
                               duration::Real; interp::Symbol = :cubic)
        new{typeof(field)}(field, convert(Vector{ComplexF64}, vals),
                           Float64(duration), interp)
    end
end


"""
    update!(m::AmplitudeModifier, t)

Set the complex amplitude `_coeff[]` of the underlying field or beam to the
envelope's value at time `t` within the instruction.
"""
@inline function update!(m::AmplitudeModifier{F}, t::Float64) where {F}
    m.field._coeff[] = sample_at(m.vals, m.duration, t, m.interp)
end

"""
    AmplitudeModifier(field::GaussianCoupling, vals)

For `GaussianCoupling`, redirect the modifier to `field._amplitude` rather than
`field._coeff`, so that the pulse amplitude and the spatial envelope both contribute
to the instantaneous Rabi rate without overwriting each other.
"""
function AmplitudeModifier(field::GaussianCoupling, vals::AbstractVector{<:Number},
                           duration::Real; interp::Symbol = :cubic)
    AmplitudeModifier(field._amplitude, vals, duration; interp = interp)
end

@inline function update!(m::AmplitudeModifier{<:Base.RefValue{ComplexF64}}, t::Float64)
    m.field[] = sample_at(m.vals, m.duration, t, m.interp)
end

"""
    SetModifier{F} <: AbstractBoundaryModifier

Sets a coupling/field amplitude to a fixed value at the start of an instruction
(`begin_instruction!`). Used by `compile(::Pulse)` for constant-amplitude pulses
and by `compile(::On)`.
"""
struct SetModifier{F} <: AbstractBoundaryModifier
    field::F
    val::ComplexF64
end

begin_instruction!(m::SetModifier) = (m.field._coeff[] = m.val)
begin_instruction!(m::SetModifier{<:Base.RefValue{ComplexF64}}) = (m.field[] = m.val)

"""
    ResetModifier{F} <: AbstractBoundaryModifier

Zeros a coupling/field amplitude at the end of an instruction (`end_instruction!`).
Used by `compile(::Pulse)` and `compile(::Off)`. Never appears in the solver loop.
"""
struct ResetModifier{F} <: AbstractBoundaryModifier
    field::F
end

end_instruction!(m::ResetModifier) = (m.field._coeff[] = zero(ComplexF64))
end_instruction!(m::ResetModifier{<:Base.RefValue{ComplexF64}}) = (m.field[] = zero(ComplexF64))
