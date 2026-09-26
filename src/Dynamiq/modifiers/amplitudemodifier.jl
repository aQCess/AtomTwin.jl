"""
    AmplitudeModifier{F} <: AbstractModifier

Modifier that sets the complex amplitude `_coeff[]` of a field or beam from a
sampled envelope, read at arbitrary time.

`vals` is the envelope the user gave, at its own resolution, spanning
`[0, duration]` — it is NOT aligned with the solver's steps. The solvers read
it at step midpoints and at their own sub-steps, wherever those land, via
[`sample_at`](@ref).

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

    # Inner constructor for a bare `Ref` target: a field's `envelope` when that
    # is not its `_coeff` (see the redirect below)
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
    AmplitudeModifier(field::Union{PlanarCoupling,GaussianCoupling,StarkShiftAC,VdWInteraction}, vals, duration)

For a field whose `update!` recomputes `_coeff` from geometry, write to its
[`envelope`](@ref) (`_amplitude`) instead, so the commanded amplitude and the
geometric factor both reach the Hamiltonian without overwriting each other.
"""
function AmplitudeModifier(field::Union{PlanarCoupling, GaussianCoupling,
                                        StarkShiftAC, VdWInteraction},
                           vals::AbstractVector{<:Number},
                           duration::Real; interp::Symbol = :cubic)
    AmplitudeModifier(envelope(field), vals, duration; interp = interp)
end

@inline function update!(m::AmplitudeModifier{<:Base.RefValue{ComplexF64}}, t::Float64)
    m.field[] = sample_at(m.vals, m.duration, t, m.interp)
end

"""
    RampModifier{F} <: AbstractModifier

Linear ramp of a beam's amplitude to `target` over `duration`, starting from
whatever amplitude the beam has when the instruction begins.

The start is captured by `begin_instruction!`, like `MoveModifier`'s start
position, because it is set by the instructions that run before this one -- an
`AmplRow`, an earlier ramp -- none of which have run when the sequence is
compiled. Reading it at compile time ramped from a stale amplitude.
"""
struct RampModifier{F} <: AbstractModifier
    field::F
    target::ComplexF64
    duration::Float64
    start::Base.RefValue{ComplexF64}
end

RampModifier(field, target::Number, duration::Real) =
    RampModifier(field, ComplexF64(target), Float64(duration), Ref(field._coeff[]))

begin_instruction!(m::RampModifier) = (m.start[] = m.field._coeff[]; nothing)

# Same expression as a two-sample linear `sample_at`, so a ramp whose start was
# already right is reproduced bit for bit.
@inline function update!(m::RampModifier, t::Float64)
    f = m.duration > 0 ? clamp(t / m.duration, 0.0, 1.0) : 1.0
    m.field._coeff[] = m.start[] * (1 - f) + m.target * f
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
