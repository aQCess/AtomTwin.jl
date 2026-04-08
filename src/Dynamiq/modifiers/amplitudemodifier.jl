"""
    AmplitudeModifier{F} <: AbstractModifier

Generic modifier that updates the complex amplitude `_coeff[]` of a field
or beam as a function of time.

# Fields

- `field::F`: Target field or beam (must have a `_coeff::Ref{ComplexF64}`).
- `vals::Vector{ComplexF64}`: Time series of complex coefficients aligned with simulation time grid.

# Constructors

- `AmplitudeModifier(field::AbstractField, vals)`
- `AmplitudeModifier(beam::AbstractBeam, vals)`
"""
struct AmplitudeModifier{F} <: AbstractModifier
    field::F
    vals::Vector{ComplexF64}

    function AmplitudeModifier(field::Union{AbstractField, AbstractBeam},
                               vals::AbstractVector{<:Number})
        # Convert once - no allocation if already ComplexF64
        new{typeof(field)}(field, convert(Vector{ComplexF64}, vals))
    end

    # Inner constructor for Ref-based target (used by GaussianCoupling redirect)
    function AmplitudeModifier(field::Base.RefValue{ComplexF64},
                               vals::AbstractVector{<:Number})
        new{typeof(field)}(field, convert(Vector{ComplexF64}, vals))
    end
end


"""
    update!(m::AmplitudeModifier, i)

Set the complex amplitude `_coeff[]` of the underlying field or beam
to `m.vals[i]` at time step `i`, if `i` is within bounds.
"""
@inline function update!(m::AmplitudeModifier{F}, i::Int) where {F}
    @inbounds if i <= length(m.vals)
        m.field._coeff[] = m.vals[i]
    end
end

"""
    AmplitudeModifier(field::GaussianCoupling, vals)

For `GaussianCoupling`, redirect the modifier to `field._amplitude` rather than
`field._coeff`, so that the pulse amplitude and the spatial envelope both contribute
to the instantaneous Rabi rate without overwriting each other.
"""
function AmplitudeModifier(field::GaussianCoupling, vals::AbstractVector{<:Number})
    AmplitudeModifier(field._amplitude, vals)
end

@inline function update!(m::AmplitudeModifier{<:Base.RefValue{ComplexF64}}, i::Int)
    @inbounds if i <= length(m.vals)
        m.field[] = m.vals[i]
    end
end

"""
    end_instruction!(m::AbstractModifier)

Called by the simulation engine once after an instruction's `evolve!` loop
completes, before the next instruction begins. Default is a no-op.
"""
end_instruction!(::AbstractModifier) = nothing

"""
    ResetModifier{F} <: AbstractModifier

Zero-timestep modifier appended by `compile(::Pulse)` and `compile(::On/Off)`
to reset a coupling amplitude to zero after the instruction completes.

`update!` is a no-op (empty `vals`); only `end_instruction!` acts.
"""
struct ResetModifier{F} <: AbstractModifier
    field::F
end

update!(::ResetModifier, ::Int) = nothing
end_instruction!(m::ResetModifier) = (m.field._coeff[] = zero(ComplexF64))
end_instruction!(m::ResetModifier{<:Base.RefValue{ComplexF64}}) = (m.field[] = zero(ComplexF64))
