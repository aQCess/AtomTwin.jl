"""
    MoveModifier <: AbstractModifier

Modifier that updates a beam position incrementally over time to realize a
smooth displacement according to a user-defined schedule.

Like `PositionModifier` it writes an absolute position, but from a total
displacement and a schedule rather than a table of points: position at
normalised time `s` is `r0_start + schedule(s) * displacement`.

# Fields

- `beam::AbstractBeam`: Beam whose position `r0` will be moved.
- `displacement::Vector{Float64}`: Total displacement over the instruction.
- `schedule::Function`: `s in [0,1]` -> fraction of `displacement` applied.
- `tspan::Vector{Float64}`: Time grid; its span sets the instruction duration.
- `dims::Vector{Int}`: Components of `r0` to move at each step.

# Constructor

    MoveModifier(beam, displacement, tspan;
                 dims = [1:3...],
                 schedule = s -> s)

- `displacement::Vector{Float64}`: Total displacement to be applied over `tspan`.
- `tspan::Vector{Float64}`: Time grid from start to end (length ≥ 2).
- `schedule::Function`: Mapping `s ∈ [0, 1]` ↦ scalar factor that shapes the
  trajectory; `s` is normalized time `(t - t0) / (t_end - t0)`.
"""
struct MoveModifier <: AbstractModifier
    beam::AbstractBeam
    displacement::Vector{Float64}   # total displacement over the instruction
    schedule::Function              # s in [0,1] -> fraction of `displacement`
    tspan::Vector{Float64}
    dims::Vector{Int}
    r0_start::Vector{Float64}       # captured on the first update of a run
    started::Base.RefValue{Bool}

    function MoveModifier(beam::AbstractBeam,
                          displacement::Vector{Float64},
                          tspan::Vector{Float64};
                          dims = [1:3...],
                          schedule = s -> s)

        length(tspan) ≥ 2 ||
            error("MoveModifier: tspan must contain at least two time points")
        @assert tspan[end] - tspan[1] > 0 "Move interval cannot be zero"

        return new(beam, copy(displacement), schedule, tspan, dims,
                   zeros(Float64, 3), Ref(false))
    end
end

"""
    update!(m::MoveModifier, t)

Set the beam position `r0` to its start plus the scheduled displacement at time
`t` within the instruction, on the components listed in `m.dims`.

**Absolute, not incremental.** The stored schedule is evaluated at `t` and the
result written, rather than a per-step increment being accumulated. The solver
no longer visits a fixed set of steps -- it chooses its step from `tol` and
sub-divides further when the error estimator asks -- so accumulating increments
would make the final position depend on how many times `update!` happened to be
called. Evaluating the schedule directly makes the trajectory a function of time
alone, which is what it physically is.

The start position is captured on the first call of an instruction, since it is
whatever the previous instruction left behind.
"""
function update!(m::MoveModifier, t::Float64)
    if !m.started[]
        @inbounds for d in 1:3
            m.r0_start[d] = m.beam.r0[d]
        end
        m.started[] = true
    end
    T = m.tspan[end] - m.tspan[1]
    s = T > 0 ? clamp(t / T, 0.0, 1.0) : 1.0
    f = m.schedule(s)
    @inbounds for d in m.dims
        m.beam.r0[d] = m.r0_start[d] + f * m.displacement[d]
    end
end

# A new instruction re-captures the start position: a second move must begin
# from wherever the first one left the beam, not from the first one's origin.
begin_instruction!(m::MoveModifier) = (m.started[] = false; nothing)
