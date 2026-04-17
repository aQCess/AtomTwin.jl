"""
    MoveModifier <: AbstractModifier

Modifier that updates a beam position incrementally over time to realize a
smooth displacement according to a user-defined schedule.

Unlike `PositionModifier`, which overwrites positions, `MoveModifier` stores
per-step increments that are added to the beam position.

# Fields

- `beam::AbstractBeam`: Beam whose position `r0` will be moved.
- `vals::Vector{Vector{Float64}}`: Per-step displacement increments.
- `tspan::Vector{Float64}`: Internal time grid (left edges of increments).
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
    vals::Matrix{Float64}   # 3 × n_steps, contiguous increments
    tspan::Vector{Float64}
    dims::Vector{Int}

    function MoveModifier(beam::AbstractBeam,
                          displacement::Vector{Float64},
                          tspan::Vector{Float64};
                          dims = [1:3...],
                          schedule = s -> s)

        length(tspan) ≥ 2 ||
            error("MoveModifier: tspan must contain at least two time points")

        T  = tspan[end] - tspan[1]
        @assert T > 0 "Move interval cannot be zero"

        n  = length(tspan) - 1
        t0 = tspan[1]
        vals = Matrix{Float64}(undef, 3, n)
        s_prev = schedule(0.0)
        for i in 1:n
            s_next = schedule((tspan[i + 1] - t0) / T)
            Δs = s_next - s_prev
            @inbounds for d in 1:3
                vals[d, i] = Δs * displacement[d]
            end
            s_prev = s_next
        end

        return new(beam, vals, tspan[1:end-1], dims)
    end
end

"""
    update!(m::MoveModifier, i)

Increment the beam position `r0` at time step `i` by the stored displacement
increment on the components listed in `m.dims`, if `i` is within bounds.
"""
function update!(m::MoveModifier, i::Int)
    if i <= size(m.vals, 2)
        @inbounds for d in m.dims
            m.beam.r0[d] += m.vals[d, i]
        end
    end
end
