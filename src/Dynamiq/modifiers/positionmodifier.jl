"""
    PositionModifier <: AbstractModifier

Modifier that overwrites the position of a beam at each time step using
precomputed trajectories.

`PositionModifier` is typically used to impose an externally defined
trajectory on a beam, rather than integrating motion from forces.

# Fields

- `beam::AbstractBeam`: Beam whose position `r0` will be updated.
- `vals::Vector{Vector{Float64}}`: Sampled target positions, uniform over the
  instruction (full 3D or more).
- `tspan::Vector{Float64}`: Time grid the samples were given on; its span sets
  the instruction duration.
- `dims::Vector{Int}`: Components of `r0` to overwrite.
- `interp::Symbol`: `:constant`, `:linear` or `:cubic` — see [`sample_at`](@ref).

The trajectory is read at whatever time the solver asks for, not at a step
index: the solver chooses its own step from `tol` and sub-divides further when
the error estimator asks.

# Constructor

- `PositionModifier(beam, vals, tspan; dims = [1:3...], interp = :cubic)`
"""
struct PositionModifier <: AbstractModifier
    beam::AbstractBeam
    vals::Vector{Vector{Float64}}
    tspan::Vector{Float64}
    dims::Vector{Int}
    duration::Float64
    interp::Symbol

    function PositionModifier(beam::AbstractBeam,
                              vals::Vector{Vector{Float64}},
                              tspan::Vector{Float64};
                              dims = [1:3...], interp::Symbol = :cubic)
        dur = length(tspan) ≥ 2 ? tspan[end] - tspan[1] : 0.0
        new{}(beam, vals, tspan, dims, dur, interp)
    end
end

"""
    update!(m::PositionModifier, t)

Overwrite the beam position components `r0[d]` with the stored trajectory
evaluated at time `t` within the instruction.

Only the indices listed in `m.dims` are updated. Each component is interpolated
independently, so the trajectory is read component-wise rather than by snapping
to the nearest stored sample.
"""
function update!(m::PositionModifier, t::Float64)
    beam = m.beam
    n = length(m.vals)
    @inbounds for d in m.dims
        beam.r0[d] = _sample_component(m.vals, n, d, m.duration, t, m.interp)
    end
end

# One component of a sampled trajectory, read at time `t`. The samples live in a
# vector-of-vectors, so the component is gathered on the fly; with the two- and
# few-sample trajectories these modifiers carry, that is cheaper than
# materialising a per-component array.
@inline function _sample_component(vals, n::Int, d::Int, duration::Float64,
                                   t::Float64, kind::Symbol)
    n == 1 && return @inbounds vals[1][d]
    duration > 0 || return @inbounds vals[1][d]
    if kind === :constant
        k = clamp(floor(Int, t / duration * n) + 1, 1, n)
        return @inbounds vals[k][d]
    end
    h = duration / (n - 1)
    x = t / h
    x <= 0     && return @inbounds vals[1][d]
    x >= n - 1 && return @inbounds vals[n][d]
    i = floor(Int, x)
    if kind === :linear || n < 4
        i = clamp(i, 0, n - 2)
        f = x - i
        @inbounds return vals[i+1][d] * (1 - f) + vals[i+2][d] * f
    end
    j = clamp(i, 1, n - 3)
    f = x - j
    @inbounds begin
        y0 = vals[j][d]; y1 = vals[j+1][d]; y2 = vals[j+2][d]; y3 = vals[j+3][d]
    end
    return 0.5 * ((2y1) + (-y0 + y2) * f +
                  (2y0 - 5y1 + 4y2 - y3) * f * f +
                  (-y0 + 3y1 - 3y2 + y3) * f * f * f)
end
