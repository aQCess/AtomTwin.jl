"""
    interpolate(y_vals::Vector{<:Real}, t_out::AbstractVector{<:Real})

Third‑order (cubic) Lagrange interpolation of a real array `y_vals` onto the
output time grid `t_out`.

`y_vals` are assumed sampled at times `0, duration/(N-1), ..., duration`,
where `duration = maximum(t_out)`.
"""
function interpolate(y_vals::Vector{<:Real}, t_out::AbstractVector{<:Real})
    N_in = length(y_vals)
    N_in < 4 && throw(ArgumentError("Need at least 4 points for cubic Lagrange interpolation"))

    duration = maximum(t_out)
    t_in = range(0.0, stop = duration, length = N_in)

    y_out = similar(t_out, Float64)

    @inbounds for k in eachindex(t_out)
        t = t_out[k]

        if t ≤ t_in[1]
            y_out[k] = y_vals[1]
            continue
        elseif t ≥ t_in[end]
            y_out[k] = y_vals[end]
            continue
        end

        i = searchsortedlast(t_in, t)
        i = clamp(i, 1, N_in-1)

        if i ≤ 1
            i1, i2, i3, i4 = 1, 2, 3, 4
        elseif i ≥ N_in-1
            i1, i2, i3, i4 = N_in-3, N_in-2, N_in-1, N_in
        else
            i1, i2, i3, i4 = i-1, i, i+1, i+2
        end

        x1, x2, x3, x4 = t_in[i1], t_in[i2], t_in[i3], t_in[i4]
        y1, y2, y3, y4 = y_vals[i1], y_vals[i2], y_vals[i3], y_vals[i4]

        w1 = 1 / ((x1-x2)*(x1-x3)*(x1-x4))
        w2 = 1 / ((x2-x1)*(x2-x3)*(x2-x4))
        w3 = 1 / ((x3-x1)*(x3-x2)*(x3-x4))
        w4 = 1 / ((x4-x1)*(x4-x2)*(x4-x3))

        num = w1*(t-x2)*(t-x3)*(t-x4)*y1 +
              w2*(t-x1)*(t-x3)*(t-x4)*y2 +
              w3*(t-x1)*(t-x2)*(t-x4)*y3 +
              w4*(t-x1)*(t-x2)*(t-x3)*y4

        den = w1*(t-x2)*(t-x3)*(t-x4) +
              w2*(t-x1)*(t-x3)*(t-x4) +
              w3*(t-x1)*(t-x2)*(t-x4) +
              w4*(t-x1)*(t-x2)*(t-x3)

        y_out[k] = num / den
    end

    return y_out
end



"""
    interpolate(y_vals::Vector{ComplexF64}, t_out::AbstractVector{<:Real})

Interpolate a complex array `y_vals` by interpolating amplitude and phase
separately onto `t_out` using cubic Lagrange.
"""
function interpolate(y_vals::Vector{ComplexF64}, t_out::AbstractVector{<:Real})
    y_vals_c = ComplexF64.(y_vals)
    amp_vals = abs.(y_vals_c)
    phase_vals = angle.(y_vals_c)

    amp_interp   = interpolate(amp_vals,   t_out)
    phase_interp = interpolate(phase_vals, t_out)

    y_out = similar(t_out, ComplexF64)
    @inbounds for k in eachindex(t_out)
        y_out[k] = amp_interp[k] * cis(phase_interp[k])
    end
    return y_out
end

"""
    interpolate_piecewise_constant(y_vals, tsteps)

Map `N` samples uniformly onto `tsteps` output points using nearest-lower
(piecewise-constant / zero-order hold) interpolation.

Sample `k` (1-based) is held for output steps where
`floor((i-1)*N/tsteps) + 1 == k`.  This matches the convention used by
pulse-shape optimisers that treat each sample as a constant segment.
"""
function interpolate_piecewise_constant(y_vals::AbstractVector, tsteps::Int)
    N = length(y_vals)
    y_out = similar(y_vals, tsteps)
    @inbounds for i in 1:tsteps
        k = floor(Int, (i - 1) * N / tsteps) + 1
        y_out[i] = y_vals[k]
    end
    return y_out
end

