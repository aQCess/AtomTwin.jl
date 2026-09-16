module AtomTwinGLMakieExt

using AtomTwin
using AtomTwin.Visualization
using GLMakie

"""
    animate(out::NamedTuple;
            options,
            limits = ((-100, 100), (-20, 20)),
            sleep_time = 0.01,
            unit = 1,
            gifname = nothing,
            framerate = 60,
            shotidx = 1)

Animate 2D trajectories extracted from simulation detector outputs using GLMakie.

The input `out` is expected to be a named tuple with at least
`out.detectors` and `out.times`, where `out.detectors[g]` stores
per-group position data and `out.times` the corresponding time axis.
For each group key in `options`, the corresponding detector array is
interpreted as either

- 2D: `[time, dims]` (single trajectory), or
- 3D: `[time, dims, traj]` (multiple trajectories),

and the `shotidx`-th trajectory is visualized. Positions are divided by
`unit` for display (e.g. to convert from meters to micrometers).

Keyword arguments

- `options`: dictionary mapping group names (keys of `out.detectors`)
  to Makie plotting keyword arguments (e.g. marker, color).
- `limits`: axis limits as `((xmin,xmax), (ymin,ymax))`.
- `sleep_time`: delay between frames (seconds) for interactive playback.
- `unit`: spatial scaling factor applied to x and y (default `1`).
- `gifname`: if `nothing`, animate interactively; otherwise, record a
  GIF to the given file path.
- `framerate`: frame rate used when recording a GIF.
- `stride`: show every `stride`-th time sample. Raising it speeds the animation
  up (and shrinks the GIF) without changing `framerate`; e.g. `stride = 10` plays
  5x faster than the default 2.
- `shotidx`: trajectory index to animate in the multi-shot case.

Returns the `Figure` containing the animation. When `gifname` is set,
frames are recorded with `record`; otherwise, frames are advanced in a
loop with `sleep(sleep_time)` between updates.
"""

# `options` may be a Dict (unordered) or a Vector of `name => style` pairs, which
# preserves the caller's intended draw order.
_style_names(o::AbstractDict) = collect(keys(o))
_style_names(o::AbstractVector) = [first(p) for p in o]
_style_for(o::AbstractDict, k) = o[k]
_style_for(o::AbstractVector, k) = last(o[findfirst(p -> first(p) == k, o)])

function AtomTwin.Visualization.animate(
    out::NamedTuple;
    options = Dict{String,NamedTuple}(),   # Dict, or an ordered Vector of `name => style` pairs
    limits = ((-40, 40), (-15, 15)),
    sleep_time = 0.001,
    unit = 1e-6,
    gifname = nothing,
    framerate = 60,
    stride = 2,
    shotidx = 1,
)
    # Map detector name to a style key based on prefixes in `options`
    function style_key(det::AbstractString)
        for k in _style_names(options)
            if startswith(det, k)
                return k
            end
        end
        return nothing
    end

    # Draw order matters: `keys(out.detectors)` is a Dict, so its order is
    # arbitrary and interleaves groups. Markers drawn later paint over earlier
    # ones, so a large static-tweezer marker could hide a smaller dynamic one
    # sitting on the same site -- making dynamic tweezers look stationary, and
    # appear/disappear as they moved between sites.
    #
    # Sort by the order the styles are declared in `options`, so the caller
    # controls layering (declare background groups first), and sort by name
    # within a group so runs are reproducible.
    groups = let ks = filter(k -> !endswith(k, "_ampl"), collect(keys(out.detectors))),
                 order = _style_names(options)
        rank(g) = begin
            k = style_key(g)
            k === nothing ? length(order) + 1 : findfirst(==(k), order)
        end
        sort(ks; by = g -> (rank(g), g))
    end

    # Build per-group trajectory vectors
    function extractor(out, g::String, shotidx)
        data = out.detectors[g]
        if ndims(data) == 3
            x = data[:, 1, shotidx] ./ unit
            y = data[:, 2, shotidx] ./ unit
            times = out.times
        else
            x = data[:, 1] ./ unit
            y = data[:, 2] ./ unit
            times = out.times
        end
        return x, y, times
    end

    xvecs = Dict{String,Vector{Float64}}()
    yvecs = Dict{String,Vector{Float64}}()
    tvecs = Dict{String,Vector{Float64}}()
    for g in groups
        xvecs[g], yvecs[g], tvecs[g] = extractor(out, g, shotidx)
    end

    # Amplitude gating. A tweezer trap is switched off by setting its beam
    # amplitude to zero (AmplRow/AmplCol), but a MotionDetector only records
    # POSITION -- so without this every trap is drawn whether or not it exists,
    # showing phantom static boxes and letting "boxes" appear to cross.
    #
    # If a detector "<name>" has a companion amplitude detector "<name>_ampl",
    # frames where |amplitude| is negligible are blanked (NaN), which Makie
    # simply does not draw.
    avecs = Dict{String,Vector{Float64}}()
    for g in groups
        key = g * "_ampl"
        haskey(out.detectors, key) || continue
        raw = out.detectors[key]
        col = ndims(raw) == 3 ? view(raw, :, 1, shotidx) : vec(raw)
        avecs[g] = abs.(col)
    end
    if !isempty(avecs)
        for (g, a) in avecs
            thresh = 1e-9 * maximum(a; init = 0.0)
            for i in eachindex(a)
                if a[i] <= thresh
                    xvecs[g][i] = NaN
                    yvecs[g][i] = NaN
                end
            end
        end
    end

    x_obs = Dict(g => Observable([xvecs[g][1]]) for g in groups)
    y_obs = Dict(g => Observable([yvecs[g][1]]) for g in groups)

    # Compute aspect-correct figure size from limits
    (xmin, xmax) = limits[1]
    (ymin, ymax) = limits[2]
    xrange = xmax - xmin
    yrange = ymax - ymin
    base_width = 600
    width  = base_width
    height = Int(round(base_width * (yrange / xrange)))

    fig = Figure(size = (width, height))
    ax = Axis(fig[1, 1]; title = "0.00 ms", limits = limits)
    ax.xgridvisible = ax.ygridvisible = true
    ax.aspect = DataAspect()  # enforce equal aspect ratio in data units

    for g in groups
        key = style_key(g)
        style = key === nothing ? NamedTuple() : _style_for(options, key)
        scatter!(ax, x_obs[g], y_obs[g]; style...)
    end

    display(fig)

    function update_frame(i)
        for g in groups
            if i <= length(xvecs[g])
                x_obs[g][] = [xvecs[g][i]]
                y_obs[g][] = [yvecs[g][i]]
            end
        end
        mint = minimum(tvecs[g][min(i, end)] for g in groups)
        t = round(mint * 1e3, digits = 2)
        ax.title[] = "$t ms"
    end

    M = minimum(length.(values(tvecs)))
    if gifname !== nothing
        record(fig, gifname, 1:stride:M; framerate = framerate) do i
            update_frame(i)
        end
    else
        for i in 1:stride:M
            update_frame(i)
            yield()
            sleep(sleep_time)
        end
    end
    return fig
end

end
