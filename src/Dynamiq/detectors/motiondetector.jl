"""
    MotionDetector{A} <: AbstractDetector{A}

Detector that records selected coordinates of a target object `obj::A`
over time.

# Fields

- `obj::A`: Target being observed (e.g. an atom or a beam).
- `dims::Vector{Int}`: 1-based component indices to record at each time step.
- `vals::Matrix{Float64}`: Recorded samples, of size `(length(tspan), length(dims))`.
- `tspan::Vector{Float64}`: Time vector associated with the samples.
- `name::String`: Logical name used to group outputs.

# Constructors

- `MotionDetector(obj, dim::Int, tspan; name = \"\")`
- `MotionDetector(obj, dims::Vector{Int}, tspan; name = \"\")`

The detector preallocates its `vals` buffer according to `tspan`. Use
`write!(det, i)` to sample the source object at time step `i`.
"""   
struct MotionDetector{A,V,T} <: AbstractDetector
    obj::A
    dims::Vector{Int}
    vals::V # Vector{Float64}, Matrix{Float64} or SubArray
    tspan::T
    name::String

    function MotionDetector(obj::A,
                            dims::Union{Int, Vector{Int}},
                            tspan::AbstractVector{Float64};
                            name::AbstractString = "") where {A}
        vals = zeros(length(tspan), length(dims))
        new{typeof(obj), typeof(vals), typeof(tspan)}(obj, collect(dims),
                         vals, tspan, name)
    end

    # Constructor with preallocated views
    function MotionDetector(obj,
                            dims::Union{Int, Vector{Int}},
                            tspan::AbstractVector{Float64},
                            vals::AbstractArray{Float64};
                            name::AbstractString = "")
        @assert length(tspan) == size(vals,1) "tspan and vals must have same length"
        new{typeof(obj), typeof(vals), typeof(tspan)}(
            obj, collect(dims), 
            vals,
            tspan,
            name)
    end
end

"""
    MotionDetectorSpec(obj; dims = [1, 2], name = \"\", tspan = nothing) -> DetectorSpec

Inert spec for constructing a `MotionDetector` at runtime.

# Arguments

- `obj`: Target object to monitor (e.g. atom or beam).

# Keywords

- `dims::AbstractVector{<:Integer} = [1, 2]`: Components to record: 1–3 the position
  x, y, z; for an atom also 4–6, the velocity v_x, v_y, v_z.
- `name::AbstractString = \"\"`: Logical name used to group outputs.
- `tspan::Union{Nothing,Vector{Float64}} = nothing`: Times at which to report,
  in place of `out.times`; each is read at the nearest output time.

Specs are declarative: they do not allocate buffers. At runtime, a resolver
binds `obj` to either the original (persistent run) or a per-run copy
(stateless run), and the detector is instantiated with the appropriate `tspan`.
"""
function MotionDetectorSpec(obj;
                            dims::AbstractVector{<:Integer} = [1, 2],
                            name::AbstractString            = "",
                            tspan::Union{Nothing,Vector{Float64}} = nothing)
    nmax = obj isa AbstractBeam ? 3 : 6          # a beam has a position, no velocity
    all(d -> 1 <= d <= nmax, dims) ||
        throw(ArgumentError("MotionDetectorSpec: dims must lie in 1:$nmax, got $(collect(dims))"))
    return DetectorSpec{typeof(MotionDetector)}(
        MotionDetector,
        obj,
        (dims = collect(dims), name = name),
        tspan,
        Float64,
        length(dims)
    )
end

"""
    write!(d::MotionDetector{NLevelAtom}, i)

Sample the position or velocity of the specified atom at time step `i`, writing each
requested component into `d.vals[i, :]`.
"""
function write!(d::MotionDetector{NLevelAtom,V,T}, i::Int) where {V,T}
    for (j, dim) in enumerate(d.dims)
        d.vals[i, j] = dim <= 3 ? d.obj.x[dim] : d.obj.v[dim - 3]
    end
end

"""
    write!(d::MotionDetector{<:AbstractBeam}, i)

Sample the reference position of the specified beam at time step `i`,
writing each requested coordinate into `d.vals[i, :]`.
"""
function write!(d::MotionDetector{<:AbstractBeam, V, T}, i::Int) where {V,T}
    for (j, dim) in enumerate(d.dims)
        d.vals[i, j] = d.obj.r0[dim]
    end
end

"""
    write!(detectors::Vector{MotionDetector{A}}, i)

Sample all motion detectors in the vector at time step `i`.
"""
function write!(detectors::Vector{MotionDetector{A,V,T}}, i::Int) where {A,V,T}
    for d in detectors
        write!(d, i)
    end
end

"""
    reset!(d::MotionDetector)

Reset all stored samples in `d.vals` to zero while preserving shape and
the associated `tspan`.
"""
function reset!(d::MotionDetector)
    fill!(d.vals, 0.0)
end