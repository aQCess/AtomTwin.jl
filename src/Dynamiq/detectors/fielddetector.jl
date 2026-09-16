"""
    FieldDetector{A} <: AbstractDetector{A}

Detector that records the complex amplitude of a field `obj::A` over time.

Typical use is to monitor the scalar coefficient `_coeff[]` of an
`AbstractField` (e.g. a coupling or detuning) during a simulation.

# Fields

- `obj::A`: Target field being observed (e.g. coupling or detuning).
- `vals::Vector{ComplexF64}`: Per-timestep samples of the complex amplitude.
- `tspan::Vector{Float64}`: Time vector associated with the recorded samples.
- `name::String`: Logical name used to group outputs.

# Constructor

- `FieldDetector(obj, tspan; name = \"\")`

The detector stores its own `vals` buffer sized to `tspan`. Use
`write!(det, i)` to sample the field at time step `i`.
"""
struct FieldDetector{A,V,T} <: AbstractDetector
    obj::A
    vals::V
    tspan::T
    name::String

    function FieldDetector(obj::A,
                           tspan::Vector{Float64};
                           name::AbstractString = "") where {A}
        new{typeof(obj),Vector{ComplexF64}, typeof(tspan)}(obj, [0.0 + 0.0im for _ in tspan], tspan, name)
    end
    
    function FieldDetector(obj::A,
                           tspan::AbstractVector{Float64},
                           vals::AbstractVector{ComplexF64};
                           name::AbstractString = "") where {A}
        new{typeof(obj), typeof(vals), typeof(tspan)}(obj, vals, tspan, name)
    end
end

"""
    FieldDetectorSpec(obj; name = \"\", tspan = nothing) -> DetectorSpec

Inert spec for constructing a `FieldDetector` at runtime.

# Arguments

- `obj`: Target object to monitor (e.g. coupling or detuning).

# Keywords

- `name::AbstractString = \"\"`: Logical name used to group outputs.
- `tspan::Union{Nothing,Vector{Float64}} = nothing`:
  Optional time vector to attach to the spec. If `nothing`, the runtime
  supplies the segment or full-run `tspan`.

Specs are declarative and do not allocate buffers. At runtime, a resolver
binds `obj` to either the original (persistent run) or a per-run copy
(stateless run), and the detector is instantiated with the appropriate `tspan`.
"""
FieldDetectorSpec(obj;
                  name::AbstractString            = "",
                  tspan::Union{Nothing,Vector{Float64}} = nothing) =
    DetectorSpec{typeof(FieldDetector)}(
        FieldDetector,
        obj,
        (name = name,),
        tspan,
        ComplexF64,
        1
    )

"""
    write!(d::FieldDetector{<:AbstractField}, i)

Sample the field amplitude at time step `i` and store it in `d.vals[i]`.

The value recorded is the scalar coefficient `_coeff[]` of the underlying
field or coupling, accessed via `base_coupling(d.obj)`.
"""
function write!(d::FieldDetector{<:AbstractField, V, T}, i::Int) where {V,T}
    d.vals[i] = base_coupling(d.obj)._coeff[]
end

"""
    write!(d::FieldDetector{<:AbstractBeam}, i)

Sample a **beam's** complex amplitude. Tweezer traps are switched on and off
during a sequence (`AmplRow`/`AmplCol` set `_coeff` to zero), so this is what
tells a consumer whether a trap is actually present at a given time — position
alone does not.
"""
function write!(d::FieldDetector{<:AbstractBeam, V, T}, i::Int) where {V,T}
    d.vals[i] = d.obj._coeff[]
end

"""
    reset!(d::FieldDetector)

Reset all recorded samples to zero while preserving the length and `tspan`.
"""
function reset!(d::FieldDetector)
    fill!(d.vals, 0.0im)
end
