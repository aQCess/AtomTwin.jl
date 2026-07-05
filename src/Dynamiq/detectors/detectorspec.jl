"""
    DetectorSpec{K}

Inert, declarative descriptor of a detector to be instantiated at runtime.

# Fields

- `kind::K`: Callable constructor (e.g. `MotionDetector`).
- `obj::Any`: Target object to resolve and observe (atom, field, beam, etc.).
- `params::NamedTuple`: Detector-specific parameters.
- `tspan::Union{Nothing,Vector{Float64}}`: Optional time grid.

`obj` always holds the primary observable object that needs resolution.
Secondary context (basis, state, system-wide settings) is carried in
`params` and retrieved from the system at build time.
"""
struct DetectorSpec{K}
    kind::K
    obj::Any         # Always the resolvable target
    params::NamedTuple
    tspan::Union{Nothing,Vector{Float64}}
    eltype::Type     # Data type for vals
    ndims::Int       # Dimensions of vals
end

"""
    build_detector(specs, tspan, vals, resolve_target, system) -> Vector{AbstractDetector}

Instantiate detectors for a simulation segment.

Since detector specs travel with the system and are deep-copied together,
atom references in `specs` already point to the correct copied atoms.
`resolve_target` is still applied for consistency and for resolving
deferred parameters.

# Arguments

- `spec::Vector{DetectorSpec}`: Detector specification.
- `tspan::Vector{Float64}`: Time vector for this segment.
- `vals`::AbstractVector{T}
- `resolve_target::Function`: Mapping from original to resolved objects
  (e.g. for deferred parameters).
- `system`: System object providing quantum state, basis, etc.
"""
function build_detector(spec::DetectorSpec,
                         tspan::AbstractVector{Float64},
                         vals::AbstractArray{T},
                         resolve_target::Function,
                         system,
                         ) where T

    # Apply resolve_target to the object reference
    # (mostly for Deferred parameters; deepcopy already handles atom refs)
    obj = resolve_target(spec.obj)
    p   = spec.params

    if spec.kind === PopulationDetector
        return PopulationDetector(
            system.state[],
            system.basis,
            obj.inner, # use NLevelAtom ref
            p.level,
            tspan,  # view into global tspan
            vals;   # view into detector-specific vals array
            name = p.name
        )
    elseif spec.kind === MotionDetector
        if obj isa AbstractAtom
            obj = obj.inner
        end
        return MotionDetector(
            obj,
            p.dims,
            tspan,
            vals;
            name = p.name
        )
    elseif spec.kind === FieldDetector
        return FieldDetector(
            obj,
            tspan,
            vals;
            name = p.name
        )
    elseif spec.kind === CoherenceDetector
        return CoherenceDetector(
            system.state[],
            system.basis,
            obj.inner, # use NLevelAtom ref
            p.levels,
            tspan,  # view into global tspan
            vals;   # view into detector-specific vals array
            name = p.name
        )
    elseif spec.kind === ExpectationDetector
        # `obj` is the operator itself (basis-global; not an atom to resolve).
        return ExpectationDetector(
            system.state[],
            obj,
            tspan,  # view into global tspan
            vals;   # view into detector-specific vals array
            name = p.name
        )
    elseif spec.kind === PhotoDetector
        # No state/atom target: counts are written by the jump it is bound to.
        return PhotoDetector(
            tspan,  # view into global tspan
            vals;   # view into detector-specific vals array (Int)
            name = p.name
        )
    end
end
