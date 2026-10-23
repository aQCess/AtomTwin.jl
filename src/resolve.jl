"""
Resolve parametric descriptions of sequences, beams, and instructions into
concrete instances for a given set of parameter values.

Parameters in beams and instructions are resolved once per `compile` call.
System Hamiltonian terms are handled by the DAG node lifecycle instead.
"""

"""
    resolve(obj, param_values; cache = nothing, rng = Random.default_rng())

Recursively resolve all `Parameter` and `ParametricExpression` objects
contained in `obj` using the dictionary `param_values`.

`param_values` maps parameter names (symbols) to concrete values or to
`Parameter` instances that override the defaults. The function walks
through `obj` (which may be a `Sequence`, instruction, beam, or container)
and returns a new object in which all parametric entries have been replaced
by concrete numerical values.

An internal `IdDict` `cache` is used to ensure that identical objects are
only resolved once and then re-used, preserving pointer sharing across
the resolved structure. A `Parameter` with a nonzero `std` draws from `rng`.
"""
function resolve(obj, param_values; cache=nothing, rng=Random.default_rng())
    cache === nothing && (cache = IdDict())
    _resolve(obj, param_values, cache, rng)
end

"""
    _resolve(x::Number, param_values, cache::IdDict, rng)

Internal helper: numbers are already concrete and are returned unchanged.
"""
function _resolve(x::Number, param_values, cache::IdDict, rng) x end

"""
    _resolve(x::String, param_values, cache::IdDict, rng)
    _resolve(x::Symbol, param_values, cache::IdDict, rng)
    _resolve(x::Function, param_values, cache::IdDict, rng)

Internal helpers: symbols and functions are treated as already-resolved
primitives and returned unchanged.
"""
function _resolve(x::String, param_values, cache::IdDict, rng) x end
function _resolve(x::Symbol, param_values, cache::IdDict, rng) x end
function _resolve(x::Function, param_values, cache::IdDict, rng) x end

"""
    _resolve(b::AbstractBeam, param_values, cache::IdDict, rng)

Resolve a concrete beam to a **private copy** (cached per original beam), so the
compiled job owns its trapping beams and never mutates the user's `System` /
`TweezerArray`. Inner-loop modifiers (`MoveModifier`, `PositionModifier`,
`AmplitudeModifier`) mutate `beam.r0` / `beam._coeff` in place; copying here is
what decouples that runtime state from the source objects. The cache keys on the
original beam's identity so that the same copy is reused everywhere it appears
(e.g. the `System` beam list and the `TweezerArray` an instruction holds).
"""
function _resolve(b::AbstractBeam, param_values, cache::IdDict, rng)
    get!(cache, b) do
        copy(b)
    end
end

"""
    _resolve(ta::TweezerArray, param_values, cache::IdDict, rng)

Resolve a `TweezerArray` to a cached copy whose beams are the per-beam cached
copies (see [`_resolve(::AbstractBeam, ...)`]). This keeps the beams handed to
move/ramp modifiers identical to those stored in the job, while leaving the
caller's array untouched.
"""
function _resolve(ta::TweezerArray, param_values, cache::IdDict, rng)
    get!(cache, ta) do
        new_beams = GaussianBeam[_resolve(b, param_values, cache, rng) for b in ta.beams]
        TweezerArray(
            ta.λ, ta.w0, ta.P_total, ta.r0,
            copy(ta.row_freqs), copy(ta.col_freqs),
            copy(ta.row_amplitudes), copy(ta.col_amplitudes),
            ta.dx, ta.dy,
            new_beams,
        )
    end
end

"""
    _resolve(x::Vector, param_values, cache::IdDict, rng)
    _resolve(x::Tuple,  param_values, cache::IdDict, rng)

Recursively resolve all elements of a vector or tuple, threading the
same `cache` so that shared substructures and deferred objects are
handled consistently.
"""
function _resolve(x::Vector, param_values, cache::IdDict, rng)
    [_resolve(e, param_values, cache, rng) for e in x]
end

function _resolve(x::Tuple, param_values, cache::IdDict, rng)
    tuple((_resolve(xi, param_values, cache, rng) for xi in x)...)
end

"""
    _resolve(seq::Sequence, param_values, cache::IdDict, rng)

Resolve all instructions in a `Sequence`, preserving its solver settings.

Returns a new `Sequence` with the same `dt`, `downsample` and `tol`, and resolved
instructions.
"""
function _resolve(seq::Sequence, param_values, cache::IdDict, rng)
    # NB: pass every field. Reconstructing with only (instructions, dt) silently
    # dropped `downsample` back to its default.
    Sequence([_resolve(inst, param_values, cache, rng) for inst in seq.instructions],
             seq.dt, seq.downsample, seq.tol)
end

"""
    _resolve(mc::MoveCol, param_values, cache::IdDict, rng)
    _resolve(inst::Pulse, param_values, cache::IdDict, rng)
    _resolve(inst::On,   param_values, cache::IdDict, rng)
    _resolve(inst::Off,  param_values, cache::IdDict, rng)

Resolve parametric fields and timing parameters appearing in instruction
objects (`MoveCol`, `Pulse`, `On`, `Off`), returning fully concrete
copies suitable for simulation.
"""
function _resolve(mc::MoveCol, param_values, cache::IdDict, rng)
    MoveCol(
        mc.tweezers,
        _resolve(mc.cols, param_values, cache, rng),
        _resolve(mc.delta, param_values, cache, rng),
        _resolve(mc.duration, param_values, cache, rng),
        _resolve(mc.sweep, param_values, cache, rng),
        mc.dt,
        mc.downsample
    )
end

"""
    _resolve(mr::MoveRow, param_values, cache::IdDict, rng)

Resolve parametric fields in a `MoveRow` instruction (delta may be a Parameter or ParametricExpression).
"""
function _resolve(mr::MoveRow, param_values, cache::IdDict, rng)
    MoveRow(
        mr.tweezers,
        _resolve(mr.rows, param_values, cache, rng),
        _resolve(mr.delta, param_values, cache, rng),
        _resolve(mr.duration, param_values, cache, rng),
        _resolve(mr.sweep, param_values, cache, rng),
        mr.dt,
        mr.downsample
    )
end

function _resolve(inst::Pulse, param_values, cache::IdDict, rng)
    Pulse(
        _resolve(inst.couplings, param_values, cache, rng),
        _resolve(inst.duration, param_values, cache, rng),
        _resolve(inst.ampl, param_values, cache, rng),
        inst.amplitudes,
        inst.dt,
        inst.downsample,
        inst.interp,
    )
end

function _resolve(inst::On, param_values, cache::IdDict, rng)
    On(_resolve(inst.couplings, param_values, cache, rng), inst.dt, inst.downsample)
end

function _resolve(inst::Off, param_values, cache::IdDict, rng)
    Off(_resolve(inst.couplings, param_values, cache, rng), inst.dt, inst.downsample)
end

"""
    _resolve(p::Parallel, param_values, cache::IdDict, rng)

Resolve each part of a `Parallel` instruction, preserving dt and downsample.
"""
function _resolve(p::Parallel, param_values, cache::IdDict, rng)
    Parallel(
        [_resolve(part, param_values, cache, rng) for part in p.parts],
        p.dt,
        p.downsample
    )
end

"""
    _resolve(b::ParametricBeam{T}, param_values, cache::IdDict, rng) where T

Resolve a `ParametricBeam{T}` by resolving all parametric arguments and
constructing a concrete beam of type `T`.
"""
function _resolve(b::ParametricBeam{T}, param_values, cache::IdDict, rng) where T
    resolved_args = [_resolve(arg, param_values, cache, rng) for arg in b.args]
    resolved_kwargs = NamedTuple{keys(b.kwargs)}(
        [_resolve(val, param_values, cache, rng) for val in values(b.kwargs)]
    )
    T(resolved_args...; resolved_kwargs...)
end

"""
    _resolve(sys::System, param_values, cache::IdDict, rng)

Resolve a `System` by resolving its atoms and beams, while passing through
`initial_state`, `state`, `basis`, `nodes`, and `detector_specs` unchanged.

Returns a new `System` instance with parametric components concretized
according to `param_values`.
"""
function _resolve(sys::System, param_values, cache::IdDict, rng)
    System(
        [_resolve(atom, param_values, cache, rng) for atom in sys.atoms],
        _resolve(sys.beams, param_values, cache, rng),
        sys.initial_state,
        sys.state,
        sys.basis,
        sys.nodes,
        sys.detector_specs
    )
end


"""
    _resolve(p::Parameter, param_values, cache::IdDict, rng)

Resolve a scalar `Parameter` to a concrete numeric value.

If `param_values` contains an entry for `p.name`, that value (or
parameter) overrides the default `p.default` / `p.std`. A nonzero
`std` leads to a random draw `mean + randn(rng)*std`, enabling static
noise sampling; otherwise the mean value is returned.
"""
function _resolve(p::Parameter, param_values, cache::IdDict, rng)
    if haskey(param_values, p.name)
        val = param_values[p.name]
        mean_val, std_val = val isa Parameter ? (val.default, val.std) : (val, 0)
    else
        mean_val, std_val = p.default, p.std
    end
    std_val != 0 ? mean_val + randn(rng) * std_val : mean_val
end

"""
    _resolve(expr::ParametricExpression, param_values, cache::IdDict, rng)

Evaluate a `ParametricExpression` by first resolving all its arguments
and then applying the encoded operation.

The operations are those of the DAG resolvers, [`_apply_op`](@ref).
"""
_resolve(expr::ParametricExpression, param_values, cache::IdDict, rng) =
    _apply_op(expr.op, [_resolve(arg, param_values, cache, rng) for arg in expr.args])

"""
    _resolve(obj, param_values, cache::IdDict, rng)

Fallback resolver: for unhandled types, return `obj` unchanged.

This ensures that objects which do not depend on parameters pass
through the resolution stage transparently.
"""
function _resolve(obj, param_values, cache::IdDict, rng)
    obj
end

