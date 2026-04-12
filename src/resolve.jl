"""
Resolve parametric descriptions of sequences, beams, and instructions into
concrete instances for a given set of parameter values.

Parameters in beams and instructions are resolved once per `compile` call.
System Hamiltonian terms are handled by the DAG node lifecycle instead.
"""

"""
    resolve(obj, param_values; cache = nothing)

Recursively resolve all `Parameter` and `ParametricExpression` objects
contained in `obj` using the dictionary `param_values`.

`param_values` maps parameter names (symbols) to concrete values or to
`Parameter` instances that override the defaults. The function walks
through `obj` (which may be a `Sequence`, instruction, beam, or container)
and returns a new object in which all parametric entries have been replaced
by concrete numerical values.

An internal `IdDict` `cache` is used to ensure that identical objects are
only resolved once and then re-used, preserving pointer sharing across
the resolved structure.
"""
function resolve(obj, param_values; cache=nothing)
    cache === nothing && (cache = IdDict())
    _resolve(obj, param_values, cache)
end

"""
    _resolve(x::Number, param_values, cache::IdDict)

Internal helper: numbers are already concrete and are returned unchanged.
"""
function _resolve(x::Number, param_values, cache::IdDict) x end

"""
    _resolve(x::String, param_values, cache::IdDict)
    _resolve(x::Symbol, param_values, cache::IdDict)
    _resolve(x::Function, param_values, cache::IdDict)
    _resolve(x::TweezerArray, param_values, cache::IdDict)

Internal helpers: symbols, functions, and `TweezerArray` instances are
treated as already-resolved primitives and returned unchanged.
"""
function _resolve(x::String, param_values, cache::IdDict) x end
function _resolve(x::Symbol, param_values, cache::IdDict) x end
function _resolve(x::Function, param_values, cache::IdDict) x end
function _resolve(x::TweezerArray, param_values, cache::IdDict) x end

"""
    _resolve(x::Vector, param_values, cache::IdDict)
    _resolve(x::Tuple,  param_values, cache::IdDict)

Recursively resolve all elements of a vector or tuple, threading the
same `cache` so that shared substructures and deferred objects are
handled consistently.
"""
function _resolve(x::Vector, param_values, cache::IdDict)
    [_resolve(e, param_values, cache) for e in x]
end

function _resolve(x::Tuple, param_values, cache::IdDict)
    tuple((_resolve(xi, param_values, cache) for xi in x)...)
end

"""
    _resolve(seq::Sequence, param_values, cache::IdDict)

Resolve all instructions in a `Sequence`, preserving its time step `dt`.

Returns a new `Sequence` with the same `dt` and resolved instructions.
"""
function _resolve(seq::Sequence, param_values, cache::IdDict)
    Sequence([_resolve(inst, param_values, cache) for inst in seq.instructions], seq.dt)
end

"""
    _resolve(mc::MoveCol, param_values, cache::IdDict)
    _resolve(inst::Pulse, param_values, cache::IdDict)
    _resolve(inst::On,   param_values, cache::IdDict)
    _resolve(inst::Off,  param_values, cache::IdDict)

Resolve parametric fields and timing parameters appearing in instruction
objects (`MoveCol`, `Pulse`, `On`, `Off`), returning fully concrete
copies suitable for simulation.
"""
function _resolve(mc::MoveCol, param_values, cache::IdDict)
    MoveCol(
        mc.tweezers,
        _resolve(mc.cols, param_values, cache),
        _resolve(mc.delta, param_values, cache),
        _resolve(mc.duration, param_values, cache),
        _resolve(mc.sweep, param_values, cache),
        mc.dt,
        mc.downsample
    )
end

"""
    _resolve(mr::MoveRow, param_values, cache::IdDict)

Resolve parametric fields in a `MoveRow` instruction (delta may be a Parameter or ParametricExpression).
"""
function _resolve(mr::MoveRow, param_values, cache::IdDict)
    MoveRow(
        mr.tweezers,
        _resolve(mr.rows, param_values, cache),
        _resolve(mr.delta, param_values, cache),
        _resolve(mr.duration, param_values, cache),
        _resolve(mr.sweep, param_values, cache),
        mr.dt,
        mr.downsample
    )
end

function _resolve(inst::Pulse, param_values, cache::IdDict)
    Pulse(
        _resolve(inst.couplings, param_values, cache),
        _resolve(inst.duration, param_values, cache),
        _resolve(inst.ampl, param_values, cache),
        inst.amplitudes,
        inst.dt,
        inst.downsample,
        inst.interp,
    )
end

function _resolve(inst::On, param_values, cache::IdDict)
    On(_resolve(inst.couplings, param_values, cache), inst.dt, inst.downsample)
end

function _resolve(inst::Off, param_values, cache::IdDict)
    Off(_resolve(inst.couplings, param_values, cache), inst.dt, inst.downsample)
end

"""
    _resolve(p::Parallel, param_values, cache::IdDict)

Resolve each part of a `Parallel` instruction, preserving dt and downsample.
"""
function _resolve(p::Parallel, param_values, cache::IdDict)
    Parallel(
        [_resolve(part, param_values, cache) for part in p.parts],
        p.dt,
        p.downsample
    )
end

"""
    _resolve(b::ParametricBeam{T}, param_values, cache::IdDict) where T

Resolve a `ParametricBeam{T}` by resolving all parametric arguments and
constructing a concrete beam of type `T`.
"""
function _resolve(b::ParametricBeam{T}, param_values, cache::IdDict) where T
    resolved_args = [_resolve(arg, param_values, cache) for arg in b.args]
    resolved_kwargs = NamedTuple{keys(b.kwargs)}(
        [_resolve(val, param_values, cache) for val in values(b.kwargs)]
    )
    T(resolved_args...; resolved_kwargs...)
end

"""
    _resolve(sys::System, param_values, cache::IdDict)

Resolve a `System` by resolving its atoms and beams, while passing through
`initial_state`, `state`, `basis`, `nodes`, and `detector_specs` unchanged.

Returns a new `System` instance with parametric components concretized
according to `param_values`.
"""
function _resolve(sys::System, param_values, cache::IdDict)
    System(
        [_resolve(atom, param_values, cache) for atom in sys.atoms],
        _resolve(sys.beams, param_values, cache),
        sys.initial_state,
        sys.state,
        sys.basis,
        sys.nodes,
        sys.detector_specs
    )
end


"""
    _resolve(p::Parameter, param_values, cache::IdDict)

Resolve a scalar `Parameter` to a concrete numeric value.

If `param_values` contains an entry for `p.name`, that value (or
parameter) overrides the default `p.default` / `p.std`. A nonzero
`std` leads to a random draw `mean + randn()*std`, enabling static
noise sampling; otherwise the mean value is returned.
"""
function _resolve(p::Parameter, param_values, cache::IdDict)
    if haskey(param_values, p.name)
        val = param_values[p.name]
        mean_val, std_val = val isa Parameter ? (val.default, val.std) : (val, 0)
    else
        mean_val, std_val = p.default, p.std
    end
    std_val != 0 ? mean_val + randn() * std_val : mean_val
end

"""
    _resolve(expr::ParametricExpression, param_values, cache::IdDict)

Evaluate a `ParametricExpression` by first resolving all its arguments
and then applying the encoded operation.

Currently supports the binary operators `:*` and `:+`, corresponding to
multiplication and addition. An error is thrown if an unknown operator
symbol is encountered.
"""
function _resolve(expr::ParametricExpression, param_values, cache::IdDict)
    resolved_args = [_resolve(arg, param_values, cache) for arg in expr.args]
    if expr.op == :*
        return resolved_args[1] * resolved_args[2]
    elseif expr.op == :+
        return resolved_args[1] + resolved_args[2]
    else
        error("Unknown operation: $(expr.op)")
    end
end

"""
    _resolve(obj, param_values, cache::IdDict)

Fallback resolver: for unhandled types, return `obj` unchanged.

This ensures that objects which do not depend on parameters pass
through the resolution stage transparently.
"""
function _resolve(obj, param_values, cache::IdDict)
    obj
end

