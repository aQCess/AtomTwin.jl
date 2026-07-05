"""
Computation graph (DAG) for system specification.

Each node encodes the recipe for constructing one Hamiltonian term, jump
operator, or other system component. Nodes are built with default parameter
values at system construction time (via `build_node!`), then updated
in-place at `compile` time and on per-shot copies at `recompile!` time.

## Node lifecycle

1. `build_node!(node, ...)` — called at system build time. Creates the
   compiled object using default parameter values and stores it in the node.
   Returns the compiled object so callers can reference it.

2. `compile_node!(node, basis, rng, param_values)` — called by `compile`
   before each simulation. Samples parameter values and updates in-place.

3. `recompile_node!(node, obj, rng, param_values)` — called by `recompile!`
   for each shot on the thread-local job copy. Thread-safe.

## Value resolution protocol

Any type that stores parametric values and produces a concrete result at
compile time should implement:

- `_resolve_node_default(x)` — evaluate using parameter defaults
- `_resolve_node_value(x, param_values, rng)` — evaluate with sampling

This protocol is used by node lifecycle methods and by value types stored
inside nodes (e.g. `BeamRabiFrequency`, `GaussianPosition`, `MaxwellBoltzmann`).
Nodes appear in `sys.nodes` and are topologically sorted by `compile` before
each simulation; insertion order is preserved for independent nodes.
"""

abstract type AbstractNode end

"""Return the compiled output of a node (nothing if not yet built)."""
node_output(::AbstractNode) = nothing

"""
    node_dependencies(node::AbstractNode) -> Vector{AbstractNode}

Return nodes that must be compiled before `node`. The default returns no
dependencies. Override for nodes whose value fields reference other nodes
(e.g. `CouplingNode` with a `BeamRabiFrequency` Ω).
"""
node_dependencies(::AbstractNode) = AbstractNode[]

"""
    _extract_beam_node_deps(val) -> Vector{AbstractNode}

Return any `BeamNode` that `val` depends on. The default returns nothing.
Overridden for value types such as `BeamRabiFrequency` in `physics/couplings.jl`.
"""
_extract_beam_node_deps(::Any) = AbstractNode[]

#=============================================================================
SCALAR RESOLUTION HELPERS
=============================================================================#

"""Evaluate a node value at build time using parameter defaults."""
_resolve_node_default(x::Number)    = x
_resolve_node_default(p::Parameter) = p.default
_resolve_node_default(x)            = x  # fallback: vectors, arrays, etc.
function _resolve_node_default(expr::ParametricExpression)
    args = [_resolve_node_default(a) for a in expr.args]
    expr.op == :* && return prod(args)
    expr.op == :+ && return sum(args)
    expr.op == :- && return length(args) == 1 ? -args[1] : args[1] - args[2]
    expr.op == :inv && return inv(args[1])
    error("Unsupported op in ParametricExpression at build time: $(expr.op). " *
          "Supported ops: :+, :-, :*, :inv.")
end

"""Evaluate a node value at compile/recompile time (samples std if nonzero)."""
_resolve_node_value(x::Number, _, _) = x
_resolve_node_value(x, _, _)         = x  # fallback: vectors, arrays, etc.

function _resolve_node_value(p::Parameter, param_values, rng)
    if haskey(param_values, p.name)
        v = param_values[p.name]
        mean_val, std_val = v isa Parameter ? (v.default, v.std) : (v, 0.0)
    else
        mean_val, std_val = p.default, p.std
    end
    return std_val != 0.0 ? mean_val + randn(rng) * std_val : mean_val
end

function _resolve_node_value(expr::ParametricExpression, param_values, rng)
    args = [_resolve_node_value(a, param_values, rng) for a in expr.args]
    expr.op == :* && return prod(args)
    expr.op == :+ && return sum(args)
    expr.op == :- && return length(args) == 1 ? -args[1] : args[1] - args[2]
    expr.op == :inv && return inv(args[1])
    error("Unsupported op in ParametricExpression: $(expr.op). " *
          "Supported ops: :+, :-, :*, :inv.")
end

#=============================================================================
COUPLING NODE  (GlobalCoupling)
=============================================================================#

"""
    CouplingNode

Node for a two-level coupling (GlobalCoupling). `Ω` may be a plain `Number`,
a `Parameter`, or a `ParametricExpression`. The compiled `GlobalCoupling` is
stored in `_field` after `build_node!`.
"""
mutable struct CouplingNode <: AbstractNode
    Ω::Any
    atom::AbstractAtom
    transition::Pair{<:AbstractLevel, <:AbstractLevel}
    active::Bool
    _field::Union{Nothing, GlobalCoupling}
end

CouplingNode(Ω, atom, transition; active=true) =
    CouplingNode(Ω, atom, transition, active, nothing)

node_output(n::CouplingNode) = n._field

function build_node!(node::CouplingNode, basis::Basis)
    node._field === nothing || return node._field
    Ω_val = ComplexF64(_resolve_node_default(node.Ω))
    idx1  = node.atom.level_indices[node.transition[1]]
    idx2  = node.atom.level_indices[node.transition[2]]
    c = GlobalCoupling(basis, node.atom.inner, idx1 => idx2, Ω_val)
    c._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
    node._field = c
    return c
end

function compile_node!(node::CouplingNode, basis::Basis, rng, param_values)
    Ω_val = ComplexF64(_resolve_node_value(node.Ω, param_values, rng))
    if node._field === nothing
        idx1 = node.atom.level_indices[node.transition[1]]
        idx2 = node.atom.level_indices[node.transition[2]]
        c = GlobalCoupling(basis, node.atom.inner, idx1 => idx2, Ω_val)
        c._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
        node._field = c
    elseif node._field.rate == 0 && Ω_val != 0
        # Previous rate was zero — can't rescale from zero; rebuild H in-place.
        idx1 = node.atom.level_indices[node.transition[1]]
        idx2 = node.atom.level_indices[node.transition[2]]
        new_op = Op(basis, node.atom.inner, idx1 => idx2, Ω_val / 2)
        copy!(node._field.H.forward, new_op.forward)
        copy!(node._field.H.reverse, new_op.reverse)
        node._field.rate = Ω_val
    else
        update!(node._field, Val(:_), Ω_val)
    end
    return node._field
end

function recompile_node!(node::CouplingNode, c::GlobalCoupling, rng, param_values)
    Ω_val = ComplexF64(_resolve_node_value(node.Ω, param_values, rng))
    if c.rate == 0 && Ω_val != 0
        copy!(c.H.forward, node._field.H.forward)
        copy!(c.H.reverse, node._field.H.reverse)
        c.rate = node._field.rate
        update!(c, Val(:_), Ω_val)
    else
        update!(c, Val(:_), Ω_val)
    end
end

#=============================================================================
GAUSSIAN COUPLING NODE  (position-dependent Rabi frequency)
=============================================================================#

"""
    GaussianCouplingNode

Node for a spatially-varying coupling driven by a `GaussianBeam` or `GeneralGaussianBeam`.

Peak Rabi frequency Ω₀ is computed at build time from `rabi_frequency(...)` unless
`Ω0_override` is provided (useful when E1 selection rules underestimate the effective
coupling due to state mixing). Each solver timestep `update!` rescales `_coeff` by the
ratio of the current to build-time field scalar.
"""
mutable struct GaussianCouplingNode <: AbstractNode
    atom::AbstractAtom
    transition::Pair{<:AbstractLevel, <:AbstractLevel}
    beam::AbstractBeam
    q_axis::Vector{Float64}
    d_red::Float64
    g::Any
    e::Any
    Ω0_override::Union{Nothing, ComplexF64}
    active::Bool
    _field::Union{Nothing, GaussianCoupling}
end

GaussianCouplingNode(atom, transition, beam, q_axis, d_red, g, e;
                     Ω0_override = nothing, active = true) =
    GaussianCouplingNode(atom, transition, beam, q_axis, d_red, g, e,
                         Ω0_override, active, nothing)

node_output(n::GaussianCouplingNode) = n._field

function build_node!(node::GaussianCouplingNode, basis::Basis)
    node._field === nothing || return node._field
    idx1 = node.atom.level_indices[node.transition[1]]
    idx2 = node.atom.level_indices[node.transition[2]]
    Ω0 = node.Ω0_override !== nothing ? node.Ω0_override :
         ComplexF64(rabi_frequency(node.atom, node.g, node.e, node.beam, node.atom.x;
                                   q_axis = node.q_axis, d_red = node.d_red))
    c = GaussianCoupling(basis, node.atom.inner, idx1 => idx2, node.beam, Ω0)
    c._amplitude[] = node.active ? ComplexF64(1.0) : ComplexF64(0.0)
    node._field = c
    return c
end

function compile_node!(node::GaussianCouplingNode, basis::Basis, ::Any, ::Any)
    c = node._field
    c === nothing && return build_node!(node, basis)
    if node.Ω0_override === nothing
        # Recompute Ω₀ and E₀ at current atom position (may differ between shots)
        Ω0 = ComplexF64(rabi_frequency(node.atom, node.g, node.e, node.beam, node.atom.x;
                                       q_axis = node.q_axis, d_red = node.d_red))
        update!(c, Val(:_), Ω0)   # rescale H entries in-place
        c.E0 = ComplexF64(Dynamiq.efield_scalar(node.beam, node.atom.x))
    end
    # Override: Ω0/E0 ratio fixed at build time; only reset amplitude
    c._amplitude[] = node.active ? ComplexF64(1.0) : ComplexF64(0.0)
    return c
end

function recompile_node!(node::GaussianCouplingNode, ::GaussianCoupling, ::Any, ::Any)
    c = node._field
    c === nothing && error("GaussianCouplingNode not compiled before recompile!")
    if node.Ω0_override === nothing
        Ω0 = ComplexF64(rabi_frequency(node.atom, node.g, node.e, node.beam, node.atom.x;
                                       q_axis = node.q_axis, d_red = node.d_red))
        update!(c, Val(:_), Ω0)
        c.E0 = ComplexF64(Dynamiq.efield_scalar(node.beam, node.atom.x))
    end
    c._amplitude[] = node.active ? ComplexF64(1.0) : ComplexF64(0.0)
    return c
end

#=============================================================================
NOISY COUPLING NODE  (NoisyField wrapping GlobalCoupling)
=============================================================================#

"""
    NoisyCouplingNode

Node for a phase-noisy coupling. Wraps a `GlobalCoupling` in a `NoisyField`
so that per-shot phase noise is applied by the modifier loop in `recompile!`.
`Ω` may be a plain `Number`, a `Parameter`, or a `ParametricExpression`.
"""
mutable struct NoisyCouplingNode <: AbstractNode
    Ω::Any
    atom::AbstractAtom
    transition::Pair{<:AbstractLevel, <:AbstractLevel}
    noise::AbstractNoiseModel
    active::Bool
    _field::Union{Nothing, NoisyField}
end

NoisyCouplingNode(Ω, atom, transition, noise; active=true) =
    NoisyCouplingNode(Ω, atom, transition, noise, active, nothing)

node_output(n::NoisyCouplingNode) = n._field

function build_node!(node::NoisyCouplingNode, basis::Basis)
    node._field === nothing || return node._field
    Ω_val = ComplexF64(_resolve_node_default(node.Ω))
    idx1  = node.atom.level_indices[node.transition[1]]
    idx2  = node.atom.level_indices[node.transition[2]]
    c  = GlobalCoupling(basis, node.atom.inner, idx1 => idx2, Ω_val)
    c._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
    nf = NoisyField(c, node.noise)
    node._field = nf
    return nf
end

function compile_node!(node::NoisyCouplingNode, basis::Basis, rng, param_values)
    Ω_val = ComplexF64(_resolve_node_value(node.Ω, param_values, rng))
    if node._field === nothing
        idx1 = node.atom.level_indices[node.transition[1]]
        idx2 = node.atom.level_indices[node.transition[2]]
        c  = GlobalCoupling(basis, node.atom.inner, idx1 => idx2, Ω_val)
        c._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
        node._field = NoisyField(c, node.noise)
    else
        update!(node._field.coupling, Val(:_), Ω_val)
    end
    return node._field
end

function recompile_node!(node::NoisyCouplingNode, nf::NoisyField, rng, param_values)
    Ω_val = ComplexF64(_resolve_node_value(node.Ω, param_values, rng))
    update!(nf.coupling, Val(:_), Ω_val)
end

#=============================================================================
PLANAR COUPLING NODE  (PlanarCoupling)
=============================================================================#

"""
    PlanarCouplingNode

Node for a position-dependent coupling (PlanarCoupling). `Ω` may be a plain
`Number`, a `Parameter`, or a `ParametricExpression`. The compiled
`PlanarCoupling` is stored in `_field` after `build_node!`.
"""
mutable struct PlanarCouplingNode <: AbstractNode
    Ω::Any
    atom::AbstractAtom
    transition::Pair{<:AbstractLevel, <:AbstractLevel}
    beam::PlanarBeam
    active::Bool
    _field::Union{Nothing, PlanarCoupling}
    _current_rate::ComplexF64
end

PlanarCouplingNode(Ω, atom, transition, beam; active=true) =
    PlanarCouplingNode(Ω, atom, transition, beam, active, nothing, zero(ComplexF64))

node_output(n::PlanarCouplingNode) = n._field

function _rescale_planar!(c::PlanarCoupling, new_rate::ComplexF64, old_rate::ComplexF64)
    if old_rate != 0
        scale = new_rate / old_rate
        for k in eachindex(c.H.forward)
            i, j, v = c.H.forward[k]
            c.H.forward[k] = (i, j, v * scale)
        end
        for k in eachindex(c.H.reverse)
            i, j, v = c.H.reverse[k]
            c.H.reverse[k] = (i, j, v * scale)
        end
    end
end

function build_node!(node::PlanarCouplingNode, basis::Basis)
    node._field === nothing || return node._field
    Ω_val = ComplexF64(_resolve_node_default(node.Ω))
    idx1  = node.atom.level_indices[node.transition[1]]
    idx2  = node.atom.level_indices[node.transition[2]]
    c = PlanarCoupling(basis, node.atom.inner, idx1 => idx2, Ω_val, node.beam)
    c._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
    node._field = c
    node._current_rate = Ω_val
    return c
end

function compile_node!(node::PlanarCouplingNode, basis::Basis, rng, param_values)
    Ω_val = ComplexF64(_resolve_node_value(node.Ω, param_values, rng))
    if node._field === nothing
        idx1 = node.atom.level_indices[node.transition[1]]
        idx2 = node.atom.level_indices[node.transition[2]]
        c = PlanarCoupling(basis, node.atom.inner, idx1 => idx2, Ω_val, node.beam)
        c._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
        node._field = c
    else
        _rescale_planar!(node._field, Ω_val, node._current_rate)
    end
    node._current_rate = Ω_val
    return node._field
end

function recompile_node!(node::PlanarCouplingNode, c::PlanarCoupling, rng, param_values)
    Ω_val = ComplexF64(_resolve_node_value(node.Ω, param_values, rng))
    _rescale_planar!(c, Ω_val, node._current_rate)
end

#=============================================================================
DETUNING NODE  (Detuning)
=============================================================================#

"""
    DetuningNode

Node for a single-level energy shift (Detuning). `delta` is the physical
detuning in rad/s (positive = blue shift in rotating-frame convention).
"""
mutable struct DetuningNode <: AbstractNode
    delta::Any
    atom::AbstractAtom
    level::AbstractLevel
    active::Bool
    _field::Union{Nothing, Detuning}
end

DetuningNode(delta, atom, level; active=true) =
    DetuningNode(delta, atom, level, active, nothing)

node_output(n::DetuningNode) = n._field

function build_node!(node::DetuningNode, basis::Basis)
    node._field === nothing || return node._field
    delta_val = _resolve_node_default(node.delta)
    idx = node.atom.level_indices[node.level]
    d = Detuning(basis, node.atom.inner, idx, -delta_val)
    d._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
    node._field = d
    return d
end

function compile_node!(node::DetuningNode, basis::Basis, rng, param_values)
    delta_val = _resolve_node_value(node.delta, param_values, rng)
    if node._field === nothing
        idx = node.atom.level_indices[node.level]
        d = Detuning(basis, node.atom.inner, idx, -delta_val)
        d._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
        node._field = d
    else
        update!(node._field, Val(:_), delta_val)
    end
    return node._field
end

function recompile_node!(node::DetuningNode, d::Detuning, rng, param_values)
    delta_val = _resolve_node_value(node.delta, param_values, rng)
    update!(d, Val(:_), delta_val)
end

#=============================================================================
HAMILTONIAN NODE  (Hamiltonian — operator supplied directly)
=============================================================================#

"""
    HamiltonianNode

Node for a Hamiltonian term supplied as an explicit operator (see
[`add_hamiltonian!`](@ref)). `H` is a concrete `Op` already dimensioned to the
system basis; the node simply wraps it in a [`Hamiltonian`](@ref) field. The
operator is parameter-free (materialised at `add_hamiltonian!` time), so
compile/recompile only toggle the active coefficient.
"""
mutable struct HamiltonianNode <: AbstractNode
    H::Op
    active::Bool
    _field::Union{Nothing, Hamiltonian}
end

HamiltonianNode(H::Op; active=true) = HamiltonianNode(H, active, nothing)

node_output(n::HamiltonianNode) = n._field

function build_node!(node::HamiltonianNode, basis::Basis)
    node._field === nothing || return node._field
    h = Hamiltonian(node.H)
    h._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
    node._field = h
    return h
end

function compile_node!(node::HamiltonianNode, basis::Basis, rng, param_values)
    if node._field === nothing
        h = Hamiltonian(node.H)
        h._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
        node._field = h
    end
    return node._field
end

function recompile_node!(node::HamiltonianNode, h::Hamiltonian, rng, param_values)
    # Operator is static and parameter-free; nothing to resample.
    return h
end

#=============================================================================
DECAY NODE  (Jump)
=============================================================================#

"""
    DecayNode

Node for a spontaneous decay or dephasing jump operator. `Gamma` is the
decay rate in rad/s. `clicks` optionally holds the name of a `PhotoDetector`
(from a `PhotoDetectorSpec`) to which this jump's events are reported, so its
firings are counted as photon clicks; `nothing` means no photon detector.
"""
mutable struct DecayNode <: AbstractNode
    Gamma::Any
    atom::AbstractAtom
    transition::Pair{<:AbstractLevel, <:AbstractLevel}
    active::Bool
    clicks::Union{Nothing, String}
    _field::Union{Nothing, Jump}
end

DecayNode(Gamma, atom, transition; active=true, clicks=nothing) =
    DecayNode(Gamma, atom, transition, active, clicks, nothing)

node_output(n::DecayNode) = n._field

function build_node!(node::DecayNode, basis::Basis)
    node._field === nothing || return node._field
    Gamma_val = _resolve_node_default(node.Gamma)
    idx1 = node.atom.level_indices[node.transition[1]]
    idx2 = node.atom.level_indices[node.transition[2]]
    j = Jump(basis, node.atom.inner, idx1 => idx2, Gamma_val)
    j._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
    node._field = j
    return j
end

function compile_node!(node::DecayNode, basis::Basis, rng, param_values)
    Gamma_val = _resolve_node_value(node.Gamma, param_values, rng)
    if node._field === nothing || (node._field._rate == 0 && Gamma_val != 0)
        idx1 = node.atom.level_indices[node.transition[1]]
        idx2 = node.atom.level_indices[node.transition[2]]
        j = Jump(basis, node.atom.inner, idx1 => idx2, Gamma_val)
        j._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
        node._field = j
    else
        update!(node._field, Val(:_), Gamma_val)
    end
    return node._field
end

function recompile_node!(node::DecayNode, j::Jump, rng, param_values)
    Gamma_val = _resolve_node_value(node.Gamma, param_values, rng)
    if j._rate == 0 && Gamma_val != 0
        copy!(j.J.forward, node._field.J.forward)
        j._rate = node._field._rate
    end
    update!(j, Val(:_), Gamma_val)
end

#=============================================================================
INTERACTION NODE  (Interaction — two-atom coupling)
=============================================================================#

"""
    InteractionNode

Node for a pairwise two-atom interaction (e.g. Rydberg blockade). `V` may be
a plain `Number`, a `Parameter`, or a `ParametricExpression`.
"""
mutable struct InteractionNode <: AbstractNode
    V::Any
    atoms::Tuple{<:AbstractAtom, <:AbstractAtom}
    transition::Pair          # (from_tuple => to_tuple) of level tuples
    active::Bool
    _field::Union{Nothing, Interaction}
    _current_value::ComplexF64
end

InteractionNode(V, atoms, transition; active=true) =
    InteractionNode(V, atoms, transition, active, nothing, zero(ComplexF64))

node_output(n::InteractionNode) = n._field

function _interaction_transitions(node::InteractionNode)
    atom1, atom2 = node.atoms
    from_tuple, to_tuple = node.transition.first, node.transition.second
    # `(from1, from2) => (to1, to2)`: atom1 does from1→to1, atom2 does from2→to2.
    # The diagonal form `(a,b) => (a,b)` gives the projector |ab⟩⟨ab|; an
    # off-diagonal form like `(0,1) => (1,0)` gives the exchange σ₁⁺σ₂⁻ + h.c.
    t1 = atom1.level_indices[from_tuple[1]] => atom1.level_indices[to_tuple[1]]
    t2 = atom2.level_indices[from_tuple[2]] => atom2.level_indices[to_tuple[2]]
    return t1, t2
end

function _rescale_interaction!(inter::Interaction, new_val::ComplexF64, old_val::ComplexF64)
    if old_val != 0
        scale = new_val / old_val
        for k in eachindex(inter.H.forward)
            i, j, v = inter.H.forward[k]; inter.H.forward[k] = (i, j, v * scale)
        end
        for k in eachindex(inter.H.reverse)
            i, j, v = inter.H.reverse[k]; inter.H.reverse[k] = (i, j, v * scale)
        end
    end
end

function build_node!(node::InteractionNode, basis::Basis)
    node._field === nothing || return node._field
    V_val = ComplexF64(_resolve_node_default(node.V))
    atom1, atom2 = node.atoms
    t1, t2 = _interaction_transitions(node)
    inter = Interaction(basis, atom1.inner => atom2.inner, t1, t2, V_val)
    inter._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
    node._field = inter
    node._current_value = V_val
    return inter
end

function compile_node!(node::InteractionNode, basis::Basis, rng, param_values)
    V_val = ComplexF64(_resolve_node_value(node.V, param_values, rng))
    if node._field === nothing
        atom1, atom2 = node.atoms
        t1, t2 = _interaction_transitions(node)
        inter = Interaction(basis, atom1.inner => atom2.inner, t1, t2, V_val)
        inter._coeff[] = ComplexF64(node.active ? 1.0 : 0.0)
        node._field = inter
    elseif node._current_value == 0 && V_val != 0
        atom1, atom2 = node.atoms
        t1, t2 = _interaction_transitions(node)
        new_op = Op(basis, atom1.inner => atom2.inner, t1, t2, V_val)
        copy!(node._field.H.forward, new_op.forward)
        copy!(node._field.H.reverse, new_op.reverse)
    else
        _rescale_interaction!(node._field, V_val, node._current_value)
    end
    node._current_value = V_val
    return node._field
end

function recompile_node!(node::InteractionNode, inter::Interaction, rng, param_values)
    V_val = ComplexF64(_resolve_node_value(node.V, param_values, rng))
    if node._current_value == 0 && V_val != 0
        copy!(inter.H.forward, node._field.H.forward)
        copy!(inter.H.reverse, node._field.H.reverse)
    end
    _rescale_interaction!(inter, V_val, node._current_value)
end

#=============================================================================
VdW INTERACTION NODE  (distance-dependent C6/r^6 interaction)
=============================================================================#

"""
    VdWInteractionNode

DAG node for a van der Waals interaction V(r) = C6 / r⁶.

`C6` may be a plain `Float64`, a `Parameter`, or a `ParametricExpression`.
The compiled `VdWInteraction` field recomputes its `_coeff` from the
instantaneous inter-atom distance at every solver timestep.
"""
mutable struct VdWInteractionNode <: AbstractNode
    C6::Any
    atoms::Tuple{<:AbstractAtom, <:AbstractAtom}
    transition::Pair
    active::Bool
    _field::Union{Nothing, VdWInteraction}
    _current_value::ComplexF64
    V_cap::Float64
end

VdWInteractionNode(C6, atoms, transition; active = true, V_cap = Inf) =
    VdWInteractionNode(C6, atoms, transition, active, nothing, zero(ComplexF64), Float64(V_cap))

node_output(n::VdWInteractionNode) = n._field

function _vdw_transitions(node::VdWInteractionNode)
    atom1, atom2 = node.atoms
    from_tuple = node.transition.first
    t1 = atom1.level_indices[from_tuple[1]] => atom1.level_indices[from_tuple[2]]
    t2 = atom2.level_indices[from_tuple[1]] => atom2.level_indices[from_tuple[2]]
    return t1, t2
end

function build_node!(node::VdWInteractionNode, basis::Basis)
    node._field === nothing || return node._field
    C6_val = Float64(real(ComplexF64(_resolve_node_default(node.C6))))
    atom1, atom2 = node.atoms
    t1, t2 = _vdw_transitions(node)
    inter = VdWInteraction(basis, atom1.inner => atom2.inner, t1, t2, C6_val;
                           V_cap = node.V_cap)
    inter._coeff[] = node.active ? ComplexF64(C6_val) : ComplexF64(0)
    node._field = inter
    node._current_value = ComplexF64(C6_val)
    return inter
end

function compile_node!(node::VdWInteractionNode, basis::Basis, rng, param_values)
    C6_val = Float64(real(ComplexF64(_resolve_node_value(node.C6, param_values, rng))))
    if node._field === nothing
        atom1, atom2 = node.atoms
        t1, t2 = _vdw_transitions(node)
        inter = VdWInteraction(basis, atom1.inner => atom2.inner, t1, t2, C6_val;
                               V_cap = node.V_cap)
        inter._coeff[] = node.active ? ComplexF64(C6_val) : ComplexF64(0)
        node._field = inter
    else
        node._field.C6    = C6_val
        node._field.V_cap = node.V_cap
        node._field._coeff[] = node.active ? ComplexF64(C6_val) : ComplexF64(0)
    end
    node._current_value = ComplexF64(C6_val)
    return node._field
end

function recompile_node!(node::VdWInteractionNode, inter::VdWInteraction, rng, param_values)
    C6_val = Float64(real(ComplexF64(_resolve_node_value(node.C6, param_values, rng))))
    inter.C6    = C6_val
    inter.V_cap = node.V_cap
    inter._coeff[] = node.active ? ComplexF64(C6_val) : ComplexF64(0)
    node._current_value = ComplexF64(C6_val)
end

#=============================================================================
BEAM NODE  (resolves ParametricBeam to a concrete AbstractBeam)
=============================================================================#

"""
    BeamNode

DAG node that resolves a `ParametricBeam` (or concrete beam) to a concrete
`AbstractBeam` at compile/recompile time.

`BeamRabiFrequency` holds a reference to a `BeamNode` and reads
`beam_node._compiled[]` when computing Rabi frequencies. Compilation order
is resolved automatically via topological sort; `BeamNode` does not need to
be pushed before its dependents.

`_resolve_beam_default`, `_resolve_beam`, `build_node!`, `compile_node!`, and
`recompile_node!` for `BeamNode` are defined in `physics/beams.jl` (after
`ParametricBeam` is available).
"""
mutable struct BeamNode <: AbstractNode
    beam::Any                                   # ParametricBeam or concrete AbstractBeam
    _compiled::Ref{Union{Nothing, AbstractBeam}}
    BeamNode(beam) = new(beam, Ref{Union{Nothing, AbstractBeam}}(nothing))
end

node_output(n::BeamNode) = n._compiled[]

#=============================================================================
FALLBACK RECOMPILE (non-parametric nodes need no update)
=============================================================================#

recompile_node!(::AbstractNode, ::Any, ::Any, ::Any) = nothing

#=============================================================================
NODE DEPENDENCY OVERRIDES  (CouplingNode, NoisyCouplingNode, PlanarCouplingNode)
=============================================================================#

# These node types store Ω::Any which may be a BeamRabiFrequency (defined in
# physics/couplings.jl). _extract_beam_node_deps is overridden there.
node_dependencies(n::CouplingNode)       = _extract_beam_node_deps(n.Ω)
node_dependencies(n::NoisyCouplingNode)  = _extract_beam_node_deps(n.Ω)
node_dependencies(n::PlanarCouplingNode) = _extract_beam_node_deps(n.Ω)

#=============================================================================
TOPOLOGICAL SORT
=============================================================================#

"""
    _topological_sort(nodes::Vector{AbstractNode}) -> Vector{AbstractNode}

Return a topologically sorted copy of `nodes` using stable Kahn's algorithm.
Nodes with no dependency relationship are emitted in their original insertion
order. Does not mutate `nodes`.

Throws if a cycle is detected or if `node_dependencies` returns a node not
present in `nodes`.
"""
function _topological_sort(nodes::Vector{AbstractNode})
    index = IdDict{AbstractNode, Int}(n => i for (i, n) in enumerate(nodes))
    n = length(nodes)
    in_degree = zeros(Int, n)
    adj = [Int[] for _ in 1:n]
    for (i, node) in enumerate(nodes)
        for dep in node_dependencies(node)
            j = get(index, dep, nothing)
            j === nothing && error(
                "node_dependencies returned a node not present in sys.nodes. " *
                "Dependent: $(typeof(node)), missing dependency: $(typeof(dep)).")
            push!(adj[j], i)
            in_degree[i] += 1
        end
    end
    # Start with all zero-in-degree nodes in insertion order
    queue = Int[i for i in 1:n if in_degree[i] == 0]
    result = AbstractNode[]
    sizehint!(result, n)
    while !isempty(queue)
        i = popfirst!(queue)
        push!(result, nodes[i])
        for j in adj[i]
            in_degree[j] -= 1
            if in_degree[j] == 0
                insert!(queue, searchsortedfirst(queue, j), j)
            end
        end
    end
    if length(result) < n
        cycle_nodes = join([string(typeof(nodes[i])) for i in 1:n if in_degree[i] > 0], ", ")
        error("Cycle detected in DAG node dependencies. Nodes involved: $cycle_nodes.")
    end
    return result
end
