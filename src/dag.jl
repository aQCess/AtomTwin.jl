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

_rate_value(n::CouplingNode) = n.Ω

function _new_field(node::CouplingNode, basis::Basis, Ω::ComplexF64)
    idx1 = node.atom.level_indices[node.transition[1]]
    idx2 = node.atom.level_indices[node.transition[2]]
    return GlobalCoupling(basis, node.atom.inner, idx1 => idx2, Ω)
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

# Updates `c`, the JOB's field, at the job's atom position `c.atom.x`. Not
# `node._field` and `node.atom`: a multi-threaded run gives each thread a deep
# copy of the job, and only that copy was re-initialised for this shot.
function recompile_node!(node::GaussianCouplingNode, c::GaussianCoupling, ::Any, ::Any)
    if node.Ω0_override === nothing
        Ω0 = ComplexF64(rabi_frequency(node.atom, node.g, node.e, node.beam, c.atom.x;
                                       q_axis = node.q_axis, d_red = node.d_red))
        update!(c, Val(:_), Ω0)
        c.E0 = ComplexF64(Dynamiq.efield_scalar(node.beam, c.atom.x))
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
end

PlanarCouplingNode(Ω, atom, transition, beam; active=true) =
    PlanarCouplingNode(Ω, atom, transition, beam, active, nothing)

node_output(n::PlanarCouplingNode) = n._field

_rate_value(n::PlanarCouplingNode) = n.Ω

function _new_field(node::PlanarCouplingNode, basis::Basis, Ω::ComplexF64)
    idx1 = node.atom.level_indices[node.transition[1]]
    idx2 = node.atom.level_indices[node.transition[2]]
    return PlanarCoupling(basis, node.atom.inner, idx1 => idx2, Ω, node.beam)
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

# A no-op unless the rate changed (a sampled rate): `update!` then clears the
# cached L†L, which `evolve!` rebuilds.
function recompile_node!(node::DecayNode, j::Jump, rng, param_values)
    Gamma_val = _resolve_node_value(node.Gamma, param_values, rng)
    if j._rate == 0 && Gamma_val != 0
        copy!(j.J.forward, node._field.J.forward)
        j._rate = node._field._rate
    end
    update!(j, Val(:_), Gamma_val)
    return j
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
end

InteractionNode(V, atoms, transition; active=true) =
    InteractionNode(V, atoms, transition, active, nothing)

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

_rate_value(n::InteractionNode) = n.V

function _new_field(node::InteractionNode, basis::Basis, V::ComplexF64)
    atom1, atom2 = node.atoms
    t1, t2 = _interaction_transitions(node)
    return Interaction(basis, atom1.inner => atom2.inner, t1, t2, V)
end

#=============================================================================
RATE-CARRYING NODES  (CouplingNode, PlanarCouplingNode, InteractionNode)

Each builds a field whose strength is baked into its operator and recorded in
`field.rate`; a new value rescales the operator in place (`update!`). They
differ only in the field they build (`_new_field`) and where the rate is
declared (`_rate_value`).
=============================================================================#

const RatedNode = Union{CouplingNode, PlanarCouplingNode, InteractionNode}

function build_node!(node::RatedNode, basis::Basis)
    node._field === nothing || return node._field
    f = _new_field(node, basis, ComplexF64(_resolve_node_default(_rate_value(node))))
    _reset_activity!(node, f)
    return node._field = f
end

function compile_node!(node::RatedNode, basis::Basis, rng, param_values)
    rate = ComplexF64(_resolve_node_value(_rate_value(node), param_values, rng))
    f = node._field
    if f === nothing
        f = node._field = _new_field(node, basis, rate)
        _reset_activity!(node, f)
    elseif f.rate == 0 && rate != 0
        # A zero-rate operator has zero entries and cannot be rescaled: rebuild.
        fresh = _new_field(node, basis, rate)
        copy!(f.H.forward, fresh.H.forward)
        copy!(f.H.reverse, fresh.H.reverse)
        f.rate = rate
    else
        update!(f, Val(:_), rate)
    end
    return f
end

recompile_node!(node::RatedNode, f, rng, param_values) =
    _set_rate!(f, node._field, ComplexF64(_resolve_node_value(_rate_value(node), param_values, rng)))

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
    V_cap::Float64
end

VdWInteractionNode(C6, atoms, transition; active = true, V_cap = Inf) =
    VdWInteractionNode(C6, atoms, transition, active, nothing, Float64(V_cap))

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
    inter._amplitude[] = node.active ? 1.0 : 0.0
    node._field = inter
    return inter
end

function compile_node!(node::VdWInteractionNode, basis::Basis, rng, param_values)
    C6_val = Float64(real(ComplexF64(_resolve_node_value(node.C6, param_values, rng))))
    if node._field === nothing
        atom1, atom2 = node.atoms
        t1, t2 = _vdw_transitions(node)
        inter = VdWInteraction(basis, atom1.inner => atom2.inner, t1, t2, C6_val;
                               V_cap = node.V_cap)
        inter._amplitude[] = node.active ? 1.0 : 0.0
        node._field = inter
    else
        recompile_node!(node, node._field, rng, param_values)
    end
    return node._field
end

# `_coeff` is recomputed from the separation by `update!`; only C6 and the
# commanded amplitude are set here.
function recompile_node!(node::VdWInteractionNode, inter::VdWInteraction, rng, param_values)
    inter.C6    = Float64(real(ComplexF64(_resolve_node_value(node.C6, param_values, rng))))
    inter.V_cap = node.V_cap
    inter._amplitude[] = node.active ? 1.0 : 0.0
    return inter
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

"""
    LightShiftNode <: AbstractNode

AC Stark shift of one level in a trap beam, added by [`add_light_shift!`](@ref).

Compiles to a [`StarkShiftAC`](@ref) field, which reads `atom.alpha` — the same
per-level polarizability the dipole force uses, tensor contribution included. The
node is built late (Phase 3), by which time `initialize!` has filled that array.
"""
mutable struct LightShiftNode <: AbstractNode
    atom::AbstractAtom
    level::AbstractLevel
    beam::Any                       # AbstractBeam or BeamNode
    reference::Union{Nothing, AbstractLevel}
    active::Bool
    _field::Any
end

LightShiftNode(atom, level, beam; reference=nothing, active=true) =
    LightShiftNode(atom, level, beam, reference, active, nothing)

node_output(n::LightShiftNode) = n._field

# A beam given as a BeamNode must be resolved first.
node_dependencies(n::LightShiftNode) = n.beam isa BeamNode ? AbstractNode[n.beam] :
                                                             AbstractNode[]

_lightshift_beam(n::LightShiftNode) = n.beam isa BeamNode ? n.beam._compiled[] : n.beam

function _build_lightshift(node::LightShiftNode, basis::Basis)
    beam = _lightshift_beam(node)
    beam === nothing && error("LightShiftNode: beam not resolved yet")
    idx  = node.atom.level_indices[node.level]
    ref  = node.reference === nothing ? nothing :
           node.atom.level_indices[node.reference]
    λ = getwavelength(beam)
    haskey(node.atom.inner.alpha, λ) || error(
        "no polarizability for λ = $(λ*1e9) nm on this atom. add_light_shift! needs " *
        "the beam to be part of the system (pass it to `System`), so that " *
        "`initialize!` computes α at its wavelength.")
    f = StarkShiftAC(basis, node.atom.inner, idx, beam; reference = ref)
    # `_coeff` too, so `gethamiltonian` shows the peak shift before any `play`.
    f._amplitude[] = f._coeff[] = node.active ? 1.0 : 0.0
    node._field = f
    return f
end

build_node!(node::LightShiftNode, basis::Basis) =
    node._field === nothing ? _build_lightshift(node, basis) : node._field

function compile_node!(node::LightShiftNode, basis::Basis, rng, param_values)
    node._field = nothing          # α and the beam may both have changed
    return _build_lightshift(node, basis)
end

# Per shot, `initialize!` refreshes atom.alpha (resampled parameters, a resampled
# quantization axis). `StarkShiftAC` caches α at construction, so refresh that in
# place rather than rebuild -- the job holds `f` by reference, and `recompile!`
# does not use the return value.
#
# α is read from `f.atom`, the job's own atom, not `node.atom.inner`: a
# multi-threaded run gives each thread a deep copy of the job, and only that copy
# was re-initialised for this shot.
function recompile_node!(node::LightShiftNode, f::StarkShiftAC, rng, param_values)
    beam = _lightshift_beam(node)
    beam === nothing && return f
    alphas = f.atom.alpha[getwavelength(beam)]
    α = node.reference === nothing ? alphas[f.level] :
        alphas[f.level] - alphas[node.atom.level_indices[node.reference]]
    f._amplitude[] = node.active ? 1.0 : 0.0
    return Dynamiq.set_alpha!(f, α)
end

"""
    QuantizationAxisNode <: AbstractNode

The system's quantization axis — the direction magnetic sublevels are defined
against. Added with [`add_quantization_axis!`](@ref); defaults to `ẑ` when no node
is present.

The tensor light shift depends on the angle between the trap polarization and this
axis, so it must be known before atom polarizabilities are computed. Like
[`BeamNode`](@ref) it is therefore resolved in compile Phase 1, ahead of
`initialize!`.

The axis may be a `Parameter` or `ParametricExpression`, which is what makes a
magic-angle sweep — or a shot-to-shot B-field misalignment — a `play` keyword
rather than a rebuild.
"""
mutable struct QuantizationAxisNode <: AbstractNode
    axis::Any                                    # 3-vector, possibly parametric
    _compiled::Ref{Union{Nothing, Vector{Float64}}}
    QuantizationAxisNode(axis) = new(axis, Ref{Union{Nothing, Vector{Float64}}}(nothing))
end

node_output(n::QuantizationAxisNode) = n._compiled[]

# Normalise to a unit 3-vector; a zero axis has no direction to define m against.
function _unit_axis(v)
    a = Float64.(collect(v))
    length(a) == 3 || error("quantization axis must be a 3-vector; got length $(length(a))")
    n = sqrt(sum(abs2, a))
    n == 0 && error("quantization axis must be nonzero")
    return a ./ n
end

build_node!(node::QuantizationAxisNode) =
    node._compiled[] = _unit_axis(_resolve_node_default(node.axis))

compile_node!(node::QuantizationAxisNode, basis, rng, param_values) =
    node._compiled[] = _unit_axis(_resolve_node_value(node.axis, param_values, rng))

recompile_node!(node::QuantizationAxisNode, ::Any, rng, param_values) =
    compile_node!(node, nothing, rng, param_values)

#=============================================================================
FALLBACK RECOMPILE (non-parametric nodes need no update)
=============================================================================#

recompile_node!(::AbstractNode, ::Any, ::Any, ::Any) = nothing

# Every switchable field starts a run, and every shot, in its declared state. A
# sequence may leave it switched on (`On` with no `Off`); without this the next
# shot -- and the next `play` of the same system -- began with it on.
_reset_activity!(node::AbstractNode, f) =
    hasproperty(node, :active) && (Dynamiq.envelope(f)[] = node.active ? 1.0 : 0.0)

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
