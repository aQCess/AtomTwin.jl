"""
Add pairwise interaction terms between atoms by coupling specified single-atom transitions.
"""

"""
    add_interaction!(system,
                     atoms::Tuple{<:AbstractAtom,<:AbstractAtom},
                     transition::Pair,
                     V;
                     active = true)

Add a two-atom interaction term between specified single-atom transitions.

Arguments
- `system::System`: system to which the interaction is added
- `atoms::Tuple{AbstractAtom,AbstractAtom}`: pair of atoms involved
- `transition::Pair`: `(from1, from2) => (to1, to2)`: atom 1 does `from1 → to1`
  and atom 2 `from2 → to2`. The diagonal form `(a, b) => (a, b)` is the projector
  `|ab⟩⟨ab|` (an energy shift `V` of the pair state); an off-diagonal form such as
  `(g, e) => (e, g)` is the exchange `σ₁⁺σ₂⁻ + h.c.` at rate `V`. The levels may
  differ between the two atoms.
- `V`: interaction strength (rad/s); may be a number, `Parameter`, or
  `ParametricExpression`

An `InteractionNode` is constructed and built against `system.basis`.
The node is pushed to `sys.nodes` and the compiled `Interaction` field
is returned.
"""
function add_interaction!(
    system,
    atoms::Tuple{<:AbstractAtom,<:AbstractAtom},
    transition::Pair,
    V;
    active=true
)
    atom1, atom2 = atoms
    node = InteractionNode(V, (atom1, atom2), transition; active=active)
    build_node!(node, system.basis)
    push!(system, node)
    return node._field
end


"""
    add_vdwinteraction!(system, atoms, transition, C6; active=true, V_cap=C6/(1e-6)^6)

Add a van der Waals interaction V(r) = C6 / r⁶ between two atoms, its magnitude
capped at |V_cap|.

The interaction strength is recomputed from the instantaneous inter-atom
distance at every solver timestep. `C6` has units rad/s·m⁶ (ħ = 1) and may be a
number, `Parameter` or `ParametricExpression`. `transition` is read as for
[`add_interaction!`](@ref): `(r, r) => (r, r)` shifts the pair state `|rr⟩`.

# Keyword arguments
- `active`: whether the interaction is initially active (default `true`).
- `V_cap`: maximum interaction strength in rad/s (default `C6 / (1 µm)⁶`, from
  the default value of a parametric `C6`).
  At separations smaller than 1 µm the interaction is clamped to `V_cap`
  (in magnitude, so an attractive `C6 < 0` is capped too),
  preventing divergences in simulations where atoms overlap.
  Set to `Inf` to disable clamping.

Returns the compiled `VdWInteraction` field.
"""
function add_vdwinteraction!(
    system,
    atoms::Tuple{<:AbstractAtom,<:AbstractAtom},
    transition::Pair,
    C6;
    active = true,
    V_cap = nothing,
)
    V_cap === nothing && (V_cap = real(ComplexF64(_resolve_node_default(C6))) / (1e-6)^6)
    node = VdWInteractionNode(C6, atoms, transition; active = active, V_cap = Float64(V_cap))
    build_node!(node, system.basis)
    push!(system, node)
    return node._field
end
