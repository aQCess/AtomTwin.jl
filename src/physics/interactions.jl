"""
Add pairwise interaction terms between atoms by coupling specified single-atom transitions.
"""

"""
    add_interaction!(system,
                     atoms::Tuple{<:AbstractAtom,<:AbstractAtom},
                     transition::Pair,
                     V;
                     noise = nothing,
                     active = true)

Add a two-atom interaction term between specified single-atom transitions.

Arguments
- `system::System`: system to which the interaction is added
- `atoms::Tuple{AbstractAtom,AbstractAtom}`: pair of atoms involved
- `transition::Pair`: pair `(from => to)` of level tuples, specifying
  the single-atom transitions (e.g. `(g,e) => (g,e)`)
- `V`: interaction strength (e.g. energy shift or coupling rate);
  may be a number, `Parameter`, or `ParametricExpression`

An `InteractionNode` is constructed and built against `system.basis`.
The node is pushed to `sys.nodes` and the compiled `Interaction` field
is returned.
"""
function add_interaction!(
    system,
    atoms::Tuple{<:AbstractAtom,<:AbstractAtom},
    transition::Pair,
    V;
    noise=nothing,
    active=true
)
    # Unpack atoms
    atom1, atom2 = atoms

    # Map levels in transition tuples to indices for each atom
    from_tuple, to_tuple = transition[1], transition[2]
    transition1 = atom1.level_indices[from_tuple[1]]=>atom1.level_indices[from_tuple[2]]
    transition2 = atom2.level_indices[from_tuple[1]]=>atom2.level_indices[from_tuple[2]]

    node = InteractionNode(V, (atom1, atom2), transition; active=active)
    build_node!(node, system.basis)
    push!(system, node)
    return node._field
end


"""
    add_vdwinteraction!(system, atoms, transition, C6; active=true, V_cap=C6/(1e-6)^6)

Add a van der Waals interaction V(r) = min(C6 / r⁶, V_cap) between two atoms.

The interaction strength is recomputed from the instantaneous inter-atom
distance at every solver timestep. `C6` has units rad/s·m⁶ (ħ = 1).

# Keyword arguments
- `active`: whether the interaction is initially active (default `true`).
- `V_cap`: maximum interaction strength in rad/s (default `C6 / (1 µm)⁶`).
  At separations smaller than 1 µm the interaction is clamped to `V_cap`,
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
    V_cap = C6 / (1e-6)^6,
)
    node = VdWInteractionNode(C6, atoms, transition; active = active, V_cap = Float64(V_cap))
    build_node!(node, system.basis)
    push!(system, node)
    return node._field
end
