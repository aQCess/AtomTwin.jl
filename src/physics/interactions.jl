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
    add_vdwinteraction!(system, atoms, transition, C6; active=true)

Add a van der Waals interaction V(r) = C6 / r⁶ between two atoms.

The interaction strength is recomputed from the instantaneous inter-atom
distance at every solver timestep. `C6` has units rad/s·m⁶ (ħ = 1).

Returns the compiled `VdWInteraction` field.
"""
function add_vdwinteraction!(
    system,
    atoms::Tuple{<:AbstractAtom,<:AbstractAtom},
    transition::Pair,
    C6;
    active = true,
)
    node = VdWInteractionNode(C6, atoms, transition; active = active)
    build_node!(node, system.basis)
    push!(system, node)
    return node._field
end
