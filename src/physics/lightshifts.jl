
"""
    add_lightshifts!(system, atom)

Add AC Stark shifts onto all levels of the atom, for every beam within 
`system.beams` – this excludes coupling beams.

Interference between beams is not taken into account
"""
function add_lightshifts!(
    system, atom::AbstractAtom; 
    q_axis::AbstractVector{<:Real} = [0.0, 0.0, 1.0],
    active = true
)
    q_axis_vec = Vector{Float64}(q_axis)
    lightshifts = Dynamiq.AbstractField[]

    for l in atom.levels
        for b in system.beams
            node = StarkShiftACNode(atom, l, b; q_axis = q_axis_vec, active = active)
            build_node!(node, system.basis)
            push!(system, node)
            push!(lightshifts, node._field)
        end
    end

    return lightshifts
end

function add_lightshifts!(
    system, atoms::Vector{AbstractAtom}; 
    q_axis::AbstractVector{<:Real} = [0.0, 0.0, 1.0],
    active = true
)
    lightshifts = Dynamiq.AbstractField[]

    for a in atoms
        ls = add_lightshifts!(system, a; q_axis = q_axis, active = active)
        append!(lightshifts, ls)
    end

    return lightshifts
end

function add_lightshifts!(
    system; 
    q_axis::AbstractVector{<:Real} = [0.0, 0.0, 1.0],
    active = true
)
    return add_lightshifts!(system, system.atoms; q_axis = q_axis, active = active)
end