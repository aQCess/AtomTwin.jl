
"""
    add_lightshifts!(system, atom)

Add AC Stark shifts onto all levels of the atom, for every beam within 
`system.beams` – this excludes coupling beams.

Interference between different beams is not yet taken into account

atom needs to be of type AtomWrapper
"""
function add_lightshifts!(
    system, atom; 
    q_axis::AbstractVector{<:Real} = [0.0, 0.0, 1.0]
)
    lvls = atom.levels
    lvl_indices = atom.level_indices
    beams = system.beams

    stark_shifts = Dynamiq.AbstractField[]

    for l in lvls
        for b in beams
            # instantiate a StarkShiftACNode(b, atom, l)

            #push!(stark_shifts, node._field)
        end
    end

end