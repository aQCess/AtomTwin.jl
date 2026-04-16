#------------------------------------------------------------------------------
# Abstract core types
#------------------------------------------------------------------------------

"""
AbstractAtom

Abstract supertype for all atomic models used in Dynamiq.
Concrete implementations represent specific internal-level structures and motion.
"""
abstract type AbstractAtom end

"""
base_atom(atom::AbstractAtom)

Return the underlying low-level atom representation for `atom`.

For most concrete types this is the atom itself, but wrappers (such as
higher-level convenience types) may override this to expose an internal core.
"""
base_atom(atom::AbstractAtom) = atom

"""
AbstractBeam

Abstract supertype for all laser or optical beam models.
Concrete subtypes describe spatial profiles, polarizations, and time dependence.
"""
abstract type AbstractBeam end


"""
AbstractField

Abstract supertype for classical fields that can couple to atoms.

Fields may be static or time dependent, and can represent composite
configurations built from multiple beams or modes.
"""
abstract type AbstractField end  # can be static or time-dependent

# Default no-op: subtypes that don't need per-step updates inherit this.
# Explicit return Nothing ensures a uniform inferred return type across all
# AbstractField subtypes, preventing boxing when dispatching through the
# abstract type (e.g. in `for f in fields; update!(f, i); end`).
update!(::AbstractField, ::Int) = nothing

"""
base_coupling(coupling::AbstractField)

Return the underlying low-level coupling object for `coupling`.

Simple field types typically return themselves, while higher-level
composite objects may override this to expose an internal representation.
"""
base_coupling(coupling::AbstractField) = coupling

"""
AbstractDissipator{A}

Abstract supertype for dissipative processes acting on atoms of type `A`.

Concrete subtypes implement, for example, spontaneous emission, dephasing,
or other Lindblad-like channels.
"""
abstract type AbstractDissipator{A} end

"""
AbstractModifier

Abstract supertype for modifiers that transform fields, beams, or atom
trajectories in time (e.g. motion, amplitude, or phase modifiers).

`update!(m, i)` is called at every solver timestep `i`.
"""
abstract type AbstractModifier end

"""
AbstractBoundaryModifier

Abstract supertype for modifiers that are called once at instruction
boundaries — never inside the solver loop.

`begin_instruction!(m)` is called once before `evolve!` starts.
`end_instruction!(m)` is called once after `evolve!` completes.

This allows coupling amplitudes to be set/reset at instruction boundaries
without incurring per-timestep dispatch overhead.
"""
abstract type AbstractBoundaryModifier end

begin_instruction!(::AbstractBoundaryModifier) = nothing
end_instruction!(::AbstractBoundaryModifier) = nothing

"""
AbstractDetector{A}

Abstract supertype for detectors that measure properties of atoms of type `A`.

Concrete implementations define how populations, coherences, motion,
or field observables are extracted from a simulation.
"""
abstract type AbstractDetector end
