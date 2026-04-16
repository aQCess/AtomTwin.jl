"""
Attach population, coherence, motion, and field detectors to a `System` to record observables.
"""

import .Dynamiq: PopulationDetectorSpec
import .Dynamiq: CoherenceDetectorSpec
import .Dynamiq: FieldDetectorSpec
# import Dynamiq: MotionDetectorSpec

"""
    PopulationDetectorSpec(atom, level::AbstractLevel; name = "") -> Dynamiq.DetectorSpec

AtomTwin convenience wrapper to define a population detector using level
objects instead of integer indices.

Arguments
- `atom::AbstractAtom`: atom whose population is to be monitored
- `level::AbstractLevel`: target level (e.g. `g`, `e`, `r`)

Keyword arguments
- `name::String`: optional detector name; defaults to the empty string

This converts `level` to its integer index via `atom.level_indices` and
constructs a `Dynamiq.PopulationDetectorSpec` with that index.
"""
function PopulationDetectorSpec(atom::AbstractAtom, level::AbstractLevel; name::String="")
    level_idx = atom.level_indices[level]
    return Dynamiq.PopulationDetectorSpec(atom; level=level_idx, name=name)
end

"""
    CoherenceDetectorSpec(atom, levels::Pair{<:AbstractLevel,<:AbstractLevel}; name = "") -> Dynamiq.DetectorSpec

AtomTwin convenience wrapper to define a coherence detector between two
levels using `AbstractLevel` objects.

Arguments
- `atom::AbstractAtom`: atom whose coherence is to be monitored
- `levels::Pair`: pair `ℓ1 => ℓ2` of levels defining the coherence

Keyword arguments
- `name::String`: optional detector name; defaults to the empty string

The levels are mapped to their integer indices via `atom.level_indices`
and passed to `Dynamiq.CoherenceDetectorSpec`.
"""
function CoherenceDetectorSpec(atom::AbstractAtom, levels::Pair{<:AbstractLevel, <:AbstractLevel}; name::String="")
    # Convert Level to integer index
    level1_idx = atom.level_indices[levels[1]]
    level2_idx = atom.level_indices[levels[2]]
    
    # Call Dynamiq's constructor with integer level
    return Dynamiq.CoherenceDetectorSpec(atom; levels=level1_idx=>level2_idx, name=name)
end

"""
    MotionDetectorSpec(atom; dims = [1, 2, 3], name = "") -> Dynamiq.DetectorSpec

AtomTwin wrapper for Dynamiq's motion detector specification.

Arguments
- `atom::AbstractAtom`: atom whose motional degrees of freedom are monitored

Keyword arguments
- `dims`: list of coordinate indices to monitor (default `[1,2,3]` for x,y,z)
- `name::String`: optional detector name

This is a thin wrapper around `Dynamiq.MotionDetectorSpec`, re-exported
for a uniform AtomTwin-facing detector API.
""" 
const MotionDetectorSpec = Dynamiq.MotionDetectorSpec


"""
    FieldDetectorSpec(field; name = "") -> Dynamiq.DetectorSpec

AtomTwin wrapper for Dynamiq's field detector specification.

Arguments
- `field::AbstractField`: field instance to be monitored

Keyword arguments
- `name::String`: optional detector name

Constructs and returns a `Dynamiq.FieldDetectorSpec(field; ...)`.
"""
FieldDetectorSpec(field::AbstractField; name::String="") =
    Dynamiq.FieldDetectorSpec(field; name=name)


"""
    add_detector!(system::System, detector_spec::DetectorSpec)

Register an observable detector to record during simulation.

# Arguments
- `system::System`: The quantum system to instrument
- `detector_spec::DetectorSpec`: Detector specification (population, coherence, motion, field)

# Detector Types
- **PopulationDetectorSpec**: Track occupation probability of atomic level
- **CoherenceDetectorSpec**: Track off-diagonal density matrix element
- **MotionDetectorSpec**: Track atomic position and velocity
- **FieldDetectorSpec**: Track electromagnetic field amplitude/phase

# Returns
- `nothing` (modifies system in-place)

# Examples
```julia
# Monitor ground state population
add_detector!(sys, PopulationDetectorSpec(atom, g; name="P_g"))

# Monitor Rabi oscillation coherence
add_detector!(sys, CoherenceDetectorSpec(atom, g=>e; name="rho_ge"))

# Monitor atomic trajectory
add_detector!(sys, MotionDetectorSpec(atom; dims=[1,2,3], name="position"))

# Monitor Rabi field strength
add_detector!(sys, FieldDetectorSpec(rabi_beam; name="Omega"))
function add_detector!(sys::System, spec::Dynamiq.DetectorSpec)
    push!(sys.detector_specs, spec)
    return sys
end
"""
function add_detector!(sys::System, spec::Dynamiq.DetectorSpec)
    push!(sys.detector_specs, spec)
    return sys
end

"""
    add_detector!(sys::System, specs::Vector{<:Dynamiq.DetectorSpec}) -> System

Attach multiple detector specifications to a `System`.

Each element of `specs` is appended to `sys.detector_specs`. The
modified system is returned.
"""
function add_detector!(sys::System, specs::Vector{<:Dynamiq.DetectorSpec})
    append!(sys.detector_specs, specs)
    return sys
end

