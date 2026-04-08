"""
Dynamiq.jl

Low-level simulation engine for AtomTwin.

This module provides:
- Units and geometry primitives for beams and atomic motion.
- Quantum state representations and atom–light interactions.
- Time-domain solvers for Schrödinger and Newton equations.
- Detector specifications for measuring populations, coherences, fields, and motion.

It is intended to be used internally by AtomTwin, but may also serve as a
standalone engine for efficient mixed quantum and classical simulations.
"""
module Dynamiq

#------------------------------------------------------------------------------
# Units
#------------------------------------------------------------------------------

include("Units.jl")
using .Units
using Random

import Base: copy

#------------------------------------------------------------------------------
# Abstract interfaces and core types
#------------------------------------------------------------------------------

export AbstractBeam
export AbstractAtom

include("abstract.jl")

export base_coupling
export base_atom

#------------------------------------------------------------------------------
# Beams and geometry
#------------------------------------------------------------------------------

export GaussianBeam
export GeneralGaussianBeam
export PlanarBeam
export getposition
export getwavelength
export getbeam

include("geometry.jl")
include("geometry/planarbeam.jl")
include("geometry/gaussianbeam.jl")

#------------------------------------------------------------------------------
# Quantum states
#------------------------------------------------------------------------------

export Basis
export Op
export productstate

include("statevec.jl")

#------------------------------------------------------------------------------
# Atom–light interactions
#------------------------------------------------------------------------------

export Jump
export PlanarCoupling
export GlobalCoupling
export BlockadeCoupling
export Detuning
export StarkShiftAC
export NLevelAtom
export Interaction, VdWInteraction, GaussianCoupling

include("atomlight.jl")

#------------------------------------------------------------------------------
# Modifiers
#------------------------------------------------------------------------------

export PositionModifier
export MoveModifier
export AmplitudeModifier
export end_instruction!

include("./modifiers/positionmodifier.jl")
include("./modifiers/movemodifier.jl")
include("./modifiers/amplitudemodifier.jl")

#------------------------------------------------------------------------------
# Detectors
#------------------------------------------------------------------------------

export DetectorSpec
export PopulationDetector, PopulationDetectorSpec
export CoherenceDetector, CoherenceDetectorSpec
export MotionDetector, MotionDetectorSpec
export FieldDetector, FieldDetectorSpec
export PhotoDetector
export reset!
export build_detectors

include("./detectors/detectorspec.jl")
include("./detectors/populationdetector.jl")
include("./detectors/coherencedetector.jl")
include("./detectors/fielddetector.jl")
include("./detectors/motiondetector.jl")
include("./detectors/photodetector.jl")

#------------------------------------------------------------------------------
# Solvers
#------------------------------------------------------------------------------

export evolve!
export newton
export tdse

include("solvers.jl")

end # module Dynamiq
