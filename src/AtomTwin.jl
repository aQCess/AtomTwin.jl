"""
# AtomTwin

A simulation and design environment for neutral-atom quantum processors.

AtomTwin is an open-source simulation and design environment for neutral-atom quantum
processors. It focuses on realistic, physics-based modeling of atoms, tweezers, and
laser control, enabling users to design, validate, and calibrate low-level instructions
without explicitly defining Hamiltonians.

## Key Features

- High-performance simulation engines for quantum and classical dynamics
- Build simulations directly from atoms, tweezers, and lasers, not abstract math
- Models of common atomic species (starting with ytterbium), extensible to others
- Firmware-like machine code (sequences) to mimic execution on real neutral-atom processors
- Animate atomic trajectories and quantum state evolution for learning and debugging
- Implemented in Julia for performance and usability
- Interactive notebooks for demonstrating key quantum protocols
- Suitable for both independent research and structured learning environments

Author: Shannon Whitlock, University of Strasbourg. Copyright © 2025-2026.
"""
module AtomTwin

using Random
using WignerSymbols
using FFTW
using RecipesBase
using LinearAlgebra

include("Dynamiq/Dynamiq.jl")

# Import from Dynamiq submodule
import .Dynamiq: GaussianBeam, GeneralGaussianBeam
using .Dynamiq: PlanarBeam
using .Dynamiq: GlobalCoupling, PlanarCoupling, Detuning, Interaction, VdWInteraction, GaussianCoupling
using .Dynamiq: Jump, AbstractAtom, NLevelAtom, Basis, Op
using .Dynamiq: productstate, build_detector, evolve!
using .Dynamiq: getposition, getwavelength, getbeams
using .Dynamiq: AbstractModifier, AbstractBeam
using .Dynamiq: DetectorSpec, AbstractDetector
using .Dynamiq: PopulationDetector, CoherenceDetector
using .Dynamiq: MoveModifier, AmplitudeModifier, AbstractBoundaryModifier, SetModifier, ResetModifier, PositionModifier, begin_instruction!, end_instruction!
using .Dynamiq: Units; export Units

import Base: copy

# Include source files
include("abstract.jl")
include("parameters.jl")
include("noise.jl")
include("dag.jl")
include("levels.jl")
include("tweezerarray.jl")
include("instructions.jl")
include("sequence.jl")
include("interp.jl")
include("compile.jl")
include("system.jl")
include("atoms/polarizability.jl")
include("atoms/atoms.jl")
include("atoms/ytterbium171atom.jl")
include("detectors.jl")
include("job.jl")
include("play.jl")
include("physics/beams.jl")
include("physics/couplings.jl")
include("physics/detunings.jl")
include("physics/dissipators.jl")
include("physics/interactions.jl")
include("resolve.jl")
include("Visualization.jl")
include("tomography.jl")

# Export beams and fields
export GaussianBeam, GeneralGaussianBeam, PlanarBeam
export GlobalCoupling, PlanarCoupling, Detuning, Interaction, VdWInteraction, GaussianCoupling

# Export quantum types
export Jump, AbstractAtom, NLevelAtom, Basis
export productstate, build_detector, evolve!

# Export utilities
export getposition, getwavelength, getbeams
export Units

# Export modifiers
export AbstractModifier, AbstractBeam
export MoveModifier, AmplitudeModifier, AbstractBoundaryModifier, SetModifier, ResetModifier, PositionModifier

# Export detectors
export DetectorSpec, AbstractDetector
export PopulationDetector, CoherenceDetector
export PopulationDetectorSpec, CoherenceDetectorSpec
export MotionDetectorSpec, FieldDetectorSpec
export add_detector!

# Export abstract types
export AbstractManifold
export AbstractInstruction

# Export parameters
export Parameter, ParametricExpression
export MixedPolarization
export estimate_PER, estimate_PER_dB

# Export noise models
export NoisyField
export laser_phase_psd, laser_freq_psd
export LaserPhaseNoiseModel

# Export levels
export HyperfineManifold, FineManifold
export Level, FineLevel, HyperfineLevel
export Superposition

# Export tweezers
export TweezerArray

# Export system and compilation
export System
export getqstate, compile, getmatrix, gethamiltonian, getbasis

# Export polarizability
export PolarizabilityModel, PolarizabilityCurve
export light_shift_coeff_Hz_per_Wcm2
export scattering_rate_per_Wcm2
export polarizability_au, polarizability_si

# Export simulation
export SimulationJob
export recompile!
export play

# Export physics utilities
export add_zeeman_detunings!
export add_coupling!, add_detuning!, rabi_frequencies
export add_decay!, add_dephasing!
export add_interaction!, add_vdwinteraction!

# Export sequence instructions
export Sequence, @sequence
export Wait
export MoveRow, RampRow, MoveCol, RampCol
export AmplCol, AmplRow
export FreqCol, FreqRow
export Pulse, On, Off
export Parallel

# Export atoms
export Atom
export Ytterbium171Atom
export Potassium39Atom, Rubidium87Atom, Strontium88Atom
export getspecies

# Export initialization / sampling
export GaussianPosition, MaxwellBoltzmann
export maxwellboltzmann, gaussian
export initialize!

# AbstractNode is exported for dispatch and isinstance checks;
# concrete node subtypes are internal implementation details.
export AbstractNode

# Export resolution
export resolve

# Export analysis
export process_tomography

end # module AtomTwin
