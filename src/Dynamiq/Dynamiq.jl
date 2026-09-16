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
export Hamiltonian
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
export AbstractBoundaryModifier
export SetModifier
export ResetModifier
export begin_instruction!
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
export ExpectationDetector, ExpectationDetectorSpec
export PhotoDetector, PhotoDetectorSpec
export reset!
export build_detectors

include("./detectors/detectorspec.jl")
include("./detectors/populationdetector.jl")
include("./detectors/coherencedetector.jl")
include("./detectors/fielddetector.jl")
include("./detectors/expectationdetector.jl")
include("./detectors/motiondetector.jl")
include("./detectors/photodetector.jl")

#------------------------------------------------------------------------------
# Solvers
#------------------------------------------------------------------------------

export evolve!
export newton
export reset_force!
export suggested_dt      # useful for users diagnosing their own step size
export tdse

include("solvers/kernels.jl")
include("solvers/integrators.jl")
include("solvers/propagators.jl")
include("solvers/control.jl")
include("solvers/solvers.jl")

#------------------------------------------------------------------------------
# Scalar evaluation at arbitrary time
#------------------------------------------------------------------------------

"""
    InterpKind

How a sampled envelope is read between its samples: `:constant`, `:linear` or
`:cubic`.

An envelope is `N` samples spanning `[0, duration]`. The solver no longer steps
on the sample grid — it chooses its own step from `tol`, and sub-divides further
when the error estimator asks — so the samples must be readable at *any* time,
not just at the points they were given on.

- `:constant` holds each sample to the next, a physical staircase. An AWG really
  does this, so it is the correct choice for hardware-defined waveforms and the
  resulting first-order convergence is the true behaviour, not a defect.
- `:linear` is second order — the same order as the exponential-midpoint
  quadrature the solver uses, so it caps the scheme.
- `:cubic` (Catmull–Rom, the default) is fourth order and stays below the
  quadrature, at 1.11x the cost of linear. Measured on a Gaussian envelope at
  101 samples: max error 2.9e-5 against linear's 6.2e-4.
"""
const InterpKind = Symbol

"""
    sample_at(vals, duration, t, kind) -> eltype(vals)

Read a uniformly-sampled envelope at time `t ∈ [0, duration]`.

`vals[1]` sits at `t = 0` and `vals[end]` at `t = duration`. Times outside the
span clamp to the end samples. The grid is uniform, so locating the interval is
arithmetic — no search.
"""
@inline function sample_at(vals::AbstractVector{T}, duration::Float64,
                           t::Float64, kind::Symbol) where {T}
    n = length(vals)
    n == 1 && return @inbounds vals[1]
    (duration > 0) || return @inbounds vals[1]

    if kind === :constant
        # Staircase: sample k holds over [k-1, k)/N of the span.
        k = clamp(floor(Int, t / duration * n) + 1, 1, n)
        return @inbounds vals[k]
    end

    h = duration / (n - 1)
    x = t / h
    # Clamp to the sample range: outside it the value is the end sample, and at
    # the ends exactly it must BE the end sample (the shifted cubic stencil
    # would otherwise extrapolate and miss it by O(h⁴)).
    x <= 0      && return @inbounds vals[1]
    x >= n - 1  && return @inbounds vals[n]
    i = floor(Int, x)

    if kind === :linear || n < 4
        i = clamp(i, 0, n - 2)
        f = x - i
        f = f < 0 ? 0.0 : (f > 1 ? 1.0 : f)
        @inbounds return vals[i+1] * (1 - f) + vals[i+2] * f
    end

    # Catmull-Rom. For x in [j, j+1) the stencil is samples j-1, j, j+1, j+2
    # (0-based) and the interpolation parameter is f = x - j, measured between
    # the MIDDLE two. In 1-based indices that is vals[j], vals[j+1], vals[j+2],
    # vals[j+3]. Near the ends the stencil would run off the array, so clamp j
    # and keep f measured from the true interval -- letting f leave [0,1] there,
    # which is the correct cubic extrapolation within the shifted stencil.
    j = clamp(i, 1, n - 3)
    f = x - j
    @inbounds begin
        y0 = vals[j]; y1 = vals[j+1]; y2 = vals[j+2]; y3 = vals[j+3]
    end
    return 0.5 * ((2y1) + (-y0 + y2) * f +
                  (2y0 - 5y1 + 4y2 - y3) * f * f +
                  (-y0 + 3y1 - 3y2 + y3) * f * f * f)
end

end # module Dynamiq
