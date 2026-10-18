# Internals: Dynamiq engine

The simulation engine behind AtomTwin. Internal, subject to change without notice.

```@docs
AtomTwin.Dynamiq
```

## Building blocks
```@docs
AtomTwin.Dynamiq.AbstractAtom
AtomTwin.Dynamiq.AbstractBeam
AtomTwin.Dynamiq.Op
AtomTwin.Dynamiq.productstate
AtomTwin.Dynamiq.base_atom
AtomTwin.Dynamiq.base_coupling
AtomTwin.Dynamiq.BlockadeCoupling
AtomTwin.Dynamiq.getbeams
AtomTwin.Dynamiq.getposition
AtomTwin.Dynamiq.getwavelength
AtomTwin.Dynamiq.reset!
AtomTwin.Dynamiq.reset_force!
AtomTwin.Dynamiq.DetectorSpec
AtomTwin.Dynamiq.MotionDetectorSpec
AtomTwin.Dynamiq.AbstractBoundaryModifier
AtomTwin.Dynamiq.AmplitudeModifier
AtomTwin.Dynamiq.SetModifier
AtomTwin.Dynamiq.ResetModifier
AtomTwin.Dynamiq.MoveModifier
AtomTwin.Dynamiq.PositionModifier
```

## Atom-light couplings
```@autodocs
Modules = [AtomTwin.Dynamiq]
Pages = ["Dynamiq/Dynamiq.jl", "Dynamiq/abstract.jl", "Dynamiq/atomlight.jl"]
Public = false
```

## Geometry
```@autodocs
Modules = [AtomTwin.Dynamiq]
Pages = ["Dynamiq/geometry.jl", "Dynamiq/geometry/gaussianbeam.jl",
         "Dynamiq/geometry/planarbeam.jl"]
Public = false
```

## State vectors and operators
```@autodocs
Modules = [AtomTwin.Dynamiq]
Pages = ["Dynamiq/statevec.jl"]
Public = false
```

## Detectors
```@autodocs
Modules = [AtomTwin.Dynamiq]
Pages = ["Dynamiq/detectors/detectorspec.jl", "Dynamiq/detectors/coherencedetector.jl",
         "Dynamiq/detectors/expectationdetector.jl", "Dynamiq/detectors/fielddetector.jl",
         "Dynamiq/detectors/motiondetector.jl", "Dynamiq/detectors/photodetector.jl",
         "Dynamiq/detectors/populationdetector.jl"]
Public = false
```

## Modifiers
```@autodocs
Modules = [AtomTwin.Dynamiq]
Pages = ["Dynamiq/modifiers/amplitudemodifier.jl", "Dynamiq/modifiers/movemodifier.jl",
         "Dynamiq/modifiers/positionmodifier.jl"]
Public = false
```

## Units
```@autodocs
Modules = [AtomTwin.Dynamiq.Units]
Pages = ["Dynamiq/Units.jl"]
Public = true
```
