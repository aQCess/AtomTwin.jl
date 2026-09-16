# Internals

These are internal methods are are subject to changes without warning. Use with caution

## Compilation
```@autodocs
Modules = [AtomTwin]
Pages = ["job.jl", "compile.jl"]
Public = false
```

## System
```@autodocs
Modules = [AtomTwin]
Pages = ["system.jl"]
Public = false
```

## Simulation
```@autodocs
Modules = [AtomTwin]
Pages = ["play.jl"]
Public = false
```

## Noise
```@autodocs
Modules = [AtomTwin]
Pages = ["noise.jl"]
Public = false
```

## Tomography
```@autodocs
Modules = [AtomTwin]
Pages = ["tomography.jl"]
Public = false
```

## Parameters
```@autodocs
Modules = [AtomTwin]
Pages = ["parameters.jl"]
Public = false
```

## Tweezers
```@autodocs
Modules = [AtomTwin]
Pages = ["tweezerarray.jl"]
Public = false
```

## Detectors
```@autodocs
Modules = [AtomTwin]
Pages = ["detectors.jl"]
Public = false
```

## Resolve
```@autodocs
Modules = [AtomTwin]
Pages = ["resolve.jl"]
Public = false
```

## Physics
```@autodocs
Modules = [AtomTwin]
Pages = ["physics/couplings.jl", "physics/detunings.jl", "physics/dissipators.jl", "physics/interactions.jl"]
Public = false
```

## Visualization
```@autodocs
Modules = [AtomTwin.Visualization]
Pages = ["Visualization.jl"]
Public = false
```

## Dynamiq engine

### Atom-light couplings
```@autodocs
Modules = [AtomTwin.Dynamiq]
Pages = ["Dynamiq/atomlight.jl"]
Public = false
```
### Geometry
```@autodocs
Modules = [AtomTwin.Dynamiq]
Pages = ["Dynamiq/geometry.jl"]
Public = false
```
### Solvers
```@autodocs
Modules = [AtomTwin.Dynamiq]
Pages = ["Dynamiq/solvers/kernels.jl", "Dynamiq/solvers/integrators.jl",
         "Dynamiq/solvers/propagators.jl", "Dynamiq/solvers/control.jl",
         "Dynamiq/solvers/solvers.jl"]
Public = false
```
### Statevectors and Operators
```@autodocs
Modules = [AtomTwin.Dynamiq]
Pages = ["Dynamiq/statevec.jl"]
Public = false
```
### Detectors
```@autodocs
Modules = [AtomTwin.Dynamiq]
Pages = ["Dynamiq/detectors/detectorspec.jl",
        "Dynamiq/detectors/coherencedetector.jl",
        "Dynamiq/detectors/fielddetector.jl",
        "Dynamiq/detectors/motiondetector.jl",
        "Dynamiq/detectors/photodetector.jl",
        "Dynamiq/detectors/populationdetector.jl"]
Public = false
```
### Modifiers
```@autodocs
Modules = [AtomTwin.Dynamiq]
Pages = ["Dynamiq/modifiers/amplitudemodifier.jl",
        "Dynamiq/modifiers/movemodifier.jl",
        "Dynamiq/modifiers/positionmodifier.jl"]
Public = false
```

## Units
```@autodocs
Modules = [AtomTwin.Dynamiq.Units]
Pages = ["Dynamiq/Units.jl"]
Public = true
```
