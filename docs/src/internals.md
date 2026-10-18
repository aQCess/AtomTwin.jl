# Internals: AtomTwin

Internal methods, subject to change without notice.

```@docs
AtomTwin
```

## Atoms, levels and polarizability
```@autodocs
Modules = [AtomTwin]
Pages = ["abstract.jl", "levels.jl", "atoms/atoms.jl", "atoms/polarizability.jl",
         "atoms/rubidium87atom.jl", "atoms/strontium88atom.jl",
         "atoms/ytterbium171atom.jl", "atoms/ytterbium174atom.jl"]
Public = false
Filter = t -> !(t === AtomTwin.AtomWrapper || t === AtomTwin.ATOM_DEFAULTS)
```

## System and DAG
```@autodocs
Modules = [AtomTwin]
Pages = ["system.jl", "dag.jl", "resolve.jl", "operators.jl", "detectors.jl"]
Public = false
```

## Physics
```@autodocs
Modules = [AtomTwin]
Pages = ["physics/beams.jl", "physics/couplings.jl", "physics/detunings.jl",
         "physics/dissipators.jl", "physics/interactions.jl"]
Public = false
```

## Sequences, compilation and simulation
```@autodocs
Modules = [AtomTwin]
Pages = ["instructions.jl", "interp.jl", "job.jl", "compile.jl", "play.jl"]
Public = false
```

## Parameters, noise and tweezers
```@autodocs
Modules = [AtomTwin]
Pages = ["parameters.jl", "noise.jl", "tweezerarray.jl"]
Public = false
```

## Tomography
```@autodocs
Modules = [AtomTwin]
Pages = ["tomography.jl"]
Public = false
```

## Visualization
```@autodocs
Modules = [AtomTwin.Visualization]
Pages = ["Visualization.jl"]
Public = false
```
