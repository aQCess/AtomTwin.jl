# System and device model

Construct full neutral-atom device models by combining atoms, levels, tweezers, static fields, and detectors into a single `System` object.

```@docs
System
getqstate
gethamiltonian
getmatrix
getbasis
AtomTwin.Dynamiq.Basis
```

***

## Levels

Specify internal energy levels and state manifolds for building atomic models.

```@docs
Level
FineLevel
HyperfineLevel
Superposition
```

### Term symbols

A level's `term` is what binds it to a polarizability model, so it is matched by
spectroscopic term rather than by display label.

```@docs
TermSymbol
@term
@l_str
```

### Manifold types
```@docs
AbstractManifold
FineManifold
HyperfineManifold
```

### Iteration and access

These methods let you iterate over levels and select sublevels:
```@docs
Base.getindex
Base.iterate
```

### Example: building a hyperfine manifold
```@example

using AtomTwin

# Hyperfine manifold with F = 3/2, J=1/2 and a simple label
manifold = HyperfineManifold(3//2, 1//2; label = "6S1/2", g_F = 0.5)

#Iterate over all mF sublevels

for level in manifold
    @show level.F, level.mF
end
```

***

## Atoms

Define the internal structure of each atom by choosing species and setting properties like the levels, position, and velocity.

```@docs
Atom
Ytterbium171Atom
Rubidium87Atom
Strontium88Atom
Potassium39Atom
NLevelAtom
```

These aliases all wrap the same low‑level constructor `AtomWrapper{S}` with different species labels and default parameters.

### Atomic Data
```@docs
PolarizabilityModel
PolarizabilityCurve
light_shift_coeff_Hz_per_Wcm2
polarizability_au
polarizability_si
```

### Utilities
```@docs
AtomTwin.AtomWrapper
AtomTwin.ATOM_DEFAULTS
initialize!
```

Use these when you need fine control over species parameters or when integrating AtomTwin with other simulation layers. Position and velocity samplers (`GaussianPosition`, `MaxwellBoltzmann`) are documented in the [DAG system](api_dag.md) section.

### Example
```@example

using AtomTwin

# Create a single ytterbium-171 atom at rest
yb = Ytterbium171Atom()

# Attach a thermal velocity sampler at 5 µK
yb = Ytterbium171Atom(; v_init = maxwellboltzmann(T = 5e-6))

# Draw a random velocity and initialize the wrapped NLevelAtom
initialize!(yb)

yb # show its basic properties
```

***

## Beams

Define beam objects for producing forces on atoms or driving transitions between internal states

```@docs
AtomTwin.Dynamiq.GaussianBeam
AtomTwin.Dynamiq.GeneralGaussianBeam
AtomTwin.Dynamiq.PlanarBeam
```

## Tweezers

Configure 2D AOD-driven tweezer arrays, including spot positions and power distribution, to specify how atoms are trapped in space.

```@docs
TweezerArray
```

## Detectors

Attach population, coherence, motion, and field detectors to a `System` so simulations produce the observables you need.

```@docs
PopulationDetectorSpec
CoherenceDetectorSpec
MotionDetectorSpec
AtomTwin.Dynamiq.FieldDetectorSpec
add_detector!
```

Actual detector objects are constructed by `play' as Dynamiq objects.
```@docs
AtomTwin.Dynamiq.PopulationDetector
AtomTwin.Dynamiq.CoherenceDetector
AtomTwin.Dynamiq.MotionDetector
AtomTwin.Dynamiq.FieldDetector
```