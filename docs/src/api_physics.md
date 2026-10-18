# Physics

## Couplings
```@docs
add_coupling!
rabi_frequencies
AtomTwin.Dynamiq.GlobalCoupling
AtomTwin.Dynamiq.PlanarCoupling
AtomTwin.Dynamiq.GaussianCoupling
```

## Detunings
```@docs
add_detuning!
add_zeeman_detunings!
AtomTwin.Dynamiq.Detuning
```

## Light shifts
A trapping beam passed to `System` light-shifts its levels automatically (see
[Solvers and accuracy](solvers.md#Trap-light-shifts)).
```@docs
add_light_shift!
AtomTwin.Dynamiq.StarkShiftAC
```

## Dissipators
```@docs
add_decay!
add_dephasing!
AtomTwin.Dynamiq.Jump
```

## Interactions
```@docs
add_interaction!
add_vdwinteraction!
AtomTwin.Dynamiq.Interaction
AtomTwin.Dynamiq.VdWInteraction
```

## Arbitrary Hamiltonian terms
```@docs
add_hamiltonian!
AtomTwin.Dynamiq.Hamiltonian
```

## Open-system operators
```@docs
getjumps
getheffective
getliouvillian
```
