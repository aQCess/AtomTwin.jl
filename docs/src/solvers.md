# Solvers and accuracy

`play` compiles a `System` and a `Sequence` into a simulation job and runs it with
one of three quantum solvers, each with a semiclassical twin that also moves the
atoms:

| state | jumps | solver | with motion |
|---|---|---|---|
| state vector | none | Schrödinger (`tdse`) | `tdse_semiclassical` |
| state vector | `add_decay!`, `add_dephasing!` | wavefunction Monte Carlo (`wfmc`), one trajectory per shot | `wfmc_semiclassical` |
| density matrix (`density_matrix = true`) | any | Lindblad master equation (`qme`) | `qme_semiclassical` |

With no `initial_state`, only the classical motion is integrated (`newton`).

## Time grid and tolerance

`dt` is the **output grid**, and the largest step a solver takes. The solvers
divide it further where their error control needs to; they never take a larger step.

- `Sequence(dt)` records every `dt` (thinned by `downsample`).
- `Sequence(; tol)` with no `dt` records one sample at the end of each instruction,
  or more often if the Hamiltonian requires it.
- `tol` (default `1e-6`) is the local error target per step.

What `tol` controls depends on the solver:

- **Schrödinger and MCWF.** Each step is a Chebyshev expansion of
  `exp(-iH dt)`, exact to machine precision for a Hamiltonian that is constant over
  the step, at any step length. A time-dependent drive (a shaped `Pulse`, a ramp, a
  move) is read once per step, at its midpoint, so `dt` must resolve it; `compile`
  warns when it may not.
- **Master equation.** Each step is divided into Strang-split sub-steps until a
  step-doubling estimate of the error is below `tol`; drives are read at every
  sub-step, and the dissipator is an exact, trace-preserving channel. Here `tol`
  controls the result.

The Taylor propagator is available as
`compile(…; integrator = AtomTwin.Dynamiq.Taylor(order))`; it is stable only for
`‖H‖·dt < 2√2` and is not supported by the MCWF solver.

## Quantum jumps

The MCWF solver places each quantum jump at the time the norm of the conditional
state crosses its random threshold. The Chebyshev vectors of a step do not depend on
the step length, so the state anywhere inside the step costs no further
matrix-vector products, and the crossing time is found on it directly. Any number of
jumps may fall in one step; a constant Hamiltonian costs one Chebyshev step per
sample plus one propagation per jump. A `PhotoDetectorSpec` bound to a decay counts
its jumps.

Shots run in parallel over Julia's threads. Pass `rng = MersenneTwister(seed)` (or
any generator) to `play` for reproducible trajectories.

## Atomic motion

Atoms move by default; `play(…; frozen = true)` holds them at their initial
positions. A run in which nothing can move takes the frozen solver automatically.
Motion is semiclassical: each atom has a classical position and velocity, integrated
by velocity Verlet around the quantum step, and is subject to

- the **dipole force** of every trapping beam, from each level's polarizability
  weighted by its population (and, for the tensor light shift, by the coherences it
  mixes);
- **radiation pressure** of plane-wave couplings (`add_coupling!(…; beam = PlanarBeam(…))`):
  the Ehrenfest force of the plane-wave phase, integrated over the solver steps. In
  MCWF the momentum follows the detection record: each emitted photon was absorbed
  from the beam that excited the atom;
- **emission recoil** of decays given a wavelength, `add_decay!(…; λ)`, along the
  transition dipole's emission pattern when `dipole` is given.

Motional quantisation is not modelled: results are accurate while
`k_B T ≫ ħω_trap`.

## Trap light shifts

A beam passed to `System` light-shifts every level with a polarizability at its
wavelength, by `U = −α I/(2c ε₀)`, and pushes the atom with the same potential.
Polarizabilities come from the species' models, matched by term symbol (`l"3P1"`),
or from `polarizabilities = Dict(term => model)` on the atom constructor. For levels
with angular momentum the **tensor** part is included in full, defined against the
system's quantization axis (`add_quantization_axis!`, default `ẑ`): it mixes
sublevels whenever the trap polarization is not along that axis.
`add_light_shift!` replaces the automatic shift on chosen levels, to reference it to
one level or to switch it with `On`/`Off`.

## Solver entry points

```@docs
AtomTwin.Dynamiq.tdse
AtomTwin.Dynamiq.newton
AtomTwin.Dynamiq.suggested_dt
```
