# Changelog

All notable changes to AtomTwin are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.2.1] — 2026-10-19

### Fixed

- **Yb ³P₁ polarizability model.** The effective lines standing in for the high-lying
  states were `J′ = 1` and `J′ = 2` only; the missing `J′ = 0` member of 6p² ³P (409 nm)
  sets the tensor part between 470 and 540 nm, where the model had it near zero or of
  the wrong sign (π/σ shift ratios off by up to 80σ). The line list now has the 6p²
  ³P₀,₁,₂ multiplet (LS strengths, one fitted amplitude) and the 5d6s ¹D₂ line at
  1032 nm, with the effective pair, (6s6d)³D₂ and (6s7s)³S₁ refitted, against 20
  calibration-free measurements (light-shift ratios, magic angles and wavelengths;
  χ² = 17.9 for 15 dof). Light shifts of Yb ³P₁ change at all wavelengths; at 767 nm
  β∥ 0.381 → 0.434 and β⊥ 1.817 → 1.803.

### Added

- `YB171_POLARIZABILITY_3P1`: Yb-171 now has a ³P₁ model, the same line list
  (`YB174_POLARIZABILITY_3P1` is an alias), recoupled to `F = 3/2, 1/2`.

---

## [0.2.0] — 2026-10-18

Numerical results change in this release: the solvers, the trap light shift, the
polarizability conventions and motion on by default all move answers. Re-run
anything you intend to compare against 0.1.x.

### Changed

**Solvers.**
- The default propagator is Chebyshev: exact for a constant Hamiltonian at any
  step, no stability limit, degree set by the spectrum. `Taylor(order)` remains
  available via `integrator`.
- `dt` is the output grid and the largest step; `tol` (default `1e-6`) sets the
  accuracy. `Sequence(; tol)` with no `dt` records one sample per instruction.
  The density-matrix solver sub-divides each step to meet `tol`, with an exact,
  trace-preserving dissipator; the statevector solvers read a time-dependent
  drive once per step, so give shaped pulses, ramps and moves a `dt` (`compile`
  warns).
- MCWF places every quantum jump at the time its norm threshold is crossed, found
  on the Chebyshev step's dense output: any number of jumps per step, no jump
  sub-stepping. Every MCWF realisation changes.
- The semiclassical solvers are second order in `dt` (split velocity Verlet), and
  the classical solver is second order with moving beams. (H. Schmit-Veiler)
- Motion is the default; `play(…; frozen = true)` holds the atoms. A run in which
  nothing can move takes the frozen solvers automatically, with identical results.
- Shaped envelopes are read with `:cubic` interpolation by default (`:lagrange`
  and `:piecewise_constant` remain).
- Moves and ramps take `round(duration/dt)` steps like every other instruction (a
  move's output grid has one sample fewer than in 0.1); an instruction shorter
  than its `downsample` keeps its own step count.

**Light shifts and polarizability.**
- A trapping beam passed to `System` light-shifts every polarizable level, by the
  same potential that traps the atom; polarizabilities are per sublevel, with the
  full tensor light shift, defined against the system's quantization axis.
  Levels find their polarizability model by term symbol.
- Polarizabilities use the standard relation U = −α I/(2cε₀), and `dipole_ea0` is
  read in the Wigner-3j (Safronova) convention, with the Clebsch–Gordan (Steck)
  convention also accepted: α was reported at half its standard value, and
  dipole-specified lines (Rb, Sr) were twice too strong. `polarizability_si` and
  `polarizability_au` take the `F`, `mF`, `I`, `ε_z` keywords of
  `light_shift_coeff_Hz_per_Wcm2`; the transition fields `Jg` and `f` are gone.
  (H. Schmit-Veiler)
- The Yb ¹S₀/³P₀ models (Höhn et al. 2023) are refitted to the measured
  state-(in)dependent wavelengths (Kroeze et al., arXiv:2509.04416), with measured
  line strengths held at their literature values: clock magic wavelength
  759.3561 nm (measured 759.356 063 3(31)), ³P₀ tune-out 576.612 nm (576.613(10));
  α(¹S₀) rises 0.3 %. The models cover the visible and near infrared, not the
  static limit.

**Motion and recoil.**
- Radiation pressure of plane-wave drives is the Ehrenfest force of the plane-wave
  phase, in all three semiclassical solvers, integrated over the solver steps; it
  was a mean force sampled once per step, in the MCWF solver only. In MCWF the
  momentum follows the detection record: each photon is absorbed from the beam
  that excited the atom. Recoil heating in imaging-type simulations goes up (Yb
  399 nm, s = 40: 146 → 221 nK per photon).
- A shaped envelope on a position-dependent field (planar or Gaussian coupling,
  light shift, van der Waals interaction) reaches every solver sub-step.

**API.**
- An explicit Rabi frequency given with a Gaussian beam is the Rabi frequency at
  the beam's focus `beam.r0`; the coupling follows the beam's profile and phase.
- A beam-derived J = 0 → 0 coupling is an error naming the explicit path; it
  silently created nothing.
- A `Pulse` on a detuning with a complex amplitude is an error, as documented.
- Atom positions and velocities, moved beams, switch states, detector outputs and
  the quantum state all reset between shots and between `play`s of a job.

### Added
- Ytterbium-174 (with ¹S₀, ³P₀ and ³P₁ polarizability models), and Strontium-88 and
  Rubidium-87 polarizability models; tensor polarizability;
  `scattering_rate_per_Wcm2`.
- `polarizabilities = Dict(term => model)` on any atom constructor, and
  `PolarizabilityModel(…; tensor_offset_Hz_per_Wm2)` for a measured tensor light
  shift without a line list.
- `add_light_shift!`, `add_quantization_axis!`, `add_hamiltonian!`,
  `add_vdwinteraction!`, transverse exchange in `add_interaction!`,
  `rabi_frequencies`.
- `add_decay!(…; λ, dipole)`: photon recoil on emission, along the transition
  dipole's emission pattern.
- Symbolic `Operator`s (`ket * bra'`), `ExpectationDetectorSpec`,
  `PhotoDetectorSpec`, `getjumps`, `getheffective`, `getliouvillian`.
- Typed term symbols: `TermSymbol`, `@term`, `l"3P1"`.
- Examples: two-qubit exchange, laser cooling in a tweezer (Doppler and Sisyphus). A "Solvers and accuracy" documentation page.
- A precompile workload, and solver benchmarks under `bench/`.

### Fixed
- `process_tomography` evolved every input state from the first input: an X gate
  with 20 % Rabi noise scored process fidelity 0.25 instead of ≈ 0.91.
- Fine → hyperfine couplings had σ⁺ and σ⁻ swapped, and the hyperfine dipole
  matrix element carried an extra (−1)^q.
- Manifold decays took one jump per sublevel pair, so an excited superposition
  decayed into a mixture; there is one coherent jump per photon polarisation.
- `add_coupling!`'s keyword form (`Ω_π`, `Ω_p`, `Ω_m`) skipped the Clebsch–Gordan
  coefficients and threw on a `Parameter`; the positional form with a Gaussian beam
  built a uniform coupling.
- `compute_coupling_strength` turned any error into a zero coupling.
- `MixedPolarization(pol, ε; k)` added no contamination to circular polarization.
- `rabi_frequencies(::GaussianBeam)` threw.
- `GeneralGaussianBeam` fell off axis as w0²/w², doubled the Gouy phase, ignored
  z in `intensity`, and ran its phase as exp(−ik·r).
- Planar-beam couplings and van der Waals interactions ignored `active`, `Pulse`,
  `On`/`Off` and `ampl`; an attractive van der Waals interaction was pinned at its
  cap.
- Parameters sampled per shot compounded, were not resampled (decay rates), or
  were not refreshed in multi-threaded runs, and a sampled beam parameter or atom
  position in a threaded run shared the system between threads (results differed
  from serial runs with the same seed); density-matrix runs with motion felt
  no state-dependent force without a population detector.
- The density-matrix dissipator was not trace-preserving.
- Moves inside `Parallel` converged at first order; a ramp after `AmplRow` started
  from a stale amplitude; ramps ignored the row × column amplitude factorisation.
- `Superposition` scalar algebra nested superpositions and mis-weighted states.
- `play` threw with no detectors, and with `shots ≥ 4` inside `Threads.@threads`;
  `isequal` on two atoms threw; the Yb-171 polarizability methods threw.

---

## [0.1.1] — 2026-04-16

### Added
- Unit test suite covering `Parameter`, `ParametricExpression`, `Level`/manifolds, `Sequence`, and `System` construction.
- GitHub Actions CI workflow (`.github/workflows/CI.yml`) running on Julia 1.11 and nightly.
- `LICENSE` file (Apache-2.0).
- Narrative documentation pages for Parameters & Noise and Visualization APIs.
- Vulnerability reporting instructions added to `CONTRIBUTING.md`.

### Changed
- `Revise` removed from package dependencies (was incorrectly listed as a runtime dependency; it is a developer tool).
- `Test` moved from `[deps]` to `[extras]`/`[targets]` per Julia packaging convention.
- `julia = "1.11"` compat bound added to `Project.toml`.
- Internal DAG node types (`BeamNode`, `CouplingNode`, etc.) removed from public exports; only `AbstractNode` is exported.
- README placeholder URLs replaced with real repository and documentation links.

---

## [0.1.0] — 2026

Initial beta release.

### Features

- **Atom / level model**: `Level`, `HyperfineLevel`, `FineLevel`, `HyperfineManifold`, `FineManifold`, `Superposition`.
- **Atomic species**: `Ytterbium171Atom`, `Potassium39Atom`, `Rubidium87Atom`, `Strontium88Atom`.
- **Gaussian beam model**: `GaussianBeam`, `GeneralGaussianBeam`, `PlanarBeam`.
- **System building**: `System`, `add_coupling!`, `add_detuning!`, `add_decay!`, `add_dephasing!`, `add_interaction!`, `add_zeeman_detunings!`.
- **Parametric simulations**: `Parameter`, `ParametricExpression`, `MixedPolarization`.
- **Laser phase noise**: `LaserPhaseNoiseModel`, `NoisyField`, `laser_freq_psd`, `laser_phase_psd`.
- **Tweezer arrays**: `TweezerArray`.
- **Instruction sequences**: `Sequence`, `@sequence`, `Wait`, `Pulse`, `On`, `Off`, `MoveRow`, `MoveCol`, `RampRow`, `RampCol`, `AmplRow`, `AmplCol`, `FreqRow`, `FreqCol`, `Parallel`.
- **Simulation engine**: `play`, `compile`, `recompile!`.
- **Detectors**: `PopulationDetectorSpec`, `CoherenceDetectorSpec`, `MotionDetectorSpec`, `FieldDetectorSpec`.
- **Analysis**: `process_tomography`, `estimate_PER`, `estimate_PER_dB`.
- **Visualization**: `AtomTwin.Visualization.animate` (requires GLMakie extension).
- **Plotting recipes** via RecipesBase (Plots.jl extension).
- 11 worked examples covering Rabi oscillations, dissipation, motion, laser noise, Rydberg blockade, time-optimal gates, Ramsey/EIT, Yb-171 Raman gates, K-39 state preparation, process tomography, and atom sorting.

### Known limitations

- Visualization is currently limited to 2D atomic trajectory animation
- The General Registry registration is pending.
