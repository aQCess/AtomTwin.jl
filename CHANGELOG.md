# Changelog

All notable changes to AtomTwin are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.2.0] — unreleased

Numerical results change in this release: the default propagator, the trap
light shift and several solver fixes all move answers, most by more than the
old tolerance. Re-run anything you intend to compare against 0.1.x.

### Changed
- The default propagator is Chebyshev (no stability limit; degree set by the
  spectrum and `tol`). `Taylor(order)` remains available via `integrator`.
- `Sequence(; tol)` with no `dt` records one output sample per instruction. The
  density-matrix solver sub-divides each step to meet `tol`; the statevector
  solvers do not sub-divide a time-dependent drive, so give shaped pulses,
  ramps and moves a `dt` there (`compile` warns).
- A trapping beam passed to `System` now light-shifts every polarizable level,
  not only pushes the atom. Every trap light shift was also 1/(cε₀) ≈ 377× too
  small before.
- Polarizabilities are per sublevel (tensor light shift), defined against the
  system's quantization axis (`add_quantization_axis!`). Levels match
  polarizability models by term symbol.
- Atom positions and velocities, beams moved or ramped by a shot, field switch
  states, detector outputs and the quantum state all reset between shots and
  between `play`s of the same job.
- Spontaneous emission recorded with `add_decay!(…; λ)` gives a photon recoil.
- Radiation pressure of plane-wave drives (`add_coupling!(…; beam = PlanarBeam(…))`)
  is the Ehrenfest force of the plane-wave phase, in all three semiclassical solvers,
  integrated over their sub-steps. It used to be a mean force `ħ R_tot Σ ŵ_b k_b`
  sampled once per step, and only in the MCWF solver: a π pulse moved almost no
  momentum, stimulated emission into an opposing beam none, and the spread of the
  absorbed momentum was set by `dt` instead of by photon statistics. In MCWF the
  momentum follows the detection record: each emitted photon was absorbed, with
  its momentum, from the beam that excited the atom, and an excitation that
  decays unseen leaves none. Recoil heating in imaging-type simulations goes up
  (Yb 399 nm, s = 40: 146 → 221 nK per photon).
- A shaped envelope on a position-dependent field — a `PlanarCoupling` or
  `GaussianCoupling`, a trap light shift, a van der Waals interaction — reaches the
  Hamiltonian at every solver sub-step. Inside MCWF sub-steps and density-matrix
  sub-step pairs it was applied as a staircase at `dt`, however fine those were: a
  detuned sin² pulse sampled at four points was off by 5–10 % in population.
- MCWF jumps are placed within their sub-step (at its start or end with the
  probabilities that make the jump time right on average) instead of always at its
  end. Photon counts on a saturated line were ~√jtol/4 low (2.5 % at the default
  jtol); they now match the master equation. Every MCWF realisation changes.
- Motion is the default: `play(…; frozen = true)` freezes the atoms. `play` used to
  freeze them unless a trapping beam could exert a force and the atom had a velocity
  or recoil, so a free atom driven by plane waves never moved, and an atom released
  off-centre at rest never oscillated. Runs where nothing can move take the frozen
  solvers automatically, with identical results. Runs that can move now take the
  semiclassical solvers, which cost more; with plane-wave drives the statevector
  solver also sub-divides each step to 0.1 rad of the driven states' rotation.
  `play(…; frozen = true)` restores the previous speed and physics.
- A `Pulse` on a detuning with a complex amplitude is an error, as documented; it
  was silently accepted.
- The density-matrix dissipator is an exact channel; MCWF sub-steps are bounded
  by the jump-omission tolerance `jtol` (default from `shots`) and by the
  spectral width.
- Shaped envelopes are read with `:cubic` interpolation by default
  (`:lagrange`, `:piecewise_constant` remain as aliases).
- An instruction shorter than its `downsample` keeps its step count and records
  one sample at its end; it was integrated with `downsample` steps instead
  (25 → 10 000 for a 250 ns pulse at `dt = 10 ns`, `downsample = 10_000`).
- Moves and ramps take `round(duration/dt)` steps, like every other timed
  instruction; a move used to take one more, so its output grid has one sample
  fewer.
- `RampRow`/`RampCol`: a vector `final_amplitude` gives one target per row
  (column), as documented, and the ramp respects the row × column amplitude
  factorisation that `AmplRow`/`AmplCol` apply.
- Polarizabilities use the standard relation U = −α I/(2cε₀), and `dipole_ea0` is
  read in the Wigner-3j (Safronova) convention, with the Clebsch–Gordan (Steck)
  convention also accepted. Two compensating factors of 2 are gone: every α
  printed at half its standard value, while dipole-specified lines (Rb, Sr) were
  twice too strong, so those traps were 2× too deep per intensity. Rb-87 5S₁/₂ now
  gives 318 a.u. static and 687 a.u. at 1064 nm. (H. Schmit-Veiler)
- The tensor polarizability α⁽²⁾ carries the 3π ε₀c³ prefactor and the same
  line-strength factors as the scalar part. For Yb-174 ³P₁ the m = 0 magic angle at
  759.3 nm is now 39.0° (measured 38.5(9)°); at 767 nm, β∥ = 0.38 and β⊥ = 1.82
  relative to ¹S₀. (H. Schmit-Veiler)
- `polarizability_si` and `polarizability_au` take the same `F`, `mF`, `I`, `ε_z`
  keywords as `light_shift_coeff_Hz_per_Wcm2` (tensor part included); for I = 0
  isotopes the fine-structure quantum numbers are used automatically. `dipole_ea0`
  lines may go to a lower-lying level. The transition fields `Jg` (a duplicate of
  `J`) and `f` (the line-strength factor) are gone. (H. Schmit-Veiler)
- The semiclassical solvers are second order in `dt` for moving atoms (split
  velocity Verlet), and the fully classical (Newton) solver is second order with
  moving beams; they were first order. `test/unit/test_solver_order.jl` checks the
  convergence order. (H. Schmit-Veiler)

### Added
- `polarizabilities = Dict(term => model)` on any atom constructor (documented,
  never implemented): a polarizability model for a state the species does not ship,
  or an override, for that atom only.
- `PolarizabilityModel(…; tensor_offset_Hz_per_Wm2)`: a measured tensor light shift,
  recoupled to each hyperfine F like a line, so a (scalar, tensor) pair measured at
  one wavelength needs no line list. The `1P1` term is registered for Yb.
- Ytterbium-174, Rubidium-87 and Strontium-88 polarizability models; tensor
  polarizability; `scattering_rate_per_Wcm2`.
- `add_light_shift!`, `add_quantization_axis!`, `add_hamiltonian!`,
  `add_vdwinteraction!`, transverse exchange in `add_interaction!`,
  `rabi_frequencies`.
- Symbolic `Operator`s (`ket * bra'`), `ExpectationDetectorSpec`,
  `PhotoDetectorSpec`, `getjumps`, `getheffective`, `getliouvillian`.
- Typed term symbols: `TermSymbol`, `@term`, `l"3P1"`.
- A precompile workload, and solver benchmarks under `bench/`.
- `add_decay!(…; λ, dipole)`: the emission recoil follows the transition dipole's
  pattern `∝ 1 − |n̂·d̂|²` instead of being isotropic. A dipole along x̂ puts 1/5 of
  its recoil energy along x, not 1/3, which matters when one dipole dominates the
  scattering. Without `dipole`, unchanged (same RNG stream).

### Fixed
- `process_tomography` evolved every input state from the first input: an X
  gate with 20% Rabi noise scored process fidelity 0.25 instead of ≈ 0.91.
- Reusing a compiled job (`play(job, sys)`) started from the previous run's
  final state, accumulated photon clicks, and kept any coupling left on.
- Planar-beam couplings, van der Waals interactions and light shifts ignored
  `active = false`, `Pulse`, `On`/`Off` and `ampl`: they were always on.
- The Chebyshev propagator could leave its spectral interval — a trap ramped or
  moved during a pulse, a van der Waals interaction between static atoms — and
  diverge without an error.
- A density-matrix run with no dissipation took one unchecked Taylor step per
  `dt`, diverging past ‖H‖·dt ≈ 2.8.
- From the second shot on, polarizabilities were computed against ẑ instead of
  the system's quantization axis.
- Density-matrix runs with atomic motion felt no state-dependent force unless a
  population detector happened to be attached.
- Parameters sampled per shot compounded from shot to shot (planar couplings,
  interactions), were never resampled (decay rates), or were not refreshed in
  multi-threaded runs (Gaussian-beam couplings).
- Moves inside `Parallel` converged at first order in `dt`; a ramp after
  `AmplRow` started from a stale amplitude.
- An attractive van der Waals interaction (C₆ < 0) was pinned at its cap at
  every distance.
- `play` threw on a system with no detectors, and when called with
  `shots ≥ 4` inside a `Threads.@threads` loop.
- `isequal` on two atoms threw; the API docs showed a non-existent
  `noise_model` keyword and argument order for `laser_freq_psd`.
- The trap light shift raised a trapped level instead of lowering it, while the
  force still attracted: every differential light shift came out mirrored about the
  bare line. (H. Schmit-Veiler, S. Whitlock)
- The Rb-87 polarizability offset was off by a factor of 2. (H. Schmit-Veiler)
- The tensor light shift was applied on the m-diagonal only, right only for trap
  polarisation along (or circular about) the quantization axis. It is now the full
  operator, which mixes Δm = ±1, ±2 and with no field is quantised along ε; the
  dipole force takes its coherences. The 54.7° magic angle makes a manifold
  degenerate only once a field quantises it along the axis.
- MCWF radiation pressure gave one momentum to the whole excited branch, the mean
  over the beams that excited it: a balanced counter-propagating pair moved no
  momentum on absorption (var(p)/N = 0.41 (ħk)² for a σ⁺/σ⁻ pair, emission only).
  Each jump now draws the beam that supplied the photon (1.4 (ħk)²).
- Manifold decays (`add_decay!(e_manifold => g_manifold, …)`) took one jump per
  sublevel pair, so an excited superposition decayed into a mixture. There is now
  one coherent jump per photon polarisation, with the same branching ratios.
- The hyperfine dipole matrix element carried an extra (−1)^q: light along
  (x̂ + ẑ)/√2 excited the dipole (−x̂ + ẑ)/√2. Harmless while every operator was
  diagonal in m; not with the full tensor light shift.
- Fine → hyperfine couplings took Δm = m_g − m_e: σ⁺ and σ⁻ were swapped.
- `GeneralGaussianBeam` carried its phase as exp(−ik·r), opposite to `GaussianBeam`
  and `PlanarBeam`.

---

## [Pre-release]

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
