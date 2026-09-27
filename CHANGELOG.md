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

### Added
- Ytterbium-174, Rubidium-87 and Strontium-88 polarizability models; tensor
  polarizability; `scattering_rate_per_Wcm2`.
- `add_light_shift!`, `add_quantization_axis!`, `add_hamiltonian!`,
  `add_vdwinteraction!`, transverse exchange in `add_interaction!`,
  `rabi_frequencies`.
- Symbolic `Operator`s (`ket * bra'`), `ExpectationDetectorSpec`,
  `PhotoDetectorSpec`, `getjumps`, `getheffective`, `getliouvillian`.
- Typed term symbols: `TermSymbol`, `@term`, `l"3P1"`.
- A precompile workload, and solver benchmarks under `bench/`.

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
