# Changelog

All notable changes to AtomTwin are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Fixed

- **Density-matrix dephasing with signed amplitudes.** A diagonal jump over several
  levels, `L = Σ vₛ|s⟩⟨s|`, now damps a coherence at `½γ|vᵢ − vⱼ|²` and turns it at
  `γ Im(vᵢvⱼ*)`. The step used `|vᵢ||vⱼ|`, so a σz-type jump left the coherence
  undamped in the density-matrix solver while MCWF damped it. The built-in
  `add_dephasing!` (one level per jump) was not affected.
- **`Sequence(; tol)` with a Taylor integrator.** The automatic step took the order-4
  bound whatever the order: `Taylor(n < 4)` stepped past `tol`.
- **Parameter expressions.** Beams and instructions resolved only `+` and `*`, so
  `p - 1` or `1/p` built but failed at `play`; all resolvers now share one table
  (`+`, `-`, `*`, `inv`, `abs`), and `-p` and `p/q` are defined. A noisy `Parameter`
  in a beam or an instruction now draws from the `rng` given to `play` rather than
  the global RNG, so such runs are reproducible (and change).
- **Wavelength units.** `polarizability_si`, `polarizability_au`,
  `light_shift_coeff_Hz_per_Wcm2` and `scattering_rate_per_Wcm2` take nanometres;
  a wavelength in metres (`759e-9`, `759nm`) returned a finite, wrong value and is now
  an error.
- **Unknown species.** `AtomWrapper{S}` for a species with no defaults took 87 amu and
  `I = 1/2` silently; it now needs `mass` and `I`.
- **Beam constructors.** The `GeneralGaussianBeam` keyword form always threw; an `Int`
  power made `GaussianBeam` throw and `GeneralGaussianBeam` return a parametric beam.
  Both take any `Real`, and `PlanarBeam` gains a keyword form.
- **API consistency.** Manifold `add_decay!` returns its `Jump`s (it returned
  `nothing`); `add_zeeman_detunings!` returns a typed vector;
  `ExpectationDetectorSpec(sys, atom, O)` takes `tspan`; the detector wrappers take
  any string `name`; `gethamiltonian` no longer swallows unknown keywords; the
  warning for a coupling to a level the atom lacks no longer throws.
- **Dynamiq.** `Basis(atoms, elements)` sets the atoms' population indices;
  `productstate` names a state missing from the basis; `coeff * op` keeps the
  operator's forward/reverse split, so the term stays Hermitian.
- **Hermiticity under a complex rate.** `rescale!` scaled the conjugate half of an
  operator by `s` instead of `conj(s)`, so a Gaussian-beam coupling refreshed at an
  atom displaced along the beam (every thermal shot) ran with a non-Hermitian
  Hamiltonian: populations reached 10³. Every MCWF or statevector run with a
  beam-derived coupling and a sampled or off-origin position changes.
- **Instant instructions and `Parallel`.** `On`, `Off`, `AmplRow/Col`, `FreqRow/Col`
  took two solver steps each (`On(c); Wait(T); Off(c)` drove for `T + 4dt`); they
  take none, and a zero-step instruction leaves no sample in `out.times`. A pulse
  shorter than its `Parallel` stayed on to the end of the `Parallel`; it ends at its
  own duration, and an `Off` inside a `Parallel` switches at the start.
- **Gaussian-beam couplings.** Built at the beam's focus, not where the atom sat
  before `compile` (the origin): an addressing beam focused on a distant site was
  silently dead. A shot beyond the beam's cutoff no longer zeroes the operator for
  good.
- **Interactions.** `add_vdwinteraction!` read `transition` differently from
  `add_interaction!` (`(g,r) => (g,r)` built `|rr⟩⟨gg| + h.c.`), could not take two
  atoms with different level objects, and threw on its default `V_cap` for a
  `Parameter` C6.
- **Angular factors.** The hyperfine dipole element fixed its sign pair by pair
  (stretched σ⁺ positive), which flipped the interference of two manifolds coupled
  to a shared one: the Rb D1 π–π Raman amplitude summed over F′ did not vanish.
  One phase convention (Edmonds) now. A hyperfine → fine-level coupling summed the
  polarizations' amplitudes coherently although they reach orthogonal nuclear-spin
  states; they add in quadrature. `rabi_frequencies(beam; d_red)` takes the
  stretched-transition element (documented); `rabi_frequencies(atom, g, e, beam)`
  converts a reduced element so the positional form matches the beam-derived one.
- **Build-time `tol`** dropped a `Parameter`-valued Ω or rate whose default was 0
  (keyword manifold coupling form, manifold decay) before `play` could set it.
- **`d_red`** was never used where it is needed: the species lookup ran first and
  threw for a line without line data. `|ΔJ| > 1`, `|ΔF| > 1` and `noise` with a
  beam are refused instead of silently giving nothing.
- **Atom model.** Duplicate levels and an F outside `|J−I| … J+I` are errors (for a
  species or an explicit `I`; a generic `Atom` has no nuclear spin to check); a
  level's J is checked against its polarizability model; a species with no models
  warns once in a beam, as does elliptical trap light (vector shift not modelled);
  a `FineLevel` takes the I = 0 tensor shift on any isotope, as its couplings do;
  `gethamiltonian` refreshes explicit `add_light_shift!` nodes for the current
  axis. Species masses are AME2020 isotope masses (0.04–0.11 % below the mass
  numbers; trap frequencies move by up to 0.05 %); the Rb D1 reduced dipole is
  4.231 ea₀.
- **Laser phase noise.** The synthesised phase noise was far too weak: `S_φ` was
  taken as `S_ν/(2πf)²` for an `S_ν` in Hz²/Hz (it is `S_ν/f²`: `S_ν = f² S_φ` for a
  frequency in Hz, de Léséleuc et al. PRA 97, 053803, the convention of Jiang et al.
  PRA 107, 042611 and of the model's own documentation; `(2πf)²` is the relation for
  an angular-frequency PSD), the one-sided PSD
  was doubled, the sinusoid amplitude was `2√(…)` for `√(2…)`, and the band started
  at `1/T`, which drops most of a white floor's phase diffusion — together ~10× too
  little phase variance for a servo bump and ~100× for a white floor. The
  synthesis now reproduces `Var[φ(T) − φ(0)] = 2π² h₀ T` for a white floor `h₀` to
  ~10 % and a servo bump's `∫S_ν/f² df` to a few %; the band starts at `1/(100T)`
  and `n_freqs` defaults to 300. **Every run with a `LaserPhaseNoiseModel`
  changes**. The model's defaults and the `rabi_with_noise` example, whose
  `powerlaw_ampl = 2.5e4` and `bump_ampl = 2e5` Hz²/Hz had been chosen to look right
  under the old synthesis (a 78 kHz-linewidth laser at face value), are rescaled by
  `1/(2π²)` to `1.25e3` and `1e4` Hz²/Hz (a 4 kHz linewidth), so the example's damping
  is what it was.
- **Instruction constructors** take any `Real` duration, `On`/`Off` a vector with
  `dt`/`downsample`, `MoveRow`/`RampRow`/`RampCol` a vector of rows and a
  `Parameter` delta, like their column twins; `FreqRow/Col` keep the array's `r0`.
  `add_dephasing!` states its rate convention (`L = √γ|l⟩⟨l|`: a coherence decays
  at γ/2); manifold `add_decay!` states that `rate` is the partial rate into that
  ground manifold.
- **MCWF step.** Chebyshev is exact for a Hermitian `H` at any step but not for the
  MCWF effective Hamiltonian: with a weak drive on a decaying line and `Γ·dt ≳ 20`
  the series diverged and the jump test renormalised it into a plausible wrong
  population (`P_e = 0.99` for 0.01); `Sequence(; tol)` was no safer. `wfmc` and
  `wfmc_semiclassical` now divide the output step into steps the non-Hermitian part
  allows (of order `10/Γ`), jump times staying exact. A coarse-`dt` MCWF run on a
  decaying line changes.
- **Radiation pressure.** The impulse entered the velocity only, so the position
  lagged by `a·T·dt/2` in all three semiclassical solvers; the force at the start
  of a step now drifts the position as the dipole force does. `newton` swallowed
  its `dt`, so a one-step classical instruction did not move the atoms. Frozen
  solvers refresh position-dependent fields only when a modifier moves a beam.
- **Crossed beams.** `force` summed same-wavelength beams as scalars, so two
  orthogonally polarized beams interfered (a σ⁺/σ⁻ or lin⊥lin lattice had the
  wrong landscape, and a twin crossed-beam trap the wrong depth); the vector
  field is summed with each beam's polarization, as `intensity` already did.
- **`GeneralGaussianBeam` frame.** The local x axis was `k̂ × x̂` (the lab **y**
  axis for a beam along z), so `w0x` and `w0y` were swapped against the lab axes
  and the documentation. Local x is now the lab x axis projected transverse to
  `k̂`; every elliptical `GeneralGaussianBeam` changes. `efield_vector` returns
  the bare field for it as for `GaussianBeam` (it carried the pulse coefficient);
  the field helpers allocate nothing in `force`.
- **Detectors and beams.** `add_detector!` refuses a second detector with the same
  `name` (the output replaced the first silently); `MotionDetectorSpec` rejects a
  beam without a position; a `PlanarBeam` is refused as a trapping beam of a
  `System` (it has no profile to trap with, and the trap machinery threw later);
  `TweezerArray` documents that its row/column weights are relative powers and
  that spot `(i, j)` has linear index `(i − 1)·ncol + j`; the coherence detector
  documents `ρ_ji` as the recorded convention.

- **A job is a snapshot.** `compile` works on a private copy of the `System`: its
  fields and atoms were the system's own, so compiling or playing one job retuned
  every other job from that system (`job1 = compile(sys, seq; Ω = A); job2 =
  compile(sys, seq; Ω = B)` ran `job1` at `B`), and a serial multi-shot run left
  the last shot's random draw in the user's system (`gethamiltonian(sys)` after a
  sampled run returned a random Hamiltonian). The system is as declared before and
  after `play`; `copy(sys)` is an independent copy (it always threw).
- **Replaying a job.** `play(job, sys; name = value)` applied the override from
  shot 2 on (`shots = 1` ignored it), and shot 1 of a replay kept the previous run's
  last draw, so a loop of replays was not an ensemble. Every shot of a replay is
  re-resolved: a fresh draw of every sampled quantity, at the parameter values the
  job was last given (`compile`, `recompile!` or `play` keywords; the job remembers
  them, so `recompile!(job; Δ = x); play(job, sys)` sweeps as documented). The single-shot path returned the job's own buffers, so a
  later replay overwrote earlier results in place; it returns copies.
- **Unknown keywords.** A `play`/`compile` keyword that names no `Parameter` of the
  system or sequence (`Omeg = …`, `delta` for `δ`, `seed = 1`) ran silently at the
  default; it is an `ArgumentError`. `play(job, …; density_matrix = …)` disagreeing
  with the job is an error (it ran the other solver silently); an empty sequence is
  an error at `compile`, not a 20-argument `MethodError`.
- **Beam-derived couplings follow the quantization axis.** `add_coupling!(…; beam)`
  decomposed the polarization about ẑ whatever `add_quantization_axis!` set, so a
  beam polarized along an x̂ bias field drove `Δm = ±1` while the Zeeman and light
  shifts of the same sublevels were computed about x̂, and a sampled axis never
  reached the couplings. They now follow the system's axis (set before or after the
  coupling, sampled per shot); `q_axis` fixes an axis for one coupling. Pairs are no
  longer dropped at `add_coupling!` by a strength the axis may still change.
- **Parametric axis.** `Parameter(:B, [0, 0, 1.0]; std = 0.02)` and `[θ, 0, 1 − θ]`
  threw at `compile`/`add_quantization_axis!`; a vector resolves per component and
  an array-valued Parameter draws one deviate per component.
- **Initial states.** `getqstate`/`initial_state` built an N-atom product state by
  summing full-dimension vectors, O(terms × dim): 825 MB for 12 atoms in `g + e`,
  out of memory at 16. One pass over the basis now. A single level is taken for
  every atom (as documented), a numeric statevector or density matrix is accepted,
  and a wrong number of entries is an error (it was silently truncated).
- **Sampled rates** were rescaled from the previous shot's operator, so a run's
  numbers depended on which thread ran which shot (serial ≠ threaded at 1e-14).
  Coupling and interaction entries are set from the rate itself; serial and
  threaded runs of one seed agree bit for bit. (A jump's entries are still
  rescaled, by `√(Γ′/Γ)`.)
- **Performance, measured against `devel` with `bench/` and the examples suite.**
  Three regressions introduced earlier on this branch were found and removed before
  the merge: the polarization lookup in `force` boxed every accumulation when the
  beam list is abstractly typed (the `atom_sorting` example took 1.1 s for 19 ms);
  the MCWF sub-division bound re-scanned the operator terms (35 ms per shot at
  d = 4096, a second Chebyshev spectral scan); and a job's private copy rebuilt the
  jumps' solver caches at every `compile` (3 ms per 12 decays at d = 4096). The
  caches are now built once at `add_decay!` and travel with the copy; the bound
  reads the spectrum the plan already has. What remains per `compile` is the copy
  of the System, 0.8 ms at d = 4096 with 24 nodes. Benchmarks 1, 3 and 4 are at
  parity; benchmark 2 pays the copy (+4 % TDSE, within noise for WFMC); the
  `rabi_with_noise` example is 2× slower from `n_freqs = 300` (was 100) and the
  wider band, which the corrected noise needs.
- **Sanitation.** `is_classical` was always `false`; the `recompile!` docstring
  claimed the system may be shared across threads; `getqstate` and `noise.jl`
  referred to a `play!` that does not exist; `compile` documented a wrong
  `initial_state` default.
- **Tests and docs.** `test_motion.jl` normalised a one-photon momentum balance by
  a mass-number recoil velocity (failing at 1e-6 after the isotope masses); the
  `rabi_with_noise` example's "analytical" damping rate was `S_ν/2` for
  `π² S_ν(Ω/2π)/2` (the old factor matched the old, too-weak noise); `api_noise.md`
  called `process_tomography` with two arguments and labelled PSD amplitudes Hz²
  (Hz²/Hz); `solvers.md` gave the order-4 Taylor bound for every order; four
  exported Dynamiq names rendered on no page; `internals.md` listed the deleted
  `interp.jl`; README and index promised K-39 polarizability models (levels and
  mass only). CONTRIBUTING described ground-truth files that do not exist. New
  tests: laser phase noise end to end (Rabi envelope `π² h₀/2`), MCWF dephasing
  against the master equation, `Taylor(n)` through `compile`/`play`, exact
  `exp(−iHT)` and `exp(𝓛T)` references for two pinned numbers, the off-resonance
  Rabi value and period, AME2020 masses.

### Added

- Beam-derived couplings between fine-structure manifolds (`FineManifold => FineManifold`,
  e.g. Yb-174 556 nm), with the `I = 0`, `F = J` angular factor.
- `SimulationJob.system`: a job's private copy of the system it was compiled from.
- `getqstate(sys, ψ::AbstractVector{<:Number})` / `(sys, ρ::AbstractMatrix)`: numeric
  initial states.
- Since 0.2.1, before this review (unrecorded until now): the Yb E1 decay-rate table
  and the `3P2` term (`getdecayrates`); line frequencies and reduced dipoles
  (`getlinefrequencies`, `reduced_dipole`), so a beam drives a line without `d_red`;
  Landé g-factors from species `g_J`, `g_I` (`getgfactors`, `lande_gF`, `zeeman_g`;
  g-factors default from species data); manifold decays recoil along each photon's
  dipole pattern and count σ and π apart (`clicks = (π = …, σ = …)`);
  `PhotoDetectorSpec(; aperture, efficiency)`, a camera that sees each photon by its
  direction; a detector's `tspan` is honoured and `MotionDetectorSpec` records the
  velocity (`dims = 4:6`; the output gains columns).

### Changed

- Since 0.2.1, before this review: the Zeeman shift uses `μ_B` from the unit
  constants (numbers change in the last digits); `gethamiltonian` includes the
  trapping beams' light shifts.

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
