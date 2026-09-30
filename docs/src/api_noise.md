# Parameters and Noise

## Parametric simulations

AtomTwin uses a lightweight symbolic parameter system to separate model
structure from numerical values. Rather than hard-coding frequencies,
detunings, and other quantities, you wrap them in a `Parameter` and pass
them to `add_coupling!`, `add_detuning!`, and similar functions. At `play`
time the values are resolved — either to their defaults or to values you
supply — without rebuilding the system.

`Parameter` objects can be combined with standard arithmetic to build
`ParametricExpression` trees:

```julia
Ω = Parameter(:Omega, 2π * 1.0e6)       # 1 MHz Rabi frequency
δ = Parameter(:delta, 0.0; std = 2π * 0.1e6) # 100 kHz shot-to-shot disorder

amp = 0.5 * Ω + δ                       # ParametricExpression
```

Pass these expressions directly to physics functions, then sweep or sample
at run time:

```julia
coupling = add_coupling!(system, atom, g => e, amp)

# sweep Omega over a grid
for Ω_val in range(0, 4π*1e6; length = 50)
    out = play(system, seq; initial_state = g, Omega = Ω_val)
end
```

```@docs
Parameter
ParametricExpression
```

## Laser phase noise

Realistic laser phase noise is added by attaching a `LaserPhaseNoiseModel`
to a coupling. The model synthesizes a time-domain noise realization from a
parameterized frequency-noise power spectral density (PSD) that combines a
Gaussian servo bump with a power-law background.

```julia
noise_model = LaserPhaseNoiseModel(
    bump_ampl   = 2.0e5,    # Hz²
    bump_center = 1.0e6,    # Hz — servo resonance at 1 MHz
    bump_width  = 2.0e5,    # Hz — Gaussian σ
    powerlaw_ampl = 0.25e5, # Hz² — white noise floor
)

coupling = add_coupling!(system, atom, g => e, Ω; noise = noise_model)
```

Each Monte Carlo shot draws an independent noise realization; averaging over
many shots (via `shots = N` in `play`) recovers ensemble-averaged dynamics.

The helper functions `laser_freq_psd` and `laser_phase_psd` evaluate the
model's PSD at arbitrary frequencies, which is useful for plotting noise
spectra before committing to a simulation:

```julia
freqs = range(0.1e6, 10e6; length = 500)
psd   = laser_freq_psd(freqs, noise_model)
```

```@docs
LaserPhaseNoiseModel
NoisyField
laser_freq_psd
laser_phase_psd
```

## Gate tomography

`process_tomography` runs a complete set of input states through a compiled
sequence and returns the Choi matrix of the resulting quantum channel. This
is useful for computing gate fidelities and diagnosing error mechanisms.

```julia
choi = process_tomography(system, seq)
```

The returned matrix is in the standard column-vectorisation convention and
can be used with any quantum-information library that accepts Choi matrices.

```@docs
process_tomography
```
