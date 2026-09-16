"""
Define laser noise models and noisy field wrappers.
"""

using Random
using .Dynamiq: AmplitudeModifier
import Base: deepcopy_internal

#=============================================================================
NOISE MODELS
=============================================================================#

"""
    LaserPhaseNoiseModel(; bump_ampl, bump_center, bump_width,
                          powerlaw_ampl, powerlaw_exp = 0.0, n_freqs = 100)

Realistic laser phase noise model using a parameterized frequency-noise
power spectral density (PSD) in Hz²/Hz.

The PSD model combines a Gaussian “servo bump” and a power-law background:

- Servo bump: centered at `bump_center` with width `bump_width` and amplitude
  `bump_ampl`.
- Power-law: `powerlaw_ampl * f^powerlaw_exp` (white frequency noise for
  `powerlaw_exp = 0`).

Fields:

- `bump_ampl::Float64`: Servo bump amplitude (Hz²).
- `bump_center::Float64`: Center frequency of the servo bump (Hz).
- `bump_width::Float64`: Gaussian width σ of the bump (Hz).
- `powerlaw_ampl::Float64`: Amplitude of the power-law background.
- `powerlaw_exp::Float64`: Exponent `d` of the power law.
- `n_freqs::Int`: Number of frequency components used when synthesizing noise.
"""
struct LaserPhaseNoiseModel <: AbstractNoiseModel
    bump_ampl::Float64          # 'hg' parameter
    bump_center::Float64        # 'fg' parameter (Hz)
    bump_width::Float64         # 'sigma' parameter (Hz)
    powerlaw_ampl::Float64      # 'h0' parameter  
    powerlaw_exp::Float64       # 'd' parameter
    n_freqs::Int                 # number of frequency components
    function LaserPhaseNoiseModel(;
            bump_ampl = 2.0e5,
            bump_center = 1.0e6,
            bump_width = 2.0e5,
            powerlaw_ampl = 0.25e5,
            powerlaw_exp = 0.0,
            n_freqs = 100)
        new(bump_ampl, bump_center, bump_width, powerlaw_ampl, powerlaw_exp, n_freqs)
    end
end

#=============================================================================
NOISY FIELD WRAPPER
=============================================================================#

"""
    NoisyField{A,T,N} <: Dynamiq.AbstractField

Wrapper that adds time-correlated noise to an underlying field or coupling.

Type parameters:

- `A`: Atom type associated with the coupling.
- `T`: Type of the wrapped field/coupling (e.g. `GlobalCoupling`, `Detuning`).
- `N <: AbstractNoiseModel`: Noise model type (e.g. [`LaserPhaseNoiseModel`](@ref)).

Fields:

- `coupling::T`: Original coupling/field object.
- `noise::N`: Noise model and parameters.
- `global_time_ref::Ref{Float64}`: Reference to the global simulation time,
  used to maintain temporal continuity across segments.
- `n_freqs::Int`: Number of frequency components used in the noise synthesis.
- `rng::AbstractRNG`: Random-number generator used to draw noise phases.
"""
mutable struct NoisyField{A, T, N} <: Dynamiq.AbstractField
    coupling::T
    noise::N
    global_time_ref::Ref{Float64}
    n_freqs::Int
    rng::AbstractRNG
    _phase_buffer::Vector{Float64}  # Reusable buffer
end


# Forwarded to the wrapped coupling. Takes a time now, like every other
# time-dependent update: the solver evaluates drives wherever its step lands,
# not at a step index.
Dynamiq.update!(field::NoisyField, t::Float64) = Dynamiq.update!(field.coupling, t)

# Constructors
NoisyField(coupling::T, noise::N; n_freqs = noise.n_freqs, rng=Random.Xoshiro(rand(UInt32))) where {T, N} =
    NoisyField{typeof(coupling.atom), T, N}(coupling, noise, Ref(0.0), n_freqs, rng, Float64[])

NoisyField(coupling::T, noise::N, global_time_ref::Ref{Float64}, n_freqs::Int, rng::AbstractRNG) where {T, N} =
    NoisyField{typeof(coupling.atom), T, N}(coupling, noise, global_time_ref, n_freqs, rng, Float64[])

base_coupling(nf::NoisyField) = nf.coupling

# Deepcopy with fresh RNG and buffer
function Base.deepcopy_internal(nf::NoisyField, stackdict::IdDict)
    if haskey(stackdict, nf)
        return stackdict[nf]
    end
    
    result = NoisyField{typeof(nf.coupling.atom), typeof(nf.coupling), typeof(nf.noise)}(
        deepcopy_internal(nf.coupling, stackdict),
        deepcopy_internal(nf.noise, stackdict),
        Ref(0.0),  # Fresh time ref
        nf.n_freqs,
        Random.Xoshiro(rand(UInt32)),  # Fresh RNG
        Float64[]  # Fresh buffer per copy
    )
    
    stackdict[nf] = result
    return result
end

# Property delegation
function Base.getproperty(nf::NoisyField, name::Symbol)
    if name in (:coupling, :noise, :global_time_ref, :n_freqs, :rng, :_phase_buffer)
        getfield(nf, name)
    else
        getproperty(getfield(nf, :coupling), name)
    end
end

#=============================================================================
NOISE GENERATION - Multiple Dispatch on Noise Type
=============================================================================#

# Simple standard deviation without additional packages
std(x) = sqrt(sum((xi - sum(x)/length(x))^2 for xi in x) / (length(x) - 1))

"""
    get_noise_spectrum(noise::LaserPhaseNoiseModel, tspan) -> (freqs, psd_values)

Compute the one-sided laser frequency-noise power spectral density `psd_values`
(units: Hz²/Hz) at a set of logarithmically spaced frequencies `freqs` (Hz),
given a [`LaserPhaseNoiseModel`](@ref) and a time span `tspan`.

- The frequency grid is chosen adaptively from the total duration and sampling
  interval (fundamental and Nyquist limits).
- The PSD combines the servo bump and power-law background components of the
  model.

Returns a tuple `(freqs, psd_values)`, which can be used for phase-noise
synthesis or for plotting.
"""
function get_noise_spectrum(noise::LaserPhaseNoiseModel, tspan::Vector{Float64})
    n_freqs = noise.n_freqs
    dt = tspan[2] - tspan[1]
    total_duration = tspan[end] - tspan[1]
    
    f_min = 1 / total_duration
    f_max = 1 / (2 * dt)
    
    freqs = exp.(range(log(f_min), log(f_max), length=n_freqs))
    
    function laser_psd_direct(f)
        servobump_freq = noise.bump_ampl * exp(-((f - noise.bump_center)^2) / (2 * noise.bump_width^2)) +
                         noise.bump_ampl * exp(-((f + noise.bump_center)^2) / (2 * noise.bump_width^2))
        powerlaw_freq = noise.powerlaw_ampl * (f + 1e-10)^noise.powerlaw_exp
        return servobump_freq + powerlaw_freq
    end
    
    psd_values = [2 * laser_psd_direct(f) for f in freqs]  # One-sided PSD
    
    return freqs, psd_values
end

"""
    generate_noise(tspan, global_offset, nf::NoisyField{A,T,LaserPhaseNoiseModel})
        -> Vector{ComplexF64}

Generate a time series of complex phase noise for a [`NoisyField`](@ref) with
a [`LaserPhaseNoiseModel`](@ref).

The phase trace is synthesized from the one-sided frequency-noise PSD
`S_ν(f)` using random phases for each frequency component and an amplitude
consistent with the PSD and frequency spacing.

Each sinusoidal term represents a conjugate ±frequency pair of the underlying real noise process.

Returns `exp.(1im * phase_noise)`, which can be multiplied into a base
amplitude profile to obtain a noisy complex envelope.
"""

function generate_noise(tspan, global_offset, nf::NoisyField{A, T, LaserPhaseNoiseModel}) where {A,T}
    vals = Vector{ComplexF64}(undef, length(tspan))
    update_noise!(vals, tspan, global_offset, nf)
    return vals
end

function update_noise!(vals::Vector{ComplexF64}, tspan, global_offset, nf::NoisyField{A, T, LaserPhaseNoiseModel}) where {A,T}
    # Compute spectrum (fast, no caching needed)
    freqs, psd_vals = get_noise_spectrum(nf.noise, tspan)
    dlogf = log(freqs[2] / freqs[1])
    
    n_t = length(tspan)
    n_f = length(freqs)
    
    # Reuse buffer
    if length(nf._phase_buffer) != n_t
        resize!(nf._phase_buffer, n_t)
    end
    fill!(nf._phase_buffer, 0.0)
    
    phase_buf = nf._phase_buffer
    
    # Generate random phases
    phases = 2π .* rand(nf.rng, n_f)
    
    # Pre-compute global time span if needed
    global_tspan = global_offset != 0.0 ? tspan .+ global_offset : tspan
    
    # Compute phase noise
    @inbounds for i in 1:n_f
        f = freqs[i]
        # NOTE:
        # psd_vals is a *one-sided* frequency-noise PSD S_ν(f) (Hz^2/Hz).
        # Each real sinusoid sin(2π f t + θ) with random phase θ represents the ±f pair
        # but has variance A^2 / 2. To reproduce the total variance
        # ∫ S_φ(f) df = ∫ S_ν(f) / (2π f)^2 df,
        # the amplitude must therefore include an extra factor of 2 (not √2).
        amplitude = 2 * sqrt(psd_vals[i] * dlogf / f) / 2π
        phase = phases[i]
       
        @simd for j in 1:n_t
            phase_buf[j] += amplitude * sin(2π * f * global_tspan[j] + phase)
        end
    end
    
    # Convert to complex exponential
    @inbounds @simd for j in 1:n_t
        vals[j] = cis(phase_buf[j])
    end
    
    return nothing
end


#=============================================================================
INTEGRATION WITH AMPLITUDE MODIFIERS
=============================================================================#

"""
    AmplitudeModifier(nf::NoisyField, amplitude, tspan, duration) -> AmplitudeModifier

Construct an `AmplitudeModifier` for a noisy field.

- Uses [`generate_noise`](@ref) to synthesize a complex-valued noise trace
  over `tspan`.
- Multiplies the base amplitude profile `amplitude` by the noise trace.
- Returns an `AmplitudeModifier` acting on the underlying `nf.coupling`.

This method is called during instruction compilation when a field is wrapped
in a [`NoisyField`](@ref).
"""
function Dynamiq.AmplitudeModifier(nf::NoisyField, amplitude::Vector{<:Number},
                                  tspan::Vector{Float64}, duration::Real)
    global_time = nf.global_time_ref[]
    noise_mod = generate_noise(tspan, global_time, nf)
    noisy_amplitude = amplitude .* noise_mod
    
    # Keep the NoisyField itself as the modifier target, NOT nf.coupling: the
    # per-shot regeneration hook in job.jl looks for `mod.field isa NoisyField`
    # to redraw the noise trace each shot. Handing it the unwrapped coupling
    # makes that check fail, so every shot reuses one trace and the trajectory
    # average shows no damping at all.
    #
    # `:linear`, not the usual `:cubic`: a noise trace is a stochastic
    # realisation synthesised at these very points, not a smooth function
    # sampled from one. Cubic would invent structure between samples and can
    # overshoot; linear passes through every drawn value.
    return AmplitudeModifier(nf, noisy_amplitude, duration; interp = :linear)
end

#=============================================================================
UTILITY FUNCTIONS
=============================================================================#

"""
    update_noisy_field_time_refs!(obj, global_time_ref, base_resolve_target=identity) -> obj

Update time references in [`NoisyField`](@ref) objects so that their noise
generation uses a shared `global_time_ref`.

- If `obj` is a `NoisyField`, returns a new `NoisyField` with the same noise
  model and RNG but with:
  - `global_time_ref` set to the provided reference, and
  - the wrapped coupling mapped through `base_resolve_target`.
- If `obj` is not a `NoisyField`, returns it unchanged.

This is called during parameter resolution inside `play`/`play!` to ensure
time-correlated noise across segments.
"""
function update_noisy_field_time_refs!(obj, global_time_ref, base_resolve_target = identity)
    if obj isa NoisyField
        mapped_field = base_resolve_target(obj.coupling)
        return NoisyField(mapped_field, obj.noise, global_time_ref, obj.n_freqs, obj.rng)
    else
        return obj
    end
end

#=============================================================================
CONSTRUCTORS - Add Different Noise Types to Fields
=============================================================================#

"""
    laser_freq_psd(freqs, noise::AbstractNoiseModel) -> psd

Compute the laser frequency-noise power spectral density S_ν(f) in Hz²/Hz
for one or more frequencies `freqs`, given a noise model.

Arguments:

- `freqs`: Single frequency (Number) or an array of frequencies (Hz).
- `noise`: Noise model, typically [`LaserPhaseNoiseModel`](@ref).

Returns:

- A single PSD value if `freqs` is a scalar, or a vector of PSD values
  matching `freqs` otherwise.

Internally this:

- Constructs a time span consistent with the requested frequency range.
- Uses [`get_noise_spectrum`](@ref) to evaluate the canonical frequency-noise
  PSD on a grid.
- Interpolates the result to the requested frequencies.

Example:
```@example
noise = LaserPhaseNoiseModel()
f = 10 .^ range(2, 7; length = 500) # 100 Hz to 10 MHz
Sν = laser_freq_psd(f, noise) # frequency noise PSD in Hz^2/Hz
```
"""
function laser_freq_psd(freqs, noise_model::AbstractNoiseModel)

    # Estimate tspan from frequency range
    dt = 1 / (2 * maximum(freqs))  # From Nyquist: f_max = 1/(2*dt)
    tmax = 1 / minimum(freqs)      # From fundamental: f_min = 1/T
    tspan = collect(range(dt,tmax; step=dt))

    # Get the canonical spectrum
    actual_freqs, actual_psd = get_noise_spectrum(noise_model, tspan)
    
    # Interpolate to requested frequencies
    query_array = freqs isa Number ? [freqs] : freqs
    result = [interp_linear(actual_freqs, actual_psd, qf) for qf in query_array]
    
    return freqs isa Number ? result[1] : result
end

"""
    laser_phase_psd(freqs, noise::AbstractNoiseModel) -> psd_phase

Compute the laser phase-noise power spectral density S_φ(f) in rad²/Hz
from the frequency-noise PSD S_ν(f) in Hz²/Hz, using

\\[
S_\\phi(f) = \\frac{S_\\nu(f)}{(2\\pi f)^2}.
\\]

Arguments:

- `freqs`: Single frequency (Number) or an array of frequencies (Hz).
- `noise`: Noise model, typically [`LaserPhaseNoiseModel`](@ref).

Returns:

- Phase-noise PSD values `S_φ(f)` with the same shape as `freqs`.

Example:
```@example
noise = LaserPhaseNoiseModel()
f = 10 .^ range(2, 7; length = 500)
Sφ = laser_phase_psd(f, noise) # phase noise PSD in rad^2/Hz
```
"""
function laser_phase_psd(freqs, noise)
    Sν = laser_freq_psd(freqs, noise)
    return Sν ./ ((2π .* freqs).^2)
end

# Simple linear interpolation helper
function interp_linear(x_vals, y_vals, x_query)
    if x_query <= x_vals[1]; return y_vals[1]; end
    if x_query >= x_vals[end]; return y_vals[end]; end
    
    idx = searchsortedfirst(x_vals, x_query)
    x1, x2 = x_vals[idx-1], x_vals[idx]
    y1, y2 = y_vals[idx-1], y_vals[idx]
    return y1 + (y2 - y1) * (x_query - x1) / (x2 - x1)
end

#=============================================================================
VISUALIZATION - Plotting recipes
=============================================================================#

RecipesBase.@recipe function f(noise_model::LaserPhaseNoiseModel; fmin=1e2, fmax=1e8, npts=1000)
    # Generate frequency range
    f = 10 .^ range(log10(fmin), log10(fmax), length=npts)
    
    # Compute PSD
    S = laser_freq_psd(f, noise_model)
    
    # Plot settings
    seriestype --> :line
    linewidth --> 2
    xscale --> :log10
    yscale --> :log10
    xticks --> 10. .^ (2:8)
    yticks --> 10. .^ (-2:10)
    minorgrid --> true
    minorgridwidth --> 1
    minorgridcolor --> :gray
    minorgridalpha --> 0.3
    xlabel --> "Frequency (Hz)"
    ylabel --> "PSD (Hz²/Hz)"
    legend --> false
    
    # Return data to plot
    f, S
end


