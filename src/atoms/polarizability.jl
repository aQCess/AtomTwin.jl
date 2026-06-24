"""
    polarizability.jl

Generic infrastructure for computing and visualizing atomic polarizabilities
from empirical transition models.
"""

using RecipesBase
using ..Units: c, ε0, a0, h, hbar

# ======================================================================
# Data model
# ======================================================================

"""
    PolarizabilityModel

Empirical polarizability model for an atomic state, defined by a set of
discrete transitions and an optional offset.

# Fields
- `state::String`: Electronic state label (e.g. `"1S0"`, `"3P0"`).
- `transitions::Vector{NamedTuple}`: List of transitions; each entry has
  - `freq_THz::Float64`: Transition frequency in THz (linear).
  - `gamma_MHz::Float64`: Linewidth in MHz (linear).
- `offset_Hz_per_Wm2::Float64`: Empirical offset in Hz/(W/m²).
- `reference::String`: Bibliographic reference for the data.
"""
struct PolarizabilityModel
    state::String
    transitions::Vector{NamedTuple{(:freq_THz, :gamma_MHz), Tuple{Float64, Float64}}}
    offset_Hz_per_Wm2::Float64
    reference::String
end

function PolarizabilityModel(state::String,
                             transitions::Vector;
                             offset_Hz_per_Wm2::Float64 = 0.0,
                             reference::String = "")
    PolarizabilityModel(state, transitions, offset_Hz_per_Wm2, reference)
end

# ======================================================================
# Core physics
# ======================================================================

"""
    _calc_light_shift(ω0, Γ, ωL) -> Float64

Light-shift contribution from a single electric-dipole transition.

# Arguments
- `ω0`: Transition angular frequency [rad/s].
- `Γ` : Radiative linewidth (angular) [rad/s].
- `ωL`: Laser angular frequency [rad/s].

# Returns
- `U/I`: Energy shift per intensity in J/(W/m²).
"""
function _calc_light_shift(ω0::Float64, Γ::Float64, ωL::Float64)
    return -3 * π * c^2 * Γ / (ω0^2 * (ω0^2 - ωL^2))
end

"""
    _calc_scattering_rate(ω0, Γ, ωL) -> Float64

Off-resonant photon-scattering contribution from a single electric-dipole
transition, per unit intensity.

Far-off-resonance result for a two-level transition (Grimm, Weidemüller &
Ovchinnikov, *Adv. At. Mol. Opt. Phys.* **42**, 95 (2000), Eq. 11), retaining
both rotating and counter-rotating terms so it is valid across the full
wavelength range, not only in the rotating-wave limit:

    Γ_sc / I = (3π c² / 2ħ ω₀³) (ωL/ω₀)³ ( Γ/(ω₀−ωL) + Γ/(ω₀+ωL) )²

# Arguments
- `ω0`: Transition angular frequency [rad/s].
- `Γ` : Radiative linewidth (angular) [rad/s].
- `ωL`: Laser angular frequency [rad/s].

# Returns
- `Γ_sc/I`: photon-scattering rate per intensity in (1/s)/(W/m²).
"""
function _calc_scattering_rate(ω0::Float64, Γ::Float64, ωL::Float64)
    return (3 * π * c^2) / (2 * hbar * ω0^3) * (ωL / ω0)^3 *
           (Γ / (ω0 - ωL) + Γ / (ω0 + ωL))^2
end

"""
    _Gamma_sc_over_I(model::PolarizabilityModel, λ_nm::Real) -> Float64

Total off-resonant photon-scattering rate per intensity Γ_sc/I for a given
state model and wavelength, as an incoherent sum over the model's transitions.

The incoherent (rate) sum is the standard far-detuned approximation: it is
accurate when the laser is far from every line relative to the line spacings,
which holds for a trap wavelength well away from all resonances. Returns
(1/s)/(W/m²).
"""
function _Gamma_sc_over_I(model::PolarizabilityModel, λ_nm::Real)
    ωL = 2π * c / (λ_nm * 1e-9)

    Γsc_over_I = 0.0
    for t in model.transitions
        ω0 = 2π * t.freq_THz  * 1e12
        Γ  = 2π * t.gamma_MHz * 1e6
        Γsc_over_I += _calc_scattering_rate(ω0, Γ, ωL)
    end
    return Γsc_over_I
end

"""
    _U_over_I(model::PolarizabilityModel, λ_nm::Real) -> Float64

Compute total light shift per intensity U/I for a given model and wavelength.

# Arguments
- `model::PolarizabilityModel`: Polarizability model for a single state.
- `λ_nm`: Laser wavelength in nanometres.

# Returns
- `U/I` in J/(W/m²).
"""
function _U_over_I(model::PolarizabilityModel, λ_nm::Real)
    ωL = 2π * c / (λ_nm * 1e-9)

    U_over_I = 0.0
    for t in model.transitions
        ω0 = 2π * t.freq_THz  * 1e12
        Γ  = 2π * t.gamma_MHz * 1e6
        U_over_I += _calc_light_shift(ω0, Γ, ωL)
    end

    U_over_I += model.offset_Hz_per_Wm2 * h
    return U_over_I
end

# ======================================================================
# Public API
# ======================================================================

"""
    light_shift_coeff_Hz_per_Wcm2(model::PolarizabilityModel, λ_nm::Real) -> Float64

Light-shift coefficient Δν/I in Hz/(W/cm²) for the given model and wavelength.

# Definition
For a beam intensity `I` in W/cm², the light shift is

    Δν = light_shift_coeff_Hz_per_Wcm2(model, λ_nm) * I

# Arguments
- `model`: Polarizability model for a single atomic state.
- `λ_nm`: Laser wavelength in nanometres.
"""
function light_shift_coeff_Hz_per_Wcm2(model::PolarizabilityModel, λ_nm::Real)
    U = _U_over_I(model, λ_nm)
    ν_over_I = U / h              # Hz/(W/m²)
    return ν_over_I * 1e4         # Hz/(W/cm²)
end

"""
    scattering_rate_per_Wcm2(model::PolarizabilityModel, λ_nm::Real) -> Float64

Off-resonant photon-scattering-rate coefficient Γ_sc/I in (1/s)/(W/cm²) for the
given state model and wavelength.

# Definition
For a beam intensity `I` in W/cm², the off-resonant scattering rate is

    Γ_sc = scattering_rate_per_Wcm2(model, λ_nm) * I        # rad/s? NO — 1/s

`Γ_sc` is a real photon-scattering *rate* in s⁻¹ (events per second), not an
angular frequency. It is computed from the same transition data used for the
light shift (see [`_Gamma_sc_over_I`](@ref)).

The light shift scales as Γ/Δ while the scattering rate scales as (Γ/Δ)²; for a
far-detuned trap Γ_sc ≪ |Δν|, which is what makes a far-off-resonance dipole trap
viable.
"""
function scattering_rate_per_Wcm2(model::PolarizabilityModel, λ_nm::Real)
    Γsc_over_I = _Gamma_sc_over_I(model, λ_nm)   # (1/s)/(W/m²)
    return Γsc_over_I * 1e4                       # (1/s)/(W/cm²)
end

"""
    polarizability_si(model::PolarizabilityModel, λ_nm::Real) -> Float64

Dynamic electric polarizability α in SI units (C·m²·V⁻¹) for the given
model and wavelength.

# Definition
Uses the relation

    U/I = -α_SI / (c ε₀)

which gives

    α_SI = -c ε₀ (U/I)

where U/I is the light shift per intensity in J/(W/m²).

# Units
Returns polarizability in C·m²·V⁻¹ (or equivalently F·m²), which is the
standard SI unit for electric polarizability.
"""
function polarizability_si(model::PolarizabilityModel, λ_nm::Real)
    U = _U_over_I(model, λ_nm)
    α_SI = - c * ε0 * U
    return α_SI
end

"""
    polarizability_au(model::PolarizabilityModel, λ_nm::Real) -> Float64

Dynamic electric polarizability α in atomic units (a₀³) for the given
model and wavelength.

# Definition
Converts from SI units via

    α_au = α_SI / (4π ε₀ a₀³)
"""
function polarizability_au(model::PolarizabilityModel, λ_nm::Real)
    α_SI = polarizability_si(model, λ_nm)
    return α_SI / (4π * ε0 * a0^3)
end


# ======================================================================
# Plot recipe
# ======================================================================

"""
    PolarizabilityCurve

Container for plotting polarizability curves with an optional inset zoom.

# Fields
- `models::Vector{PolarizabilityModel}`: List of polarizability models to plot.
- `λ_main`: Wavelength range for main plot in nm (default: `420:0.1:800`).
- `λ_inset`: Wavelength range for inset zoom in nm (default: `550.5:0.1:556.2`).
- `ylim_main`: y-axis limits for main plot (default: `(-30, 10)`).
- `ylim_inset`: y-axis limits for inset (default: `(-8, 3.5)`).
- `unit::Symbol`: Plot unit, either `:Hz_per_Wcm2` (default) or `:au`.

# Usage

using Plots
## Default plot with inset

curve = PolarizabilityCurve([model_1S0, model_3P0])
plot(curve)
## Custom ranges

curve = PolarizabilityCurve([model_1S0, model_3P0],
λ_main = 400:0.2:900,
λ_inset = 555:0.05:556,
ylim_inset = (-5, 2))
plot(curve)
No inset (set λ_inset = nothing)

curve = PolarizabilityCurve([model_1S0, model_3P0], λ_inset = nothing)
plot(curve)
"""
struct PolarizabilityCurve
    models::Vector{PolarizabilityModel}
    λ_main::AbstractRange{<:Real}
    λ_inset::Union{Nothing, AbstractRange{<:Real}}
    ylim_main::Tuple{Real, Real}
    ylim_inset::Tuple{Real, Real}
    inset_position::Tuple{Real, Real, Real, Real}
    unit::Symbol
end

function PolarizabilityCurve(models::Vector{PolarizabilityModel};
                             λ_main = 420.0:0.1:800.0,
                             λ_inset = 550.5:0.1:556.2,
                             ylim_main = (-30, 10),
                             ylim_inset = (-8, 3.5),
                             inset_position = (0.69, 0.02, 0.28, 0.30),
                             unit::Symbol = :Hz_per_Wcm2)
    PolarizabilityCurve(models, λ_main, λ_inset, ylim_main, ylim_inset, inset_position, unit)
end

PolarizabilityCurve(model::PolarizabilityModel; kwargs...) =
    PolarizabilityCurve([model]; kwargs...)


## see ext/AtomTwinPlots.jl for plot recipes