"""
    polarizability.jl

Generic infrastructure for computing and visualizing atomic polarizabilities
from empirical transition models.
"""

using RecipesBase
using ..Units: c, ε0, a0, h, hbar, e
using WignerSymbols

# ======================================================================
# Data model
# ======================================================================

"""
    PolarizabilityModel

Empirical polarizability model for an atomic state, defined by a set of
discrete transitions and an optional offset.

# Fields
- `state::String`: Electronic state label(e.g. `"1S0"`, `"3P0"`). We refer to this as the initial state.
- `transitions::Vector{NamedTuple}`: List of transitions; each entry has
  - `freq_THz::Float64`: Transition frequency in THz (linear). If negative, then state_f < state in energy.
  - `gamma_MHz::Float64`: Effective line width in MHz (linear) — see below.
  - `state_f::String`: Electronic configuration of final state.
  - `J_f::Rational`: Total angular momentum of final state.
- `J_i`::Rational: Total angular momentum of intial state
- `offset_Hz_per_Wm2::Float64`: Empirical offset in Hz/(W/m²).
- `reference::String`: Bibliographic reference for the data.


Note that `freq_THz` is positive (negative) if the final state is above (below) the initial state in energy

# Specifying transitions

Each transition weights the dynamic polarizability by its **line strength**. Two
equivalent ways to supply that weight are accepted by the constructor:

1. `(freq_THz = …, gamma_MHz = …)` — the transition's effective line-strength
   width. This equals the natural linewidth **only for a `J=0 → J'=1` line**
   (e.g. every Yb `¹S₀` line), where the upper level decays to a single ground
   level. For such lines just use the natural linewidth.

2. `(freq_THz = …, dipole_ea0 = …)` — the reduced dipole matrix element
   `|⟨Jg‖er‖Je⟩|` in units of `e·a₀` (Steck's D-line convention), optionally with
   `Jg = …` (ground total angular momentum, default `1/2`). This is the robust
   choice for a **multiplet** such as an alkali D₁/D₂ doublet, where the two lines
   share a ground state but have unequal line strengths (D₂ carries twice the
   strength of D₁). The constructor converts the dipole to the effective width via

       Γ_eff = (2/9π) · |d|² ω₀³ / ((2Jg+1) ε₀ c³ ħ)                    (rad/s)

   so that AtomTwin's two-level light-shift sum reproduces the multi-level scalar
   polarizability α(ω) = (1/(2Jg+1)) Σᵢ (2/3) |dᵢ|² ω₀ᵢ / (ħ(ω₀ᵢ²−ω²)) exactly.

Do **not** feed natural linewidths for an alkali doublet: the near-equal D₁/D₂
natural widths under-weight D₂ and mis-split the light shift off-resonance, while
still agreeing at the static limit (a silent error). Use `dipole_ea0` instead.
"""
struct PolarizabilityModel
    state::String
    transitions::Vector{NamedTuple{(:freq_THz, :gamma_MHz, :state_f, :J_f), Tuple{Float64, Float64, String, Rational}}}
    J_i::Rational
    offset_Hz_per_Wm2::Float64
    reference::String
end

"""
    _dipole_to_gamma_MHz(freq_THz, dipole_ea0, Jg) -> Float64

Effective line-strength width (MHz, linear) for a transition specified by its
reduced dipole matrix element `dipole_ea0` = `|⟨Jg‖er‖Je⟩|` in `e·a₀`, such that
the two-level light-shift form reproduces the multi-level scalar polarizability.

    Γ_eff = (2/9π) · |d|² ω₀³ / ((2Jg+1) ε₀ c³ ħ)      [rad/s]

`Jg` is the ground-state total angular momentum (the `1/(2Jg+1)` line-strength
normalisation). See [`PolarizabilityModel`](@ref).
"""
function _dipole_to_gamma_MHz(freq_THz::Real, dipole_ea0::Real, Jg::Real)
    ω0 = 2π * freq_THz * 1e12
    d  = dipole_ea0 * e * a0                       # C·m
    Γ_eff = (2 / (9π)) * d^2 * ω0^3 / ((2Jg + 1) * ε0 * c^3 * hbar)   # rad/s
    return Γ_eff / (2π * 1e6)                       # → MHz (linear)
end

# Normalise a single user transition entry to the internal (freq_THz, gamma_MHz)
# form. A `gamma_MHz` entry is taken as-is; a `dipole_ea0` entry is converted to
# its effective line-strength width (optional `Jg`, default 1/2 for alkalis).
function _normalize_transition(t)
    if haskey(t, :gamma_MHz)
        return (freq_THz = Float64(t.freq_THz), gamma_MHz = Float64(t.gamma_MHz), state_f = String(t.state_f), J_f = Rational(t.J_f))
    elseif haskey(t, :dipole_ea0)
        Jg = haskey(t, :Jg) ? t.Jg : 0.5
        return (freq_THz = Float64(t.freq_THz),
                gamma_MHz = _dipole_to_gamma_MHz(t.freq_THz, t.dipole_ea0, Jg),
                state_f = "",
                J_f = 1//2)
    else
        error("PolarizabilityModel transition must have either `gamma_MHz` or " *
              "`dipole_ea0`; got keys $(keys(t))")
    end
end

function PolarizabilityModel(state::String,
                             transitions::Vector;
                             J_i::Union{Int, Rational},
                             offset_Hz_per_Wm2::Float64 = 0.0,
                             reference::String = "")
    J_i_r = Rational(J_i)
    norm = [_normalize_transition(t) for t in transitions]
    PolarizabilityModel(state, norm, J_i_r, offset_Hz_per_Wm2, reference)
end


# ======================================================================
# Core physics
# ======================================================================

"""
    _calc_light_shift(ω0, Γ, ωL) -> Float64

[DEPRECATED]

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
    _U_over_I(model::PolarizabilityModel, λ_nm::Real) -> Float64

[DEPRECATED] Use _U_over_I_si instead

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
    _degeneracy_factor(J::Rational, J_prime::Rational, transition_freq::Real) -> Rational

Computes the degeneracy factor of a transitions for polarizability calculations.
Factor depends on wether transitions goes to a higher or lower lying state, which is 
found by computing the sign of `transition_freq`.
"""
function _degeneracy_factor(J_i::Rational, J_f::Rational, transition_freq::Real)
    s = sign(transition_freq)
    if s == 1
        return (2J_f + 1)//(2J_i + 1)
    else
        return -1
    end
end


"""
    _light_shift_quotient(ω0::Float64, Γ::Float64, ωL::Float64)

Only supports linear polarized light for now.

Returns Γ / (ω0^2 * (ω0^2 - ωL^2)), which is proportional to the 
scalar light-shift contribution from a single electric-dipole transition.
"""
function _light_shift_quotient(ω0::Float64, Γ::Float64, ωL::Float64)
    return Γ / (ω0^2 * (ω0^2 - ωL^2))
end



# ======================================================================
# Public API
# ======================================================================

"""
    polarizability_si(model::PolarizabilityModel, λ_nm::Real) -> Float64

Dynamic electric scalar polarizability α in SI units (C·m²·V⁻¹) for the given
model and wavelength.

# Definition
Uses the relation

    U/I = -α_SI / (2 c ε₀)

which gives

    α_SI = - 2 c ε₀ (U/I)

where U/I is the light shift per intensity in J/(W/m²).

# Units
Returns scalar polarizability in C·m²·V⁻¹ (or equivalently F·m²), which is the
standard SI unit for electric polarizability.
"""
function polarizability_si(model::PolarizabilityModel, λ_nm::Real)
    ωL = 2π * c / (λ_nm * 1e-9)
    α0_SI = 0.0
    Ji = model.J_i
    for t in model.transitions
        ω0 = 2π * t.freq_THz * 1e12
        Γ = 2π * t.gamma_MHz * 1e6
        Jf = t.J_f

        deg_factor = _degeneracy_factor(Ji, Jf, ω0)

        α0_SI += _light_shift_quotient(ω0, Γ, ωL) * deg_factor
    end

    α0_SI *= 2π * ε0 * c^3
    α0_SI += -model.offset_Hz_per_Wm2 * h * 2 * c * ε0

    return α0_SI
end

"""
    polarizability_au(model::PolarizabilityModel, λ_nm::Real) -> Float64

Dynamic electric scalar polarizability α in atomic units (a₀³) for the given
model and wavelength.

# Definition
Converts from SI units via

    α_au = α_SI / (4π ε₀ a₀³)
"""
function polarizability_au(model::PolarizabilityModel, λ_nm::Real)
    α0_SI = polarizability_si(model, λ_nm)
    return α0_SI / (4π * ε0 * a0^3)
end

"""
    U_over_I_si(model::PolarizabilityModel, λ_nm::Real) -> Float64

Compute total scalar light shift per intensity U0/I for a given model and wavelength.

# Arguments
- `model::PolarizabilityModel`: Polarizability model for a single state.
- `λ_nm`: Laser wavelength in nanometres.

# Returns
- `U/I` in J/(W/m²).
"""
function U_over_I_si(model::PolarizabilityModel, λ_nm::Real)
    α0 = polarizability_si(model, λ_nm)
    U_I = - α0 / (2 * c * ε0)
    return U_I
end



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
    U = U_over_I_si(model, λ_nm)
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
    tensor_polarizability_si(model::PolarizabilityModel, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1) -> Float64

Dynamic electric tensor polarizability α⁽²⁾ in SI units (C·m²·V⁻¹) for the given
model and wavelength.

# Definition
Uses the relation

    U2/I (2 c ε₀) = -α2_SI ⋅ 

which gives

    α_SI = - 2 c ε₀ (U/I)

where U/I is the light shift per intensity in J/(W/m²).

# Units
Returns scalar polarizability in C·m²·V⁻¹ (or equivalently F·m²), which is the
standard SI unit for electric polarizability.
"""
function tensor_polarizability_si(model::PolarizabilityModel, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1)
    ωL = 2π * c / (λ_nm * 1e-9)
    α2_SI = 0.0
    Ji = model.J_i

    for t in model.transitions
        ω0 = 2π * t.freq_THz * 1e12
        Γ = 2π * t.gamma_MHz * 1e6
        Jf = t.J_f

        am_factors = (-1)^(-2*Ji - Jf - F - I ) * sqrt((40*F*(2*F + 1)*(2*F - 1))/(3(F + 1)*(2*F + 3))) * (2*Ji + 1)
        quotient = _light_shift_quotient(ω0, Γ, ωL, Ji, Jf)
        deg_factor = _degeneracy_factor(Ji, Jf, ω0)
        wigner_symbols = wigner6j(1, 1, 2, Ji, Ji, Jf) * wigner6j(Ji, Ji, 2, F, F, I)
        α2_SI += am_factors * quotient * wigner_symbols * deg_factor
    end

    α2_SI *= 3π * ε0 * c^3
    return α2_SI
end

function tensor_polarizability_au(model::PolarizabilityModel, λ_nm::Real, F::Rational = 0//1, I::Rational = 0//1)
    α2_SI = tensor_polarizability_si(model, λ_nm; F=F, I=I)
    return α2_SI / (4π * ε0 * a0^3)
end

"""
    tensor_U_over_I_si(model::PolarizabilityModel, λ_nm::Real) -> Float64

Compute total tensor light shift per intensity U0/I for a given model and wavelength.

# Arguments
- `model::PolarizabilityModel`: Polarizability model for a single state.
- `λ_nm`: Laser wavelength in nanometres.

# Returns
- `U/I` in J/(W/m²).
"""
function tensor_U_over_I_si(model::PolarizabilityModel, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1, mF::Rational = 0//1, e_z::Real = 1.0)
    α2 = tensor_polarisability_si(model, λ_nm; F=F, I=I)
    polar_factor = (3*e_z^2 - 1)/2
    num = (3*mF^2 - F*(F + 1))
    denom = (F*(2*F - 1))

    # denominator only vanishes for F = 0, 1/2s
    geometric_factor = 0.0
    if (denom != 0)
        geometric_factor = polar_factor * num/denom
    end

    U_I = - α2 * geometric_factor / (2 * c * ε0)
    return U_I
end


"""
    tensor_light_shift_coeff_Hz_per_Wcm2(model::PolarizabilityModel, λ_nm::Real) -> Float64

Tensor light-shift coefficient Δν/I in Hz/(W/cm²) for the given model and wavelength.

# Definition
For a beam intensity `I` in W/cm², the light shift is

    Δν = light_shift_coeff_Hz_per_Wcm2(model, λ_nm) * I

# Arguments
- `model`: Polarizability model for a single atomic state.
- `λ_nm`: Laser wavelength in nanometres.
"""
function tensor_light_shift_coeff_Hz_per_Wcm2(model::PolarizabilityModel, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1, mF::Rational = 0//1, e_z::Real = 1.0)
    U = tensor_U_over_I(model, λ_nm; F=F, I=I, mF=mF, e_z=e_z)
    ν_over_I = U / h              # Hz/(W/m²)

    return ν_over_I * 1e4         # Hz/(W/cm²)
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