using AtomTwin
using Plots
using ..Units: c, ε0, a0, h, hbar, e
using StatsBase
using PyFormattedStrings
using WignerSymbols

## Default plot with inset

# ## Need to redefine many structs

"""
    PolarizabilityModel_st

Empirical polarizability model for an atomic state, defined by a set of
discrete transitions and an optional offset.

# Fields
- `state::String`: Electronic state label(e.g. `"1S0"`, `"3P0"`). This is taken to be the initial state.
- `transitions::Vector{NamedTuple}`: List of transitions; each entry has
  - `freq_THz::Float64`: Transition frequency in THz (linear). If negative, then state_f < state in energy.
  - `gamma_MHz::Float64`: Effective line width in MHz (linear) — see below.
  - `state_f::String`: Electronic configuration of final state.
  - `J_f::Rational`: Total angular momentum of final state.
- `J_i`: Total angular momentum of intial state
- `offset_Hz_per_Wm2::Float64`: Empirical offset in Hz/(W/m²).
- `reference::String`: Bibliographic reference for the data.


Note that `freq_THz` is positive (negative) if the final state is above (below) the initial state in energy
"""


struct PolarizabilityModel_st
    state::String
    transitions::Vector{NamedTuple{(:freq_THz, :gamma_MHz, :state_f, :J_f),Tuple{Float64,Float64,String,Rational}}}
    J_i::Rational
    offset_Hz_per_Wm2::Float64
    reference::String
end

function PolarizabilityModel_st(state::String,
    transitions::Vector;
    J_i::Rational,
    offset_Hz_per_Wm2::Float64=0.0,
    reference::String="")
    PolarizabilityModel_st(state, transitions, J_i, offset_Hz_per_Wm2, reference)
end

const YB171_POLARIZABILITY_1S0_st = PolarizabilityModel_st(
    "1S0",
    [
        (freq_THz=539.386800, gamma_MHz=0.183, state_f="(6s6p) 3P1", J_f=1),     # (6s6p) 3P1
        (freq_THz=751.526389, gamma_MHz=29.127, state_f="(6s6p) 1P1", J_f=1),    # (6s6p) 1P1
        (freq_THz=865.111516, gamma_MHz=11.052, state_f="(7/2,5/2) J=1", J_f=1),    # (7/2,5/2) J=1
    ];
    J_i=0//1,
    offset_Hz_per_Wm2=-0.8e-4,
    reference="Phys. Rev. A 108, 053325 (2023)",
)

const YB171_POLARIZABILITY_3P0_st = PolarizabilityModel_st(
    "3P0",
    [
        (freq_THz=215.870446, gamma_MHz=0.308, state_f="(6s5d) 3D1", J_f=1),     # (6s5d) 3D1
        (freq_THz=461.867846, gamma_MHz=1.516, state_f="(6s7s) 3D1", J_f=1),     # (6s7s) 3S1
        (freq_THz=675.141040, gamma_MHz=4.081, state_f="(6s6d) 3D1", J_f=1),     # (6s6d) 3D1
        (freq_THz=729.293151, gamma_MHz=0.625, state_f="(6s8s) 3D1", J_f=1),     # (6s8s) 3S1
        (freq_THz=797.204099, gamma_MHz=23.567, state_f="Empirical J=1", J_f=1),    # Empirical J=1
    ];
    J_i=0//1,
    offset_Hz_per_Wm2=0.0,
    reference="T. O. Höhn, PhD thesis (2024)", #PhD thesis
)

const YB171_POLARIZABILITY_3P1_st = PolarizabilityModel_st(
    "3P1",
    [
        (freq_THz=-539.386800, gamma_MHz=0.183, state_f="(6s2) 1S0", J_f=0),    # (6s2) 1S0     #
        (freq_THz=194.778008, gamma_MHz=0.170, state_f="(6s5d) 3D1", J_f=1),     # (6s5d) 3D1
        (freq_THz=202.657933, gamma_MHz=0.280, state_f="(6s5d) 3D2", J_f=2),     # (6s5d) 3D2
        (freq_THz=440.775408, gamma_MHz=3.954, state_f="(6s7s) 3S1", J_f=1),     # (6s7s) 3S1   #
        (freq_THz=654.048602, gamma_MHz=2.783, state_f="(6s6d) 3D1", J_f=1),     # (6s6d) 3D1   #
        (freq_THz=654.927593, gamma_MHz=5.215, state_f="(6s6d) 3D2", J_f=2),     # (6s6d) 3D2   #
        (freq_THz=708.200713, gamma_MHz=1.718, state_f="(6s8s) 3S1", J_f=1),     # (6s8s) 3S1
        (freq_THz=778.975785, gamma_MHz=22.313, state_f="Empirical J=1", J_f=1),    # Empirical J=1
        (freq_THz=778.975785, gamma_MHz=36.687, state_f="Empirical J=2", J_f=2),    # Empirical J=2
    ];
    J_i=1//1,
    offset_Hz_per_Wm2=0.0,
    reference="T. O. Höhn, PhD thesis (2024)",
)


const models_st = Dict(
    "1S0" => YB171_POLARIZABILITY_1S0_st,
    "3P0" => YB171_POLARIZABILITY_3P0_st,
    "3P1" => YB171_POLARIZABILITY_3P1_st
)



"""
    _degeneracy_factor(J::Rational, J_prime::Rational, transition_freq::Real) -> Rational

Computes the sign of `transition_freq`, and returns a different factor based on the sign

"""
function _degeneracy_factor(J_i::Rational, J_f::Rational, transition_freq::Real)
    s = sign(transition_freq)
    if s == 1
        return (2J_f + 1)//(2J_i + 1)
    else
        return -(2J_i + 1)//(2J_f + 1)
    end
end



function _light_shift_quotient(ω0::Float64, Γ::Float64, ωL::Float64, J_i::Rational, J_f::Rational)
    deg_factor = _degeneracy_factor(J_i, J_f, ω0) 
    return Γ / (ω0^2 * (ω0^2 - ωL^2)) * deg_factor
end


"""
    polarizability_scalar_si(model::PolarizabilityModel, λ_nm::Real) -> Float64

Dynamic electric scalar polarizability α in SI units (C·m²·V⁻¹) for the given
model and wavelength.

Contains the empirical polarizability offset.

# Units
Returns polarizability in C·m²·V⁻¹ (or equivalently F·m²), which is the
standard SI unit for electric polarizability.
"""
function scalar_polarisability_si(model::PolarizabilityModel_st, λ_nm::Real)
    ωL = 2π * c / (λ_nm * 1e-9)
    α0 = 0.0
    Ji = model.J_i
    for t in model.transitions
        ω0 = 2π * t.freq_THz * 1e12
        Γ = 2π * t.gamma_MHz * 1e6
        Jf = t.J_f

        α0 += _light_shift_quotient(ω0, Γ, ωL, Ji, Jf)
    end

    α0 *= 2π * ε0 * c^3
    α0 += -model.offset_Hz_per_Wm2 * h * 2 * c * ε0
    return α0
end


"""
    scalar_U_over_I(model::PolarizabilityModel, λ_nm::Real) -> Float64

Compute total scalar light shift per intensity U0/I for a given model and wavelength.

# Arguments
- `model::PolarizabilityModel`: Polarizability model for a single state.
- `λ_nm`: Laser wavelength in nanometres.

# Returns
- `U/I` in J/(W/m²).
"""
function scalar_U_over_I(model::PolarizabilityModel_st, λ_nm::Real)
    α0 = scalar_polarisability_si(model, λ_nm)
    U_I = - α0 / (2 * c * ε0)
    return U_I
end


"""
    scalar_light_shift_coeff_Hz_per_Wcm2(model::PolarizabilityModel, λ_nm::Real) -> Float64

Scalar light-shift coefficient Δν/I in Hz/(W/cm²) for the given model and wavelength.

# Definition
For a beam intensity `I` in W/cm², the light shift is

    Δν = light_shift_coeff_Hz_per_Wcm2(model, λ_nm) * I

# Arguments
- `model`: Polarizability model for a single atomic state.
- `λ_nm`: Laser wavelength in nanometres.
"""
function scalar_light_shift_coeff_Hz_per_Wcm2(model::PolarizabilityModel_st, λ_nm::Real)
    U = scalar_U_over_I(model, λ_nm)
    ν_over_I = U / h              # Hz/(W/m²)
    return ν_over_I * 1e4         # Hz/(W/cm²)
end


"""
    polarizability_scalar_si(model::PolarizabilityModel, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1) -> Float64

Dynamic electric tensor polarizability α in SI units (C·m²·V⁻¹) for a given
model, wavelength and quantum numbers F, I.

Contains the empirical polarizability offset.

# Units
Returns polarizability in C·m²·V⁻¹ (or equivalently F·m²), which is the
standard SI unit for electric polarizability.
"""
function tensor_polarisability_si(model::PolarizabilityModel_st, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1)
    ωL = 2π * c / (λ_nm * 1e-9)
    α2 = 0.0
    Ji = model.J_i
    for t in model.transitions
        ω0 = 2π * t.freq_THz * 1e12
        Γ = 2π * t.gamma_MHz * 1e6
        Jf = t.J_f

        am_factors = (-1)^(-2*Ji - Jf - F - I ) * sqrt((40*F*(2*F + 1)*(2*F - 1))/(3(F + 1)*(2*F + 3))) * (2*Ji + 1)
        quotient = _light_shift_quotient(ω0, Γ, ωL, Ji, Jf)
        wigner_symbols = wigner6j(1, 1, 2, Ji, Ji, Jf) * wigner6j(Ji, Ji, 2, F, F, I)
        α2 += am_factors * quotient * wigner_symbols
    end

    α2 *= 3π * ε0 * c^3
    return α2
end

# For testing, returns each transition's contribution separately
function tensor_polarisability_si(model::PolarizabilityModel_st, λ_nm::Real, verbose::Bool; F::Rational = 0//1, I::Rational = 0//1)
    ωL = 2π * c / (λ_nm * 1e-9)
    α2 = 0.0
    Ji = model.J_i
    contribs = zeros(size(model.transitions))

    idx = 1
    for t in model.transitions
        ω0 = 2π * t.freq_THz * 1e12
        Γ = 2π * t.gamma_MHz * 1e6
        Jf = t.J_f

        am_factors = (-1)^(-2*Ji - Jf - F - I ) * sqrt((40*F*(2*F + 1)*(2*F - 1))/(3*(F + 1)*(2*F + 3))) * (2*Ji + 1)
        quotient = _light_shift_quotient(ω0, Γ, ωL, Ji, Jf)
        wigner_symbols = wigner6j(1, 1, 2, Ji, Ji, Jf) * wigner6j(Ji, Ji, 2, F, F, I)

        α2 += am_factors * quotient * wigner_symbols
        contribs[idx] = am_factors * quotient * wigner_symbols
        idx += 1
    end

    α2 *= 3π * ε0 * c^3
    contribs = contribs .* (3π * ε0 * c^3)
    return α2, contribs
end

"""
# For testing, accepts a single transition
function tensor_polarisability_si_single_transition(t::@NamedTuple{freq_THz::Float64, gamma_MHz::Float64, state_f::String, J_f::Rational}, Ji::Rational, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1)
    ωL = 2π * c / (λ_nm * 1e-9)
    α2 = 0.0

    ω0 = 2π * t.freq_THz * 1e12
    Γ = 2π * t.gamma_MHz * 1e6
    Jf = t.J_f

    am_factors = (-1)^(-2*Ji - Jf - F - I ) * sqrt((40*F*(2*F + 1)*(2*F - 1))/(3*(F + 1)*(2*F + 3))) * (2*Ji + 1)
    quotient = _light_shift_quotient(ω0, Γ, ωL, Ji, Jf)
    wigner_symbols = wigner6j(1, 1, 2, Ji, Ji, Jf) * wigner6j(Ji, Ji, 2, F, F, I)

    α2 += am_factors * quotient * wigner_symbols


    α2 *= 3π * ε0 * c^3
    return α2
end
"""


# For testing, accepts a single transition, sums over F' states and neglects hyperfine splitting
function tensor_polarisability_si_single_transition(t::@NamedTuple{freq_THz::Float64, gamma_MHz::Float64, state_f::String, J_f::Rational}, Ji::Rational, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1)
    ωL = 2π * c / (λ_nm * 1e-9)
    α2 = 0.0

    ω0 = 2π * t.freq_THz * 1e12
    Γ = 2π * t.gamma_MHz * 1e6
    Jf = t.J_f

    quotient = _light_shift_quotient(ω0, Γ, ωL, Ji, Jf)
    Ff_vals = abs(Jf-I):1//1:(Jf+I)
    for Ff in Ff_vals
        am_factors = (-1)^(F + Ff) * (2*Ff + 1) * sqrt((40*F*(2*F + 1)*(2*F - 1))/(3*(F + 1)*(2*F + 3))) * (2*Ji + 1)
        wigner_terms = wigner6j(1, 1, 2, F, F, Ff) * wigner6j(Ji, Jf, 1, Ff, F, I)^2

        α2 += am_factors * wigner_terms
    end
    α2 *= quotient
    α2 *= 3π * ε0 * c^3

    return α2
end


"""
    tensor_U_over_I(model::PolarizabilityModel, λ_nm::Real) -> Float64

Compute total tensor light shift per intensity U/I for a given 
model, wavelength, quantum numbers F, I, and polarisation along the quantization axis

Tensor polarizability vanishes for F = 0, 1/2

# Arguments
- `model::PolarizabilityModel`: Polarizability model for a single state.
- `λ_nm`: Laser wavelength in nanometres.

# Returns
- `U/I` in J/(W/m²).
"""
function tensor_U_over_I(model::PolarizabilityModel_st, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1, mF::Rational = 0//1, e_z::Real = 1.0)
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

# For testing, returns each transition's contribution separately
function tensor_U_over_I(model::PolarizabilityModel_st, λ_nm::Real, verbose::Bool; F::Rational = 0//1, I::Rational = 0//1, mF::Rational = 0//1, e_z::Real = 1.0)
    α2, contribs = tensor_polarisability_si(model, λ_nm; F=F, I=I, verbose=verbose)
    polar_factor = (3*e_z^2 - 1)/2
    num = (3*mF^2 - F*(F + 1))
    denom = (F*(2*F - 1))

    # denominator only vanishes for F = 0, 1/2s
    geometric_factor = 0.0
    if (denom != 0)
        geometric_factor = polar_factor * num/denom
    end

    U_I = - α2 * geometric_factor / (2 * c * ε0)
    U_contribs = contribs .* geometric_factor ./ (-2 * c * ε0)
    return U_I, U_contribs
end


# For testing, accepts a single transition
function tensor_U_over_I_single_transition(transition::@NamedTuple{freq_THz::Float64, gamma_MHz::Float64, state_f::String, J_f::Rational}, Ji::Rational, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1, mF::Rational = 0//1, e_z::Real = 1.0)
    α2 = tensor_polarisability_si_single_transition(transition, Ji, λ_nm; F=F, I=I)
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
function tensor_light_shift_coeff_Hz_per_Wcm2(model::PolarizabilityModel_st, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1, mF::Rational = 0//1, e_z::Real = 1.0)
    U = tensor_U_over_I(model, λ_nm; F=F, I=I, mF=mF, e_z=e_z)
    ν_over_I = U / h              # Hz/(W/m²)

    return ν_over_I * 1e4         # Hz/(W/cm²)
end

# For testing, returns each transition's contribution separately
function tensor_light_shift_coeff_Hz_per_Wcm2(model::PolarizabilityModel_st, λ_nm::Real, verbose::Bool; F::Rational = 0//1, I::Rational = 0//1, mF::Rational = 0//1, e_z::Real = 1.0)
    U, U_contribs = tensor_U_over_I(model, λ_nm; F=F, I=I, mF=mF, e_z=e_z, verbose=verbose)
    ν_over_I = U / h              # Hz/(W/m²)
    ν_contribs = U_contribs ./ h

    return ν_over_I * 1e4, ν_contribs .* 1e4          # Hz/(W/cm²)
end

# For testing, accepts a single transition
function tensor_light_shift_coeff_Hz_per_Wcm2_single_transition(transition::@NamedTuple{freq_THz::Float64, gamma_MHz::Float64, state_f::String, J_f::Rational}, ::Rational{Int64}, ::Float64, ::Rational{Int64}, Ji::Rational, λ_nm::Real; F::Rational = 0//1, I::Rational = 0//1, mF::Rational = 0//1, e_z::Real = 1.0)
    U = tensor_U_over_I_single_transition(transition, Ji, λ_nm; F=F, I=I, mF=mF, e_z=e_z)
    ν_over_I = U / h              # Hz/(W/m²)

    return ν_over_I * 1e4         # Hz/(W/cm²)
end



# ## Modified PolarizabilityCurve structs and plot recipes

struct PolarizabilityCurve_st
    models::Vector{PolarizabilityModel_st}
    λ_main::AbstractRange{<:Real}
    λ_inset::Union{Nothing,AbstractRange{<:Real}}
    ylim_main::Tuple{Real,Real}
    ylim_inset::Tuple{Real,Real}
    inset_position::Tuple{Real,Real,Real,Real}
    unit::Symbol
end

function PolarizabilityCurve_st(models::Vector{PolarizabilityModel_st};
    λ_main=420.0:0.1:800.0,
    λ_inset=550.5:0.1:556.2,
    ylim_main=(-30, 10),
    ylim_inset=(-8, 3.5),
    inset_position=(0.69, 0.02, 0.28, 0.30),
    unit::Symbol=:Hz_per_Wcm2)
    PolarizabilityCurve_st(models, λ_main, λ_inset, ylim_main, ylim_inset, inset_position, unit)
end

PolarizabilityCurve_st(model::PolarizabilityModel_st; kwargs...) =
    PolarizabilityCurve_st([model]; kwargs...)

@recipe function f(curve::PolarizabilityCurve_st)
    λs_main = collect(curve.λ_main)

    # Compute ylabel
    ylabel_txt = curve.unit == :au ? "Polarizability α (a.u.)" : "Δν / I  [Hz/(W/cm²)]"

    # Main plot settings
    xlabel --> "Wavelength λ (nm)"
    ylabel --> ylabel_txt
    legend --> :bottomright
    linewidth --> 2
    framestyle --> :box
    minorgrid --> true
    size --> (900, 500)
    left_margin --> 5Plots.mm
    bottom_margin --> 5Plots.mm
    right_margin --> 3Plots.mm
    top_margin --> 3Plots.mm

    # Plot main curves on subplot 1
    for model in curve.models
        if curve.unit == :Hz_per_Wcm2
            vals_main = [scalar_light_shift_coeff_Hz_per_Wcm2(model, λ) for λ in λs_main]
        end

        @series begin
            subplot --> 1
            ylims --> curve.ylim_main
            label --> model.state
            λs_main, vals_main
        end
    end

    # Inset zoom (if λ_inset is provided)
    if curve.λ_inset !== nothing
        λs_inset = collect(curve.λ_inset)

        inset_subplots --> (1, bbox(curve.inset_position...))

        for model in curve.models
            if curve.unit == :Hz_per_Wcm2
                vals_inset = [scalar_light_shift_coeff_Hz_per_Wcm2(model, λ) for λ in λs_inset]
            end

            @series begin
                subplot --> 2
                ylims --> curve.ylim_inset
                label --> ""
                guidefontsize --> 10
                legend --> false
                framestyle --> :box
                linewidth --> 2
                λs_inset, vals_inset
            end
        end
    end
end



