"""
    polarizability.jl

Generic infrastructure for computing and visualizing atomic polarizabilities
from empirical transition models.
"""

using RecipesBase
using ..Units: c, ε0, a0, h, hbar, e

# ======================================================================
# Data model
# ======================================================================

"""
    PolarizabilityModel

Empirical polarizability model for an atomic state, defined by a set of
discrete transitions and optional scalar and tensor offsets.

# Fields
- `state::String`: Electronic state label (e.g. `"1S0"`, `"3P0"`).
- `transitions::Vector{NamedTuple}`: List of transitions; each entry has
  - `freq_THz::Float64`: Transition frequency in THz (linear). **Signed**: positive
    if the final state lies ABOVE the model's state in energy, negative if below.
  - `gamma_MHz::Float64`: Effective line width in MHz (linear) — see below.
  - `J::Rational{Int}`: Total angular momentum of the model's own state.
  - `J_f::Rational{Int}`: Total angular momentum of the final state.
  - `source::Symbol`: Provenance — `:measured`, `:ls_estimated`, or `:fitted`.
- `J::Rational{Int}`: Total electronic angular momentum of the state itself.
- `offset_Hz_per_Wm2::Float64`: Empirical offset in Hz/(W/m²).
- `tensor_offset_Hz_per_Wm2::Float64`: Wavelength-independent `J`-level tensor
  light-shift coefficient in Hz/(W/m²), recoupled to each hyperfine `F` like a line
  (see the constructor).
- `reference::String`: Bibliographic reference for the data.

# Specifying transitions

Each transition weights the dynamic polarizability by its **line strength**. Two
equivalent ways to supply that weight are accepted by the constructor:

1. `(freq_THz = …, gamma_MHz = …)` — the transition's effective line
   width. This equals the natural linewidth **only for a `J=0 → J'=1` line**
   (e.g. every Yb `¹S₀` line), where the upper level decays to a single ground
   level. For such lines just use the natural linewidth.

2. `(freq_THz = …, dipole_ea0 = …)` — the reduced dipole matrix element
   `|⟨Jg‖er‖Je⟩|` in units of `e·a₀` (Steck's D-line convention). This is the robust
   choice for a **multiplet** such as an alkali D₁/D₂ doublet, where the two lines
   share a ground state but have unequal line strengths (D₂ carries twice the
   strength of D₁). 

   The constructor converts the dipole to the effective width, taking into account 
   different normalization conventions for the reduced dipole matrix element:

    1. Wigner-3j, dipole_convention = "wigner3j" (default)

    In this convention, |(Jg‖d‖Je)|² is symmetric in Jg ↔ Je and equals the 
    line strength S directly: S = |(Jg‖d‖Je)|².

    2. Clebsch-Gordan, dipole_convention = "clebschgordan"

    In this convention, |⟨Jg‖d‖Je⟩|² is NOT symmetric in Jg ↔ Je and is related 
    to the line strength by: S = (2Jg + 1) |⟨Jg‖d‖Je⟩|²

    These boil down to two conventions for the Wigner-Ekhart theorem, where 
    the (2Jg+1) is related to the numerical factor required to convert from 
    a Wigner-3j symbol to the equivalent Clebsch-Gordan coefficient. 
    The transition line width is computed from the Wigner-3j convention 
    reduced dipole matrix element |(Jg‖d‖Je)| as:

       Γ_eff = (1/3π) ·  ω₀³ |(J‖d‖J′)|² / ((2Je+1) ε₀ c³ ħ)         [rad/s]

   so that AtomTwin's two-level light-shift sum reproduces the multi-level scalar
   polarizability α(ω) = (1/(2Jg+1)) Σᵢ (2/3) |dᵢ|² ω₀ᵢ / (ħ(ω₀ᵢ²−ω²)) exactly.

Do **not** feed natural linewidths for an alkali doublet: the near-equal D₁/D₂
natural widths under-weight D₂ and mis-split the light shift off-resonance, while
still agreeing at the static limit (a silent error). Use `dipole_ea0` instead.

# Provenance

`source` records where a width came from, so a refit can tell free parameters from
spectroscopy: `:measured` (experimental), `:ls_estimated` (deduced from a lifetime
via an LS-coupling branching ratio — the asterisked entries in the Yb literature),
`:fitted` (a free parameter of an empirical model). Defaults to `:measured`.
"""
struct PolarizabilityModel
    state::String
    transitions::Vector{NamedTuple{(:freq_THz, :gamma_MHz, :J, :J_f, :source),
                                   Tuple{Float64, Float64, Rational{Int}, Rational{Int},
                                            Symbol}}}
    J::Rational{Int}
    offset_Hz_per_Wm2::Float64
    tensor_offset_Hz_per_Wm2::Float64
    reference::String
end

"""
    _dipole_to_gamma_MHz(freq_THz, dipole_ea0, Je) -> Float64

Effective line-strength width (MHz, linear) for a transition specified by its
reduced dipole matrix element `dipole_ea0` = `|(Jg‖er‖Je)|` in `e·a₀`, such that
the two-level light-shift form reproduces the multi-level scalar polarizability.

    Γ_eff = (1/3π) ·  ω₀³ |(J‖d‖J′)|² / ((2Je+1) ε₀ c³ ħ)         [rad/s]

`dipole_ea0` is assumed to be in the Wigner-3j convention (see @ref PolarizabilityModel)   
`Je` is the excited-state total angular momentum.
"""
function _dipole_to_gamma_MHz(freq_THz::Real, dipole_ea0::Real, Je::Real)
    ω0 = 2π * abs(freq_THz) * 1e12  # gamma is a positive quantity
    d  = dipole_ea0 * e * a0                       # C·m
    Γ_eff = (1 / (3π)) * d^2 * ω0^3 / ((2Je + 1) * ε0 * c^3 * hbar)   # rad/s
    return Γ_eff / (2π * 1e6)                       # → MHz (linear)
end

# Normalise a single user transition entry to the internal
# (freq_THz, gamma_MHz, J, J_f, source) form.
#
#   * `dipole_ea0` is converted by `_dipole_to_gamma_MHz` according 
#   to the convention the dipole value is given in (see @ref PolarizabilityModel)
#
# Hence declaring `J` on a dipole-specified model is safe: it documents the state
# and feeds the tensor part, without disturbing the scalar sum.
function _normalize_transition(t, J_model::Rational{Int})
    freq  = haskey(t, :freq_THz) ? t.freq_THz : 1.0   # only the sign is used
    J     = haskey(t, :J)   ? Rational{Int}(t.J)   : J_model
    J_f   = haskey(t, :J_f) ? Rational{Int}(t.J_f) : 1//1
    src   = haskey(t, :source) ? Symbol(t.source) : :measured
    src in (:measured, :ls_estimated, :fitted) || error(
        "PolarizabilityModel transition `source` must be :measured, :ls_estimated " *
        "or :fitted; got $(repr(src))")

    if haskey(t, :gamma_MHz)
        γ = Float64(t.gamma_MHz)
    elseif haskey(t, :dipole_ea0)
        Je, Jg = freq > 0 ? (J_f, J) : (J, J_f)
        convention = haskey(t, :dipole_convention) ? Symbol(t.dipole_convention) : :wigner3j
        convention in (:wigner3j, :clebschgordan) || error(
            "PolarizabilityModel transition `dipole_convention` must be :wigner3j, " *
            "or :clebschgordan; got $(repr(convention))")

        if convention == :wigner3j
            γ  = _dipole_to_gamma_MHz(t.freq_THz, t.dipole_ea0, Je)
        elseif convention == :clebschgordan
            γ  = _dipole_to_gamma_MHz(t.freq_THz, t.dipole_ea0*sqrt(2Jg+1), Je)
        end
    else
        error("PolarizabilityModel transition must have either `gamma_MHz` or " *
              "`dipole_ea0`; got keys $(keys(t))")
    end
    return (freq_THz = Float64(t.freq_THz), gamma_MHz = γ,
            J = J, J_f = J_f, source = src)
end

"""
    PolarizabilityModel(state, transitions; J = 0, offset_Hz_per_Wm2 = 0.0,
                        tensor_offset_Hz_per_Wm2 = 0.0, reference = "")

Build a model for `state` from a list of `transitions`. `J` is the total electronic
angular momentum of `state` itself; it is the default for each line's own `J`, and
leaving it at `0` reproduces the pre-existing `J=0 → J'=1` behaviour exactly.

The two offsets add wavelength-independent scalar and tensor light-shift
coefficients (Hz per W/m², the sign of the light shift: negative attracts). The
tensor one is the `J`-level coefficient `U₂/I` — the one that multiplies
`(3m_J² − J(J+1))/(J(2J−1)) · (3ε_z² − 1)/2` for a nuclear-spin-free state — and is
carried to each hyperfine `F` by the same recoupling as a line. Together they
represent a MEASURED light shift at one wavelength, e.g. with no line list:

```julia
# Yb ¹P₁ in a π-polarised 532 nm trap (Muzi Falconi et al. 2025): m_J = ±1 magic,
# m_J = 0 at −11.6 MHz/mK relative to ¹S₀, whose light shift is u = U/(hI) in Hz
# per W/m².
PolarizabilityModel("1P1", NamedTuple[]; J = 1,
                    offset_Hz_per_Wm2 = 0.814u, tensor_offset_Hz_per_Wm2 = 0.186u)
```

A tensor offset needs `J ≥ 1`; for `J ≤ 1/2` it is an `ArgumentError`.
"""
function PolarizabilityModel(state::String,
                             transitions::Vector;
                             J = 0//1,
                             offset_Hz_per_Wm2::Float64 = 0.0,
                             tensor_offset_Hz_per_Wm2::Float64 = 0.0,
                             reference::String = "")
    Jm   = Rational{Int}(J)
    (tensor_offset_Hz_per_Wm2 == 0.0 || Jm ≥ 1) || throw(ArgumentError(
        "a tensor offset needs J ≥ 1; a state with J = $Jm has no tensor light shift"))
    norm = [_normalize_transition(t, Jm) for t in transitions]
    for t in norm
        t.J == Jm || error(
            "transition declares J = $(t.J) but the model's state has J = $Jm; " *
            "every line of a model shares the model's initial state.")
    end
    PolarizabilityModel(state, norm, Jm, offset_Hz_per_Wm2, tensor_offset_Hz_per_Wm2,
                        reference)
end

# ======================================================================
# Core physics
# ======================================================================

"""
    _line_strength_factor(J, J_f, freq_THz) -> Rational{Int}

Angular-momentum weight `f(J,J')` of one line in the scalar polarizability sum.

Writing the scalar polarizability in terms of decay rates rather than reduced
dipole matrix elements gives

    α⁽⁰⁾(ω) = 2π ε₀ c³ Σ_J'  f(J,J') · Γ(J,J') / (ω₀² (ω₀² − ω²))

where `Γ(J,J')` is always the rate of the *downward* (excited → ground) decay, and

    f(J,J') = (2J'+1)/(2J+1)   if the final state is ABOVE  (freq_THz > 0)
            = −1               if the final state is BELOW  (freq_THz < 0)

The asymmetry is real, not a convention: the reduced matrix element is not
symmetric under label exchange,
`|⟨J‖d‖J'⟩|² = ((2J'+1)/(2J+1))·|⟨J'‖d‖J⟩|²`, and `Γ` is only defined downward.
The `−1` is why the two states of a two-level atom take opposite light shifts.

For the `J=0 → J'=1` lines that every ¹S₀-type model is built from, `f = 3`

Reference: Höhn-model derivation in the AtomTwin polarizability notes; Steck,
*Quantum and Atom Optics*, §7.3.4 (reduced-matrix-element conventions).
"""
function _line_strength_factor(J::Rational{Int}, J_f::Rational{Int}, freq_THz::Real)
    return freq_THz ≥ 0 ? (2J_f + 1) // (2J + 1) : -1 // 1
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
        ω0 = 2π * abs(t.freq_THz) * 1e12   # |ω₀|: a line below the state still
        Γ  = 2π * t.gamma_MHz     * 1e6    # scatters at its own resonance
        Γsc_over_I += _calc_scattering_rate(ω0, Γ, ωL)
    end
    return Γsc_over_I
end

"""
    _calc_light_shift(ω0, Γ, ωL, f = 3) -> Float64

DEPRECATED

Light-shift contribution from a single electric-dipole transition.

# Arguments
- `ω0`: Transition angular frequency [rad/s]. Use `|ω₀|`; the sign of the
  transition frequency enters through `f` (see `_line_strength_factor`).
- `Γ` : Radiative linewidth (angular) [rad/s].
- `ωL`: Laser angular frequency [rad/s].
- `f` : Angular-momentum line-strength factor; `3` for a `J=0 → J'=1` line.

# Returns
- `U/I`: Energy shift per intensity in J/(W/m²).
"""
function _calc_light_shift(ω0::Float64, Γ::Float64, ωL::Float64, f::Real = 3)
    return -π * c^2 * f * Γ / (ω0^2 * (ω0^2 - ωL^2))
end

"""
    _U_over_I(model::PolarizabilityModel, λ_nm::Real) -> Float64

DEPRECATED

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
        ω0 = 2π * abs(t.freq_THz) * 1e12     # |ω₀|; the sign is carried by f
        Γ  = 2π * t.gamma_MHz     * 1e6
        U_over_I += _calc_light_shift(ω0, Γ, ωL, t.f)
    end

    U_over_I += model.offset_Hz_per_Wm2 * h
    return U_over_I
end

# ======================================================================
# Scalar polarizability
# ======================================================================

"""
    _alpha0_si(model, λ_nm) -> Float64

Dynamic scalar polarizability α⁽⁰⁾(ω) in SI units (C·m²·V⁻¹).

    α⁽⁰⁾(ω) = 2π ε₀ c³ Σ_J' f(J,J') Γ(J,J') / [ω₀² (ω₀² − ω²)]

where the sum runs over all dipole-allowed transitions from the state `J` to
final states `J'`, and the line strength factor `f(J,J')` is the physically
relevant angular-momentum factor for the transition. The offset term is added
as a constant background contribution in the same units.

# Arguments
- `model::PolarizabilityModel`: Polarizability model for a single state.
- `λ_nm`: Laser wavelength in nanometres.

# Returns
- Scalar polarizability in SI units.
"""
function _alpha0_si(model, λ_nm)
    ωL = 2π * c / (λ_nm * 1e-9)
    α0 = 0.0
    Ji = model.J
    for t in model.transitions
        ω0 = 2π * t.freq_THz * 1e12
        Γ = 2π * t.gamma_MHz * 1e6
        Jf = t.J_f

        f_phys = _line_strength_factor(Ji, Jf, t.freq_THz)

        α0 += f_phys * Γ / (ω0^2 * (ω0^2 - ωL^2))
    end

    α0 *= 2π * ε0 * c^3
    α0 += -model.offset_Hz_per_Wm2 * h * 2 * c * ε0

    return α0
end

# ======================================================================
# Tensor polarizability
# ======================================================================
#
# The light shift of a hyperfine state |F, m_F> decomposes into scalar, vector and
# tensor parts. For LINEARLY polarised light the vector term vanishes, leaving
#
#   U/I = -(1/2 eps0 c) [ alpha0 + alpha2 * ((3|e_z|^2-1)/2) * geom(F, mF) ]
#
# with geom(F,mF) = (3 mF^2 - F(F+1)) / (F(2F-1)), and e_z the projection of the
# polarisation onto the quantisation axis. alpha2 vanishes identically for J = 0
# and J = 1/2 (the 6-j triangle rule) and geom is undefined for F = 0, 1/2 -- two
# independent reasons the tensor shift is absent for those states.
#
# Reference: the AtomTwin polarizability notes (Schmit-Veiler 2026), eqs. (2)-(3);
# Steck, Quantum and Atom Optics, sec. 7.3.

"""
    _tensor_prefactor(F) -> Float64

The `sqrt(40 F (2F+1) (2F-1) / (3 (F+1) (2F+3)))` factor of the tensor
polarizability. Zero for `F = 0` and `F = 1/2`, where the tensor shift is absent.
"""
function _tensor_prefactor(F::Rational{Int})
    (F == 0 || F == 1//2) && return 0.0
    num = 40 * F * (2F + 1) * (2F - 1)
    den = 3 * (F + 1) * (2F + 3)
    return sqrt(Float64(num / den))
end

"""
    _alpha2_si(model, λ_nm; F, I) -> Float64

Dynamic **tensor** polarizability `α⁽²⁾(F; ω)` in SI units (C·m²·V⁻¹).

    α⁽²⁾ = 3π ε₀ c³ Σ_J' (−1)^(−2J−J'−F−I) √(40F(2F+1)(2F−1)/(3(F+1)(2F+3))) (2J+1)
                        × f(J,J') Γ(J,J') / (ω₀² (ω₀² − ω²))
                        × {1 1 2; J J J'} {J J 2; F F I}

`F` is the hyperfine quantum number of the state and `I` the nuclear spin; for a
zero-spin isotope `I = 0` and `F = J`.

Returns `0.0` whenever the tensor part is absent — `J ≤ 1/2` (the `{1 1 2; J J J'}`
triangle rule) or `F ≤ 1/2` (the prefactor). The sum runs over the model's lines
with the same `f(J,J')` energy-ordering convention as the scalar part.

A model's `tensor_offset_Hz_per_Wm2` adds its `J`-level coefficient as a
polarizability, `α = −cε₀h·offset`, carried to `F` by the lines' own `F` dependence:
`(−1)^(J−F−I) pre(F){J J 2; F F I} / (pre(J){J J 2; J J 0})`, with `pre` the
square-root prefactor above.

# Normalisation — read before comparing with a paper

`α⁽²⁾` is defined only up to how the `(3m_F²−F(F+1))/(F(2F−1))` sublevel factor is
split off, and the literature is not uniform. AtomTwin's matches Kestler *et al.*,
*Phys. Rev. A* **105**, 012821 (2022): for linear polarisation along the
quantisation axis,

    α(m_F = 0)   = α⁽⁰⁾ − 2 α⁽²⁾
    α(|m_F| = 1) = α⁽⁰⁾ +   α⁽²⁾

which is what `_tensor_geometry` × `_polarization_factor` reproduces, and which
`SR88_POLARIZABILITY_3P1` is validated against.

!!! warning "Do not import α⁽²⁾ from a paper without checking its convention"
    α⁽²⁾ as a *number* is convention dependent: it and the geometric factor can be
    rescaled reciprocally. The AtomTwin polarizability notes write this sum with a
    `3π ε₀ c³` prefactor and an explicit `(2J+1)`, which makes their α⁽²⁾ exactly
    `(2J+1)×` the one here — they normalise the reduced matrix elements the other
    way (Steck §7.3.4 covers the two conventions). Our scalar already matches
    Kestler's table, which pins the mapping and leaves α⁽²⁾ no freedom.

    Two things are convention **free**, and both are what to test against:

    - the measured magic wavelengths, which are physical zero crossings;
    - the sublevel splitting `α(m_F=0) − α(|m_F|=1) = −3α⁽²⁾`, an absolute energy.

    Note the splitting identity holds under *any* rescaling of α⁽²⁾, so it checks
    the geometry but **cannot** catch a wrong normalisation. Only an absolute
    comparison does — hence the α₀ and α₂ anchors in `test/unit/test_physics.jl`.

"""
function _alpha2_si(model::PolarizabilityModel, λ_nm::Real;
                    F::Rational{Int}, I::Rational{Int} = 0//1)
    J = model.J
    (J == 0 || J == 1//2) && return 0.0      # tensor rank unreachable
    pre = _tensor_prefactor(F)
    pre == 0.0 && return 0.0

    ωL = 2π * c / (λ_nm * 1e-9)
    α2 = 0.0
    for t in model.transitions
        J_f = t.J_f
        w1 = wigner6j(1, 1, 2, J, J, J_f)
        w2 = wigner6j(J, J, 2, F, F, I)
        (w1 == 0 || w2 == 0) && continue

        ω0 = 2π * abs(t.freq_THz) * 1e12
        Γ  = 2π * t.gamma_MHz     * 1e6
        # (−1)^(−2J−J'−F−I): the exponent is an integer whenever the 6-j symbols
        # are non-zero, so round before exponentiating rather than going complex.
        phase = iseven(round(Int, -2J - J_f - F - I)) ? 1.0 : -1.0

        # physical line strength factor f(J, J')

        f_phys = _line_strength_factor(J, J_f, t.freq_THz)

        α2 += phase * pre * (2J + 1) * f_phys * Γ / (ω0^2 * (ω0^2 - ωL^2)) * w1 * w2
    end
    
    α2_si = 3π * ε0 * c^3 * α2
    # A measured J-level tensor coefficient, recoupled to F exactly as each line
    # term above: the F dependence of a line is pre(F)·{J J 2; F F I}·(−1)^(−F−I),
    # so relative to the nuclear-spin-free F = J it is the ratio of those.
    if model.tensor_offset_Hz_per_Wm2 != 0.0
        α2J = -2c * ε0 * h * model.tensor_offset_Hz_per_Wm2      # α = −2cε₀ U/I
        wF  = wigner6j(J, J, 2, F, F, I)
        wJ  = wigner6j(J, J, 2, J, J, 0)
        sgn = iseven(round(Int, J - F - I)) ? 1.0 : -1.0
        α2_si += α2J * sgn * pre / _tensor_prefactor(J) * wF / wJ
    end
    return α2_si
end

"""
    _tensor_geometry(F, mF) -> Float64

The `(3mF² − F(F+1)) / (F(2F−1))` sublevel factor of the tensor light shift.
Zero for `F = 0, 1/2`, where the denominator vanishes and there is no tensor shift.

Summed over a complete manifold this is zero: the tensor shift splits sublevels
without moving the manifold's centre of gravity.
"""
function _tensor_geometry(F::Rational{Int}, mF::Rational{Int})
    den = F * (2F - 1)
    den == 0 && return 0.0
    return Float64((3 * mF^2 - F * (F + 1)) / den)
end

"""
    _polarization_factor(ε_z) -> Float64

The `(3|ε_z|² − 1)/2` factor, with `ε_z` the projection of the (linear)
polarisation unit vector onto the quantisation axis. It is `1` for polarisation
along the axis, `−1/2` for polarisation perpendicular to it, and vanishes at the angle `acos(1/√3) ≈ 54.7356°`.
"""
_polarization_factor(ε_z::Real) = (3 * abs2(ε_z) - 1) / 2

# ======================================================================
# Public API
# ======================================================================

"""
    polarizability_si(model::PolarizabilityModel, λ_nm::Real) -> Float64

Dynamic electric polarizability α in SI units (C·m²·V⁻¹) for the given
model and wavelength. This polarizability relates to the physical shift in 
energy U per unit intensity I through the relation:

    U/I = -α_SI / ( 2 c ε₀)

# Arguments
- `model`: Polarizability model for a single atomic state.
- `λ_nm`: Laser wavelength in nanometres.

# Keyword arguments (tensor light shift)
Supplying `F` adds the **tensor** contribution for the sublevel `|F, mF⟩`:

- `F`: hyperfine quantum number. Omit (the default) for the scalar shift alone.
- `mF`: magnetic sublevel, `-F ≤ mF ≤ F`.
- `I`: nuclear spin; `0` for a zero-spin isotope, where `F = J`.
- `ε_z`: projection of the (linear) polarisation unit vector on the quantisation
  axis, i.e. `cos θ`. `1` means polarisation along the axis.

The tensor term vanishes identically for `J ≤ 1/2` or `F ≤ 1/2`, so passing `F`
for a `¹S₀` or `³P₀` state is harmless and returns the scalar result.

Valid only for linear polarisation: the vector (rank-1) term, which would enter
for elliptical light, is not included.

# Units
Returns polarizability in C·m²·V⁻¹ (or equivalently F·m²), which is the
standard SI unit for electric polarizability.
"""
function polarizability_si(model::PolarizabilityModel, λ_nm::Real;
                                       F  = nothing,
                                       mF = 0//1,
                                       I  = 0//1,
                                       ε_z::Real = 1.0)
    α = _alpha0_si(model, λ_nm)
    if F !== nothing
        Fr, mFr, Ir = Rational{Int}(F), Rational{Int}(mF), Rational{Int}(I)
        abs(mFr) <= Fr || throw(ArgumentError("need |mF| ≤ F; got mF = $mFr, F = $Fr"))
        α2 = _alpha2_si(model, λ_nm; F = Fr, I = Ir)
        if α2 != 0.0
            α += α2 * _polarization_factor(ε_z) * _tensor_geometry(Fr, mFr)
        end
    end
    return α
end

"""
    polarizability_au(model::PolarizabilityModel, λ_nm::Real) -> Float64

Dynamic electric polarizability α in atomic units (a₀³) for the given
model and wavelength.

# Definition
Converts from SI units via

    α_au = α_SI / (4π ε₀ a₀³)
"""
function polarizability_au(model::PolarizabilityModel, λ_nm::Real;
                                       F  = nothing,
                                       mF = 0//1,
                                       I  = 0//1,
                                       ε_z::Real = 1.0)
    α_SI = polarizability_si(model, λ_nm; F=F, mF=mF, I=I, ε_z=ε_z)
    return α_SI / (4π * ε0 * a0^3)
end

"""
    light_shift_coeff_Hz_per_Wcm2(model, λ_nm; F = nothing, mF = 0, I = 0, ε_z = 1) -> Float64

Light-shift coefficient Δν/I in Hz/(W/cm²) for the given model and wavelength.

# Definition
For a beam intensity `I` in W/cm², the light shift is

    Δν = light_shift_coeff_Hz_per_Wcm2(model, λ_nm) * I

Uses the relation

    U/I = -α_SI / ( 2 c ε₀)

# Arguments
- `model`: Polarizability model for a single atomic state.
- `λ_nm`: Laser wavelength in nanometres.

# Keyword arguments (tensor light shift)
Supplying `F` adds the **tensor** contribution for the sublevel `|F, mF⟩`:

- `F`: hyperfine quantum number. Omit (the default) for the scalar shift alone.
- `mF`: magnetic sublevel, `-F ≤ mF ≤ F`.
- `I`: nuclear spin; `0` for a zero-spin isotope, where `F = J`.
- `ε_z`: projection of the (linear) polarisation unit vector on the quantisation
  axis, i.e. `cos θ`. `1` means polarisation along the axis.

The tensor term vanishes identically for `J ≤ 1/2` or `F ≤ 1/2`, so passing `F`
for a `¹S₀` or `³P₀` state is harmless and returns the scalar result.

Valid only for linear polarisation: the vector (rank-1) term, which would enter
for elliptical light, is not included.
"""
function light_shift_coeff_Hz_per_Wcm2(model::PolarizabilityModel, λ_nm::Real;
                                       F  = nothing,
                                       mF = 0//1,
                                       I  = 0//1,
                                       ε_z::Real = 1.0)
    α = polarizability_si(model, λ_nm; F=F, mF=mF, I=I, ε_z=ε_z)
    # One convention across the whole file: `polarizability_si` defines
    # α_SI = − 2 c ε₀ (U/I), so every α converts with 1/(2 c ε₀).
    U = -α / (2 * c * ε0)
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
# ======================================================================
# Atom-facing forwarders
# ======================================================================
#
# One generic method per public function, dispatching through
# `getpolarizabilitymodels`. These replaced a ~45-line block copy-pasted into
# every species file, which had also drifted: `polarizability_si` — the one
# function `_init_species_data!` actually calls — was missing from all of them.

"""
    _model_for(atom, term) -> PolarizabilityModel

The species' polarizability model for `term`, or an error naming what is
available. `term` may be a [`TermSymbol`](@ref) or its name.
"""
function _model_for(atom::AbstractAtom, term)
    models = _polarizability_models(atom)
    key = termname(term)
    haskey(models, key) && return models[key]
    isempty(models) && error("$(getspecies(atom)) has no polarizability models.")
    error("no polarizability model for '$key' in $(getspecies(atom)); " *
          "known terms: $(join(sort(collect(keys(models))), ", "))")
end

for f in (:light_shift_coeff_Hz_per_Wcm2, :scattering_rate_per_Wcm2,
          :polarizability_au, :polarizability_si)
    @eval begin
        """
            $($f)(atom, term, λ_nm; kwargs...)

        As the model-level [`$($f)`](@ref), for the state `term` of `atom`'s
        species. `term` may be a [`TermSymbol`](@ref) (`l"3P1"`) or its name
        (`"3P1"`).
        """
        $f(atom::AbstractAtom, term, λ_nm::Real; kwargs...) =
            $f(_model_for(atom, term), λ_nm; kwargs...)
    end
end

"""
    print_polarizability_model(model::PolarizabilityModel; io::IO = stdout)

Print a single `PolarizabilityModel` in a structured, human-readable format.
"""
function print_polarizability_model(model::PolarizabilityModel; io::IO = stdout)
    println(io, "PolarizabilityModel")
    println(io, "  state: ", model.state)
    println(io, "  J: ", model.J)
    println(io, "  offset_Hz_per_Wm2: ", model.offset_Hz_per_Wm2)
    println(io, "  reference: ", isempty(model.reference) ? "(none)" : model.reference)
    println(io, "  transitions:")

    if isempty(model.transitions)
        println(io, "    (none)")
        return nothing
    end

    for (idx, t) in enumerate(model.transitions)
        print(io, "    [", idx, "]: ")
        print(io, "freq_THz: ", t.freq_THz, ", ")
        print(io, "gamma_MHz: ", t.gamma_MHz, "MHz, ")
        print(io, "J: ", t.J, ", ")
        print(io, "J_f: ", t.J_f, ", ")
        print(io, "source: ", t.source)
        println()
    end
    return nothing
end

function Base.show(io::IO, model::PolarizabilityModel)
    print_polarizability_model(model; io = io)
end