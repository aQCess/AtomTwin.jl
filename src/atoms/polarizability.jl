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
discrete transitions and an optional offset.

# Fields
- `state::String`: Electronic state label (e.g. `"1S0"`, `"3P0"`).
- `transitions::Vector{NamedTuple}`: List of transitions; each entry has
  - `freq_THz::Float64`: Transition frequency in THz (linear). **Signed**: positive
    if the final state lies ABOVE the model's state in energy, negative if below.
  - `gamma_MHz::Float64`: Effective line width in MHz (linear) — see below.
  - `J::Rational{Int}`: Total angular momentum of the model's own state.
  - `J_f::Rational{Int}`: Total angular momentum of the final state.
  - `f::Rational{Int}`: The line's weight in the scalar sum (see below).
  - `source::Symbol`: Provenance — `:measured`, `:ls_estimated`, or `:fitted`.
- `J::Rational{Int}`: Total electronic angular momentum of the state itself.
- `offset_Hz_per_Wm2::Float64`: Empirical offset in Hz/(W/m²).
- `reference::String`: Bibliographic reference for the data.

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

# Angular momentum and the line-strength factor

Each line enters the scalar sum with a weight `f`. How that weight is obtained
depends on which of the two specifications above was used, and the distinction
matters:

- A **`gamma_MHz`** line is a physical decay rate `Γ(J,J')`, so its angular-momentum
  weight is still outstanding and is computed as `f(J,J')` (see
  [`_line_strength_factor`](@ref)). With the defaults `J = 0`, `J_f = 1` this gives
  `f = 3`, which is what the scalar sum assumed implicitly before these fields
  existed — hence every pre-existing `¹S₀`/`³P₀` model is unchanged. A state with
  `J > 0`, notably `³P₁`, **must** declare `J` and `J_f`: a line *below* it in
  energy (negative `freq_THz`) carries `f = −1`, the opposite sign, which is why
  the two states of a two-level atom take opposite light shifts.

- A **`dipole_ea0`** line already has its `1/(2Jg+1)` normalisation folded into
  `Γ_eff` by the conversion above, precisely so the sum closes with a fixed `f = 3`.
  Applying `f(J,J')` again would double-count the angular momentum. Such lines
  therefore keep `f = 3`, and declaring `J` on the model is safe — it documents the
  state and feeds the tensor part without disturbing the scalar sum. A `dipole_ea0`
  line must lie above the state; use `gamma_MHz` for one below.

# Provenance

`source` records where a width came from, so a refit can tell free parameters from
spectroscopy: `:measured` (experimental), `:ls_estimated` (deduced from a lifetime
via an LS-coupling branching ratio — the asterisked entries in the Yb literature),
`:fitted` (a free parameter of an empirical model). Defaults to `:measured`.
"""
struct PolarizabilityModel
    state::String
    transitions::Vector{NamedTuple{(:freq_THz, :gamma_MHz, :J, :J_f, :f, :source),
                                   Tuple{Float64, Float64, Rational{Int}, Rational{Int},
                                         Rational{Int}, Symbol}}}
    J::Rational{Int}
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

# Normalise a single user transition entry to the internal
# (freq_THz, gamma_MHz, J, J_f, f, source) form.
#
# The stored `f` is the line's weight in the scalar sum, and WHICH weight is right
# depends on how the line was specified:
#
#   * `gamma_MHz` is a physical decay rate Γ(J,J'), so the angular-momentum factor
#     is still outstanding: f = _line_strength_factor(J, J_f, freq).
#
#   * `dipole_ea0` is converted by `_dipole_to_gamma_MHz`, which ALREADY folds the
#     1/(2Jg+1) line-strength normalisation into Γ_eff precisely so the sum closes
#     with the fixed f = 3. Applying f(J,J') on top would double-count the angular
#     momentum — for Rb that is a factor 1/3 on D₁ and 2/3 on D₂. So dipole lines
#     keep f = 3 and carry their angular momentum inside Γ_eff.
#
# Hence declaring `J` on a dipole-specified model is safe: it documents the state
# and feeds the tensor part, without disturbing the scalar sum.
function _normalize_transition(t, J_model::Rational{Int})
    J   = haskey(t, :J)   ? Rational{Int}(t.J)   : J_model
    J_f = haskey(t, :J_f) ? Rational{Int}(t.J_f) : 1//1
    src = haskey(t, :source) ? Symbol(t.source) : :measured
    src in (:measured, :ls_estimated, :fitted) || error(
        "PolarizabilityModel transition `source` must be :measured, :ls_estimated " *
        "or :fitted; got $(repr(src))")

    if haskey(t, :gamma_MHz)
        γ = Float64(t.gamma_MHz)
        f = _line_strength_factor(J, J_f, t.freq_THz)
    elseif haskey(t, :dipole_ea0)
        Jg = haskey(t, :Jg) ? t.Jg : 1//2   # historical default (alkali D lines)
        γ  = _dipole_to_gamma_MHz(t.freq_THz, t.dipole_ea0, Jg)
        f  = 3 // 1                          # already folded into Γ_eff
        t.freq_THz ≥ 0 || error(
            "a `dipole_ea0` line must lie above the state (freq_THz > 0); the " *
            "dipole normalisation assumes an upward transition. Use `gamma_MHz` " *
            "with a negative `freq_THz` for a line below the state.")
    else
        error("PolarizabilityModel transition must have either `gamma_MHz` or " *
              "`dipole_ea0`; got keys $(keys(t))")
    end
    return (freq_THz = Float64(t.freq_THz), gamma_MHz = γ,
            J = J, J_f = J_f, f = f, source = src)
end

"""
    PolarizabilityModel(state, transitions; J = 0, offset_Hz_per_Wm2 = 0.0, reference = "")

Build a model for `state` from a list of `transitions`. `J` is the total electronic
angular momentum of `state` itself; it is the default for each line's own `J`, and
leaving it at `0` reproduces the pre-existing `J=0 → J'=1` behaviour exactly.
"""
function PolarizabilityModel(state::String,
                             transitions::Vector;
                             J = 0//1,
                             offset_Hz_per_Wm2::Float64 = 0.0,
                             reference::String = "")
    Jm   = Rational{Int}(J)
    norm = [_normalize_transition(t, Jm) for t in transitions]
    for t in norm
        t.J == Jm || error(
            "transition declares J = $(t.J) but the model's state has J = $Jm; " *
            "every line of a model shares the model's initial state.")
    end
    PolarizabilityModel(state, norm, Jm, offset_Hz_per_Wm2, reference)
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

For the `J=0 → J'=1` lines that every ¹S₀-type model is built from, `f = 3` — the
value that was hard-coded into this kernel before the angular momenta were tracked,
which is why those models were correct. A `J>0` state such as ³P₁ has lines both
above and below it and genuinely needs the sign.

Reference: Höhn-model derivation in the AtomTwin polarizability notes; Steck,
*Quantum and Atom Optics*, §7.3.4 (reduced-matrix-element conventions).
"""
function _line_strength_factor(J::Rational{Int}, J_f::Rational{Int}, freq_THz::Real)
    return freq_THz ≥ 0 ? (2J_f + 1) // (2J + 1) : -1 // 1
end

"""
    _calc_light_shift(ω0, Γ, ωL, f = 3) -> Float64

Light-shift contribution from a single electric-dipole transition.

# Arguments
- `ω0`: Transition angular frequency [rad/s]. Use `|ω₀|`; the sign of the
  transition frequency enters through `f` (see [`_line_strength_factor`](@ref)).
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
        ω0 = 2π * abs(t.freq_THz) * 1e12     # |ω₀|; the sign is carried by f
        Γ  = 2π * t.gamma_MHz     * 1e6
        U_over_I += _calc_light_shift(ω0, Γ, ωL, t.f)
    end

    U_over_I += model.offset_Hz_per_Wm2 * h
    return U_over_I
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

    α⁽²⁾ = 2π ε₀ c³ Σ_J' (−1)^(−2J−J'−F−I) √(40F(2F+1)(2F−1)/(3(F+1)(2F+3))) (2J+1)
                        × f(J,J') Γ(J,J') / (ω₀² (ω₀² − ω²))
                        × {1 1 2; J J J'} {J J 2; F F I}

`F` is the hyperfine quantum number of the state and `I` the nuclear spin; for a
zero-spin isotope `I = 0` and `F = J`.

Returns `0.0` whenever the tensor part is absent — `J ≤ 1/2` (the `{1 1 2; J J J'}`
triangle rule) or `F ≤ 1/2` (the prefactor). The sum runs over the model's lines
with the same `f(J,J')` energy-ordering convention as the scalar part.

# Normalisation — read before comparing with a paper

`α⁽²⁾` is defined only up to how the `(3m_F²−F(F+1))/(F(2F−1))` sublevel factor is
split off, and the literature is not uniform. AtomTwin's matches Kestler *et al.*,
*Phys. Rev. A* **105**, 012821 (2022): for linear polarisation along the
quantisation axis,

    α(m_F = 0)   = α⁽⁰⁾ − 2 α⁽²⁾
    α(|m_F| = 1) = α⁽⁰⁾ +   α⁽²⁾

which is what `_tensor_geometry` × `_polarization_factor` reproduces, and which
`SR88_POLARIZABILITY_3P1` is validated against.

!!! warning "α⁽²⁾ and U/I must share one convention"
    The `3π ε₀ c³` here pairs with converting a polarizability to a shift via
    `1/(c ε₀)` — AtomTwin's own convention, set by `polarizability_si`
    (`α_SI = −c ε₀ (U/I)`). The notes instead write `U/I = −α/(2 ε₀ c)`, with the
    missing half absorbed into their `α`. Mixing the two — `3π` with `1/(2 c ε₀)` —
    silently halves the tensor splitting while leaving `α⁽²⁾` itself looking
    correct against a published table, which is exactly how this was nearly shipped.
    The invariant that catches it is
    `α(m_F=0) − α(|m_F|=1) = −3 α⁽²⁾`, tested in `test/unit/test_physics.jl`.
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

        # The PHYSICAL f(J,J'), never the stored `t.f`. For a `gamma_MHz` line the
        # two coincide, but a `dipole_ea0` line stores f = 3 because the scalar sum
        # wants the 1/(2Jg+1) weight that `_dipole_to_gamma_MHz` already folded into
        # Γ_eff. The tensor sum carries its own angular-momentum algebra in the two
        # 6-j symbols and the (2J+1), so it needs the true ratio: using the stored 3
        # inflates a line by 3/f — ×9 for J'=0, ×3 for J'=1, ×9/5 for J'=2. On Sr ³P₁,
        # dominated by 5p² ³P₂, that aggregated to a deceptively clean ≈×2.
        f_phys = _line_strength_factor(J, J_f, t.freq_THz)

        α2 += phase * pre * (2J + 1) * f_phys * Γ / (ω0^2 * (ω0^2 - ωL^2)) * w1 * w2
    end
    # 2π (not the notes' 3π) because AtomTwin converts α → shift with 1/(c ε₀)
    # while the notes use 1/(2 ε₀ c). The pair (2π, 1/(cε₀)) is what reproduces
    # the measured Sr magic wavelengths; see the docstring's warning.
    return 2π * ε0 * c^3 * α2
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
along the axis, `−1/2` for polarisation perpendicular to it, and vanishes at the
magic angle `acos(1/√3) ≈ 54.7356°`.
"""
_polarization_factor(ε_z::Real) = (3 * abs2(ε_z) - 1) / 2

# ======================================================================
# Public API
# ======================================================================

"""
    light_shift_coeff_Hz_per_Wcm2(model, λ_nm; F = nothing, mF = 0, I = 0, ε_z = 1) -> Float64

Light-shift coefficient Δν/I in Hz/(W/cm²) for the given model and wavelength.

# Definition
For a beam intensity `I` in W/cm², the light shift is

    Δν = light_shift_coeff_Hz_per_Wcm2(model, λ_nm) * I

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
    U = _U_over_I(model, λ_nm)
    if F !== nothing
        Fr, mFr, Ir = Rational{Int}(F), Rational{Int}(mF), Rational{Int}(I)
        abs(mFr) <= Fr || throw(ArgumentError("need |mF| ≤ F; got mF = $mFr, F = $Fr"))
        α2 = _alpha2_si(model, λ_nm; F = Fr, I = Ir)
        if α2 != 0.0
            # Same U/I ↔ α convention as the scalar part: `polarizability_si` defines
            # α_SI = −c ε₀ (U/I), so a polarizability converts to a shift with
            # 1/(c ε₀) — NOT the 1/(2 ε₀ c) of the notes, whose α carries the other
            # half. Mixing the two silently halves the tensor splitting.
            U += -α2 * _polarization_factor(ε_z) * _tensor_geometry(Fr, mFr) / (c * ε0)
        end
    end
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