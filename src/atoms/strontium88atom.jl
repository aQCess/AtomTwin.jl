"""
    strontium88atom.jl

Sr polarizability data and convenience methods for the optical lattice clock
(the 5s² ¹S₀ – 5s5p ³P₀ transition at 698 nm, magic wavelength 813.4 nm).

The scalar polarizability is an electronic quantity, so it is essentially
isotope-independent; the same model serves the fermionic clock isotope ⁸⁷Sr and
the bosonic ⁸⁸Sr. It is attached to the `Strontium88Atom` species that AtomTwin
already defines.

References:
- M. S. Safronova, S. G. Porsev, U. I. Safronova, M. G. Kozlov, C. W. Clark,
  "Blackbody-radiation shift in the Sr optical atomic clock",
  Phys. Rev. A 87, 012509 (2013) — Tables II & IV: recommended reduced E1
  dipole matrix elements and experimental transition energies.
- The magic wavelength itself is the measured value 813.4280(5) nm
  (Ushijima et al., and CODATA-tracked clock comparisons).
"""

# ======================================================================
# Data (Safronova et al. 2013, Tables II & IV)
# ======================================================================
#
# Each clock state is a J=0 level, so every listed line is J=0 → J'=1 and we can
# supply the recommended reduced dipole matrix element directly via `dipole_ea0`
# with `Jg = 0`. Transition energies are the experimental values (Table IV,
# column E_expt) in cm⁻¹, converted to THz (1 cm⁻¹ = 29.9792458 GHz).
#
# The line lists reproduce the reference STATIC scalar polarizabilities almost
# exactly (α(¹S₀) = 197.2 vs 197.14 a.u.; α(³P₀) = 449 vs 444.5 a.u.). The
# `offset` rows absorb the paper's "Other" + "Core + Vc" contributions — the sum
# of the many far-lying transitions that are not listed individually.
#
# CALIBRATION NOTE. The magic wavelength is a near-degenerate crossing of two
# large, nearly equal polarizabilities (both ≈ 280 a.u.), so it amplifies small
# errors: each ³P₀ matrix element, varied within its stated 1σ, moves the
# crossing by ±2–4 nm, and the crossing is likewise sensitive to the weakly
# dispersive tail that a static line list cannot capture. A pure line-list model
# built from the table lands at ≈806.7 nm — within the reference data's own
# uncertainty but 0.8% below the measured 813.428 nm. We therefore keep every
# spectroscopic matrix element untouched and anchor the ONE genuinely
# underdetermined quantity — the ³P₀ tail offset — to the measured magic
# wavelength (34.6 → 39.3 a.u., a +4.7 a.u. shift well inside the tail's
# uncertainty). This is the standard semi-empirical construction, not a fit to
# invented data: the dominant lines are fixed by spectroscopy; only the unmeasured
# tail is set by one experimental number.

# Convert an offset given in atomic-unit polarizability to AtomTwin's
# offset_Hz_per_Wm2, so the two forms stay in sync if the tails are re-tuned:
#   α_SI = α_au · 4π ε₀ a₀³;  U/I = −α_SI/(c ε₀);  offset_Hz_per_Wm2 = (U/I)/h.
_au_to_offset_Hz_per_Wm2(α_au) =
    -(α_au * 4π * ε0 * a0^3) / (c * ε0) / h

"""
    SR88_POLARIZABILITY_1S0

Empirical polarizability model for the Sr-88 5s² ¹S₀ ground clock state.
Dominated by the 461 nm ¹P₁ line; higher ¹P₁ lines and the core are the offset.
"""
const SR88_POLARIZABILITY_1S0 = PolarizabilityModel(
    "1S0",
    [
        (freq_THz = 650.4897, dipole_ea0 = 5.248, Jg = 0),  # 5s5p ¹P₁   (461 nm)
        (freq_THz = 434.8190, dipole_ea0 = 0.158, Jg = 0),  # 5s5p ³P₁   (689 nm, weak)
        (freq_THz = 1022.2323, dipole_ea0 = 0.281, Jg = 0), # 5s6p ¹P₁   (293 nm)
        (freq_THz = 1234.3055, dipole_ea0 = 0.517, Jg = 0), # 4d5p ¹P₁   (243 nm)
    ];
    offset_Hz_per_Wm2 = _au_to_offset_Hz_per_Wm2(4.60 + 5.29),  # Other + Core+Vc
    reference = "Phys. Rev. A 87, 012509 (2013), Tables II & IV",
)

"""
    SR88_POLARIZABILITY_3P0

Empirical polarizability model for the Sr-88 5s5p ³P₀ excited clock state.
The near-IR ³D₁ (2.6 µm) and 679 nm ³S₁ lines dominate the dispersion near the
magic wavelength; the tail offset is anchored to the measured 813.428 nm crossing
(see the calibration note above).
"""
const SR88_POLARIZABILITY_3P0 = PolarizabilityModel(
    "3P0",
    [
        (freq_THz = 115.1803, dipole_ea0 = 2.675, Jg = 0),  # 5s4d ³D₁   (2603 nm)
        (freq_THz = 441.3245, dipole_ea0 = 1.962, Jg = 0),  # 5s6s ³S₁   (679 nm)
        (freq_THz = 620.2406, dipole_ea0 = 2.450, Jg = 0),  # 5s5d ³D₁   (483 nm)
        (freq_THz = 632.0524, dipole_ea0 = 2.605, Jg = 0),  # 5p²  ³P₁   (474 nm)
        (freq_THz = 692.7304, dipole_ea0 = 0.516, Jg = 0),  # 5s7s ³S₁   (433 nm)
        (freq_THz = 760.5135, dipole_ea0 = 1.161, Jg = 0),  # 5s6d ³D₁   (394 nm)
    ];
    offset_Hz_per_Wm2 = _au_to_offset_Hz_per_Wm2(39.31),  # tail anchored to 813.428 nm
    reference = "Phys. Rev. A 87, 012509 (2013), Tables II & IV; " *
                "tail anchored to the measured 813.428 nm magic wavelength",
)

"""
    SR88_POLARIZABILITY

Dictionary of all Sr-88 polarizability models, keyed by state label.
"""
const SR88_POLARIZABILITY = Dict(
    "1S0" => SR88_POLARIZABILITY_1S0,
    "3P0" => SR88_POLARIZABILITY_3P0,
)


# ======================================================================
# Convenience methods for AtomWrapper{:Strontium88}
# (the `Strontium88Atom` alias itself is defined in atoms.jl)
# ======================================================================

getpolarizabilitymodels(::Strontium88Atom) = SR88_POLARIZABILITY

"""
    light_shift_coeff_Hz_per_Wcm2(atom::Strontium88Atom, state, λ_nm) -> Float64

Light-shift coefficient for a Sr-88 atom in the given state at wavelength λ_nm (nm).

Returns Δν/I in Hz/(W/cm²).
"""
function light_shift_coeff_Hz_per_Wcm2(atom::Strontium88Atom,
                                       state::String,
                                       λ_nm::Real)
    model = SR88_POLARIZABILITY[state]
    return light_shift_coeff_Hz_per_Wcm2(model, λ_nm)
end

"""
    scattering_rate_per_Wcm2(atom::Strontium88Atom, state, λ_nm) -> Float64

Off-resonant photon-scattering-rate coefficient Γ_sc/I in (1/s)/(W/cm²) for a
Sr-88 atom in the given state at wavelength λ_nm (nm). See the model-level
[`scattering_rate_per_Wcm2`](@ref) for the physics.
"""
function scattering_rate_per_Wcm2(atom::Strontium88Atom,
                                  state::String,
                                  λ_nm::Real)
    model = SR88_POLARIZABILITY[state]
    return scattering_rate_per_Wcm2(model, λ_nm)
end

"""
    polarizability_au(atom::Strontium88Atom, state, λ_nm) -> Float64

Dynamic polarizability in atomic units for a Sr-88 atom in the given state
at wavelength λ_nm (nm).
"""
function polarizability_au(atom::Strontium88Atom,
                           state::String,
                           λ_nm::Real)
    model = SR88_POLARIZABILITY[state]
    return polarizability_au(model, λ_nm)
end
