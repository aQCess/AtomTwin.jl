"""
    rubidium87atom.jl

Rb-87 ground-state polarizability data and convenience methods.

The D lines share the 5S₁/₂ ground state but carry unequal line strengths (D₂ has
twice the strength of D₁), so the model is specified by the reduced dipole matrix
elements |⟨Jg‖er‖Je⟩| rather than the near-equal natural linewidths — see
[`PolarizabilityModel`](@ref). Higher 6P/7P lines and a static ionic-core term
are included so the dynamic polarizability is accurate into the blue (below the
D lines), not only for a red-detuned trap.

References:
- D. A. Steck, "Rubidium 87 D Line Data" (2019) — D-line dipoles & frequencies.
- M. S. Safronova & C. W. Clark, Phys. Rev. A 69, 022509 (2004) — 6P/7P dipoles.
- B. Arora, M. S. Safronova & C. W. Clark, Phys. Rev. A 76, 052509 (2007).
"""

# ======================================================================
# Data (Steck D lines; Safronova/Clark higher lines + core)
# ======================================================================

"""
    RB87_POLARIZABILITY_5S12

Empirical ground-state (5S₁/₂) dynamic-polarizability model for Rb-87, built from
reduced dipole matrix elements. The D₁/D₂ doublet dominates; the 6P and 7P lines
plus a static ionic-core term (`offset_Hz_per_Wm2`, α_core ≈ 9.08 a.u.) extend
accuracy below the D lines. Reproduces the accepted static scalar polarizability
(α₀ ≈ 318 a.u.) and the correct α(λ) from the IR through the visible.

Line frequencies are from NIST wavelengths; dipoles are `|⟨5S₁/₂‖er‖nP_J⟩|` in
`e·a₀`. The `Jg = 1/2` per line is explicit for clarity (it is also the default).
"""
const RB87_POLARIZABILITY_5S12 = PolarizabilityModel(
    "5S1/2",
    [
        (freq_THz = 377.107, dipole_ea0 = 4.227, Jg = 1//2),  # D1  5S1/2 → 5P1/2  (795 nm)
        (freq_THz = 384.231, dipole_ea0 = 5.977, Jg = 1//2),  # D2  5S1/2 → 5P3/2  (780 nm)
        (freq_THz = 710.960, dipole_ea0 = 0.342, Jg = 1//2),  #     5S1/2 → 6P1/2  (422 nm)
        (freq_THz = 713.477, dipole_ea0 = 0.553, Jg = 1//2),  #     5S1/2 → 6P3/2  (420 nm)
        (freq_THz = 834.474, dipole_ea0 = 0.118, Jg = 1//2),  #     5S1/2 → 7P1/2  (359 nm)
        (freq_THz = 835.526, dipole_ea0 = 0.207, Jg = 1//2),  #     5S1/2 → 7P3/2  (359 nm)
    ];
    # Static ionic-core (Rb⁺) + valence-tail polarizability, α_core ≈ 9.08 a.u.:
    #   offset_Hz_per_Wm2 = -α_core·(4πε₀a₀³) / (c ε₀ h).
    offset_Hz_per_Wm2 = -8.5118e-5,
    reference = "Steck 2019 (D lines); Safronova & Clark PRA 69, 022509 (2004) (6P/7P + core)",
)


"""
    RB87_POLARIZABILITY

Dictionary of all Rb-87 polarizability models, keyed by state label.
"""
const RB87_POLARIZABILITY = Dict(
    "5S1/2" => RB87_POLARIZABILITY_5S12,
)


# ======================================================================
# Convenience methods for AtomWrapper{:Rubidium87}
# ======================================================================

getpolarizabilitymodels(::Rubidium87Atom) = RB87_POLARIZABILITY

"""
    light_shift_coeff_Hz_per_Wcm2(atom::Rubidium87Atom, state, λ_nm) -> Float64

Light-shift coefficient for an Rb-87 atom in the given state at wavelength λ_nm (nm).

Returns Δν/I in Hz/(W/cm²).
"""
function light_shift_coeff_Hz_per_Wcm2(atom::Rubidium87Atom,
                                       state::String,
                                       λ_nm::Real)
    model = RB87_POLARIZABILITY[state]
    return light_shift_coeff_Hz_per_Wcm2(model, λ_nm)
end

"""
    scattering_rate_per_Wcm2(atom::Rubidium87Atom, state, λ_nm) -> Float64

Off-resonant photon-scattering-rate coefficient Γ_sc/I in (1/s)/(W/cm²) for an
Rb-87 atom in the given state at wavelength λ_nm (nm). See the model-level
[`scattering_rate_per_Wcm2`](@ref) for the physics.
"""
function scattering_rate_per_Wcm2(atom::Rubidium87Atom,
                                  state::String,
                                  λ_nm::Real)
    model = RB87_POLARIZABILITY[state]
    return scattering_rate_per_Wcm2(model, λ_nm)
end

"""
    polarizability_au(atom::Rubidium87Atom, state, λ_nm) -> Float64

Dynamic polarizability in atomic units for an Rb-87 atom in the given state
at wavelength λ_nm (nm).
"""
function polarizability_au(atom::Rubidium87Atom,
                           state::String,
                           λ_nm::Real)
    model = RB87_POLARIZABILITY[state]
    return polarizability_au(model, λ_nm)
end
