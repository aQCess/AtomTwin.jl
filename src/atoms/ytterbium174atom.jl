"""
    YB174_POLARIZABILITY

Dictionary of all Yb-174 polarizability models, keyed by state label.

The Yb polarizability model is isotope independent, so this is just a reference to the Yb-171 polarizability dictionary.
"""
const YB174_POLARIZABILITY = YB171_POLARIZABILITY


# ======================================================================
# Convenience methods for AtomWrapper{:Ytterbium174}
# ======================================================================

"""
    Ytterbium174Atom

Convenience type for a Yb-174 atom with built-in polarizability models.
"""
const Ytterbium174Atom = AtomWrapper{:Ytterbium174}

# The Yb polarizability model is isotope independent
getpolarizabilitymodels(::Ytterbium174Atom) = YB174_POLARIZABILITY

"""
    light_shift_coeff_Hz_per_Wcm2(atom::Ytterbium174Atom, state, λ_nm) -> Float64

Light-shift coefficient for a Yb-174 atom in the given state at wavelength λ_nm (nm).

Returns Δν/I in Hz/(W/cm²).
"""
function light_shift_coeff_Hz_per_Wcm2(atom::Ytterbium174Atom,
                                       state::String,
                                       λ_nm::Real)
    model = YB174_POLARIZABILITY[state]
    return light_shift_coeff_Hz_per_Wcm2(model, λ_nm)
end

"""
    scattering_rate_per_Wcm2(atom::Ytterbium174Atom, state, λ_nm) -> Float64

Off-resonant photon-scattering-rate coefficient Γ_sc/I in (1/s)/(W/cm²) for a
Yb-174 atom in the given state at wavelength λ_nm (nm). See the model-level
[`scattering_rate_per_Wcm2`](@ref) for the physics.
"""
function scattering_rate_per_Wcm2(atom::Ytterbium174Atom,
                                  state::String,
                                  λ_nm::Real)
    model = YB174_POLARIZABILITY[state]
    return scattering_rate_per_Wcm2(model, λ_nm)
end

"""
    polarizability_au(atom::Ytterbium174Atom, state, λ_nm) -> Float64

Dynamic polarizability in atomic units for a Yb-174 atom in the given state
at wavelength λ_nm (nm).
"""
function polarizability_au(atom::Ytterbium174Atom,
                           state::String,
                           λ_nm::Real)
    model = YB174_POLARIZABILITY[state]
    return polarizability_au(model, λ_nm)
end