"""
    ytterbium171atom.jl

Yb-171 polarizability data and convenience methods.

References:
- Phys. Rev. A 108, 053325 (2023)
"""

# ======================================================================
# Data (Table I from the reference)
# ======================================================================

"""
    YB171_POLARIZABILITY_1S0

Empirical polarizability model for the Yb-171 1S₀ state.
"""
const YB171_POLARIZABILITY_1S0 = PolarizabilityModel(
    "1S0",
    [
        (freq_THz = 539.386800, gamma_MHz = 0.183,  J_f = 1//1),  # (6s6p) ³P₁   556 nm
        (freq_THz = 751.526389, gamma_MHz = 29.127, J_f = 1//1),  # (6s6p) ¹P₁   399 nm
        (freq_THz = 865.111516, gamma_MHz = 11.052, J_f = 1//1),  # (7/2,5/2) J=1  347 nm
    ];
    J = 0//1,                       # (6s²) ¹S₀
    offset_Hz_per_Wm2 = -0.8e-4,
    reference = "Phys. Rev. A 108, 053325 (2023)",
)

"""
    YB171_POLARIZABILITY_3P0

Empirical polarizability model for the Yb-171 3P₀ state.
"""
const YB171_POLARIZABILITY_3P0 = PolarizabilityModel(
    "3P0",
    [
        (freq_THz = 215.870446, gamma_MHz = 0.308,  J_f = 1//1),  # (6s5d) ³D₁  1389 nm
        (freq_THz = 461.867846, gamma_MHz = 1.516,  J_f = 1//1),  # (6s7s) ³S₁   649 nm
        (freq_THz = 675.141040, gamma_MHz = 4.081,  J_f = 1//1),  # (6s6d) ³D₁   444 nm
        (freq_THz = 729.293151, gamma_MHz = 0.625,  J_f = 1//1),  # (6s8s) ³S₁   411 nm
        (freq_THz = 797.204099, gamma_MHz = 22.889, J_f = 1//1,   # effective line 376 nm
         source = :fitted),
    ];
    J = 0//1,                       # (6s6p) ³P₀
    offset_Hz_per_Wm2 = 0.0,
    reference = "Phys. Rev. A 108, 053325 (2023)",
)


"""
    Ytterbium171

Term symbols for Yb-171, for the `term =` argument of a level or manifold.
Usually written with the `l"..."` macro, which needs no species prefix:

```julia
g = Level("ground"; term = l"1S0")
e = HyperfineManifold(3//2, 1; label = "³P₁", term = l"3P1")
```

`Ytterbium171._3P1` names the same object when the species is worth spelling out.
Either way a typo is an error where it is written, not a silent α = 0.
"""
module Ytterbium171
import ..TermSymbol, ..TERM_REGISTRY, ..@term
@term "1S0" 0//1      # (6s²)  ¹S₀  ground
@term "3P0" 0//1      # (6s6p) ³P₀  clock
@term "3P1" 1//1      # (6s6p) ³P₁  intercombination
end

"""
    YB171_POLARIZABILITY

Dictionary of all Yb-171 polarizability models, keyed by state label.
"""
const YB171_POLARIZABILITY = Dict(
    "1S0" => YB171_POLARIZABILITY_1S0,
    "3P0" => YB171_POLARIZABILITY_3P0,
)


# ======================================================================
# Convenience methods for AtomWrapper{:Ytterbium171}
# ======================================================================

"""
    Ytterbium171Atom

Convenience type for a Yb-171 atom with built-in polarizability models.
"""
const Ytterbium171Atom = AtomWrapper{:Ytterbium171}

getpolarizabilitymodels(::Ytterbium171Atom) = YB171_POLARIZABILITY

"""
    light_shift_coeff_Hz_per_Wcm2(atom::Ytterbium171Atom, state, λ_nm) -> Float64

Light-shift coefficient for a Yb-171 atom in the given state at wavelength λ_nm (nm).

Returns Δν/I in Hz/(W/cm²).
"""
function light_shift_coeff_Hz_per_Wcm2(atom::Ytterbium171Atom,
                                       state::String,
                                       λ_nm::Real)
    model = YB171_POLARIZABILITY[state]
    return light_shift_coeff_Hz_per_Wcm2(model, λ_nm)
end

"""
    scattering_rate_per_Wcm2(atom::Ytterbium171Atom, state, λ_nm) -> Float64

Off-resonant photon-scattering-rate coefficient Γ_sc/I in (1/s)/(W/cm²) for a
Yb-171 atom in the given state at wavelength λ_nm (nm). See the model-level
[`scattering_rate_per_Wcm2`](@ref) for the physics.
"""
function scattering_rate_per_Wcm2(atom::Ytterbium171Atom,
                                  state::String,
                                  λ_nm::Real)
    model = YB171_POLARIZABILITY[state]
    return scattering_rate_per_Wcm2(model, λ_nm)
end

"""
    polarizability_au(atom::Ytterbium171Atom, state, λ_nm) -> Float64

Dynamic polarizability in atomic units for a Yb-171 atom in the given state
at wavelength λ_nm (nm).
"""
function polarizability_au(atom::Ytterbium171Atom,
                           state::String,
                           λ_nm::Real)
    model = YB171_POLARIZABILITY[state]
    return polarizability_au(model, λ_nm)
end