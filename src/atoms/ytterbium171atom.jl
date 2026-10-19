"""
    ytterbium171atom.jl

Yb-171 polarizability data and convenience methods.

References:
- T. O. Höhn et al., Phys. Rev. A 108, 053325 (2023): the model form and line list.
- R. M. Kroeze, S. L. Kristensen, S. Pucher, "¹⁷¹Yb Reference Data",
  arXiv:2509.04416 (2025): line strengths and the measured wavelengths fitted to.
"""

# ======================================================================
# Data
# ======================================================================
#
# The ¹S₀ and ³P₀ models keep Höhn et al.'s line list and effective-line form.
# Line strengths with a measured value are held at it exactly (Kroeze et al.,
# Tables 8, 10, 12); the rest -- strengths estimated from a lifetime and an
# LS-coupling branching ratio, the ¹S₀ offset and the ³P₀ effective line -- were
# refitted to the measured state-(in)dependent wavelengths (Kroeze Table 7), with
# the clock magic wavelength weighted at its experimental uncertainty and the
# others at a 0.015 nm model floor (χ²/dof ≈ 1). Every refitted strength stays
# within 0.1σ of its prior. Reproduced (measured → model, nm):
#
#   clock magic (E1, ¹⁷¹Yb)  759.356 063 3(31) → 759.356 063 3
#   ¹S₀–³P₀ magic            459.5960(5)       → 459.595 94
#   ¹S₀–³P₀ magic (¹⁷⁴Yb)    552.610 74(19)    → 552.5975
#   ¹S₀ tune-out (¹⁷⁴Yb)     553.2936(5)       → 553.3110
#   ³P₀ tune-out (¹⁷⁴Yb)     576.613(10)       → 576.6122
#
# The previous (published) parameters put the clock magic wavelength at 759.395 nm
# and the ³P₀ tune-out at 575.54 nm. Fit and data: AtomTwinLLM
# studies/yb-clock-polarizability-refit.
#
# Validity: the visible and near-infrared, ~450-800 nm. The effective line stands in
# for many high-lying states and is not a static limit: the models give
# Δα(ω = 0) ≈ 137 a.u. against the measured 36.2612(7) a.u. (Sherman et al. 2012),
# so they must not be used for blackbody or DC Stark shifts.

"""
    YB171_POLARIZABILITY_1S0

Empirical polarizability model for the Yb-171 1S₀ state.
"""
const YB171_POLARIZABILITY_1S0 = PolarizabilityModel(
    "1S0",
    [
        # Γ/2π (MHz): measured, held fixed
        (freq_THz = 539.386800, gamma_MHz = 0.1824,     J_f = 1//1),  # (6s6p) ³P₁   556 nm  τ = 872.6(19) ns
        (freq_THz = 751.526389, gamma_MHz = 29.13,      J_f = 1//1),  # (6s6p) ¹P₁   399 nm  τ = 5.464(11) ns
        # lifetime-based estimate (Blagoev & Komarovskii 1994), refitted
        (freq_THz = 865.111516, gamma_MHz = 11.2729428, J_f = 1//1),  # (7/2,5/2) J=1  347 nm
    ];
    J = 0//1,                       # (6s²) ¹S₀
    offset_Hz_per_Wm2 = -0.797696924e-4,   # fitted
    reference = "Höhn et al., PRA 108, 053325 (2023); refitted to Kroeze et al., arXiv:2509.04416",
)

"""
    YB171_POLARIZABILITY_3P0

Empirical polarizability model for the Yb-171 3P₀ state.
"""
const YB171_POLARIZABILITY_3P0 = PolarizabilityModel(
    "3P0",
    [
        (freq_THz = 215.870446, gamma_MHz = 0.306,      J_f = 1//1),  # (6s5d) ³D₁  1389 nm  measured, fixed
        (freq_THz = 461.867846, gamma_MHz = 1.50182782, J_f = 1//1),  # (6s7s) ³S₁   649 nm  1.50(24) MHz, refitted
        (freq_THz = 675.141040, gamma_MHz = 4.0326703,  J_f = 1//1),  # (6s6d) ³D₁   444 nm  LS estimate, refitted
        (freq_THz = 729.293151, gamma_MHz = 0.62437427, J_f = 1//1),  # (6s8s) ³S₁   411 nm  LS estimate, refitted
        (freq_THz = 796.547798, gamma_MHz = 23.2660865, J_f = 1//1,   # effective line 376 nm
         source = :fitted),
    ];
    J = 0//1,                       # (6s6p) ³P₀
    offset_Hz_per_Wm2 = 0.0,
    reference = "Höhn et al., PRA 108, 053325 (2023); refitted to Kroeze et al., arXiv:2509.04416",
)


"""
    YB171_POLARIZABILITY_3P1

Polarizability model for the Yb (6s6p) ³P₁ state, the upper level of the 556 nm
intercombination line. The electronic structure is isotope-independent, so the same
model serves Yb-174 ([`YB174_POLARIZABILITY_3P1`](@ref)).

`J = 1`: the state has a **tensor** polarizability, recoupled to `F` with the nuclear
spin — `F = J = 1` for ¹⁷⁴Yb; `F = 3/2` and `1/2` for ¹⁷¹Yb, where `F = 1/2` has none.

# Line list

Höhn's thesis (Appendix A), changed as follows:

1. The (6s²)¹S₀ line is `J′ = 0`; the thesis has `J′ = 1`.
2. **6p² ³P₀,₁,₂ added** (409, 387, 374 nm, NIST energies), with the LS line-strength
   ratios 1/3 : 1/4 : 5/12 and one fitted amplitude. The thesis represents the
   high-lying states by an effective `J′ = 1`/`J′ = 2` pair only; the missing `J′ = 0`
   member (α₂ = −α₀) is what sets the tensor part between 470 and 540 nm.
3. **(5d6s)¹D₂ added** (1032.45 nm), its width fitted to the magic wavelengths of
   Zheng et al. near 1036 nm.
4. Refitted with 2–3: the effective pair (×0.17), (6s6d)³D₂ (×1.10 on its LS estimate)
   and (6s7s)³S₁ (4.26 MHz, against 3.60 MHz from Porsev's β = 0.360(25)); these three
   are strongly correlated in the fit.
   (6s6d)³D₁ and (6s8s)³S₁ keep their LS estimates.

# Data and accuracy

Fitted to data that need no intensity calibration: ¹⁷⁴Yb π/σ light-shift ratios at
473–532 nm (Ma et al. 2022) and 532 nm (Abdel Karim et al. 2025; Jenkins et al.
2022), the magic angles at 759 nm (Höhn et al. 2026; Lis et al. 2023), the ¹⁷¹Yb
magic wavelengths near 483, 486.78 and 783.8 nm (Norcia et al. 2023, 2024; Ma et al.
2022), the 760 nm sublevel ratio of Huie et al. (2023) and the ¹⁷⁴Yb magic wavelengths
near 1036 nm (Zheng et al. 2020): 20 data, 5 parameters, χ² = 17.7, all within 2.4σ.
The data, the fit and its residuals are kept with AtomTwinLLM (`studies/yb-3p1-polarizability-model`).

Absolute shifts in the literature rest on intensity calibrations. The model's scale
agrees with Norcia et al.'s 783.8 nm sublevel ratios; Huie et al.'s waist-calibrated
values are ≈3.7× smaller, their ratio matches. The (6s7s)³S₁ width sits 2.6σ above
Porsev's branching ratio, and no checked datum lies between 540 and 750 nm.
"""
const YB171_POLARIZABILITY_3P1 = PolarizabilityModel(
    "3P1",
    [
        # 556 nm line down to the ground state — BELOW ³P₁, hence negative; J′ = 0.
        (freq_THz = -539.386800, gamma_MHz = 0.183,  J_f = 0//1),  # (6s²)  ¹S₀
        (freq_THz =  194.778008, gamma_MHz = 0.170,  J_f = 1//1),  # (6s5d) ³D₁
        (freq_THz =  202.657933, gamma_MHz = 0.280,  J_f = 2//1,   # (6s5d) ³D₂
         source = :ls_estimated),
        (freq_THz =  290.3687,   gamma_MHz = 0.01012, J_f = 2//1,   # (5d6s) ¹D₂  1032 nm
         source = :fitted),
        (freq_THz =  440.775408, gamma_MHz = 4.263,  J_f = 1//1,   # (6s7s) ³S₁
         source = :fitted),
        (freq_THz =  654.048602, gamma_MHz = 2.783,  J_f = 1//1,   # (6s6d) ³D₁
         source = :ls_estimated),
        (freq_THz =  654.927593, gamma_MHz = 5.721,  J_f = 2//1,   # (6s6d) ³D₂
         source = :fitted),
        (freq_THz =  708.200713, gamma_MHz = 1.718,  J_f = 1//1,   # (6s8s) ³S₁
         source = :ls_estimated),
        # (6p²)³P multiplet: LS strengths 1/3 : 1/4 : 5/12, one fitted amplitude.
        (freq_THz =  732.8398,   gamma_MHz = 32.49,  J_f = 0//1,   # (6p²) ³P₀  409 nm
         source = :fitted),
        (freq_THz =  773.8667,   gamma_MHz = 9.565,  J_f = 1//1,   # (6p²) ³P₁  387 nm
         source = :fitted),
        (freq_THz =  802.4953,   gamma_MHz = 10.67,  J_f = 2//1,   # (6p²) ³P₂  374 nm
         source = :fitted),
        # The rest of the high-lying manifold, effective.
        (freq_THz =  778.975785, gamma_MHz = 3.543,  J_f = 1//1,   # empirical J′=1
         source = :fitted),
        (freq_THz =  778.975785, gamma_MHz = 5.830,  J_f = 2//1,   # empirical J′=2
         source = :fitted),
    ];
    J = 1//1,                       # (6s6p) ³P₁
    offset_Hz_per_Wm2 = 0.0,
    reference = "T. O. Höhn, PhD thesis (2024) App. A; 6p² ³P and 5d6s ¹D₂ added and refitted " *
                "to calibration-free ³P₁ light-shift data (AtomTwin 0.2.1)",
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
@term "1P1" 1//1      # (6s6p) ¹P₁  399 nm (no shipped polarizability model)
end

"""
    YB171_POLARIZABILITY

Dictionary of all Yb-171 polarizability models, keyed by state label.
"""
const YB171_POLARIZABILITY = Dict(
    "1S0" => YB171_POLARIZABILITY_1S0,
    "3P0" => YB171_POLARIZABILITY_3P0,
    "3P1" => YB171_POLARIZABILITY_3P1,
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
