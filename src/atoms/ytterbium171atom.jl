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
@term "3P2" 2//1      # (6s6p) ³P₂  metastable (no shipped polarizability model)
end

"""
    YB171_DECAY_RATES

Einstein A coefficients (s⁻¹) of the E1 decay channels of selected Yb levels, keyed
`upper => lower`. Levels outside the 6s6p configuration carry it in the key
(`"(6s6d)3D1"`, `"(5d6s)3D1"`). Electronic structure only, so it serves both isotopes
([`YB174_DECAY_RATES`](@ref)).

Pass a rate to `add_decay!`. Between ¹⁷¹Yb hyperfine manifolds, `add_decay!(F′ => F, rate)`
gives every upper sublevel the full `rate`, so when the lower level has several `F`, split
`A` with the hyperfine branching `(2F+1)(2J′+1){J J′ 1; F′ F I}²` (it sums to 1 over `F`).

| channel | A (s⁻¹) | source |
|---|---|---|
| ¹P₁ → ¹S₀ | 1.830e8 | τ = 5.464(11) ns, Kroeze Table 8 |
| ¹P₁ → (5d6s)³D₁, ³D₂ | 9.56, 2.19 | ⟨‖d‖⟩ = 0.27(10), 0.32(6) a.u., Porsev 1999 |
| ³P₁ → ¹S₀ | 1.146e6 | Kroeze Table 10 |
| (5d6s)³D₁ → ³P₀,₁,₂ | 1.920e6, 1.060e6, 2.98e4 | τ = 332(11) ns × β, Kroeze Table 12 |
| (5d6s)³D₂ → ³P₁,₂ | 1.897e6, 2.77e5 | τ = 460(30) ns (Bowers 1996) × β from Porsev 1999 |
| (6s7s)³S₁ → ³P₀,₁,₂, ¹P₁ | 9.42e6, 2.609e7, 3.681e7, 1.52e5 | β / τ, τ = 13.8(17) ns, Kroeze Table 13 |
| (6s6d)³D₁ → ³P₀,₁,₂ | 2.612e7, 1.841e7, 9.28e5 | ⟨‖d‖⟩ = 1.84, 1.62 a.u. (Porsev 2014); ³P₂ from LS |

The ¹P₁ leak totals 6.4e-8 per photon (Porsev 1999), against a measured lower bound of
1.2(4)e-7 (Honda 1999). For (6s6d)³D₁ the ³P₂ element is the LS value 1.84/√20 a.u.; the
LS ratio reproduces the ³P₁ element to 2 %. The resulting τ = 22.0 ns agrees with the
22.7 ns implied by the measured-lifetime line in `YB171_POLARIZABILITY_3P0`.
Not included: (6s6d)³D₁ → 6s7p ³P_J (5.8–8 µm), for which no calculation is published; a
Coulomb-approximation estimate puts it near 1 % of the decays.

References: R. M. Kroeze et al., arXiv:2509.04416 (2025); S. G. Porsev et al., PRA 60,
2781 (1999); S. G. Porsev et al., PRA 89, 012711 (2014); C. J. Bowers et al., PRA 53,
3103 (1996); K. Honda et al., PRA 59, R934 (1999).
"""
const YB171_DECAY_RATES = Dict{Pair{String,String},Float64}(
    ("1P1"       => "1S0")       => 1 / 5.464e-9,
    ("1P1"       => "(5d6s)3D1") => 9.56,
    ("1P1"       => "(5d6s)3D2") => 2.19,
    ("3P1"       => "1S0")       => 1.1460e6,
    ("(5d6s)3D1" => "3P0")       => 1.920e6,
    ("(5d6s)3D1" => "3P1")       => 1.060e6,
    ("(5d6s)3D1" => "3P2")       => 2.98e4,
    ("(5d6s)3D2" => "3P1")       => 1.897e6,
    ("(5d6s)3D2" => "3P2")       => 2.768e5,
    ("(6s7s)3S1" => "3P0")       => 0.130 / 13.8e-9,
    ("(6s7s)3S1" => "3P1")       => 0.360 / 13.8e-9,
    ("(6s7s)3S1" => "3P2")       => 0.508 / 13.8e-9,
    ("(6s7s)3S1" => "1P1")       => 0.0021 / 13.8e-9,
    ("(6s6d)3D1" => "3P0")       => 2.612e7,
    ("(6s6d)3D1" => "3P1")       => 1.841e7,
    ("(6s6d)3D1" => "3P2")       => 9.28e5,
)

"""
    YB171_LINE_FREQUENCIES

Measured transition frequencies (Hz) of the lines in [`YB171_DECAY_RATES`](@ref), keyed the
same way: the line positions the polarizability models use (Kroeze et al.; NIST level
energies). Those models fit line strengths and effective lines, not these positions; the
strengths here come from the decay rates alone. A frequency and its decay rate give the
line's reduced dipole ([`reduced_dipole`](@ref)), from which `add_coupling!` derives Rabi
frequencies from a beam. Lines to ³P₂ and from ¹P₁ to ³D_J are not listed.
"""
const YB171_LINE_FREQUENCIES = Dict{Pair{String,String},Float64}(
    ("1P1"       => "1S0") => 751.526389e12,
    ("3P1"       => "1S0") => 539.386800e12,
    ("(5d6s)3D1" => "3P0") => 215.870446e12,
    ("(5d6s)3D1" => "3P1") => 194.778008e12,
    ("(5d6s)3D2" => "3P1") => 202.657933e12,
    ("(6s7s)3S1" => "3P0") => 461.867846e12,
    ("(6s7s)3S1" => "3P1") => 440.775408e12,
    ("(6s6d)3D1" => "3P0") => 675.141040e12,
    ("(6s6d)3D1" => "3P1") => 654.048602e12,
)

"""
    YB171_G_FACTORS

g-factors of ¹⁷¹Yb (units of μ_B): the nuclear `g_I`, from the ¹S₀ Zeeman shift
(−0.7505 kHz/G per m_F, Kroeze et al. Table 5), and the electronic `g_J` of ¹S₀ and ³P₁
(NIST ASD). Through [`lande_gF`](@ref) they reproduce Kroeze's measured ³P₁ F′ = 3/2
and ½ g-factors (1392.674 and 2786.10 kHz/G) to 1e-6. ³P₀ is not listed: its g-factor
comes from hyperfine mixing with ³P₁ ([`getgfactors`](@ref)).
"""
const YB171_G_FACTORS = (g_I = -0.7505 / 1399.6245, g_J = Dict("1S0" => 0.0, "3P1" => 1.49282))

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
getdecayrates(::Ytterbium171Atom) = YB171_DECAY_RATES
getgfactors(::Ytterbium171Atom) = YB171_G_FACTORS
getlinefrequencies(::Ytterbium171Atom) = YB171_LINE_FREQUENCIES
