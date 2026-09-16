"""
    ytterbium174atom.jl

Yb-174 polarizability data — the spin-zero (bosonic) ytterbium isotope.

`I = 0`, so there is no hyperfine structure and `F = J`. That does **not** remove
the tensor light shift: `³P₁` has `J = 1`, so its three `m_J` sublevels are split
by a trap unless the polarisation sits at the magic angle. This is the isotope the
767 nm trap-depth spectroscopy example uses.

The electronic structure is shared with Yb-171 — only the nuclear spin and mass
differ — so the `¹S₀` and `³P₀` line lists are the Yb-171 ones.

References:
- T. O. Höhn, E. Staub, G. Brochier, N. Darkwah Oppong, M. Aidelsburger,
  "State-dependent potentials for the ¹S₀ and ³P₀ clock states of neutral
  ytterbium atoms", Phys. Rev. A 108, 053325 (2023).
- T. O. Höhn, "State-dependent potentials and clock ground-state cooling in an
  ytterbium quantum simulator", PhD thesis, LMU München (2024) — Appendix A, the
  ³P₁ transition list.
- T. O. Höhn et al., "Determining the ³P₀ excited-state tune-out wavelength of
  ¹⁷⁴Yb in a triple-magic lattice", PRX Quantum 7, 010303 (2026).
- S. G. Porsev, Phys. Rev. A 60, 2781 (1999) — the ab-initio branching ratio
  behind the corrected (6s7s)³S₁ linewidth.
- H. Schmit-Veiler, "Polarizability models for ytterbium", technical report,
  CESQ/University of Strasbourg (2026) — the corrections below, and the refit.
"""

# ======================================================================
# Term symbols
# ======================================================================

"""
    Ytterbium174

Term symbols for Yb-174. Usually written `l"3P1"`; see [`Ytterbium171`](@ref).
"""
module Ytterbium174
import ..TermSymbol, ..TERM_REGISTRY, ..@term
@term "1S0" 0//1      # (6s²)  ¹S₀  ground
@term "3P0" 0//1      # (6s6p) ³P₀  clock
@term "3P1" 1//1      # (6s6p) ³P₁  556 nm intercombination
end

# ======================================================================
# Data
# ======================================================================

"""
    YB174_POLARIZABILITY_1S0

Ground-state model, shared with Yb-171 (the electronic structure is isotope
independent; only mass and nuclear spin differ).
"""
const YB174_POLARIZABILITY_1S0 = YB171_POLARIZABILITY_1S0

"""
    YB174_POLARIZABILITY_3P0

Clock-state model, shared with Yb-171. See [`YB171_POLARIZABILITY_3P0`](@ref).
"""
const YB174_POLARIZABILITY_3P0 = YB171_POLARIZABILITY_3P0

"""
    YB174_POLARIZABILITY_3P1

Polarizability model for the Yb (6s6p) ³P₁ state — the upper level of the 556 nm
intercombination line.

`J = 1`, so this state has a **tensor** polarizability and its sublevels shift
differently in a trap. For ¹⁷⁴Yb, `I = 0` and `F = J = 1`.

# Two corrections to the published data

The line list is Appendix A of Höhn's thesis with two changes, both established in
the accompanying technical report:

1. **The (6s²)¹S₀ line is `J′ = 0`, not `J′ = 1`.** The published curves skip the
   556 nm resonance at 0°, which is the fingerprint of a dipole-forbidden line —
   but `F = 1 → F′ = 0` with `Δm_F = 0` is allowed, and only `F′ = 1` would
   forbid it. Correcting `J′` reproduces the thesis figures closely; leaving it
   wrong misplaces the dominant near-resonant term.

2. **(6s7s)³S₁ has `γ = 3.604 MHz`, not 3.954 MHz.** The thesis estimated the
   branching ratio under LS coupling (β = 0.395). Porsev's ab-initio calculation
   gives β = 0.360(25), which accounts for spin-orbit coupling and is the value
   the group's own later reference data adopts.

The two empirical lines stand in for the many high-lying states; their shared
frequency and 0.378/0.622 branching are taken from the thesis, and the total
width was refit against the tabulated magic wavelengths after correction 1.

# Accuracy — read before trusting a magic wavelength

Mean deviation from the tabulated magic wavelengths is **3.19 nm** (2.07 nm
excluding the poorly reproduced 532 nm point). That is far worse than the
`¹S₀`/`³P₀` models, and the cause is the data, not the method: unlike the clock
states, `³P₁` has no precision magic- or tune-out-wavelength measurements to
constrain the fit. Treat a predicted `³P₁` magic wavelength as indicative.

Concretely, at the model's own fit targets the differential shift comes out near
but not at zero, and the magic *angle* predicted for `F=1, m_F=0` at 767 nm is
44.3° against the report's 40.9°. Both are the same few-nm-scale error expressed
in different coordinates.

The residual is the **model's**, not the tensor machinery's. AtomTwin's α⁽²⁾ is
`1/(2J+1)` of the notes' — they normalise the reduced matrix elements the other
way (see [`_alpha2_si`](@ref)) — so reproducing the report's own fit residuals
requires that rescaling, and the report's fitted empirical linewidths partly
absorb it. Refitting `γ_eff` against the magic-wavelength table in AtomTwin's
convention would sharpen these predictions; the line list here is the report's
unaltered.

For a tensor model constrained by precision measurements, see
[`SR88_POLARIZABILITY_3P1`](@ref).
"""
const YB174_POLARIZABILITY_3P1 = PolarizabilityModel(
    "3P1",
    [
        # 556 nm line down to the ground state — BELOW ³P₁, hence negative.
        # J′ = 0 is correction (1) above; the thesis has J′ = 1.
        (freq_THz = -539.386800, gamma_MHz = 0.183,  J_f = 0//1),  # (6s²)  ¹S₀
        (freq_THz =  194.778008, gamma_MHz = 0.170,  J_f = 1//1),  # (6s5d) ³D₁
        (freq_THz =  202.657933, gamma_MHz = 0.280,  J_f = 2//1,   # (6s5d) ³D₂
         source = :ls_estimated),
        # γ from Porsev's β = 0.360, not the thesis' LS estimate — correction (2).
        (freq_THz =  440.775408, gamma_MHz = 3.604,  J_f = 1//1),  # (6s7s) ³S₁
        (freq_THz =  654.048602, gamma_MHz = 2.783,  J_f = 1//1,   # (6s6d) ³D₁
         source = :ls_estimated),
        (freq_THz =  654.927593, gamma_MHz = 5.215,  J_f = 2//1,   # (6s6d) ³D₂
         source = :ls_estimated),
        (freq_THz =  708.200713, gamma_MHz = 1.718,  J_f = 1//1,   # (6s8s) ³S₁
         source = :ls_estimated),
        # Effective lines for the high-lying manifold; total width refit here.
        (freq_THz =  778.975785, gamma_MHz = 20.322, J_f = 1//1,   # empirical J′=1
         source = :fitted),
        (freq_THz =  778.975785, gamma_MHz = 33.439, J_f = 2//1,   # empirical J′=2
         source = :fitted),
    ];
    J = 1//1,                       # (6s6p) ³P₁
    offset_Hz_per_Wm2 = 0.0,
    reference = "T. O. Höhn, PhD thesis (2024) App. A, with J′ and (6s7s)³S₁ " *
                "corrected per Schmit-Veiler (2026); Porsev, PRA 60, 2781 (1999)",
)

"""
    YB174_POLARIZABILITY

All Yb-174 polarizability models, keyed by term name.
"""
const YB174_POLARIZABILITY = Dict(
    "1S0" => YB174_POLARIZABILITY_1S0,
    "3P0" => YB174_POLARIZABILITY_3P0,
    "3P1" => YB174_POLARIZABILITY_3P1,
)

# ======================================================================
# Species
# ======================================================================

"""
    Ytterbium174Atom

A Yb-174 atom: `I = 0`, so `F = J` and there is no hyperfine structure.
"""
const Ytterbium174Atom = AtomWrapper{:Ytterbium174}

getpolarizabilitymodels(::Ytterbium174Atom) = YB174_POLARIZABILITY
