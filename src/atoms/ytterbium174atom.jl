"""
    ytterbium174atom.jl

Yb-174 polarizability data — the spin-zero (bosonic) ytterbium isotope.

`I = 0`, so there is no hyperfine structure and `F = J`. That does **not** remove
the tensor light shift: `³P₁` has `J = 1`, so its three `m_J` sublevels are split
by a trap unless the polarisation sits at the magic angle. This is the isotope the
767 nm trap-depth spectroscopy example uses.

The electronic structure is shared with Yb-171 — only the nuclear spin and mass
differ — so the `¹S₀`, `³P₀` and `³P₁` line lists are the Yb-171 ones.

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

Term symbols for Yb-174. Usually written `l"3P1"`; see [`AtomTwin.Ytterbium171`](@ref).
"""
module Ytterbium174
import ..TermSymbol, ..TERM_REGISTRY, ..@term
@term "1S0" 0//1      # (6s²)  ¹S₀  ground
@term "3P0" 0//1      # (6s6p) ³P₀  clock
@term "3P1" 1//1      # (6s6p) ³P₁  556 nm intercombination
@term "1P1" 1//1      # (6s6p) ¹P₁  399 nm (no shipped polarizability model)
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

The (6s6p) ³P₁ model, shared with Yb-171. See [`YB171_POLARIZABILITY_3P1`](@ref).
"""
const YB174_POLARIZABILITY_3P1 = YB171_POLARIZABILITY_3P1

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
