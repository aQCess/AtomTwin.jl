"""
Add Rabi couplings between internal levels and manifolds, applying selection rules
and Clebsch–Gordan factors to obtain the correct matrix elements.
"""

"""
    update!(c::GlobalCoupling, ::Val, val)

Set the Rabi rate of a `GlobalCoupling` (`PlanarCoupling`, `Interaction`): every
operator entry is written from the new rate. Called by `compile`/`recompile!` for
parameter bindings.

The entries are set, not rescaled from the previous value: a single-transition
operator has one value on every entry (`rate/2`, or `V` for an interaction, as
built), so this is exact, revives an operator left at rate 0, and leaves a
sampled rate with no memory of the shots that ran before it (serial and threaded
runs of one seed agree bit for bit).
"""
function update!(c::Union{GlobalCoupling, PlanarCoupling, Interaction}, ::Val, val::Number)
    new_rate = ComplexF64(val)
    _set_entries!(c.H, new_rate * _entry_scale(c))
    c.rate = new_rate
end

_entry_scale(::Union{GlobalCoupling, PlanarCoupling}) = 0.5     # H = Ω/2 (|e⟩⟨g| + h.c.)
_entry_scale(::Interaction) = 1.0                                # H = V |..⟩⟨..| (+ h.c.)

"""
    update!(c::GaussianCoupling, ::Val, new_Ω0)

Set the operator entries of `c.H` to a new peak Rabi frequency `new_Ω0` (one value,
`Ω₀/2`, on every entry). Called by `compile_node!` when the atom's position (and
hence Ω₀) has changed between shots; a shot that put the atom beyond the beam's
cutoff (Ω₀ = 0) does not kill the next one.
"""
function update!(c::GaussianCoupling, ::Val, new_Ω0::Number)
    new_Ω0 = ComplexF64(new_Ω0)
    _set_entries!(c.H, new_Ω0 / 2)
    c.Ω0 = new_Ω0
end

function _set_entries!(op::Dynamiq.Op, v::ComplexF64)
    for k in eachindex(op.forward)
        i, j, _ = op.forward[k]
        op.forward[k] = (i, j, v)
    end
    cv = conj(v)
    for k in eachindex(op.reverse)
        i, j, _ = op.reverse[k]
        op.reverse[k] = (i, j, cv)
    end
    return op
end

"""
    _add_coupling!(system, atom, level::Pair{<:AbstractLevel,<:AbstractLevel}, Ω;
                   beam = nothing, noise = nothing, active = true)

Internal helper: add a single coupling DAG node between two specified
atomic sublevels.

Constructs the appropriate node type:
- `PlanarCouplingNode` if `beam isa PlanarBeam`
- `GaussianCouplingNode` for any other beam: `Ω` is the Rabi frequency at the
  beam's reference point (`beam.r0`, the focus), and the coupling follows the
  beam's profile and phase at the atom
- `NoisyCouplingNode` if `noise isa AbstractNoiseModel`
- `CouplingNode` otherwise

`Ω` may be a number, `Parameter`, or `ParametricExpression`; it is stored
in the node and resolved at `play` time. The node is built with the default
(or zero if inactive) coefficient, pushed to `sys.nodes`, and its compiled
field is returned. Returns `nothing` if either level is not found.
"""
function _add_coupling!(
    system,
    atom::AbstractAtom,
    level::Pair{<:AbstractLevel,<:AbstractLevel},
    Ω;
    beam=nothing,
    noise=nothing,
    active=true,
)
    if get(atom.level_indices, level[1], nothing) === nothing ||
       get(atom.level_indices, level[2], nothing) === nothing
        @warn "Level not found in the atom; no coupling added" level_1 = level[1] level_2 = level[2]
        return nothing
    end
    # A beam-derived coupling has no noisy form yet; dropping the model silently
    # gave every shot the same noiseless drive.
    beam !== nothing && noise !== nothing && throw(ArgumentError(
        "add_coupling!: `noise` is not applied to a beam-derived coupling; give the " *
        "noise model without `beam`, or the beam without `noise`."))
    # A parametric beam is not an AbstractBeam: it used to fall through to a plain
    # global coupling, losing the profile, phase and polarization without a word.
    beam isa ParametricBeam && throw(ArgumentError(
        "add_coupling!: a parametric beam (a Parameter waist or power, a MixedPolarization) " *
        "is resolved per shot only by the manifold forms, add_coupling!(sys, atom, " *
        "g_manifold => e_manifold, beam); give a single level pair a concrete beam."))

    node = if beam isa PlanarBeam
        PlanarCouplingNode(Ω, atom, level, beam; active=active)
    elseif beam isa AbstractBeam
        GaussianCouplingNode(atom, level, beam, [0.0, 0.0, 1.0], 0.0, level[1], level[2];
                             Ω0_override = Ω, active = active)
    elseif noise isa AbstractNoiseModel
        NoisyCouplingNode(Ω, atom, level, noise; active=active)
    else
        CouplingNode(Ω, atom, level; active=active)
    end
    build_node!(node, system.basis)
    push!(system, node)
    return node._field
end

"""
    Efield_spherical(beam, r; q_axis=[0,0,1])

Project the lab-frame E field of `beam` at position `r` into the
spherical components (E0, Eplus, Eminus) with respect to `q_axis`.
Returns (E0, Eplus, Eminus) as ComplexF64.
"""
function Efield_spherical(beam, r::AbstractVector{<:Real};
                               q_axis = [0.0, 0.0, 1.0])

    # lab-frame E field at r (V/m), without the beam's amplitude envelope
    Ex, Ey, Ez = AtomTwin.Dynamiq.efield_vector(beam, r)
    E = ComplexF64[Ex, Ey, Ez]

    # --- orthonormal basis aligned with q_axis (shared with the tensor light shift) ---
    exp, eyp, ezp = _quantization_frame(q_axis)

    # project E into primed basis
    Exp = sum(E .* exp)
    Eyp = sum(E .* eyp)
    Ezp = sum(E .* ezp)

    # spherical components in quantization basis {e_x', e_y', e_z'}
    E0     = Ezp
    Eplus  = -(Exp - 1im*Eyp)/sqrt(2)
    Eminus =  (Exp + 1im*Eyp)/sqrt(2)

    return E0, Eplus, Eminus
end

# Magnetic projection of a sublevel along the quantization axis.
_mproj(l::HyperfineLevel) = l.mF
_mproj(l::FineLevel) = l.mJ

function rabi_frequency(
    atom::AbstractAtom,
    g::L,
    e::L,
    beam,
    r::AbstractVector{<:Real};
    q_axis = [0.0, 0.0, 1.0],      # quantization axis in lab frame
    d_red::Float64,                # ⟨J_g || er || J_e⟩_R in C·m
    norm::Bool = false
)::ComplexF64 where {L<:Union{HyperfineLevel,FineLevel}}

    # selection rule
    Δm = _mproj(e) - _mproj(g)
    if abs(Δm) > 1
        return 0.0
    end
    q = Int(Δm)  # 0, ±1

    # spherical components in quantization basis {e_x', e_y', e_z'}
    E0, Eplus, Eminus = Efield_spherical(beam, r; q_axis = q_axis)
    Eq = q == 0 ? E0 : (q == +1 ? Eplus : Eminus)

    # angular factor in units of ⟨J'||er||J⟩_R
    ang = dipole_matrix_element(atom, g, e; norm=norm)

    # physical dipole matrix element (C·m) and complex Rabi frequency
    d_ge = ang * d_red
    return -d_ge * Eq / hbar
end

"""
    reduced_dipole(atom, g, e) -> Float64

The reduced dipole matrix element `|⟨J′‖er‖J⟩|` (C·m) of the E1 line between the terms
of levels `g` and `e` (in either order), from the species' decay rate `A` and line
frequency `ν` ([`getdecayrates`](@ref), [`getlinefrequencies`](@ref)):

    |⟨J′‖er‖J⟩|² = 3π ε₀ ħ c³ (2J′ + 1) A / ω³,    ω = 2πν, J′ the upper level.

It is the `d_red` that `add_coupling!` uses with a beam when none is given.

```julia
S0 = HyperfineManifold(1//2, 0; term = l"1S0")
P1 = HyperfineManifold(3//2, 1; term = l"3P1")
reduced_dipole(Ytterbium171Atom(), S0[1//2], P1[3//2])   # 556 nm line
```
"""
function reduced_dipole(atom::AbstractAtom, g, e)
    rates, freqs = getdecayrates(atom), getlinefrequencies(atom)
    for (u, l) in ((e, g), (g, e))
        key = u.term => l.term
        (haskey(rates, key) && haskey(freqs, key)) || continue
        ω = 2π * freqs[key]
        return sqrt(3π * ε0 * hbar * c^3 * (2u.J + 1) * rates[key] / ω^3)
    end
    error("reduced_dipole: no line data for $(g.term) ↔ $(e.term) on this atom; pass `d_red`.")
end

"""
    rabi_frequencies(beam; q_axis=[0,0,1], r0=nothing, d_red::Float64)

Polarization-resolved Rabi frequencies `(Ω_π, Ω_σ⁺, Ω_σ⁻) = −d_red E_q / ħ` from the
spherical components of the beam's field, for the positional
`add_coupling!(sys, atom, g => e, Ω_π, Ω_σ⁺, Ω_σ⁻)`, which multiplies each sublevel
pair by its Clebsch–Gordan coefficient and nothing else.

`d_red` is therefore the dipole matrix element of the transition whose
Clebsch–Gordan coefficient is 1 — the stretched one — **not** the reduced element
`⟨J′‖d‖J⟩` of [`reduced_dipole`](@ref), which is larger by the Wigner–Eckart factor
(`√(2J′+1)` for a `J = 0 → 1` line). Pass that element through
`rabi_frequencies(atom, g, e, beam)` below, which converts it, or the Hamiltonian
is `√3` too strong on the 556 nm line.

By default evaluates at `beam.r0` (Gaussian beams) or the origin
(`PlanarBeam`, which is spatially uniform).
"""
function rabi_frequencies(
    beam;
    q_axis = [0.0, 0.0, 1.0],
    r0 = hasproperty(beam, :r0) ? beam.r0 : zeros(3),
    d_red::Float64,
)
    E0, Eplus, Eminus = Efield_spherical(beam, r0; q_axis = q_axis)

    Ω_π  = -d_red * E0     / hbar
    Ω_σp = -d_red * Eplus  / hbar
    Ω_σm = -d_red * Eminus / hbar

    return Ω_π, Ω_σp, Ω_σm
end

"""
    rabi_frequencies(atom, g::AbstractManifold, e::AbstractManifold, beam;
                     q_axis = [0,0,1], r0 = nothing, d_red = nothing)

The `(Ω_π, Ω_σ⁺, Ω_σ⁻)` that make the positional
`add_coupling!(sys, atom, g => e, Ω_π, Ω_σ⁺, Ω_σ⁻)` build the Hamiltonian of the
beam-derived `add_coupling!(sys, atom, g => e, beam)`: the reduced element `d_red`
(by default the species' [`reduced_dipole`](@ref)) times the Wigner–Eckart factor
of the `g → e` manifold pair ([`dipole_matrix_element`](@ref) over the
Clebsch–Gordan coefficient, the same for every sublevel pair).
"""
function rabi_frequencies(atom::AbstractAtom, g::M, e::M, beam;
                          q_axis = [0.0, 0.0, 1.0],
                          r0 = hasproperty(beam, :r0) ? beam.r0 : zeros(3),
                          d_red::Union{Float64,Nothing} = nothing,
                          ) where {M<:Union{HyperfineManifold,FineManifold}}
    d = @something d_red reduced_dipole(atom, first(g.levels), first(e.levels))
    K = 0.0
    for gl in g.levels, el in e.levels
        cg = compute_coupling_strength(gl, el, atom, 1.0, 1.0, 1.0)
        abs(cg) > 1e-12 || continue
        K = dipole_matrix_element(atom, gl, el) / cg
        break
    end
    return rabi_frequencies(beam; q_axis, r0, d_red = d * K)
end



"""
    dipole_matrix_element(atom::AbstractAtom, g::HyperfineLevel, e::HyperfineLevel; norm=false)

Return the real-valued electric dipole transition amplitude between two hyperfine states

    ⟨F′ m′ | d_q | F m⟩ in units of the reduced matrix element ⟨J || er || J'⟩_R

in the Racah phase convention (Condon–Shortley phases).

The implementation includes:

  • Zeeman (3j) angular factor
  • Hyperfine recoupling (6j) factor

The phases are the Edmonds/Condon–Shortley ones throughout, so the relative signs
of couplings from two manifolds to a shared one (a Raman pair, a repumper) are
consistent; an element need not be positive. The explicit-Ω forms
(`compute_coupling_strength`) use bare Clebsch–Gordan coefficients instead, so
there the cross-manifold signs are the caller's.

`d_q = ê_q·d` are the spherical components that pair with the field coefficients
`E_q = ê_q*·E` of `Efield_spherical` (E = Σ_q E_q ê_q), so d·E = Σ_q E_q d_q: there is
no (−1)^q, and light polarised along ε̂ excites the dipole along ε̂.

The formula used is

    ⟨F′ m′ | d_q | F m⟩ =
        (-1)^(F′ - m′)
        ( F′  1   F
         -m′  q   m )
        (-1)^(F′ + J + 1 + I)
        √[(2F′+1)(2F+1)]
        { J′  F′  I
          F   J   1 }

where

    q = m′ - m
    |q| ≤ 1

The reduced fine-structure matrix element ⟨J′||d||J⟩ is set to 1,
so the function returns relative transition amplitudes.

Keyword
-------
norm=false :
    If true and a stretched σ⁺ transition exists,
    the result is normalized so that the stretched
    transition equals +1.

Returns
-------
A real Float64 amplitude in Racah convention.
Forbidden transitions return 0.0.
"""
dipole_matrix_element(atom::AbstractAtom, g::HyperfineLevel, e::HyperfineLevel; norm=false) =
    _e1_angular(g.F, g.mF, g.J, e.F, e.mF, e.J, atom.I; norm)

# A fine-structure level is the I = 0 case, F = J: the 6j reduces to the J-level
# Wigner–Eckart factor.
dipole_matrix_element(atom::AbstractAtom, g::FineLevel, e::FineLevel; norm=false) =
    _e1_angular(g.J, g.mJ, g.J, e.J, e.mJ, e.J, 0//1; norm)

function _e1_angular(F, m, J, Fp, mp, Jp, I; norm=false)
    q = mp - m
    abs(q) > 1 && return 0.0

    # --- 3j ---
    threej = (-1)^(Fp - mp) *
             wigner3j(Fp, 1, F,
                      -mp, q, m)

    threej == 0 && return 0.0

    # --- 6j ---
    sixj = wigner6j(Jp, Fp, I,
                    F,  J,  1)

    pref = (-1)^(Fp + J + 1 + I) *
           sqrt((2Fp + 1)*(2F + 1))

    amp = threej * pref * sixj

    # The phases above are one convention for every (F, F′) pair, which is what
    # two pairs sharing a manifold need: a sign fixed pair by pair (say, to make
    # each stretched σ⁺ element positive) is not a basis change and flips the
    # interference of the two paths through the shared manifold. `norm` divides
    # by the magnitude of the stretched σ⁺ element, when the pair has one.
    if norm
        mps = F + 1
        if abs(mps) ≤ Fp
            threej_s = wigner3j(Fp, 1, F, -mps, 1, F)
            threej_s == 0 || (amp /= abs(threej_s * pref * sixj))
        end
    end

    return amp == 0 ? 0.0 : amp
end

"""
    compute_coupling_strength(g::HyperfineLevel, e::HyperfineLevel,
                              atom, Ω_π, Ω_σ⁺, Ω_σ⁻)

Compute the electric-dipole coupling strength between two hyperfine
levels `g` and `e` (F↔F) for given polarization-resolved Rabi
frequencies.

The appropriate Clebsch–Gordan coefficient
`⟨F_g, m_Fg; 1, Δm | F_e, m_Fe⟩` is evaluated using `clebschgordan`,
and multiplied by `Ω_π`, `Ω_σ⁺`, or `Ω_σ⁻` depending on
`Δm = m_Fe − m_Fg ∈ {0, ±1}`. Returns `0.0` if the transition is
forbidden. A Rabi frequency that cannot be multiplied by the coefficient (a
`Function`, say) is an error, not a zero coupling.
"""
function compute_coupling_strength(g::HyperfineLevel, e::HyperfineLevel, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
    Δm = e.mF - g.mF
    abs(Δm) > 1 && return 0.0
    cg = clebschgordan(g.F, g.mF, 1, Δm, e.F, e.mF)
    return Δm == 0  ? Ω_π * cg :
           Δm == +1 ? Ω_σ⁺ * cg :
                      Ω_σ⁻ * cg
end


"""
    compute_coupling_strength(g::FineLevel, e::FineLevel,
                              atom, Ω_π, Ω_σ⁺, Ω_σ⁻)

Compute the electric-dipole coupling strength between two fine-structure
levels `g` and `e` (J↔J) for given polarization-resolved Rabi
frequencies.

Uses the Clebsch–Gordan coefficient
`⟨J_g, m_Jg; 1, Δm | J_e, m_Je⟩` and selects `Ω_π`, `Ω_σ⁺`, or `Ω_σ⁻`
according to `Δm = m_Je − m_Jg`. Returns `0.0` for forbidden
transitions (`|Δm| > 1`).
"""
function compute_coupling_strength(g::FineLevel, e::FineLevel, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
    Δm = e.mJ - g.mJ
    abs(Δm) > 1 && return 0.0
    cg = clebschgordan(g.J, g.mJ, 1, Δm, e.J, e.mJ)
    return Δm == 0  ? Ω_π * cg :
           Δm == +1 ? Ω_σ⁺ * cg :
                      Ω_σ⁻ * cg
end

"""
    compute_coupling_strength(g::HyperfineLevel, e::FineLevel,
                              atom, Ω_π, Ω_σ⁺, Ω_σ⁻)

Compute the effective electric-dipole coupling strength from a hyperfine
level `g` (F,m_F) to a fine-structure level `e` (J, unresolved
hyperfine) by summing over compatible projections.

The algorithm decomposes the hyperfine state into `|J_g, m_J; I, m_I⟩`
components using a `clebschgordan` factor, then applies dipole
Clebsch–Gordan coefficients for `J_g → J_e` with
`Δm = m_Je − m_J`. Each `Δm = 0, ±1` reaches the fine level in a different
nuclear-spin state `|I, m_F − m_J⟩`; those are orthogonal and do not interfere, so
the contributions `Ω_q c_hyper c_dipole` add in quadrature (the nuclear spin is
traced over). With one contributing polarization the result is that term, sign and
all, and may be parametric; with several it is the positive quadrature sum of
numbers (a parametric Rabi frequency then needs the polarizations split into
separate couplings).
"""
function compute_coupling_strength(g::HyperfineLevel, e::FineLevel, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
    F, mF, Jg = g.F, g.mF, g.J
    Jp, mJp = e.J, e.mJ
    I = atom.I
    terms = Any[]
    for m_J in -Jg:1:Jg
        m_I = mF - m_J
        abs(m_I) > I && continue
        Δm = mJp - m_J
        abs(Δm) > 1 && continue
        c_hyper  = clebschgordan(Jg, m_J, I, m_I, F, mF)
        c_dipole = clebschgordan(Jg, m_J, 1, Δm, Jp, mJp)
        c_hyper * c_dipole == 0 && continue
        push!(terms, (Δm == 0  ? Ω_π  :
                      Δm == +1 ? Ω_σ⁺ : Ω_σ⁻) * c_hyper * c_dipole)
    end
    return _incoherent_sum(terms)
end

# Amplitudes into orthogonal final states: one term is itself, several add in quadrature.
function _incoherent_sum(terms)
    filter!(t -> !(t isa Number && t == 0), terms)     # a polarization not driven
    isempty(terms) && return 0.0
    length(terms) == 1 && return terms[1]
    all(t -> t isa Number, terms) ||
        throw(ArgumentError("a parametric Rabi frequency reaches this fine level through " *
                            "several polarizations, which add in quadrature; give each " *
                            "polarization its own coupling"))
    return sqrt(sum(abs2, terms))
end

"""
    compute_coupling_strength(g::FineLevel, e::HyperfineLevel,
                              atom, Ω_π, Ω_σ⁺, Ω_σ⁻)

Compute the effective coupling strength from a fine-structure level `g`
(J, m_J, hyperfine unresolved) to a hyperfine level `e` (F, m_F).

The mirror of the hyperfine → fine method: `e` is decomposed into
`|J_e, m_J′; I, m_I⟩` components, and `g` couples to each with the dipole
Clebsch–Gordan coefficient `⟨J_g, m_J; 1, Δm | J_e, m_J′⟩`, `Δm = m_J′ − m_J`;
the contributions of different polarizations start from orthogonal nuclear-spin
states and add in quadrature.
"""
function compute_coupling_strength(g::FineLevel, e::HyperfineLevel, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
    Jg, mJg = g.J, g.mJ
    F, mF, Je = e.F, e.mF, e.J
    I = atom.I
    terms = Any[]
    for m_J in -Je:1:Je
        m_I = mF - m_J
        abs(m_I) > I && continue
        Δm = m_J - mJg
        abs(Δm) > 1 && continue
        c_hyper  = clebschgordan(Je, m_J, I, m_I, F, mF)
        c_dipole = clebschgordan(Jg, mJg, 1, Δm, Je, m_J)
        c_hyper * c_dipole == 0 && continue
        push!(terms, (Δm == 0  ? Ω_π  :
                      Δm == +1 ? Ω_σ⁺ : Ω_σ⁻) * c_hyper * c_dipole)
    end
    return _incoherent_sum(terms)
end


"""
    compute_coupling_strength(g::AbstractLevel, e::AbstractLevel,
                              atom, Ω_π, Ω_σ⁺, Ω_σ⁻)

Fallback coupling-strength implementation.

For level types without a more specific method, returns `0.0`, meaning
no allowed dipole coupling is assumed.
"""
function compute_coupling_strength(g::AbstractLevel, e::AbstractLevel, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
    return 0.0
end


"""
    _add_couplings(system, atom, g_levels, e_levels,
                   Ω_π, Ω_σ⁺, Ω_σ⁻, noise, active, tol; beam = nothing)

Internal helper: add all allowed couplings between two sets of sublevels.

For each pair `(g, e)` in `g_levels × e_levels`, the corresponding
matrix element is computed via `compute_coupling_strength`. Pairs with
`abs(Ω) < tol` are skipped. Remaining couplings are created via
`_add_coupling!` and collected into a vector of `Dynamiq.AbstractField`
instances.

Returns the vector of constructed coupling fields.
"""
function _add_couplings(
    system, atom, g_levels, e_levels,
    Ω_π, Ω_σ⁺, Ω_σ⁻, noise, active, tol; beam=nothing
)
    couplings = Dynamiq.AbstractField[]
    for g in g_levels, e in e_levels
        Ω = compute_coupling_strength(g, e, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
        (abs(Ω) isa Number && abs(Ω) < tol) && continue
        coupling = _add_coupling!(system, atom, g=>e, Ω; beam=beam, noise=noise, active=active)
        isnothing(coupling) || push!(couplings, coupling)
    end
    return couplings
end

############## BEAM Ω VALUE TYPE ##############

"""
    BeamRabiFrequency

A value type that computes a Rabi frequency from a beam's E-field at the
atom's position. Used as the `Ω` field of a `CouplingNode`.

Holds a reference to a `BeamNode` whose `_compiled[]` field is populated
before this value is evaluated (insertion-order guarantee in `sys.nodes`).
Atom position is read from `atom.inner.x`, which is updated in-place by
`initialize!` before coupling nodes are compiled.

Future derived-value types (e.g. `BeamACStarkShift` for light-shift detunings)
follow the same pattern: store physics inputs, implement `_resolve_node_default`
and `_resolve_node_value`.
"""
struct BeamRabiFrequency
    beam_node::BeamNode
    atom::AbstractAtom
    g::AbstractLevel
    e::AbstractLevel
    q_axis::Any                          # see `_axis_of`
    d_red::Float64
    x0::Union{Nothing, Vector{Float64}}  # where the field is read: the atom's position, or a fixed point
end

BeamRabiFrequency(beam_node, atom, g, e, q_axis, d_red) =
    BeamRabiFrequency(beam_node, atom, g, e, q_axis, d_red, nothing)

_extract_beam_node_deps(f::BeamRabiFrequency) = AbstractNode[f.beam_node]
# Bound to the system's axis node by `compile` (see `_bind_quantization_axis!`).
_bound_to_axis(f::BeamRabiFrequency, axis) = f.q_axis === nothing ?
    BeamRabiFrequency(f.beam_node, f.atom, f.g, f.e, axis, f.d_red, f.x0) : f

_resolve_node_default(f::BeamRabiFrequency) =
    ComplexF64(rabi_frequency(f.atom, f.g, f.e, f.beam_node._compiled[], something(f.x0, f.atom.x);
                              q_axis = _axis_of(f.q_axis), d_red = f.d_red))

_resolve_node_value(f::BeamRabiFrequency, param_values, rng) = _resolve_node_default(f)

############## PUBLIC USER API (DISPATCHES) ##############

"""
    add_coupling!(system, atom,
                  level::Pair{HyperfineManifold,HyperfineManifold},
                  Ω_π, Ω_σ⁺, Ω_σ⁻;
                  noise = nothing, active = true, tol = 1e-10, beam = nothing)

Add all allowed electric-dipole couplings between the sublevels of two
hyperfine manifolds.

The manifolds `ground` and `excited` are expanded into their constituent
levels (`.levels`), selection rules and Clebsch–Gordan coefficients are
applied via `compute_coupling_strength`, and any coupling with magnitude
below `tol` is discarded. Optional `noise` wraps each coupling in a
`NoisyField`. Returns a vector of the created couplings.
"""
function add_coupling!(
    system, atom::AbstractAtom,
    level::Pair{HyperfineManifold, HyperfineManifold},
    Ω_π, Ω_σ⁺, Ω_σ⁻; noise=nothing, active=true, tol=1e-10, beam=nothing
)
    ground, excited = level
    _add_couplings(system, atom, ground.levels, excited.levels,
        Ω_π, Ω_σ⁺, Ω_σ⁻, noise, active, tol; beam=beam)
end

function add_coupling!(
    system, atom::AbstractAtom,
    level::Pair{M, M},
    beam;
    q_axis::Union{Nothing, AbstractVector{<:Real}} = nothing,
    d_red::Union{Float64, Nothing} = nothing,
    noise=nothing, active=true, tol=1e-10,
) where {M<:Union{HyperfineManifold, FineManifold}}
    ground, excited = level
    _check_e1_allowed(first(ground.levels), first(excited.levels))
    # A concrete beam: couplings that follow its field at the atom.
    beam isa AbstractBeam && return add_coupling!(system, atom, level; beam, q_axis, d_red, active, tol)
    noise === nothing || throw(ArgumentError(
        "add_coupling!: `noise` is not applied to a beam-derived coupling; give the " *
        "noise model without `beam`, or the beam without `noise`."))
    # A parametric beam is resolved per shot (`BeamNode`), its Rabi frequencies at the
    # atom's initial position. (`@something`: the species' line data is looked up only
    # when no `d_red` is given.)
    d_red = @something d_red reduced_dipole(atom, first(ground.levels), first(excited.levels))
    q_axis_vec = q_axis === nothing ? nothing : Vector{Float64}(q_axis)
    couplings = Dynamiq.AbstractField[]

    # Create a single BeamNode for this beam (shared across all sublevel pairs)
    beam_node = BeamNode(beam)
    build_node!(beam_node)
    push!(system, beam_node)

    for g in ground.levels, e in excited.levels
        Δm = _mproj(e) - _mproj(g)
        abs(Δm) > 1 && continue  # selection rule, independent of beam params

        Ω = BeamRabiFrequency(beam_node, atom, g, e, q_axis_vec, d_red)
        node = CouplingNode(Ω, atom, g => e; active=active)
        build_node!(node, system.basis)
        push!(system, node)
        push!(couplings, node._field)
    end

    return couplings
end

"""
    add_coupling!(system, atom, level::Pair{HyperfineManifold,HyperfineManifold};
                  beam=nothing, Ω_π=nothing, Ω_p=nothing, Ω_m=nothing,
                  d_red=nothing, q_axis = nothing, active=true, tol=1e-10)

Add electric-dipole couplings between hyperfine manifolds, or between fine-structure
manifolds (`Pair{FineManifold,FineManifold}`, the `I = 0` case with `F = J`). A mixed
pair (a hyperfine ground manifold to a fine-structure excited one) has no
beam-derived form: give it explicit `Ω_π, Ω_σ⁺, Ω_σ⁻`.

- If only `beam` is given: Rabi frequencies are derived from the beam E-field
  via `d_red`, by default the species' value for the line ([`reduced_dipole`](@ref)).
  So `add_coupling!(sys, yb, S0 => P1; beam)` drives the ¹⁷¹Yb 556 nm line with the
  beam's intensity and polarization. The positional form `add_coupling!(sys, yb,
  S0 => P1, beam)` is the same, and also takes a parametric beam (a `Parameter`
  power or a `MixedPolarization`), resolved per shot with its Rabi frequencies at
  the atom's initial position.
- If `Ω_π`/`Ω_p`/`Ω_m` are given without `beam`: uniform `GlobalCoupling`s
  are created with those amplitudes; a polarization left out is not driven.
- If both `beam` and `Ω_π`/`Ω_p`/`Ω_m` are given: the explicit values replace the
  beam-derived ones (useful when selection rules underestimate the effective
  coupling, as for a hyperfine-mixed clock line). They are the Rabi frequencies
  at the beam's reference point (its focus); the coupling follows the beam's
  profile and phase at the atom.
- If neither `beam` nor any Ω is provided: an error is raised.

The beam's polarization is decomposed into π and σ± about the system's quantization
axis ([`add_quantization_axis!`](@ref), ẑ if none; it may be set after the coupling
and may be a `Parameter`), the axis the Zeeman and tensor light shifts use. `q_axis`
fixes a different axis for this coupling alone.

With a beam the couplings follow the field at the atom: a `PlanarBeam` gives
`PlanarCoupling`s (phase `exp(ik·x)`, and the radiation pressure and recoil of the
semiclassical motion), any other beam `GaussianCoupling`s (its profile and phase).

`Ω_π`, `Ω_p` and `Ω_m` are polarization-resolved, exactly as the positional
`Ω_π, Ω_σ⁺, Ω_σ⁻`: each pair takes its Clebsch–Gordan coefficient
`⟨F m; 1 Δm | F′ m′⟩`, so the two forms build the same Hamiltonian. Each may be a
number, `Parameter` or `ParametricExpression`.
"""
function add_coupling!(
    system, atom::AbstractAtom,
    level::Pair{M, M};
    beam::Union{AbstractBeam, Nothing} = nothing,
    Ω_π = nothing,
    Ω_p  = nothing,
    Ω_m  = nothing,
    q_axis::Union{Nothing, AbstractVector{<:Real}} = nothing,
    d_red::Union{Float64, Nothing} = nothing,
    active = true,
    tol = 1e-10,
) where {M<:Union{HyperfineManifold, FineManifold}}
    beam === nothing && Ω_π === nothing && Ω_p === nothing && Ω_m === nothing &&
        error("add_coupling!: provide at least one of `beam` or `Ω_π`/`Ω_p`/`Ω_m`")

    ground, excited = level
    q_axis_vec = q_axis === nothing ? nothing : Vector{Float64}(q_axis)
    override_map = Dict{Int,Any}(0 => Ω_π, 1 => Ω_p, -1 => Ω_m)
    couplings = Dynamiq.AbstractField[]
    # A plane wave's couplings read the field at the origin (`PlanarCoupling` adds
    # exp(ik·x)); one node resolves it, shared by every sublevel pair.
    beam_node = beam isa PlanarBeam ? BeamNode(beam) : nothing
    beam_node === nothing || (build_node!(beam_node); push!(system, beam_node))

    for g in ground.levels, e in excited.levels
        Δm = _mproj(e) - _mproj(g)
        abs(Δm) > 1 && continue
        override = override_map[Δm]

        if override === nothing
            beam === nothing && continue            # this polarization is not driven
            _check_e1_allowed(g, e)
            d_red = @something d_red reduced_dipole(atom, g, e)
            Ω = nothing
            ## the Rabi frequency at the beam's reference point: the origin for a plane
            ## wave, the focus for a Gaussian beam (the coupling follows the profile at
            ## the atom, wherever `compile` puts it)
            x0 = beam isa PlanarBeam ? zeros(3) : Vector{Float64}(beam.r0)
            Ω0 = rabi_frequency(atom, g, e, beam, x0; q_axis = _axis_of(q_axis_vec), d_red = d_red)
        else
            Ω  = compute_coupling_strength(g, e, atom, override, override, override)
            Ω0 = _resolve_node_default(Ω)
        end
        # `tol` drops a pair the polarization or selection rules make negligible, never
        # a parametric Ω whose default happens to be 0, nor a pair whose strength
        # depends on an axis the system may still set or sample: those are knobs
        # `compile` and `play` set later.
        fixed = override === nothing ? q_axis_vec !== nothing : override isa Number
        fixed && abs(Ω0) < tol && continue

        if beam isa PlanarBeam
            Ωp = Ω === nothing ? BeamRabiFrequency(beam_node, atom, g, e, q_axis_vec, d_red, zeros(3)) : Ω
            push!(couplings, _add_coupling!(system, atom, g => e, Ωp; beam, active))
        elseif beam !== nothing
            node = GaussianCouplingNode(atom, g => e, beam, q_axis_vec,
                                        something(d_red, 0.0), g, e;
                                        Ω0_override = Ω, active = active)
            build_node!(node, system.basis)
            push!(system, node)
            push!(couplings, node._field)
        else
            push!(couplings, _add_coupling!(system, atom, g => e, Ω; active = active))
        end
    end
    return couplings
end

# A beam-derived Rabi frequency carries the J-level 6j symbol {J′ F′ I; F J 1},
# which vanishes for J = J′ = 0 and for |ΔJ| > 1: every pair would get Ω = 0 and be
# dropped without a word. The ¹⁷¹Yb/⁸⁷Sr clock line is E1 only through hyperfine
# mixing, which this model does not compute, so the user must supply the strength.
function _check_e1_allowed(g, e)
    (hasproperty(g, :J) && hasproperty(e, :J)) || return nothing
    if g.J == 0 && e.J == 0
        throw(ArgumentError(
            "add_coupling!: $(g.label) → $(e.label) is J = 0 → 0, E1-forbidden at the J level, " *
            "so a beam-derived Rabi frequency is identically zero. For a clock line made " *
            "allowed by hyperfine mixing (¹⁷¹Yb, ⁸⁷Sr) give the polarization-resolved Rabi " *
            "frequencies: add_coupling!(sys, atom, g => e, Ω_π, Ω_σ⁺, Ω_σ⁻; beam), or the " *
            "Ω_π/Ω_p/Ω_m keywords with `beam`."))
    end
    abs(g.J - e.J) <= 1 || throw(ArgumentError(
        "add_coupling!: $(g.label) → $(e.label) has |ΔJ| = $(abs(g.J - e.J)) > 1: not an " *
        "E1 line, so a beam-derived Rabi frequency is identically zero."))
    if hasproperty(g, :F) && hasproperty(e, :F)
        abs(g.F - e.F) <= 1 || throw(ArgumentError(
            "add_coupling!: $(g.label) → $(e.label) has |ΔF| = $(abs(g.F - e.F)) > 1: not an " *
            "E1 line, so a beam-derived Rabi frequency is identically zero."))
    end
    return nothing
end


"""
    add_coupling!(system, atom,
                  level::Pair{FineManifold,FineManifold},
                  Ω_π, Ω_σ⁺, Ω_σ⁻;
                  noise = nothing, active = true, tol = 1e-10, beam = nothing)

Add all allowed electric-dipole couplings between sublevels of two
fine-structure manifolds, using polarization-resolved Rabi frequencies
`Ω_π`, `Ω_σ⁺`, and `Ω_σ⁻`.

Dispatches to `_add_couplings` with the fine-manifold level sets.
"""
function add_coupling!(
    system, atom::AbstractAtom,
    level::Pair{FineManifold, FineManifold},
    Ω_π, Ω_σ⁺, Ω_σ⁻; noise=nothing, active=true, tol=1e-10, beam=nothing
)
    ground, excited = level
    _add_couplings(system, atom, ground.levels, excited.levels,
        Ω_π, Ω_σ⁺, Ω_σ⁻, noise, active, tol; beam=beam)
end

"""
    add_coupling!(system, atom,
                  level::Pair{HyperfineManifold,FineManifold},
                  Ω_π, Ω_σ⁺, Ω_σ⁻;
                  noise = nothing, active = true, tol = 1e-10, beam = nothing)

Add all allowed electric-dipole couplings between a hyperfine manifold
(typically ground) and a fine-structure or Rydberg manifold (excited).

Expands both manifolds to their sublevels and uses
`compute_coupling_strength` to include appropriate CG weights and
selection rules for each `(g,e)` pair.
"""
function add_coupling!(
    system, atom::AbstractAtom,
    level::Pair{HyperfineManifold, FineManifold},
    Ω_π, Ω_σ⁺, Ω_σ⁻; noise=nothing, active=true, tol=1e-10, beam=nothing
)
    ground, excited = level
    _add_couplings(system, atom, ground.levels, excited.levels,
        Ω_π, Ω_σ⁺, Ω_σ⁻, noise, active, tol; beam=beam)
end

"""
    add_coupling!(system, atom,
                  level::Pair{FineManifold,HyperfineManifold},
                  Ω_π, Ω_σ⁺, Ω_σ⁻;
                  noise = nothing, active = true, tol = 1e-10, beam = nothing)

Add all allowed electric-dipole couplings from a fine-structure manifold
(ground) to a hyperfine manifold (excited).

This is the manifold-level counterpart of the mixed-level
`compute_coupling_strength` method and delegates to `_add_couplings`.
"""
function add_coupling!(
    system, atom::AbstractAtom,
    level::Pair{FineManifold, HyperfineManifold},
    Ω_π, Ω_σ⁺, Ω_σ⁻; noise=nothing, active=true, tol=1e-10, beam=nothing
)
    ground, excited = level
    _add_couplings(system, atom, ground.levels, excited.levels,
        Ω_π, Ω_σ⁺, Ω_σ⁻, noise, active, tol; beam=beam)
end

"""
    add_coupling!(system, atom,
                  level::Pair{<:AbstractLevel,<:AbstractLevel},
                  Ω_π, Ω_σ⁺, Ω_σ⁻;
                  beam = nothing, noise = nothing, active = true, tol = 1e-10)

Add a selection-rule–aware electric-dipole coupling between two specific
sublevels `g` and `e`.

The effective Rabi frequency `Ω` is computed via
`compute_coupling_strength(g, e, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)`. If
`abs(Ω) < tol`, no coupling is added and an empty `GlobalCoupling[]`
vector is returned. Otherwise, a single coupling is created via
`_add_coupling!` and returned in a one-element `GlobalCoupling` array.
"""
function add_coupling!(
    system, atom::AbstractAtom,
    level::Pair{<:AbstractLevel,<:AbstractLevel},
    Ω_π, Ω_σ⁺, Ω_σ⁻; beam=nothing, noise=nothing, active=true, tol=1e-10
)
    g, e = level
    Ω = compute_coupling_strength(g, e, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
    abs(Ω) < tol && return GlobalCoupling[]
    coupling = _add_coupling!(system, atom, g=>e, Ω; beam=beam, noise=noise, active=active)
    return isnothing(coupling) ? GlobalCoupling[] : GlobalCoupling[coupling]
end

"""
    add_coupling!(system, atom,
                  level::Pair{<:AbstractLevel,<:AbstractLevel},
                  Ω;
                  beam = nothing, noise = nothing, active = true, tol = 1e-10)

Add a direct Rabi coupling of strength `Ω` between two specific levels,
without applying selection rules or Clebsch–Gordan coefficients.

This is useful for effective two-level models or custom matrix elements
where `Ω` has already been computed externally. If `Ω` is a number with
`abs(Ω) < tol`, no coupling is added and `GlobalCoupling[]` is returned.
Otherwise `_add_coupling!` is invoked and either the created coupling
or an empty vector is returned.
"""
function add_coupling!(
    system, atom::AbstractAtom,
    level::Pair{<:AbstractLevel,<:AbstractLevel},
    Ω; beam=nothing, noise=nothing, active=true, tol=1e-10
)
    if Ω isa Number
        if abs(Ω) < tol
            return GlobalCoupling[]
        end
    end
    coupling = _add_coupling!(system, atom, level, Ω; beam=beam, noise=noise, active=active)
    return isnothing(coupling) ? GlobalCoupling[] : coupling
end


add_coupling!(system, atoms::Vector{<:AbstractAtom}, args...; kwargs...) = [add_coupling!(system, atom, args...; kwargs...) for atom in atoms]
