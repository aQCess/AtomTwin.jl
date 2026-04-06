"""
Add Rabi couplings between internal levels and manifolds, applying selection rules
and Clebsch–Gordan factors to obtain the correct matrix elements.
"""

"""
    update!(c::GlobalCoupling, ::Val, val)

Update the Rabi rate of a `GlobalCoupling` by rescaling its operator entries.
Called by `compile`/`recompile!` for parameter bindings.
"""
function update!(c::GlobalCoupling, ::Val, val::Number)
    new_rate = ComplexF64(val)
    if c.rate != 0
        scale = new_rate / c.rate
        for k in eachindex(c.H.forward)
            i, j, v = c.H.forward[k]
            c.H.forward[k] = (i, j, v * scale)
        end
        for k in eachindex(c.H.reverse)
            i, j, v = c.H.reverse[k]
            c.H.reverse[k] = (i, j, v * scale)
        end
    end
    c.rate = new_rate
end

"""
    update!(c::GaussianCoupling, ::Val, new_Ω0)

Rescale the operator entries of `c.H` to reflect a new peak Rabi frequency `new_Ω0`.
Called by `compile_node!` when the atom's position (and hence Ω₀) has changed between shots.
"""
function update!(c::GaussianCoupling, ::Val, new_Ω0::Number)
    new_Ω0 = ComplexF64(new_Ω0)
    if c.Ω0 != 0
        scale = new_Ω0 / c.Ω0
        for k in eachindex(c.H.forward)
            i, j, v = c.H.forward[k]
            c.H.forward[k] = (i, j, v * scale)
        end
        for k in eachindex(c.H.reverse)
            i, j, v = c.H.reverse[k]
            c.H.reverse[k] = (i, j, v * scale)
        end
    end
    c.Ω0 = new_Ω0
end

"""
    _add_coupling!(system, atom, level::Pair{<:AbstractLevel,<:AbstractLevel}, Ω;
                   beam = nothing, noise = nothing, active = true)

Internal helper: add a single coupling DAG node between two specified
atomic sublevels.

Constructs the appropriate node type:
- `PlanarCouplingNode` if `beam isa PlanarBeam`
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
        @warn "Level not found in atom.level_indices" level_1=level[1], level_2=level[2], atom=atom
        return nothing
    end

    node = if beam isa PlanarBeam
        PlanarCouplingNode(Ω, atom, level, beam; active=active)
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

    # lab-frame E field at r (V/m)
    Ex, Ey, Ez = AtomTwin.Dynamiq.Efield(beam, r)
    E = ComplexF64[Ex, Ey, Ez]

    # --- orthonormal basis aligned with q_axis (same as rabi_frequency) ---
    ezp = normalize(collect(q_axis))
    ref  = abs(ezp[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    exp  = normalize(ref - dot(ref, ezp)*ezp)   # Gram-Schmidt
    eyp  = cross(ezp, exp)

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

function rabi_frequency(
    atom::AbstractAtom,
    g::HyperfineLevel,
    e::HyperfineLevel,
    beam,
    r::AbstractVector{<:Real};
    q_axis = [0.0, 0.0, 1.0],      # quantization axis in lab frame
    d_red::Float64,                # ⟨J_g || er || J_e⟩_R in C·m
    norm::Bool = false
)::ComplexF64

    # selection rule
    Δm = e.mF - g.mF
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
    rabi_frequencies(beam; q_axis=[0,0,1], r0=beam.r0, d_red::Float64)

Polarization-resolved Rabi frequencies for an effective E1 operator,
independent of J/F. Returns (Ω_π, Ω_σ⁺, Ω_σ⁻) computed from the
spherical components of the beam E-field and a reduced dipole `d_red`.

By default it returns the peak Rabi frequencies (at beam center).
"""
function rabi_frequencies(
    beam;
    q_axis = [0.0, 0.0, 1.0],
    r0 = beam.r0,
    d_red::Float64,
)
    E0, Eplus, Eminus = Efield_spherical(beam, r0; q_axis = q_axis)

    Ω_π  = -d_red * E0     / hbar
    Ω_σp = -d_red * Eplus  / hbar
    Ω_σm = -d_red * Eminus / hbar

    return Ω_π, Ω_σp, Ω_σm
end



"""
    dipole_matrix_element(atom::AbstractAtom, g::HyperfineLevel, e::HyperfineLevel; norm=false)

Return the real-valued electric dipole transition amplitude between two hyperfine states

    ⟨F′ m′ | d_q | F m⟩ in units of the reduced matrix element ⟨J || er || J'⟩_R

in the Racah phase convention (Condon–Shortley phases).

The implementation includes:

  • Zeeman (3j) angular factor  
  • Hyperfine recoupling (6j) factor  
  • The spherical tensor phase factor (-1)^q  
  • A global phase choice such that a stretched σ⁺ transition
    (if it exists) is positive

The formula used is

    ⟨F′ m′ | d_q | F m⟩ =
        (-1)^q
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
function dipole_matrix_element(atom::AbstractAtom, g::HyperfineLevel, e::HyperfineLevel; norm=false)

    F,m = g.F,g.mF
    Fp,mp = e.F,e.mF
    J,Jp = g.J, e.J
    I = atom.I

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

    amp = (-1)^q * threej * pref * sixj

    # ---- Fix global phase via stretched σ⁺ ----
    ms = F
    qs = 1
    mps = ms + qs

    if abs(mps) ≤ Fp
        threej_s = (-1)^(Fp - mps) *
                   wigner3j(Fp, 1, F,
                            -mps, qs, ms)

        if threej_s != 0
            sixj_s = sixj
            pref_s = pref
            stretched = (-1)^qs * threej_s * pref_s * sixj_s

            if stretched != 0
                amp *= sign(stretched)
                norm && (amp /= abs(stretched))
            end
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
forbidden or if the CG coefficient cannot be evaluated.
"""
function compute_coupling_strength(g::HyperfineLevel, e::HyperfineLevel, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
    Δm = e.mF - g.mF
    abs(Δm) > 1 && return 0.0
    try
        cg = clebschgordan(g.F, g.mF, 1, Δm, e.F, e.mF)

        return Δm == 0  ? Ω_π * cg :
               Δm == +1 ? Ω_σ⁺ * cg :
               Δm == -1 ? Ω_σ⁻ * cg : 0.0
    catch
        return 0.0
    end
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
transitions (`|Δm| > 1`) or evaluation failures.
"""
function compute_coupling_strength(g::FineLevel, e::FineLevel, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
    Δm = e.mJ - g.mJ
    abs(Δm) > 1 && return 0.0
    try
        cg = clebschgordan(g.J, g.mJ, 1, Δm, e.J, e.mJ)
        return Δm == 0  ? Ω_π * cg :
               Δm == +1 ? Ω_σ⁺ * cg :
               Δm == -1 ? Ω_σ⁻ * cg : 0.0
    catch
        return 0.0
    end
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
`Δm = m_Je − m_J`. Contributions for `Δm = 0, ±1` are weighted by
`Ω_π`, `Ω_σ⁺`, `Ω_σ⁻` respectively and accumulated. Returns the total
effective Rabi frequency.
"""
function compute_coupling_strength(g::HyperfineLevel, e::FineLevel, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
    F, mF, Jg = g.F, g.mF, g.J
    Jp, mJp = e.J, e.mJ
    I = atom.I
    total = 0.0
    for m_J in -Jg:1:Jg
        m_I = mF - m_J
        abs(m_I) > I && continue
        try
            c_hyper = clebschgordan(Jg, m_J, I, m_I, F, mF)
            Δm = mJp - m_J
            abs(Δm) > 1 && continue
            c_dipole = clebschgordan(Jg, m_J, 1, Δm, Jp, mJp)
            Ω = Δm == 0  ? Ω_π * c_hyper * c_dipole :
                Δm == +1 ? Ω_σ⁺ * c_hyper * c_dipole :
                Δm == -1 ? Ω_σ⁻ * c_hyper * c_dipole : 0.0
            total += Ω
        catch
            continue
        end
    end
    return total
end

"""
    compute_coupling_strength(g::FineLevel, e::HyperfineLevel,
                              atom, Ω_π, Ω_σ⁺, Ω_σ⁻)

Compute the effective coupling strength from a fine-structure level to
a hyperfine level.

This is defined symmetrically via
`compute_coupling_strength(e, g, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)`.
"""
function compute_coupling_strength(g::FineLevel, e::HyperfineLevel, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
    # Symmetric to above, swap roles.
    return compute_coupling_strength(e, g, atom, Ω_π, Ω_σ⁺, Ω_σ⁻)
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
    q_axis::Vector{Float64}
    d_red::Float64
end

function _resolve_node_default(f::BeamRabiFrequency)
    ComplexF64(rabi_frequency(f.atom, f.g, f.e, f.beam_node._compiled[], f.atom.x;
                              q_axis=f.q_axis, d_red=f.d_red))
end

function _resolve_node_value(f::BeamRabiFrequency, param_values, rng)
    ComplexF64(rabi_frequency(f.atom, f.g, f.e, f.beam_node._compiled[], f.atom.x;
                              q_axis=f.q_axis, d_red=f.d_red))
end

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
    level::Pair{HyperfineManifold, HyperfineManifold},
    beam;
    q_axis::AbstractVector{<:Real} = [0.0, 0.0, 1.0],
    d_red::Float64,
    noise=nothing, active=true, tol=1e-10,
)
    ground, excited = level
    q_axis_vec = Vector{Float64}(q_axis)
    couplings = Dynamiq.AbstractField[]

    # Create a single BeamNode for this beam (shared across all sublevel pairs)
    beam_node = BeamNode(beam)
    build_node!(beam_node)
    push!(system, beam_node)

    for g in ground.levels, e in excited.levels
        Δm = e.mF - g.mF
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
    add_coupling!(system, atom, level::Pair{HyperfineManifold,HyperfineManifold},
                  beam::Union{GaussianBeam,GeneralGaussianBeam};
                  q_axis, d_red=nothing, Ω_π=nothing, Ω_p=nothing, Ω_m=nothing,
                  active=true, tol=1e-10)

Add position-dependent electric-dipole couplings driven by a Gaussian beam.

`_coeff` is recomputed each solver timestep as `Ω₀ × efield_scalar(beam, atom.x) / E₀`,
where Ω₀ and E₀ are evaluated at the atom's build-time position. This requires no
CG-coefficient recalculation per step.

If `Ω_π`, `Ω_p`, or `Ω_m` are provided they override the Ω₀ computed by
`rabi_frequency()` for the π (Δm=0), σ⁺ (Δm=+1), and σ⁻ (Δm=-1) components
respectively. Use this when E1 selection rules underestimate the effective coupling
due to state mixing. When all three are overridden, `d_red` is not required.
"""
function add_coupling!(
    system, atom::AbstractAtom,
    level::Pair{HyperfineManifold, HyperfineManifold},
    beam::Union{GaussianBeam, GeneralGaussianBeam};
    q_axis::AbstractVector{<:Real} = [0.0, 0.0, 1.0],
    d_red::Union{Float64, Nothing} = nothing,
    Ω_π = nothing,
    Ω_p = nothing,
    Ω_m = nothing,
    active = true,
    tol = 1e-10,
)
    ground, excited = level
    q_axis_vec = Vector{Float64}(q_axis)
    couplings = Dynamiq.AbstractField[]
    override_map = Dict{Int,Any}(0 => Ω_π, 1 => Ω_p, -1 => Ω_m)

    for g in ground.levels, e in excited.levels
        Δm = e.mF - g.mF
        abs(Δm) > 1 && continue
        override = override_map[Δm]
        Ω0 = if override !== nothing
            ComplexF64(override)
        else
            isnothing(d_red) && error("d_red is required when not providing a Ω override for Δm=$Δm")
            ComplexF64(rabi_frequency(atom, g, e, beam, atom.x; q_axis = q_axis_vec, d_red = d_red))
        end
        abs(Ω0) < tol && continue
        node = GaussianCouplingNode(atom, g => e, beam, q_axis_vec,
                                    something(d_red, 0.0), g, e;
                                    Ω0_override = override !== nothing ? Ω0 : nothing,
                                    active = active)
        build_node!(node, system.basis)
        push!(system, node)
        push!(couplings, node._field)
    end
    return couplings
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
