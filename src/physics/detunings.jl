"""
Add level detunings and Zeeman shifts to the system Hamiltonian.
"""

"""
    add_detuning!(system, atom, level, delta; active = true)

Add a single detuning term for a given atomic level.

`delta` is the physical detuning in rad/s. It may be a plain number,
a `Parameter`, or a `ParametricExpression`. Builds a `DetuningNode` and
appends it to `system.nodes`. Returns the compiled `Detuning` object.

# Time-dependent detuning

The returned `Detuning` is `Switchable`, so a detuning can be made
time-dependent by driving it with a shaped `Pulse` — the same mechanism used to
shape couplings. Create it inactive with unit base value, then supply the
per-step detuning samples (in rad/s) as the pulse `amplitudes`:

```julia
δ = add_detuning!(sys, atom, level, 1.0; active = false)   # base 1.0 ⇒ amplitudes ARE δ(t)
@sequence seq begin
    Pulse(δ, T; amplitudes = A .* sin.(2π*f .* tgrid), interp = :piecewise_constant)
end
```

Because the base value is `1.0`, the pulse `amplitudes` map one-to-one to the
physical detuning δ(t). The amplitudes **must be real** (a detuning is a real,
diagonal energy shift; a complex amplitude would make the term non-Hermitian and
is rejected at compile time). `Pulse`'s `amplitudes` keyword reads as an
amplitude envelope; for a detuning it is really a frequency-vs-time *sweep*.
"""
function add_detuning!(system, atom::AbstractAtom, level::AbstractLevel, delta; active=true, tol=1e-10)
    node = DetuningNode(delta, atom, level; active=active)
    build_node!(node, system.basis)
    push!(system, node)
    return node._field
end

"""
    update!(d::Detuning, ::Val, val)

Update the energy shift of a `Detuning` operator to `val` (rad/s) in-place.
Called by `compile_node!` and `recompile_node!` for `DetuningNode`.
"""
function update!(d::Detuning, ::Val, val::Number)
    for k in eachindex(d.H.forward)
        i, j, _ = d.H.forward[k]
        d.H.forward[k] = (i, j, ComplexF64(-val))
    end
end

"""
    add_zeeman_detunings!(system, atom, manifold; B=0.0, delta=0.0, active=true)

Add Zeeman-shifted detunings for all levels in a manifold.

For each level the total detuning (node value) is

    delta_total = delta - zeeman_coeff * B

where `zeeman_coeff = mF * g_F * μ_B` (hyperfine) or `mJ * g_J * μ_B`
(fine). `B` and `delta` may be plain numbers or `Parameter`s. `B` is
in units of Tesla.

Returns a vector of the created `Detuning` objects.
"""
function add_zeeman_detunings!(system, atom::AbstractAtom, manifold::AbstractManifold;
                               B=0.0, delta=0.0, active=true, tol=1e-10)
    detunings = []

    for level in manifold
        if manifold isa HyperfineManifold
            zeeman_coeff = Float64(level.mF) * level.g_F * BOHR_MAGNETON_RAD_S_TESLA
        elseif manifold isa FineManifold
            zeeman_coeff = Float64(level.mJ) * level.g_J * BOHR_MAGNETON_RAD_S_TESLA
        else
            zeeman_coeff = 0.0
        end

        # Physical detuning for this level: delta - zeeman_coeff * B
        # Using Parameter arithmetic when B or delta is a Parameter
        delta_total = delta + (-zeeman_coeff) * B

        node = DetuningNode(delta_total, atom, level; active=active)
        build_node!(node, system.basis)
        push!(system, node)
        push!(detunings, node._field)
    end

    return detunings
end

#------------------------------------------------------------------------------
# Quantization axis
#------------------------------------------------------------------------------

"""
    add_quantization_axis!(system, axis)

Set the system's quantization axis — the direction magnetic sublevels are defined
against, normally the magnetic bias field. Defaults to `ẑ` if never called.

    add_quantization_axis!(sys, [0, 0, 1])
    add_quantization_axis!(sys, B_vec)          # need not be normalised

This is what gives the **tensor light shift** its geometry: the shift depends on
`ε_z`, the projection of a trap beam's polarization onto this axis, so a linearly
polarized trap at the magic angle `acos(1/√3) ≈ 54.74°` produces no tensor shift
at all. Each beam's `pol` is projected onto the axis automatically — no angle is
passed by hand.

`axis` may be a `Parameter` or `ParametricExpression`, so a magic-angle scan or a
shot-to-shot field misalignment is a `play` keyword rather than a rebuilt system.

Only one axis may be set per system.

See also [`add_zeeman_detunings!`](@ref), which currently takes the field
*magnitude* separately.
"""
function add_quantization_axis!(system, axis)
    for n in system.nodes
        n isa QuantizationAxisNode && error(
            "the system already has a quantization axis; only one may be set")
    end
    node = QuantizationAxisNode(axis)
    build_node!(node)
    push!(system, node)
    return node
end

"""
    getquantizationaxis(system) -> Vector{Float64}

The system's quantization axis as a unit vector, or `ẑ` if none was set.
"""
function getquantizationaxis(system)
    for n in system.nodes
        if n isa QuantizationAxisNode
            ax = node_output(n)
            ax === nothing || return ax
        end
    end
    return [0.0, 0.0, 1.0]
end

#------------------------------------------------------------------------------
# Trap light shifts
#------------------------------------------------------------------------------

"""
    add_light_shift!(system, atom, levels, beam; reference = nothing, active = true)

Override the automatic trap light shift on `levels`.

!!! note "You usually do not need this"
    A trapping beam shifts the levels it traps **automatically** — just pass the
    beam to `System` and give the atom polarizability data:

        trap = GaussianBeam(λ = 767nm, w0 = 1µm, P = 20mW, pol = [sind(θ), 0, cosd(θ)])
        sys  = System(yb, trap)          # the shift is already included

    Reach for `add_light_shift!` only to change the default: to reference the
    shift against a particular level, or to make it switchable with `On`/`Off`.
    Levels named here are excluded from the automatic shift, so nothing is
    counted twice.

`levels` may be a single level, a manifold, or a vector of either. The shift is
taken from the same per-level `α` that drives the dipole force, tensor
contribution included — so in a non-magic trap the sublevels of a manifold shift
by different amounts, and the transition frequency moves with the local intensity.
That is what makes trap-depth spectroscopy work.

`beam` must already belong to the system (pass it to `System`), because α is
computed at the wavelengths the system's beams use.

# Reference level

By default each level is shifted by its own `α I/ħ`. Pass `reference` to measure
against one level instead, which sets that level's shift to zero and leaves the
others as differences — convenient when only a transition frequency matters.

Returns the created nodes, so they can be switched with `On`/`Off` like a coupling.
"""
function add_light_shift!(system, atom::AbstractAtom, levels, beam;
                          reference = nothing, active::Bool = true)
    lvls = _lightshift_levels(levels)

    # α is normally filled by `initialize!` during `compile`, but the node has to
    # build now so that `gethamiltonian(sys)` can be inspected before any `play`.
    # Fill it on demand if this is the first light shift added.
    b = beam isa BeamNode ? beam._compiled[] : beam
    if b !== nothing && !haskey(atom.inner.alpha, getwavelength(b))
        initialize!(atom, atom.inner; beams = [b],
                    q_axis = getquantizationaxis(system))
    end

    nodes = LightShiftNode[]
    for l in lvls
        node = LightShiftNode(atom, l, beam; reference = reference, active = active)
        build_node!(node, system.basis)
        push!(system, node)
        push!(nodes, node)
    end
    return nodes
end

add_light_shift!(system, atom::AbstractAtom, level::AbstractLevel, beam; kwargs...) =
    add_light_shift!(system, atom, [level], beam; kwargs...)

# Accept a level, a manifold, or any mixed collection of them.
_lightshift_levels(l::AbstractLevel)  = AbstractLevel[l]
_lightshift_levels(m::AbstractManifold) = AbstractLevel[l for l in m.levels]
function _lightshift_levels(v)
    out = AbstractLevel[]
    for x in v
        append!(out, _lightshift_levels(x))
    end
    return out
end

"""
    _auto_light_shifts(sys, atoms, trapping_beams) -> Vector{AbstractField}

The AC Stark shifts every trapping beam imposes on every level it can shift.

A trap that holds an atom also shifts the atom's levels; that is one physical
effect, not two, so it needs no `add_*!` call. `compile` calls this after the
atoms are initialised (α is filled) and folds the result into the field list
beside the coupling and detuning terms.

Scope and omissions, all deliberate:

- Only `sys.beams` — the beams handed to `System`. A coupling beam's effect on the
  atom is already its coupling term; adding a Stark shift for it too would
  double-count.
- A level with `α = 0` at a beam's wavelength contributes nothing, so it is
  skipped rather than given a zero operator.
- A level explicitly given an `add_light_shift!` node is skipped here, so an
  explicit request wins over the automatic one and the shift is never applied
  twice.

The shift is **absolute**: each level takes its own `α I/ħ`. The common-mode part
is a global phase, and the Chebyshev propagator sets its degree from the spectral
half-width `(Emax−Emin)/2` after recentering on `Ē`, so an offset costs nothing.
"""
function _auto_light_shifts(sys, atoms, trapping_beams)
    fields = Dynamiq.AbstractField[]
    isempty(trapping_beams) && return fields

    # Levels already covered by an explicit add_light_shift!: (atom, level index).
    explicit = Set{Tuple{UInt, Int}}()
    for n in sys.nodes
        n isa LightShiftNode || continue
        push!(explicit, (objectid(n.atom), n.atom.level_indices[n.level]))
    end

    for (k, a) in enumerate(sys.atoms)
        inner = atoms[k]
        for beam in trapping_beams
            λ = getwavelength(beam)
            haskey(inner.alpha, λ) || continue
            αs = inner.alpha[λ]
            for idx in eachindex(αs)
                αs[idx] == 0.0 && continue
                (objectid(a), idx) in explicit && continue
                f = StarkShiftAC(sys.basis, inner, idx, beam)
                f._coeff[] = ComplexF64(1.0)
                push!(fields, f)
            end
        end
    end
    return fields
end
