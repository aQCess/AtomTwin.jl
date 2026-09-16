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
