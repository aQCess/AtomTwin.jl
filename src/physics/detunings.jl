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
