"""
Add dissipative jump operators to the system.
"""

"""
    add_decay!(system, atom, level::Pair, Gamma; active = true, clicks = nothing)

Add a single spontaneous-emission jump operator between two specific levels.

`Gamma` (rad/s) may be a plain number or a `Parameter`. Builds a `DecayNode`
and appends it to `system.nodes`. Returns the compiled `Jump` object.

Pass `clicks = spec`, where `spec` is a `PhotoDetectorSpec` already attached with
`add_detector!`, to count this jump's firings as photon clicks in that detector
(statevector / wavefunction Monte Carlo runs only).
"""
function add_decay!(system, atom::AbstractAtom, level::Pair{<:AbstractLevel, <:AbstractLevel}, Gamma;
                    active=true, clicks=nothing)
    node = DecayNode(Gamma, atom, level; active=active, clicks=_clicks_name(clicks))
    build_node!(node, system.basis)
    push!(system, node)
    return node._field
end

# Accept either a PhotoDetectorSpec (extract its name) or nothing.
_clicks_name(::Nothing) = nothing
function _clicks_name(spec::Dynamiq.DetectorSpec)
    spec.kind === Dynamiq.PhotoDetector ||
        error("add_decay!(...; clicks=...) expects a PhotoDetectorSpec.")
    return spec.params.name
end

"""
    update!(j::Jump, ::Val, val)

Rescale the jump operator to a new decay rate `val` (rad/s) in-place.
Updates J.forward entries by sqrt of the rate ratio and clears cached
non-Hermitian Hamiltonian and LdagL diagonal.
"""
function update!(j::Jump, ::Val, val::Number)
    new_rate = Float64(val)
    if j._rate > 0
        scale = sqrt(new_rate / j._rate)
        for k in eachindex(j.J.forward)
            i, r, v = j.J.forward[k]
            j.J.forward[k] = (i, r, v * scale)
        end
    end
    j._rate = new_rate
    j.Hnh = nothing
    j.LdagL_diag = nothing
end

"""
    add_dephasing!(system, atom, level, gamma; active = true)

Add a pure dephasing jump operator for a single level.
"""
function add_dephasing!(system, atom::AbstractAtom, level::L, gamma; active=true) where {L<:AbstractLevel}
    node = DecayNode(gamma, atom, level => level; active=active)
    build_node!(node, system.basis)
    push!(system, node)
    return node._field
end

"""
    add_dephasing!(system, atom, manifold, gamma; active = true)

Add pure dephasing jump operators for all levels in a manifold.
"""
function add_dephasing!(system, atom::AbstractAtom, manifold::M, gamma; active=true) where {M<:AbstractManifold}
    [add_dephasing!(system, atom, level, gamma; active=active) for level in manifold]
end

"""
    add_decay!(system, atom, levels::Pair{HyperfineManifold,HyperfineManifold}, rate; ...)

Add spontaneous decay from an excited hyperfine manifold to a ground manifold,
with branching ratios set by Clebsch-Gordan coefficients.
"""
function add_decay!(system, atom::AbstractAtom, levels::Pair{HyperfineManifold,HyperfineManifold},
                    rate; active=true, tol=1e-10)
    excited, ground = levels
    for e in excited
        transitions = [(g, abs2(clebschgordan(e.F, e.mF, 1, g.mF - e.mF, g.F, g.mF)))
                       for g in ground
                       if abs(e.F - g.F) <= 1 && abs(e.mF - g.mF) <= 1]
        total_strength = sum(strength for (_, strength) in transitions)
        for (g, strength) in transitions
            jump_rate = rate * strength / total_strength
            jump_rate > tol && add_decay!(system, atom, e => g, jump_rate; active=active)
        end
    end
end

"""
    add_decay!(system, atom, levels::Pair{HyperfineManifold,<:AbstractLevel}, rate; ...)

Add decay from an excited hyperfine manifold to a single leak level.
"""
function add_decay!(system, atom::AbstractAtom, levels::Pair{HyperfineManifold,<:AbstractLevel}, rate; active=true)
    excited, leak_level = levels
    for e in excited
        add_decay!(system, atom, e => leak_level, rate; active=active)
    end
end

"""
    add_decay!(system, atom, levels::Pair{FineManifold,FineManifold}, rate; ...)

Add spontaneous decay with Clebsch-Gordan branching ratios between fine-structure manifolds.
"""
function add_decay!(system, atom::AbstractAtom,
                    levels::Pair{FineManifold,FineManifold},
                    rate; active=true, tol=1e-10)
    excited, ground = levels
    for e in excited
        transitions = [(g, abs2(clebschgordan(e.J, e.mJ, 1, g.mJ - e.mJ, g.J, g.mJ)))
                       for g in ground
                       if abs(e.J - g.J) <= 1 && abs(e.mJ - g.mJ) <= 1]
        total_strength = sum(strength for (_, strength) in transitions)
        for (g, strength) in transitions
            jump_rate = rate * strength / total_strength
            jump_rate > tol && add_decay!(system, atom, e => g, jump_rate; active=active)
        end
    end
end

"""
    add_decay!(system, atom, levels::Pair{FineManifold,<:AbstractLevel}, rate; ...)

Add decay from an excited fine-structure manifold into a single leak level.
"""
function add_decay!(system, atom::AbstractAtom,
                    levels::Pair{FineManifold,<:AbstractLevel},
                    rate; active=true)
    excited, leak_level = levels
    for e in excited
        add_decay!(system, atom, e => leak_level, rate; active=active)
    end
end
