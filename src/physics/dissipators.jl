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

Pass `λ` (metres) — the wavelength of the emitted photon — to include the
**recoil kick** `ħk` the atom takes on each spontaneous emission, with
`k = 2π/λ` in a random direction. Without it the decay is radiatively correct
but momentum-free, so an atom never heats from its own fluorescence. Recoil
matters whenever the recoil energy is not negligible against the trap depth:
131 scattering events at 556 nm eject a Yb-174 atom from a 23 µK tweezer.

```julia
add_decay!(sys, yb, e => g, Γ; λ = 556e-9, clicks = pd)
```

The kick is isotropic unless `dipole` gives the transition dipole (a 3-vector in the
lab frame, complex for circular polarisation; normalised internally). Kicks then
follow its emission pattern `∝ 1 − |n̂·d̂|²`, and so do the photons a detector with an
aperture collects (`PhotoDetectorSpec(; aperture)`), with or without `λ`. This matters when one dipole dominates
the scattering: a dipole along x̂ puts 1/5 of its recoil energy along x, against
1/3 for isotropic emission.

```julia
add_decay!(sys, yb, ex => g, Γ; λ = 556e-9, dipole = [1, 0, 0])   # linear, along x
add_decay!(sys, yb, ep => g, Γ; λ = 556e-9, dipole = [1, im, 0])  # σ⁺ about z
```
"""
function add_decay!(system, atom::AbstractAtom, level::Pair{<:AbstractLevel, <:AbstractLevel}, Gamma;
                    active=true, clicks=nothing, λ=nothing, dipole=nothing)
    return _add_decay!(system, atom, [level => 1.0], Gamma; active, clicks, λ, dipole)
end

# One jump `L = √Gamma Σ c |g⟩⟨e|` over `branches = [(e => g) => c, …]`.
# `polarization` (a photon q about the quantization axis) instead of `dipole` defers the
# recoil dipole to the axis a run uses: see `_set_photon_dipoles!`.
function _add_decay!(system, atom::AbstractAtom, branches, Gamma;
                     active=true, clicks=nothing, λ=nothing, dipole=nothing, polarization=nothing)
    level = first(branches).first
    node = DecayNode(Gamma, atom, level; active=active, clicks=_clicks_name(clicks), branches=branches,
                     polarization=polarization)
    build_node!(node, system.basis)
    push!(system, node)
    # The photon's wavelength (its recoil) and dipole (its direction) live on the atom,
    # keyed by the first branch's level-index pair, the same key `Jump` carries.
    key = atom.level_indices[level.first] => atom.level_indices[level.second]
    λ === nothing || (atom.inner.lambda[key] = Float64(λ))
    if dipole === nothing
        delete!(atom.inner.dipole, key)
    else
        d = ComplexF64.(collect(dipole))
        (length(d) == 3 && norm(d) > 0) || error("add_decay!: `dipole` must be a nonzero 3-vector.")
        atom.inner.dipole[key] = Tuple(d ./ norm(d))
    end
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
    # Unchanged: keep the cached operators, which are the expensive part.
    new_rate == j._rate && return j
    j._rate > 0 && Dynamiq.rescale!(j.J, sqrt(new_rate / j._rate))
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

There is one jump per photon polarisation q, `L_q = √rate Σₘ c_q(m) |g_{m+q}⟩⟨e_m|`,
so a superposition of excited sublevels decays into the matching superposition of
ground sublevels (coherence transfer).

With `λ`, each jump's recoil kick follows the emission pattern of its polarisation
about the quantization axis: π photons `∝ sin²θ`, σ photons `∝ (1 + cos²θ)/2`. `clicks` is one `PhotoDetectorSpec` for every
photon, or a named tuple `(π = spec₁, σ = spec₂)` to count the two kinds apart (a
polarisation left out is not counted):

```julia
add_decay!(sys, yb, P1 => S0, Γ; λ = 556e-9, clicks = (π = pdπ, σ = pdσ))
```
"""
function add_decay!(system, atom::AbstractAtom, levels::Pair{HyperfineManifold,HyperfineManifold},
                    rate; active=true, tol=1e-10, clicks=nothing, λ=nothing)
    excited, ground = levels
    branches = [e => [(g, clebschgordan(e.F, e.mF, 1, g.mF - e.mF, g.F, g.mF), g.mF - e.mF)
                      for g in ground
                      if abs(e.F - g.F) <= 1 && abs(e.mF - g.mF) <= 1]
                for e in excited]
    _add_polarized_decays!(system, atom, branches, rate; tol, active, clicks, λ)
end

# `branches = [e => [(g, CG, q), …], …]`. The emission amplitudes are normalised over
# each excited sublevel, and grouped by q into one coherent jump each: the three
# polarisations are orthogonal photon modes, so they decay incoherently.
# Each jump carries its photon's polarisation, which sets its recoil pattern.
function _add_polarized_decays!(system, atom, branches, rate; tol, clicks, kwargs...)
    byq = Dict(q => Pair{<:Pair,Float64}[] for q in -1:1)
    for (e, gs) in branches
        total_strength = sum(abs2(c) for (_, c, _) in gs)
        for (g, c, q) in gs
            rate * abs2(c) / total_strength > tol &&
                push!(byq[Int(q)], (e => g) => Float64(c / sqrt(total_strength)))
        end
    end
    for q in -1:1
        isempty(byq[q]) || _add_decay!(system, atom, identity.(byq[q]), rate;
                                       clicks = _polarization_clicks(clicks, q), polarization = q,
                                       kwargs...)
    end
end

"""
    _set_photon_dipoles!(system, q_axis)

Give every polarised decay the dipole of its photon about `q_axis`:
`ẑ′` for π, `(x̂′ ± iŷ′)/√2` for σ. Called for the axis a run uses (`compile`, and
`recompile!` for a sampled axis), as the tensor light shift is.
"""
function _set_photon_dipoles!(system, q_axis)
    ex, ey, ez = _quantization_frame(q_axis)
    for n in system.nodes
        (n isa DecayNode && n.polarization !== nothing) || continue
        key = n.atom.level_indices[n.transition.first] => n.atom.level_indices[n.transition.second]
        q = n.polarization
        n.atom.inner.dipole[key] = Tuple(q == 0 ? ComplexF64.(ez) : (ex .+ q * im .* ey) ./ √2)
    end
end

_polarization_clicks(clicks, q) = clicks
function _polarization_clicks(clicks::NamedTuple, q)
    issubset(keys(clicks), (:π, :σ)) ||
        error("add_decay!: `clicks` as a named tuple takes the keys π and σ.")
    return get(clicks, q == 0 ? :π : :σ, nothing)
end

"""
    add_decay!(system, atom, levels::Pair{HyperfineManifold,<:AbstractLevel}, rate; ...)

Add decay from an excited hyperfine manifold to a single leak level.
"""
function add_decay!(system, atom::AbstractAtom, levels::Pair{HyperfineManifold,<:AbstractLevel}, rate;
                    active=true, clicks=nothing, λ=nothing)
    excited, leak_level = levels
    for e in excited
        add_decay!(system, atom, e => leak_level, rate; active=active, clicks=clicks, λ=λ)
    end
end

"""
    add_decay!(system, atom, levels::Pair{FineManifold,FineManifold}, rate; ...)

Add spontaneous decay with Clebsch-Gordan branching ratios between fine-structure
manifolds, one coherent jump per photon polarisation as for hyperfine manifolds
(including the recoil patterns and `clicks` by polarisation).
"""
function add_decay!(system, atom::AbstractAtom,
                    levels::Pair{FineManifold,FineManifold},
                    rate; active=true, tol=1e-10, clicks=nothing, λ=nothing)
    excited, ground = levels
    branches = [e => [(g, clebschgordan(e.J, e.mJ, 1, g.mJ - e.mJ, g.J, g.mJ), g.mJ - e.mJ)
                      for g in ground
                      if abs(e.J - g.J) <= 1 && abs(e.mJ - g.mJ) <= 1]
                for e in excited]
    _add_polarized_decays!(system, atom, branches, rate; tol, active, clicks, λ)
end

"""
    add_decay!(system, atom, levels::Pair{FineManifold,<:AbstractLevel}, rate; ...)

Add decay from an excited fine-structure manifold into a single leak level.
"""
function add_decay!(system, atom::AbstractAtom,
                    levels::Pair{FineManifold,<:AbstractLevel},
                    rate; active=true, clicks=nothing, λ=nothing)
    excited, leak_level = levels
    for e in excited
        add_decay!(system, atom, e => leak_level, rate; active=active, clicks=clicks, λ=λ)
    end
end
