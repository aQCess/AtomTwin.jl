"""
Features:

- Semiclassical simulation of atomic dynamics including motion.
- Accurate treatment of multilevel effects and light shifts.
- Efficient sparse representation of operators.
- Efficient simulation of closed and open system dynamics using high-order
  Trotter-like or Magnus-like decompositions.
- Support for both global and local Hamiltonian terms.
"""

using StatsBase   # sample, Weights
using Polyester   # @batch
# using ExponentialUtilities

# The two hot kernels -- sparse operator application and the density-matrix
# commutator -- and the radiation forces.

#------------------------------------------------------------------------------
# Operator application (apply!)
#------------------------------------------------------------------------------

"""
    apply!(w, terms, v, scale; hermitian_pairs = true)

Accumulate `scale * Σⱼ (cⱼ Hⱼ.forward + conj(cⱼ) Hⱼ.reverse) * v` into `w`.
`w` is added to, not overwritten.

With `hermitian_pairs = false` only the `forward` part is applied, for
non-Hermitian terms such as a Monte Carlo effective Hamiltonian.
"""
Base.@propagate_inbounds function apply!(w::Vector{ComplexF64},
                                        terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                                        v::Vector{ComplexF64},
                                        scale::ComplexF64;
                                        hermitian_pairs::Bool = true)
    @inbounds for (coeff_ref, op) in terms
        coeff = coeff_ref[]
        iszero(coeff) && continue           # switched-off coupling
        ac = scale * coeff
        _accum!(w, op.forward, v, ac)
        if hermitian_pairs
            _accum!(w, op.reverse, v, scale * conj(coeff))
        end
    end
    return w
end

# The COO walk, unrolled two at a time: both `v[j]` and `w[i]` are indirect, so
# two independent chains per iteration hide the dependent loads. 1.15x at
# d = 4096, neutral below. `@simd` does not help -- the scatter defeats it.
@inline function _accum!(w::Vector{ComplexF64}, coo::Vector{Tuple{Int,Int,ComplexF64}},
                         v::Vector{ComplexF64}, ac::ComplexF64)
    n = length(coo)
    k = 1
    @inbounds while k + 1 <= n
        (i1, j1, x1) = coo[k]
        (i2, j2, x2) = coo[k+1]
        @fastmath w[i1] += ac * x1 * v[j1]
        @fastmath w[i2] += ac * x2 * v[j2]
        k += 2
    end
    @inbounds while k <= n
        (i, j, x) = coo[k]
        @fastmath w[i] += ac * x * v[j]
        k += 1
    end
    return w
end

"""
    apply_commutator!(W, terms, R, scale)

Accumulate `scale * Σⱼ [cⱼ Hⱼ, R]` into `W`, where each `Hⱼ` is stored as an
`Op` in split forward/reverse form. `W` is added to, not overwritten.

**`R` must be Hermitian**: only the `HR` half is computed and `RH` is taken as
its adjoint. The sole caller ([`fquantum!`](@ref)) satisfies this. A
non-Hermitian `R` gives a silently wrong result.
"""
Base.@propagate_inbounds function apply_commutator!(W::Matrix{ComplexF64},
                                                   terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                                                   R::Matrix{ComplexF64},
                                                   scale::ComplexF64)
    n = size(R, 1)
    # Accumulate (HR)^T contiguously, then reflect. Written directly, `HR` is
    # `W[i, col] += a * R[j, col]` -- `col` walks across a row, jumping `n`
    # complex numbers per iteration on a column-major layout. Transposed,
    # `(HR)^T[col, i] = Σⱼ a * conj(R[col, j])` reads and writes down columns.
    # ~15x at d = 256, for identical flops.
    ws = _commutator_ws(n)
    Wt = ws.Wt
    fill!(Wt, 0)
    @inbounds for (coeff_ref, op) in terms
        c  = coeff_ref[]
        # A sequence carries every term it will ever use from the first step,
        # most switched off at any instant (k39: 79 terms, 24 driven). 5.4x.
        iszero(c) && continue
        cc = conj(c)
        # `view`, not 2D indexing: the compiler sees a contiguous vector. ~1.5x.
        for (i, j, v) in op.forward
            α = c * v                      # unscaled -- see the reflect below
            wcol = view(Wt, :, i); rcol = view(R, :, j)
            @simd for col in 1:n
                wcol[col] += α * conj(rcol[col])
            end
        end
        for (i, j, u) in op.reverse
            β = cc * u
            wcol = view(Wt, :, i); rcol = view(R, :, j)
            @simd for col in 1:n
                wcol[col] += β * conj(rcol[col])
            end
        end
    end
    # W += scale * (HR - (HR)^dagger), reading (HR)^T from `Wt` columnwise.
    #
    # `scale` is applied HERE, not folded into the accumulation: `[H,R] =
    # HR - (HR)^dagger` holds for the UNSCALED product. Carried inside, an
    # imaginary `scale` gives `scale*(HR + (HR)^dagger)` -- the anticommutator.
    @inbounds for j in 1:n
        @simd for i in 1:n
            hr = Wt[j, i]              # (HR)^T[j,i] = HR[i,j]
            W[i, j] += scale * (hr - conj(Wt[i, j]))
        end
    end
    return W
end


#------------------------------------------------------------------------------
# Core propagators and forces
#------------------------------------------------------------------------------

"""
    RadiationPressure

Radiation-pressure bookkeeping for the semiclassical solvers.

A `PlanarCoupling` is `c_b H_b + h.c.` with `c_b ∝ exp(i k_b·x)`, so its force on
the atom is the Ehrenfest force

    F = −ħ ∇⟨H⟩ = ħ Σ_b k_b R_b,    R_b = 2 Im(c_b ⟨H_b⟩) / ⟨1⟩,

`R_b` being the rate at which beam `b` moves population up its transition.
Absorption from a beam pushes along its `k`, stimulated emission back into it pushes
against it, and a π pulse moves exactly ħk. The flow oscillates at the Rabi
frequency, so the solvers integrate it over their sub-steps (trapezoid) instead of
sampling it once per `dt`. For the density matrix this mean force is exact for the
ensemble mean.

**MCWF.** A trajectory's state is conditioned on its detection record, and the
momentum must follow. The atom is carried as two branches: the excited one (the
upper levels of its planar drives, population `P`) sits `D` above the rest in
momentum, and `E = D·P` is tracked per atom. Absorption from beam `b` adds `ħk_b`
per unit of population to `E`; stimulated emission, and the no-jump evolution that
drains `P` while no photon is seen, remove population from the branch with its
momentum `D`. The latter is a Bayesian update, not a force: it moves the mean
momentum by `−D` per unit of population it drains. A jump from the excited branch
collapses the atom onto it, `p → p − E + D`. Over one emission cycle a single beam
then transfers exactly `ħk`, as quantised absorption does; a pulse that leaves the
atom partly excited, followed by no emission, transfers nothing. A drive between
two levels of the branch (a ladder g → e → r) moves momentum within it, not
population into it. Not captured: the branch structure of a superposition hit by a
*different* beam (a classical momentum per branch cannot follow the momentum
superposition it creates).

Fields: the planar drives (concretely typed), the atom each acts on, the atoms and
their ħ/m; at the last evaluation (`…prev`: the one before) the force per atom
`(Fx, Fy, Fz)` in units of ħ (m⁻¹s⁻¹) and `R_b` per drive; for MCWF
(`conditional`) per atom the basis states and levels of the excited branch (`up`,
`uplev`) and `P`, and per drive whether it acts within the branch (`internal`) — `E`
lives on the atom, `atom._Eb`, so it carries across instructions; for the density
matrix the force at the start of the user
step (`F0`) and the step's integral (`step`), committed only once the adaptive step
is accepted.
"""
struct RadiationPressure{P, A}
    drives::Vector{P}
    atomidx::Vector{Int}
    atoms::Vector{A}
    hm::Vector{Float64}
    F::Vector{NTuple{3,Float64}}
    Fprev::Vector{NTuple{3,Float64}}
    R::Vector{Float64}
    Rprev::Vector{Float64}
    conditional::Bool
    up::Vector{Vector{Int}}
    uplev::Vector{Vector{Int}}
    internal::Vector{Bool}
    P::Vector{Float64}
    Pprev::Vector{Float64}
    F0::Vector{NTuple{3,Float64}}
    step::Vector{NTuple{3,Float64}}
end

const _ZERO3 = (0.0, 0.0, 0.0)
@inline _add3(a::NTuple{3,Float64}, b::NTuple{3,Float64}) = (a[1] + b[1], a[2] + b[2], a[3] + b[3])
@inline _axpy3(s::Float64, a::NTuple{3,Float64}, b::NTuple{3,Float64}) =
    (muladd(s, a[1], b[1]), muladd(s, a[2], b[2]), muladd(s, a[3], b[3]))
@inline function _kick!(atom, s::Float64, I::NTuple{3,Float64})   # v += s I
    atom.v[1] = muladd(s, I[1], atom.v[1]); atom.v[2] = muladd(s, I[2], atom.v[2])
    atom.v[3] = muladd(s, I[3], atom.v[3])
    return atom
end

# Below this the excited branch is empty and its momentum `D = E/P` undefined.
const _PMIN = 1e-12

"""
    radiation_pressure(fields, atoms; conditional = false) -> RadiationPressure or nothing

`nothing` when no field is a `PlanarCoupling`: every radiation call then dispatches
to a no-op, so a run without plane-wave drives pays nothing. `conditional = true`
(MCWF) also tracks the excited branch of each atom.
"""
function radiation_pressure(fields, atoms::Vector{A}; conditional::Bool = false) where {A}
    drives = PlanarCoupling{A}[f for f in fields if f isa PlanarCoupling{A}]
    isempty(drives) && return nothing
    n = length(atoms)
    atomidx = Vector{Int}(undef, length(drives))
    for b in eachindex(drives)
        i = findfirst(a -> a === drives[b].atom, atoms)
        i === nothing && error("a PlanarCoupling acts on an atom that is not being evolved")
        atomidx[b] = i
    end
    ## The excited branch (MCWF only): the upper levels of the atom's drives, and the
    ## basis states in which the atom occupies one -- the rows of the drives' forward
    ## parts, the same for every drive into the same level.
    up    = [Int[] for _ in 1:n]
    uplev = [Int[] for _ in 1:n]
    if conditional
        for b in eachindex(drives)
            a, lev = atomidx[b], drives[b].transition.second
            lev in uplev[a] && continue
            push!(uplev[a], lev)
            for (i, _, _) in drives[b].H.forward
                push!(up[a], i)
            end
        end
        foreach(sort!, up)
    end
    internal = Bool[conditional && drives[b].transition.first in uplev[atomidx[b]]
                    for b in eachindex(drives)]
    RadiationPressure(drives, atomidx, atoms, [hbar / a.m for a in atoms],
                      fill(_ZERO3, n), fill(_ZERO3, n), zeros(length(drives)),
                      zeros(length(drives)), conditional, up, uplev, internal, zeros(n),
                      zeros(n), fill(_ZERO3, n), fill(_ZERO3, n))
end

# ⟨H_b⟩ without the coefficient: statevector ⟨ψ|H_b|ψ⟩, density matrix Tr(H_b ρ).
@inline function _drive_expect(d, psi::Vector{ComplexF64})
    z = zero(ComplexF64)
    @inbounds for (i, j, x) in d.H.forward
        z = muladd(conj(psi[i]) * x, psi[j], z)
    end
    z
end
@inline function _drive_expect(d, ρ::Matrix{ComplexF64})
    z = zero(ComplexF64)
    @inbounds for (i, j, x) in d.H.forward
        z = muladd(x, ρ[j, i], z)
    end
    z
end

"""
    radiation_eval!(rp, state, inv_n2)

Evaluate the force per atom into `rp.F` and `R_b` per drive into `rp.R` for `state`
with ⟨1⟩ = 1/`inv_n2`; for a conditional (MCWF) state also each atom's
excited-branch population `rp.P`. The values they replace move to `Fprev`, `Rprev`,
`Pprev`. Cost: one pass over the forward entries of the switched-on planar drives,
plus one over the excited-branch basis states.
"""
@inline function radiation_eval!(rp::RadiationPressure, state, inv_n2::Float64)
    F = rp.F
    @inbounds for a in eachindex(F)
        rp.Fprev[a] = F[a]
        F[a] = _ZERO3
    end
    @inbounds for b in eachindex(rp.drives)
        d = rp.drives[b]
        c = d._coeff[]
        rp.Rprev[b] = rp.R[b]
        if iszero(c)                                # beam off
            rp.R[b] = 0.0
            continue
        end
        f = 2 * inv_n2 * imag(c * _drive_expect(d, state))
        rp.R[b] = f
        k = d.beam.k
        a = rp.atomidx[b]
        F[a] = _add3(F[a], (f * k[1], f * k[2], f * k[3]))
    end
    if rp.conditional && state isa Vector
        @inbounds for a in eachindex(rp.atoms)
            p = 0.0
            for i in rp.up[a]
                p += abs2(state[i])
            end
            rp.Pprev[a] = rp.P[a]
            rp.P[a] = p * inv_n2
        end
    end
    return rp
end
radiation_eval!(::Nothing, state, inv_n2) = nothing

"""
    radiation_step!(rp, state, inv_n2, h)

Advance the radiation impulse over one (sub-)step of length `h` that ended in
`state`: the trapezoid of the force at its two ends goes into the atoms' velocities.
`rp.F` must hold the force at the step's start; on return it holds the force at its
end, which is the next step's start (the force is invariant under the
renormalisation that follows). Conditional (MCWF): also the excited branches, see
[`RadiationPressure`](@ref).
"""
@inline function radiation_step!(rp::RadiationPressure, state, inv_n2::Float64, h::Float64)
    radiation_eval!(rp, state, inv_n2)
    @inbounds for a in eachindex(rp.atoms)
        F0 = rp.Fprev[a]; F1 = rp.F[a]
        I  = (0.5h * (F0[1] + F1[1]), 0.5h * (F0[2] + F1[2]), 0.5h * (F0[3] + F1[3]))
        rp.conditional && (I = _branch_step!(rp, a, h, I))
        _kick!(rp.atoms[a], rp.hm[a], I)
    end
    return rp
end
radiation_step!(::Nothing, state, inv_n2, h) = nothing

# Atom `a`'s excited branch over one MCWF sub-step, returning the impulse `I` plus
# the drain's back-action. Absorption (R_b > 0) feeds `E` with ħk_b; stimulated
# emission (R_b < 0) and the no-jump drain remove population, with momentum
# D = E/P, in proportion; a drive within the branch moves ħk_b per unit of
# population inside it. The population balance P₀ + A = P₁ + S + ρ gives the drain
# ρ, which moves the mean momentum by −D ρ.
@inline function _branch_step!(rp::RadiationPressure, a::Int, h::Float64,
                               I::NTuple{3,Float64})
    Eb = rp.atoms[a]._Eb
    A = 0.0; S = 0.0; E = (Eb[1], Eb[2], Eb[3])
    @inbounds for b in eachindex(rp.drives)
        rp.atomidx[b] == a || continue
        r = 0.5h * (rp.Rprev[b] + rp.R[b])
        if rp.internal[b]
            E = _axpy3(r, rp.drives[b].beam.k, E)
        elseif r > 0
            A += r
            E = _axpy3(r, rp.drives[b].beam.k, E)
        else
            S -= r
        end
    end
    tot = rp.Pprev[a] + A
    if tot > _PMIN
        P1 = rp.P[a]
        u  = 1 / tot
        I  = _axpy3(-(tot - S - P1) * u, E, I)
        E  = (E[1] * P1 * u, E[2] * P1 * u, E[3] * P1 * u)
    end
    Eb[1] = E[1]; Eb[2] = E[2]; Eb[3] = E[3]
    return I
end

"""
    radiation_jump!(rp, jump, state, inv_n2)

Re-evaluate the forces and branch populations for the post-jump `state` (the
pre-jump ones move to `…prev`). A jump of an atom out of one of its branch levels
collapses it onto the branch (`p → p − E + D`); otherwise — another atom's jump, or
one from outside the branch — the branch is reweighted to its post-jump population
(`p → p + D ΔP`).
"""
function radiation_jump!(rp::RadiationPressure, jump, state, inv_n2::Float64)
    radiation_eval!(rp, state, inv_n2)          # `Pprev`: pre-jump
    @inbounds for a in eachindex(rp.atoms)
        P0 = rp.Pprev[a]; Eb = rp.atoms[a]._Eb
        if P0 > _PMIN
            D = (Eb[1] / P0, Eb[2] / P0, Eb[3] / P0)
            hit = rp.atoms[a] === jump.atom && jump.transition.first in rp.uplev[a]
            _kick!(rp.atoms[a], ((hit ? 1.0 : rp.P[a]) - P0) * rp.hm[a], D)
            Eb[1] = D[1] * rp.P[a]; Eb[2] = D[2] * rp.P[a]; Eb[3] = D[3] * rp.P[a]
        else
            fill!(Eb, 0.0)
        end
    end
    return rp
end
radiation_jump!(::Nothing, jump, state, inv_n2) = nothing

"""
    radiation_resume!(rp, psi)

At the start of an MCWF instruction: evaluate the forces and branch populations
for `psi`, and empty the momentum of any excited branch that holds no population
(where `D = E/P` is undefined).
"""
function radiation_resume!(rp::RadiationPressure, psi::Vector{ComplexF64})
    radiation_eval!(rp, psi, 1 / real(dot(psi, psi)))
    @inbounds for a in eachindex(rp.atoms)
        rp.P[a] > _PMIN || fill!(rp.atoms[a]._Eb, 0.0)
    end
    return rp
end
radiation_resume!(::Nothing, psi) = nothing

# Density matrix: the adaptive controller may retake a user step, so the step's
# impulse is kept aside (`F0`, `step`) and committed only once it is accepted. There
# are no jumps, hence no branches: this is the ensemble mean force, which is exact
# for the mean.
function radiation_begin!(rp::RadiationPressure, ρ::Matrix{ComplexF64})
    radiation_eval!(rp, ρ, 1 / real(tr(ρ)))
    copyto!(rp.F0, rp.F)
    fill!(rp.step, _ZERO3)
    return rp
end
radiation_begin!(::Nothing, ρ) = nothing

function radiation_retry!(rp::RadiationPressure)
    copyto!(rp.F, rp.F0)
    fill!(rp.step, _ZERO3)
end
radiation_retry!(::Nothing) = nothing

function radiation_pair!(rp::RadiationPressure, ρ::Matrix{ComplexF64}, h2::Float64)
    @inbounds for a in eachindex(rp.atoms)
        rp.step[a] = _axpy3(0.5h2, rp.F[a], rp.step[a])
    end
    radiation_eval!(rp, ρ, 1 / real(tr(ρ)))
    @inbounds for a in eachindex(rp.atoms)
        rp.step[a] = _axpy3(0.5h2, rp.F[a], rp.step[a])
    end
    return rp
end
radiation_pair!(::Nothing, ρ, h2) = nothing

function radiation_commit!(rp::RadiationPressure)
    @inbounds for a in eachindex(rp.atoms)
        _kick!(rp.atoms[a], rp.hm[a], rp.step[a])
    end
    fill!(rp.step, _ZERO3)
end
radiation_commit!(::Nothing) = nothing

"""
    recoil!(J, rng)

Apply an isotropic recoil kick of magnitude `ħk` to `J`'s atom, with
`k = 2π / J.atom.lambda[J.transition]`.
"""
Base.@inline function recoil!(J::Jump, rng::AbstractRNG)
    _x, _y, _z = randn(rng), randn(rng), randn(rng)
    _norm = sqrt(_x * _x + _y * _y + _z * _z)

    k = 2 * pi / J.atom.lambda[J.transition]
    c = hbar * k / J.atom.m / _norm

    J.atom.v[1] += c * _x
    J.atom.v[2] += c * _y
    J.atom.v[3] += c * _z
end

