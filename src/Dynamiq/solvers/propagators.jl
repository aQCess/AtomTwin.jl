# The propagators themselves: the `fquantum!` family, the Lindblad dissipator,
# jump application, and the per-dimension scratch caches they share.

#------------------------------------------------------------------------------
# Core quantum propagators (fquantum!)
#------------------------------------------------------------------------------

"""
    stability_limit(order) -> Float64

Largest `θ = ‖H‖·dt` for which the order-`p` Taylor propagator is
norm-contractive on the imaginary axis: 1.732, 2.828, 1.764, 3.395 for orders
3, 4, 7, 8.

Truncated Taylor contracts only for `p ≡ 0, 3 (mod 4)`; [`check_order`](@ref)
rejects the rest, so only those four appear. The order-4 value is `2√2`, the RK4
imaginary-axis limit -- for a linear autonomous problem order-4 Taylor *is* RK4.
"""
Base.@inline stability_limit(order::Int) =
    order == 3 ? 1.732 : order == 4 ? 2.828 : order == 7 ? 1.764 : 3.395

"""
    suggested_dt(terms, tol, order) -> Float64

Time step giving a local error of about `tol`, derived from the Hamiltonian
rather than supplied by the user. Returns `Inf` when there is no Hamiltonian, so
the caller must fall back to the instruction duration.

`dt = θ_max / ‖H‖`, with `θ_max` from [`propagator_theta`](@ref) for the given
integrator and `‖H‖` a `gershgorin_bound` upper bound -- so the step is
conservative and the true error at most the target.
"""
function suggested_dt(terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                      tol::Float64, order::Int = 4;
                      integrator::AbstractIntegrator = Chebyshev())
    nrm = gershgorin_bound(terms; peak = true)
    nrm <= 0 && return Inf
    # Spectrum-aware: a Hermitian operator takes a far larger step than an MCWF
    # effective Hamiltonian.
    spec = SpectralSpec(-nrm, nrm, antihermitian_bound(terms; peak = true))
    return propagator_theta(integrator, tol, order, spec) / nrm
end

# `steptol` on the wave-function solvers (`tdse`, `wfmc` and their semiclassical
# variants) is accepted for a uniform interface and ignored: those paths use no
# operator splitting, so there is no splitting error to control. Only the
# density-matrix solvers act on it, through `strang_substeps!`.

"""
    taylor_order(integrator) -> Int

Truncation order the Taylor kernels should use under `integrator`.

Only [`Taylor`](@ref) carries one. Chebyshev sets its degree from `ΔE·dt` and
`tol`, so where a Taylor kernel still runs beneath it -- the Strang half-steps of
the density-matrix path -- the default 4 applies.
"""
taylor_order(t::Taylor)   = t.order
taylor_order(::Chebyshev)  = 4
taylor_order(::AbstractIntegrator) = 4

"""
    propagator_theta(integrator, tol, order) -> Float64

Largest `θ = ‖H‖·dt` the propagator may take for a local error near `tol`.

Integrator-specific: Taylor's error is `O(θ^(p+1))` under a hard stability limit,
while Chebyshev has no stability limit and absorbs a larger `θ` by raising its
degree.
"""
propagator_theta(::Taylor, tol::Float64, order::Int) =
    min(tol^(1 / (order + 1)),        # accuracy: error is O(θ^(p+1))
        0.25 * stability_limit(order))  # never ride the stability edge

# Chebyshev is spectrally exact for a Hermitian `H`, so accuracy does not bound
# the step. The bound comes from the non-Hermitian part, whose expansion diverges
# as `r = Im/ΔE` grows; measured, `θ_max` falls monotonically with `r`:
#
#     r        0.2    0.5    0.8    0.94   0.98
#     θ_max   >400   33.5   16.8   13.7   12.7
#
# `THETA_NH / r` follows that shape with a factor-of-two margin. Empirical.
const THETA_HERM = 64.0    # Hermitian operator
const THETA_NH   = 8.0     # fully non-Hermitian

propagator_theta(::Chebyshev, tol::Float64, order::Int) = THETA_HERM

"""
    propagator_theta(integrator, tol, order, spec) -> Float64

Largest `θ = ‖H‖·dt` the propagator may take.

The four-argument form reads the anti-Hermitian radius from `spec` and relaxes
the bound toward `THETA_HERM` as the operator approaches Hermitian. The
three-argument form has no spectrum available and returns the Hermitian value.
"""
function propagator_theta(::Chebyshev, tol::Float64, order::Int, spec::SpectralSpec)
    ΔE = (spec.Emax - spec.Emin) / 2
    ΔE <= 0 && return THETA_HERM
    r = spec.Iradius / ΔE
    r <= THETA_NH / THETA_HERM && return THETA_HERM
    return min(THETA_HERM, THETA_NH / r)
end

propagator_theta(i::Taylor, tol::Float64, order::Int, ::SpectralSpec) =
    propagator_theta(i, tol, order)

"""
    gershgorin_bound(terms) -> Float64

Rigorous upper bound on `‖Σⱼ cⱼHⱼ‖₂` from the Gershgorin circle theorem:
the spectral radius is at most the largest absolute row sum.

Costs one pass over the COO triples — O(nnz), no matrix assembled, no
allocation beyond a single row-sum accumulator. Both the `forward` and
`reverse` halves contribute, since the applied operator is
`cⱼHⱼ.forward + conj(cⱼ)Hⱼ.reverse`.

The bound is loose (it ignores cancellation) but it is an *upper* bound, so
`θ = gershgorin_bound(H)·dt` below the stability limit is a guarantee, never a
false reassurance.
"""
function gershgorin_bound(terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}};
                          peak::Bool = false)
    isempty(terms) && return 0.0
    dim = terms[1][2].dim
    rowsum = _gershgorin_ws(dim)   # thread-cached: up to 3 calls per instruction
    fill!(rowsum, 0.0)
    @inbounds for (coeff_ref, op) in terms
        # `peak = true` bounds over the whole sequence, not this instant.
        # Couplings are built inactive (coeff 0) and switched on by a Pulse/On,
        # which sets the envelope to at most 1 -- the rate lives in `op` -- so
        # `max(|coeff|, 1)` covers any amplitude the sequence can later apply.
        # Without it the bound is taken at coeff = 0 and underestimates `‖H‖`.
        c = peak ? max(abs(coeff_ref[]), 1.0) : abs(coeff_ref[])
        for (i, _, v) in op.forward
            rowsum[i] += c * abs(v)
        end
        for (i, _, u) in op.reverse
            rowsum[i] += c * abs(u)
        end
    end
    maximum(rowsum)
end

# Taylor-4 is norm-contractive only for θ = ‖H‖dt < 2√2. Warn well before that:
# the bound is loose, so firing at 2.0 leaves headroom without crying wolf.
const _THETA_WARN = 2.0

"""
    warn_if_step_too_large(terms, dt, integrator)

Warn, before integrating, when the step is near or past `integrator`'s stability
limit. Too large a `dt` otherwise produces `NaN` populations with no indication
of why.

Only [`Taylor`](@ref) has such a limit. [`Chebyshev`](@ref) has none -- it
absorbs a larger `θ = ‖H‖·dt` by raising its expansion degree -- so warning on it
would be false: measured at `θ = 63`, Chebyshev is accurate to 3.5e-13 where
Taylor-4 is wrong by 6.5e+05.
"""
warn_if_step_too_large(::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                       ::Float64, ::AbstractIntegrator) = 0.0

function warn_if_step_too_large(terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                                dt::Float64, integrator::Taylor)
    # `peak = true`: an instantaneous bound would miss a drive not yet switched on.
    nrm = gershgorin_bound(terms; peak = true)
    θ = nrm * dt
    θ < _THETA_WARN && return θ
    order = taylor_order(integrator)
    lim = stability_limit(order)
    suggested = 0.5 * lim / max(nrm, eps())
    @warn """
    Solver step size is near or past the stability limit.

      ‖H‖·dt ≈ $(round(θ, sigdigits = 4))   (order $order is stable only below $lim)

    Above the limit the propagator amplifies the state every step and the result
    diverges to NaN. Reduce the sequence step to about $(round(suggested, sigdigits = 3)) s
    (or smaller) — note ‖H‖ here is a Gershgorin upper bound, so this is conservative.
    """
    return θ
end

"""
    check_order(order)

Validate the Taylor truncation order of [`fquantum!`](@ref): 3, 4, 7 or 8,
default 4.

The truncated Taylor series of `exp(-iH dt)` is norm-contractive on the imaginary
axis only for `p ≡ 0, 3 (mod 4)`. For `p ≡ 1, 2 (mod 4)` it amplifies at *every*
step size however small, so `order = 2` or `order = 6` would silently give a
divergent integrator. See [`stability_limit`](@ref).
"""
Base.@inline function check_order(order::Int)
    order in (3, 4, 7, 8) && return nothing
    throw(ArgumentError(
        "order = $order is unstable: the Taylor propagator is norm-contractive " *
        "only for p ≡ 0, 3 (mod 4). Orders 1, 2, 5, 6, 9, ... amplify the norm " *
        "at every step size. Use order ∈ (3, 4, 7, 8); 4 is the default."))
end

"""
    fquantum!(dt, qstate, Hlist, _q1, _q2; order = 4)

Advance the state vector `qstate` by `dt` under `Hlist`, in place, by the
truncated Taylor series of `exp(-i H dt)` to `order` terms.

`Hlist` entries are `(coeff, H)` with `coeff[]` a possibly time-dependent scalar
and `H::Op` in split forward/reverse storage, so the applied operator is
`coeff[]*H.forward + conj(coeff[])*H.reverse`. `_q1`/`_q2` are caller-supplied
scratch of the same length as `qstate`.

`order` must be 3, 4, 7 or 8 -- see [`check_order`](@ref).
"""
function fquantum!(dt::Float64,
                   qstate::Vector{ComplexF64},
                   Hlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                   _q1::Vector{ComplexF64},
                   _q2::Vector{ComplexF64};
                   order::Int = 4)

    check_order(order)
    _q1 .= qstate
    for k in 1:order
        fill!(_q2, 0.0)
        a = ComplexF64(-1.0im / k * dt)
        apply!(_q2, Hlist, _q1, a)
        _q1, _q2 = _q2, _q1
        qstate .+= _q1
    end
end

"""
    fquantum!(dt, qstate, Hlist, Hnhlist, _q1, _q2; order = 4)

As above, with an additional non-Hermitian list `Hnhlist` applied one-sided --
the effective Hamiltonian of a wavefunction Monte Carlo trajectory.

- `Hlist`: Hermitian terms, applied as
  `coeff[] * H.forward + conj(coeff[]) * H.reverse`.
- `Hnhlist`: Non-Hermitian terms, applied as `coeff[] * H.forward` only.
"""
function fquantum!(dt::Float64,
                   qstate::Vector{ComplexF64},
                   Hlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                   Hnhlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                   _q1::Vector{ComplexF64},
                   _q2::Vector{ComplexF64};
                   order::Int = 4)

    check_order(order)
    _q1 .= qstate

    for k in 1:order
        fill!(_q2, 0.0)
        a = ComplexF64(-1.0im * dt / k)
        apply!(_q2, Hlist,   _q1, a)                          # Hermitian part
        apply!(_q2, Hnhlist, _q1, a; hermitian_pairs = false) # non-Hermitian part
        _q1, _q2 = _q2, _q1
        qstate .+= _q1
    end
end


"""
    fquantum!(dt, ρ, Hlist, _ρ1, _ρ2; order = 4)

Advance the density matrix `ρ` by `dt` under `-i[H, ρ]` alone, in place, by the
same truncated Taylor series. `ρ` must be Hermitian -- see
[`apply_commutator!`](@ref).
"""
function fquantum!(dt::Float64,
                   ρ::Matrix{ComplexF64},
                   Hlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                   _ρ1::Matrix{ComplexF64},
                   _ρ2::Matrix{ComplexF64};
                   order::Int = 4)

    check_order(order)
    _ρ1 .= ρ

    for k in 1:order
        fill!(_ρ2, 0.0)
        a = ComplexF64(-1im * dt / k)
        apply_commutator!(_ρ2, Hlist, _ρ1, a)
        _ρ1, _ρ2 = _ρ2, _ρ1
        ρ .+= _ρ1
    end
end

"""
    fquantum!(dt, ρ, Hlist, Jlist, _ρ1, _ρ2; order = 4)

Advance `ρ` by `dt` under the Lindblad master equation

    dρ/dt = -i[H, ρ] + Σⱼ γⱼ (Lⱼ ρ Lⱼ† - ½{Lⱼ†Lⱼ, ρ})

by Strang splitting of the Hamiltonian and dissipative flows.

`Jlist` entries are `(coeff, L, LdagL_diag)` with rate `γ = |coeff[]|²`; only
`L.forward` and the cached diagonal of `L†L` are used.
"""
function fquantum!(dt::Float64,
                   ρ::Matrix{ComplexF64},
                   Hlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                   Jlist::Vector{Tuple{Base.RefValue{ComplexF64},Op,Vector{Float64}}},
                   _ρ1::Matrix{ComplexF64},
                   _ρ2::Matrix{ComplexF64};
                   order::Int = 4)

    check_order(order)
    # Strang, second order overall: `fdissipator2!` is exact for every topology
    # except a cascade, which is itself second order.
    #
    # Left UNFUSED deliberately. Fusing the adjacent half-steps across the loop
    # would save one `fquantum!` per step but leave the state half a Hamiltonian
    # step ahead at step boundaries -- exactly where detectors read.
    fquantum!(dt / 2, ρ, Hlist, _ρ1, _ρ2; order = order)
    fdissipator2!(dt, ρ, Jlist, _ρ1, _ρ2)
    fquantum!(dt / 2, ρ, Hlist, _ρ1, _ρ2; order = order)
end

"""
    fdissipator2!(dt, ρ, Jlist, _ρ1, _ρ2)

Apply one step of the Lindblad dissipator `Σⱼ 𝒟ⱼ[ρ]` as an exact quantum channel
per source level.

AtomTwin's jump operators are single transitions, so each `Lⱼ†Lⱼ` is diagonal and
supported on the transition's *source* level. Grouping channels by source, each
group is one exact Kraus set needing two distinct rates per level `s`:

    Γ[s]    = Σⱼ γⱼ (Lⱼ†Lⱼ)[s]         every channel touching `s`
    Γout[s] = Σⱼ γⱼ |Lⱼ[p,s]|², p ≠ s   only those moving population off `s`

`Γ` damps coherences, `Γout` populations. They differ for a channel whose source
and target coincide -- pure dephasing, `L = |s⟩⟨s|`, which destroys coherence at
rate γ while returning all the population it removes. The step is

    ρ[s,s] ← exp(−Γout[s]·dt) · ρ[s,s]
    ρ[i,j] ← exp(−Γ[i]·dt/2) exp(−Γ[j]·dt/2) · ρ[i,j],   i ≠ j
    ρ      ← ρ + Σⱼ (γⱼ/Γout[s])·(1 − exp(−Γout[s]·dt)) · Lⱼ ρ Lⱼ†

Trace is preserved identically, so the step is CPTP by construction with no
rescaling and no step-size restriction.

Exact to machine precision unless a channel leaving a level feeds another
decaying level: only a *cascade* is approximate, and there it is second order.
Physicality holds at any step size regardless.

Allocation-free -- the per-level rate totals live in a cached workspace.
"""
function fdissipator2!(dt::Float64,
                       ρ::Matrix{ComplexF64},
                       Jlist::Vector{Tuple{Base.RefValue{ComplexF64},Op,Vector{Float64}}},
                       _ρ1::Matrix{ComplexF64},
                       _ρ2::Matrix{ComplexF64})
    isempty(Jlist) && return ρ
    n = size(ρ, 1)

    ws = _dissipator_ws2(n)
    Γ, Γout, m0 = ws.Γ, ws.Γout, ws.m0

    # Charging a self-transition to `Γout` would over-damp the populations and
    # drop the surrounding Strang integration to first order. See the docstring.
    fill!(Γ, 0.0)
    fill!(Γout, 0.0)
    @inbounds for (coeff, L, LdagL_diag) in Jlist
        γ = abs2(coeff[])
        for k in 1:n
            Γ[k] += γ * LdagL_diag[k]
        end
        # `LdagL_diag` cannot tell a transfer from a self-transition; the
        # transition structure can.
        for (p, j, vL) in L.forward
            p == j && continue
            Γout[j] += γ * abs2(vL)
        end
    end

    # Off-diagonals decay at `Γ`, the diagonal only at `Γout`.
    @inbounds for k in 1:n
        m0[k] = exp(-0.5 * dt * Γ[k])
    end

    ρ_in = _ρ1
    copyto!(ρ_in, ρ)

    # Column views, with the diagonal lifted out of the inner loop so it carries
    # no branch.
    @inbounds for j in 1:n
        ρcol = view(ρ, :, j); incol = view(ρ_in, :, j)
        mj = m0[j]
        @simd for i in 1:n
            ρcol[i] = m0[i] * mj * incol[i]
        end
        ρ[j, j] = exp(-dt * Γout[j]) * ρ_in[j, j]
    end

    # Self-transition (dephasing) cross term. `m0[i]*m0[j]` damps a coherence at
    # `½(Γ[i]+Γ[j])`, right for a channel that MOVES population. A
    # self-transition damps at `½γ(vᵢ-vⱼ)²` instead -- `L = v|s⟩⟨s|` cancels on a
    # coherence whose two states see the same amplitude. Expanding,
    #
    #   ½γ(vᵢ-vⱼ)² = ½γvᵢ² + ½γvⱼ² - γvᵢvⱼ
    #
    # the first two terms are already in `Γ`, hence in `m0`; only the cross term
    # is missing, so undo it rather than recompute the factor. Without this a
    # coherence between two BOTH-excited states decays at the full rate instead
    # of not at all -- invisible on one two-level atom (one of vᵢ, vⱼ is always
    # 0), O(1) wrong with N atoms.
    @inbounds for (coeff, L, _) in Jlist
        γ = abs2(coeff[])
        γ <= 0 && continue
        for (p, i, vi) in L.forward
            p == i || continue                 # self-transitions only
            ai = abs(vi)
            for (q, j, vj) in L.forward
                q == j || continue
                i == j && continue             # diagonal already handled
                ρ[i, j] *= exp(dt * γ * ai * abs(vj))
            end
        end
    end

    # Gain: Σⱼ Mⱼ ρ Mⱼ†. The population leaving `s` is `1 − exp(−Γout[s]·dt)`,
    # shared between transferring channels in proportion to their rates.
    @inbounds for (coeff, L, LdagL_diag) in Jlist
        γ = abs2(coeff[])
        for (p, j, vL) in L.forward
            p == j && continue            # self-transition: no population moves
            Γs = Γout[j]                  # j is this transition's source level
            Γs <= 0 && continue
            w = (γ / Γs) * (1 - exp(-Γs * dt))
            cj = w * vL
            for (q, k, vR) in L.forward
                q == k && continue
                ρ[p, q] += cj * ρ_in[j, k] * conj(vR)
            end
        end
    end
    return ρ
end

# Scratch for `force`: the distinct-wavelength list, sized by the beam count.
const _FORCE_WS = ThreadCache{NamedTuple{(:seen,),Tuple{Vector{Float64}}}}(
    n -> (seen = zeros(Float64, n),))

_force_ws(n::Int) = get_ws!(_FORCE_WS, n)

# Per-dimension scratch for `gershgorin_interval`: disc centres and radii.
const _GERSH_INT_WS = ThreadCache{NamedTuple{(:diag, :rad),NTuple{2,Vector{Float64}}}}(
    n -> (diag = zeros(Float64, n), rad = zeros(Float64, n)))

_gersh_interval_ws(n::Int) = get_ws!(_GERSH_INT_WS, n)

# Per-dimension scratch for `gershgorin_bound`: the row-sum accumulator.
const _GERSH_WS = ThreadCache{Vector{Float64}}(n -> zeros(Float64, n))

_gershgorin_ws(n::Int) = get_ws!(_GERSH_WS, n)

# Per-dimension scratch for `apply_commutator!`: the transposed accumulator that
# lets the kernel write its one half contiguously. Here rather than beside the
# kernel because `ThreadCache` must already be defined at load time.
const _COMM_WS = ThreadCache{NamedTuple{(:Wt,),Tuple{Matrix{ComplexF64}}}}(
    n -> (Wt = zeros(ComplexF64, n, n),))

_commutator_ws(n::Int) = get_ws!(_COMM_WS, n)

# Per-dimension scratch for `fdissipator2!`: per-level total rates and the
# attenuation diagonal.
const _DISS_WS2 = ThreadCache{NamedTuple{(:Γ, :Γout, :m0),NTuple{3,Vector{Float64}}}}(
    n -> (Γ = zeros(Float64, n), Γout = zeros(Float64, n), m0 = zeros(Float64, n)))

_dissipator_ws2(n::Int) = get_ws!(_DISS_WS2, n)

#------------------------------------------------------------------------------
# Jump selection
#------------------------------------------------------------------------------

"""
    jump!(psi, J, _prob, _psi1, _psi2, rng)

Apply one randomly selected jump from `J` to `psi` in place, and return the
`Jump` chosen. With several jumps, each is weighted by `‖Jₖ psi‖²`.

`psi` is projected but NOT renormalised -- the caller ([`quantum_jump!`](@ref))
does that. A jump of vanishing norm leaves `psi` untouched.
"""
Base.@inline function jump!(psi::Vector{ComplexF64},
                            J::Vector{Jump},
                            _prob::Weights,
                            _psi1::Vector{ComplexF64},
                            _psi2::Vector{ComplexF64},
                            rng::AbstractRNG)
    if length(J) == 1
        jump = J[1]
    else
        for (j, jump) in enumerate(J)
            _psi1 .= psi
            mul!(jump.J, _psi1, _psi2)
            _prob[j] = norm(_psi1)^2
        end
        jump = sample(rng, J, _prob)
    end
    _psi1 .= psi
    mul!(jump.J, _psi1, _psi2)
    if norm(_psi1) != 0.0   # guard: a vanishing jump would blow up the renorm
        psi .= _psi1
    end
    return jump
end

