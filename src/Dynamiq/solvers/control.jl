# Step control: how finely one user step must be divided. WFMC divides a priori
# from a jump-rate bound (`jump_substeps`); QME divides adaptively, by Richardson
# estimation under a PI filter (`strang_substeps!`, `_strang_adapt`). Also holds
# the MCWF jump machinery and the modifier resampling helpers.

#------------------------------------------------------------------------------
# Wavefunction Monte Carlo evolution
#------------------------------------------------------------------------------

"""
    quantum_jump!(psi, jumps, photo_detectors, i, steps, downsample, _prob, _q1, _q2, rng)

Test for a quantum jump after one MCWF step, fire one if `‖ψ‖² < rand()`, and
renormalise.

The test runs once per step, so at most one jump fires; [`jump_substeps`](@ref)
sizes the step so two-jump events stay within `jtol`.

A fired jump writes a click to any `PhotoDetector` bound to it, in the
downsampled bin containing raw step `i`. Trailing steps beyond the last full bin
are attributed to that bin, so no click is lost or written out of bounds.
"""
@inline function quantum_jump!(psi::Vector{ComplexF64}, jumps, photo_detectors,
                               i::Int, steps::Int, downsample::Int,
                               _prob, _q1, _q2, rng)
    n = norm(psi)
    if n^2 < rand(rng)
        fired = jump!(psi, jumps, _prob, _q1, _q2, rng)
        @inbounds for d in photo_detectors
            ## `any(===(fired), d.jumps)` rather than a single identity test: a
            ## manifold decay has one jump per sublevel channel and all of them
            ## belong to this detector.
            for j in d.jumps
                if j === fired
                    write!(d, min(cld(i, downsample), length(d.vals)))
                    break
                end
            end
        end
        n = norm(psi)
    end
    inv_n = 1.0 / n            # explicit loop: `psi ./= n` allocates per step
    @inbounds @simd for k in eachindex(psi)
        psi[k] *= inv_n
    end
    return
end

"""
    jump_rate_bound(jumps) -> Float64

Largest per-state jump rate `Γ_max = λ_max(Σⱼ γⱼ Lⱼ†Lⱼ)`, or 0 when there are no
jumps.

`Σⱼ γⱼ Lⱼ†Lⱼ` is diagonal (AtomTwin's jumps are single transitions), so this is
the largest cached `LdagL_diag` entry scaled by its rate -- no
eigendecomposition. Being state-independent, it bounds the jump probability over
a step of `h` by `1 − exp(−Γ_max·h)` before the run starts.
"""
function jump_rate_bound(jumps)
    Γmax = 0.0
    for j in jumps
        (hasproperty(j, :_coeff) && hasproperty(j, :LdagL_diag)) || continue
        j.LdagL_diag === nothing && continue
        γ = abs2(j._coeff[])
        for x in j.LdagL_diag
            Γmax = max(Γmax, γ * x)
        end
    end
    return Γmax
end

"""
    THETA_JUMP

Rotation `ΔE·h` tolerated between MCWF jump tests. See [`jump_substeps`](@ref).
"""
const THETA_JUMP = 0.5

"""
    jump_substeps(dt, Γmax, jtol, ΔE = 0.0) -> Int

How many equal sub-steps of `dt` the MCWF jump test needs. Returns 1 when there
is nothing to bound.

Two separate errors set the sub-step `h`, and the smaller bound wins:

1. **Two-jump omission.** At most one jump fires per sub-step, so the
   probability of missing a second is `Δp²` where `Δp` bounds the per-sub-step
   jump probability (Dörner et al., Comput. Phys. Commun. 234 (2019) 44).
   `Δp = √jtol` bounds the omission by `jtol` with no fitted constant, giving
   `h ≤ −log1p(−√jtol)/Γmax`.

2. **Jump-time resolution.** A jump is applied at the END of the sub-step it is
   detected in, so its time carries an error of order `h`. What makes that
   matter is not the decay rate but how far the state rotates meanwhile: the
   error is `O(ΔE·h)` with `ΔE` the spectral half-width of `H_eff`. This needs
   `ΔE·h ≲ 1` and is the binding constraint whenever `ΔE ≫ Γmax` — a
   far-detuned drive, a large Zeeman or light shift.

Bounding only (1) makes the trajectory mean depend on the user's sampling `dt`,
which it must not: with `Γmax = 2π×182 kHz` and a 2 MHz detuning, `jtol = 1e-2`
permits `ΔE·h = 3.3` and overestimates the off-resonant population **7×**, while
the master equation is exact at every `dt`. See `bugs/mcwf-jtol-underresolves.jl`
in the harness. `ΔE = 0` recovers the old behaviour for callers that have no
spectral estimate.

`THETA_JUMP` is the tolerated rotation per sub-step. It is not a fitted
constant: at `ΔE·h = 1` the timing error is a radian, and the measured bias is
already below shot noise there.
"""
@inline function jump_substeps(dt::Float64, Γmax::Float64, jtol::Float64,
                               ΔE::Float64 = 0.0)
    dt > 0 || return 1
    h_max = Inf

    # (1) two-jump omission
    if Γmax > 0
        Δp    = sqrt(min(jtol, 0.25))
        h_max = -log1p(-Δp) / Γmax
    end

    # (2) jump-time resolution: the state must not rotate far between tests.
    ΔE > 0 && (h_max = min(h_max, THETA_JUMP / ΔE))

    (isfinite(h_max) && h_max < dt) || return 1
    return max(1, ceil(Int, dt / h_max))
end

#------------------------------------------------------------------------------
# Master equation evolution (QME)
#------------------------------------------------------------------------------

# Re-evaluate every time-dependent modifier at instruction time `t`. A no-op
# when there are none, which is the common case.
@inline function _resample!(modifiers, t::Float64)
    isempty(modifiers) && return
    @inbounds for m in modifiers
        update!(m, t)
    end
    return
end

# A staircase is a LEFT-held value: sample `k` is in force from its own time
# until the next one, which is what an AWG does and what
# `interpolate_piecewise_constant` did. Reading it at the step midpoint shifts
# the bin boundaries by half a step and changes the waveform -- measured as a
# 5.2e-4 drift in the time-optimal Rydberg gate's checksum. Midpoint sampling is
# the right quadrature for a SMOOTH envelope (it is what makes the scheme second
# order); for a staircase the left edge is not an approximation but the
# definition.
@inline _modifier_time(m, tmid::Float64, h::Float64) =
    _is_staircase(m) ? tmid - 0.5h : tmid
@inline _is_staircase(m) = false
@inline _is_staircase(m::AmplitudeModifier) = m.interp === :constant
@inline _is_staircase(m::PositionModifier)  = m.interp === :constant

# Does any active modifier hold a piecewise-constant envelope? Used by the
# sub-step controller to decide whether the local error can be assumed smooth
# across a user step.
@inline function _has_staircase(modifiers)
    @inbounds for m in modifiers
        _is_staircase(m) && return true
    end
    return false
end

@inline function _resample!(modifiers, tmid::Float64, h::Float64)
    isempty(modifiers) && return
    @inbounds for m in modifiers
        _resample_one!(m, tmid, h)
    end
    return
end

# Function barrier. `modifiers` is a `Vector{AbstractModifier}` -- it holds a mix
# of concrete types, so iterating it is a dynamic dispatch and the loop body
# cannot be specialised. Calling through this boundary lets Julia specialise on
# the concrete type once per element, which takes the per-call cost from 48 B to
# zero; `sample_at` and `update!` are already allocation-free when called with a
# known type.
@inline function _resample_one!(m, tmid::Float64, h::Float64)
    update!(m, _modifier_time(m, tmid, h))
    return
end

# How many times one user step may be retaken while the controller refines `k`.
#
# Each attempt costs three sub-step flows regardless of `k` -- the estimate uses
# only the first pair -- while the advance costs `2k`, so refining is cheap next
# to the work it sizes. The cap is not binding on any measured case.
const MAX_SUBSTEP_RETRIES = 10

"""
    SolverProbe

Instrumentation hook for the adaptive controller. A probe receives a NamedTuple
per sub-step pair carrying what [`strang_substeps!`](@ref) saw: the step and
attempt index, the pair index, `k`, `h`, the error ratio `r`, the state before
and after, and the operator lists.

Implement `probe!(p::MyProbe, info)` on a subtype to observe a run without
patching the solver. The default [`NoProbe`](@ref) compiles the call away.
"""
abstract type SolverProbe end

"""
    NoProbe <: SolverProbe

The default probe: observes nothing, and inlines away so an uninstrumented run
pays neither a call nor a branch.
"""
struct NoProbe <: SolverProbe end

"""
    probe!(probe, info)

Deliver one instrumentation record to `probe`. No-op for [`NoProbe`](@ref).
"""
@inline probe!(::NoProbe, info) = nothing

"""
    StrangControl(n)

Scratch and state for the adaptive Strang stepper ([`strang_substeps!`](@ref)).

`k` is the number of sub-step PAIRS one user step is divided into. It persists
across steps: the local error varies smoothly, so last step's `k` is the right
starting guess for this one.

`e1`/`e2` are the previous two accepted inverse error ratios `tol/rel` for the
PI filter in `_strang_adapt`; `0.0` means no history yet.
"""
mutable struct StrangControl
    k::Int                      # current sub-step pairs per user step
    seeded::Bool                # has `k` been set by a measurement yet?
    e1::Float64                 # previous accepted eps = tol/rel  (0 = none)
    e2::Float64                 # the one before that             (0 = none)
    warned::Bool                # has the tolerance-not-met warning fired?
    stepno::Int                 # user-step index, for instrumentation only
    probe::SolverProbe          # `NoProbe()` unless a run is instrumented
    a::Matrix{ComplexF64}       # incoming state / two-half-steps result
    b::Matrix{ComplexF64}       # one-whole-step reference
    t1::Matrix{ComplexF64}
    t2::Matrix{ComplexF64}
end
StrangControl(n::Int) = StrangControl(1, false, 0.0, 0.0, false, 0, NoProbe(),
                                      (zeros(ComplexF64, n, n) for _ in 1:4)...)

const _STRANG_CTL = ThreadCache{StrangControl}(StrangControl)

"""
    strang_control(n) -> StrangControl

Per-dimension, per-thread controller state, allocated once and reused.
"""
strang_control(n::Int) = get_ws!(_STRANG_CTL, n)

"""
    strang_reset!(ctl)

Clear the adaptive history at the start of an instruction. `k` and `seeded` are
NOT cleared: the previous instruction's converged pair count is the best
available guess for this one.
"""
@inline function strang_reset!(ctl::StrangControl)
    ctl.e1 = 0.0
    ctl.e2 = 0.0
    ctl.warned = false
    ctl.stepno = 0
    return ctl
end

# One Richardson-controlled sub-step pair. Takes two sub-steps of `h` in place
# (these ARE the advance) plus one `2h` step from the same start into `ctl.b`,
# and returns the tolerance-scaled error ratio `r = rel/tol`; `r <= 1` accepts.
#
# Each sub-step reads the drives at its OWN midpoint -- freezing a coefficient
# across the pair would hide the Magnus error from the estimate.
@inline function _strang_pair!(ρ::Matrix{ComplexF64}, h::Float64, tmid::Float64,
                               Hlist, Jlist, ctl::StrangControl,
                               _ρ1, _ρ2, order::Int, tol::Float64, modifiers)
    copyto!(ctl.b, ρ)
    _resample!(modifiers, tmid - 0.5h, h)
    fquantum!(h, ρ, Hlist, Jlist, _ρ1, _ρ2; order = order)
    _resample!(modifiers, tmid + 0.5h, h)
    fquantum!(h, ρ, Hlist, Jlist, _ρ1, _ρ2; order = order)

    _resample!(modifiers, tmid, 2h)
    # `ctl.t1`/`ctl.t2` need no clearing: `fquantum!` and `fdissipator2!` both
    # fully write their scratch before reading it.
    fquantum!(2h, ctl.b, Hlist, Jlist, ctl.t1, ctl.t2; order = order)

    diff = 0.0
    @inbounds for m in eachindex(ρ); diff += abs2(ρ[m] - ctl.b[m]); end
    # Richardson gives the local error of the coarse (2h) member as
    # `diff / (1 - 2^-p)`, p = 2. The advance takes an h-PAIR instead, whose
    # error is smaller by 4.
    err_2h = sqrt(diff) / (1 - 0.25)
    # ABSOLUTE error: `tr ρ = 1` identically, so 1 is the natural reference.
    # Dividing by `‖ρ‖_F` -- the square root of the purity, which runs over
    # `[1/√d, 1]` -- would let a decohering run silently tighten its own tol.
    rel = err_2h / 4
    # Non-finite means the pair overflowed past the stability limit; report an
    # unbounded rejection. (`rel == 0` is fine and accepts: pure dephasing
    # reproduces the step exactly, since `D(h)∘D(h) = D(2h)`.)
    isfinite(rel) || return Inf
    return rel / tol
end

# Two sub-steps of `h` with no error estimate: the advance alone, at 2/3 the cost
# of `_strang_pair!`. Used for interior pairs whose ends are both estimated.
@inline function _strang_pair_plain!(ρ::Matrix{ComplexF64}, h::Float64, tmid::Float64,
                                     Hlist, Jlist, _ρ1, _ρ2, order::Int, modifiers)
    _resample!(modifiers, tmid - 0.5h, h)
    fquantum!(h, ρ, Hlist, Jlist, _ρ1, _ρ2; order = order)
    _resample!(modifiers, tmid + 0.5h, h)
    fquantum!(h, ρ, Hlist, Jlist, _ρ1, _ρ2; order = order)
    return nothing
end

"""
    strang_substeps!(dt, ρ, Hlist, Jlist, tol, ctl, _ρ1, _ρ2, order) -> Int

Advance `ρ` by `dt` in `ctl.k` equal Strang sub-step pairs, adapting `k` so each
pair's local error stays within `tol`. Returns the `k` actually used.

`tol` bounds one sub-step pair, as a step tolerance normally does; error
accumulated over the run grows with the number of steps in the usual way.

**Sub-steps of a fixed `dt`, not a free step.** `dt` is also the control grid:
modifiers are resampled onto it and advanced once per user step. Dividing by an
integer keeps every modifier firing where it was compiled to fire, which a
freely-varying step would silently break.

**Step doubling for the estimate.** Each pair takes two steps of `h` plus one of
`2h`; Richardson gives the `2h` local error as
`‖x_{2h} − x_{h,h}‖ / (1 − 2^{-p})`, p = 2. Measured against the exact channel on
a driven, detuned, decaying two-level atom, estimate/true is 1.0019 → 1.0000 as
`h` falls and stays within 1.09–1.28 even at `θ = ‖H‖h = 2` — erring high, so the
controller errs toward smaller steps. Cost is one extra flow per pair; the
symmetrized defect estimator (Auzinger et al., arXiv:1806.07771) is as accurate
but needs three more, measured at 4.2–5.2x a step.

**Only the two end pairs are estimated**, with the interior covered by the
`STRANG_BRACKET` margin — see the comment on the pair loop. A staircase envelope
breaks the smoothness that rests on, and forces every pair to be estimated.
"""
function strang_substeps!(dt::Float64,
                          ρ::Matrix{ComplexF64},
                          Hlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                          Jlist::Vector{Tuple{Base.RefValue{ComplexF64},Op,Vector{Float64}}},
                          tol::Float64,
                          ctl::StrangControl,
                          _ρ1::Matrix{ComplexF64},
                          _ρ2::Matrix{ComplexF64},
                          order::Int,
                          modifiers = (),
                          t0::Float64 = 0.0)
    isempty(Jlist) && (fquantum!(dt, ρ, Hlist, _ρ1, _ρ2; order = order); return 1)
    # No Hamiltonian: `fdissipator2!` applies the dissipator as an exact channel
    # at any step size, so there is no splitting error to control and no `k` that
    # would improve it.
    isempty(Hlist) && (fdissipator2!(dt, ρ, Jlist, _ρ1, _ρ2); return 1)

    # `k` counts sub-step PAIRS: the advance is 2k sub-steps of `h = dt/2k`, and
    # every two of them form a step-doubling pair against one extra `2h` flow.
    # Counting single sub-steps would need a special construction at `k = 1`,
    # whose two half-steps the advance then discards.
    ctl.stepno += 1

    # A pair may miss `tol`, in which case the whole user step is retaken from
    # here at a finer `k`.
    copyto!(ctl.a, ρ)

    # Reuse the previous step's `k` once measured -- the dynamics vary slowly
    # within an instruction. A cold start begins at 1 and doubles until the
    # estimate clears the decorrelation floor (`STRANG_DECORR`).
    k = ctl.seeded ? ctl.k : 1
    ctl.seeded = true

    rworst = 0.0
    @inbounds for attempt in 1:MAX_SUBSTEP_RETRIES
        h       = dt / (2k)
        rworst  = 0.0
        ok      = true
        copyto!(ρ, ctl.a)

        # Estimate the two END pairs and advance the interior unestimated. The
        # local error varies smoothly across a user step, so the ends bracket
        # the interior: measured, the worst pair exceeds `max(r_first, r_last)`
        # by a median of 1.001 and never by more than 1.9x. The interior is then
        # covered by a margin rather than a probe -- the test below is
        # `r <= 1/STRANG_BRACKET`, which absorbs that worst case.
        #
        # A STAIRCASE envelope breaks the smoothness by construction: its sample
        # boundaries are not aligned to the solver's steps, so `r(p)` jumps
        # wherever one falls. Estimate every pair then; the test is a static
        # property of the modifier list.
        probe_all = _has_staircase(modifiers)
        pfirst = 1
        plast  = k
        for p in 1:k
            estimate = probe_all || p == pfirst || p == plast
            r = if estimate
                _strang_pair!(ρ, h, t0 + (2p - 1) * h, Hlist, Jlist, ctl,
                              _ρ1, _ρ2, order, tol, modifiers)
            else
                _strang_pair_plain!(ρ, h, t0 + (2p - 1) * h, Hlist, Jlist,
                                    _ρ1, _ρ2, order, modifiers)
                0.0
            end
            r > rworst && (rworst = r)

            probe!(ctl.probe, (step = ctl.stepno, attempt = attempt,
                               pair = p, k = k, h = h, r = r,
                               t = t0 + (2p - 1) * h,
                               estimated = estimate,
                               rho = ρ, rho_in = ctl.a, ctl = ctl,
                               H = Hlist, J = Jlist, tol = tol, order = order))

            # The standard accept/reject rule (Hairer & Wanner). A diverging pair
            # blows the tolerance out on its own, so no separate norm guard is
            # needed -- and a norm test would be wrong anyway, since `‖ρ‖_F` is
            # conserved by a unitary step. Bail on the FIRST failure: later
            # pairs would integrate on from an already-rejected state.
            if r > 1 / STRANG_BRACKET
                ok = false
                break
            end
        end

        ok && break

        knew = _strang_adapt(ctl, k, rworst, rworst * tol)
        # Only an ACCEPTED step may be kept: committing a rejected one leaves
        # the trajectory on a state the estimator has already refused.
        copyto!(ρ, ctl.a)
        # Cannot refine further, or the retry budget is spent: take the finest
        # advance available rather than freezing time.
        if knew == k || attempt == MAX_SUBSTEP_RETRIES
            k = max(knew, k)
            h = dt / (2k)
            rworst = 0.0
            for p in 1:k
                r = _strang_pair!(ρ, h, t0 + (2p - 1) * h, Hlist, Jlist, ctl,
                                  _ρ1, _ρ2, order, tol, modifiers)
                r > rworst && (rworst = r)
            end
            # Committed whether or not it met `tol`. A silent miss is the failure
            # mode to avoid -- the run completes and looks plausible -- so warn
            # once.
            if rworst > 1 && !ctl.warned && isfinite(rworst)
                ctl.warned = true
                @warn """
                Solver could not meet the step tolerance.

                  requested  steptol = $(tol)
                  achieved   ≈ $(round(rworst * tol, sigdigits = 3))   ($(round(rworst, sigdigits = 3))x)
                  at $(k) sub-step pairs after $(MAX_SUBSTEP_RETRIES) refinements

                The step was taken anyway -- stopping would freeze time -- so
                results from this instruction carry that error. Reduce the
                sequence step, or raise `steptol` if this accuracy is adequate.
                """
            end
            break
        end
        k = knew
    end

    # Propose the `k` this step's error justifies for the next one.
    ctl.k = _strang_adapt(ctl, k, rworst, rworst * tol)
    # Only an ACCEPTED step enters the filter's memory: a forced advance may be
    # above `tol`, and feeding that forward has the integral term chase an error
    # the controller never approved.
    if isfinite(rworst) && rworst > 0 && rworst <= 1
        ctl.e2 = ctl.e1
        ctl.e1 = 1 / rworst
    else
        ctl.e2 = 0.0
        ctl.e1 = 0.0
    end
    return k
end

# Step-size filter: map the measured error ratio to the pair count to try next.
#
# The control variable is `h = dt/2k`, so multiplying `h` by `q` DIVIDES `k` by
# it. Elementary form is `h <- h*(tol/rel)^(1/(p+1))`, p = 2 for Strang. The
# order is the method's, known a priori -- fitting it from measurements is
# unstable and must not be attempted.
#
# PI/PID (Soderlind, ACM TOMS 29 (2003) 1-26; Hairer & Wanner II.4) adds memory
# of the previous ratios, damping the oscillation a proportional-only controller
# shows near a stability boundary. `eps = tol/rel` is the INVERSE scaled error,
# so every exponent is positive.
#
# `k` is an integer, so round UP on refinement -- the rounding only ever buys
# accuracy. The safety factor and `qmin`/`qmax` are the standard Hairer-Wanner
# values, bounding how far one measurement may be extrapolated.
const STRANG_SAFETY = 0.9
const STRANG_QMIN   = 0.2
const STRANG_QMAX   = 10.0

# `PI42` of Soderlind (ACM TOMS 29 (2003) 1-26): beta1 = 0.6, beta2 = -0.2,
# divided by `p + 1 = 3` for the order-2 Strang scheme. Published values, not
# fitted here.
const STRANG_BETA1 = 0.6 / 3
const STRANG_BETA2 = -0.2 / 3

# Scaled error above which the Richardson quotient carries no gradient.
#
# Past the stability limit the two flows being differenced are unrelated points
# on the state manifold rather than a fine and a coarse approximation, so `rel`
# saturates at a value fixed by the estimator's own constants -- independent of
# the system, the state and `tol`. Above this floor the order-3 law is fitting
# noise, so a saturated measurement is recognised rather than extrapolated from.
const STRANG_DECORR = 1 / 6

# Margin by which the first+last bracket may underestimate the worst pair.
#
# Only the ends of a user step are estimated. Measured over 514 sweeps across
# tol = 1e-4 ... 1e-8 on the N=5 blockade, the worst pair exceeded
# `max(r_first, r_last)` by at most 2.03x (median 1.001), so accepting on
# `r <= 1/STRANG_BRACKET` bounds every pair, at the cost of running `k` about
# 1.26x finer.
#
# A trigger on the two ends DISAGREEING was tried instead and measures the wrong
# quantity -- a symmetric interior peak gives ends that agree exactly while being
# the worst case. Correlation with the actual underestimate was -0.022.
const STRANG_BRACKET = 2.0

# `r` here is compared against the same tightened threshold the caller accepts
# on, so that the filter's notion of "this step passed" matches the integrator's.
@inline function _strang_adapt(ctl::StrangControl, k::Int, r::Float64,
                               rel::Float64 = -1.0)
    r *= STRANG_BRACKET
    # Blew up: no ratio to extrapolate from, so halve the step unconditionally.
    isfinite(r) || return clamp(2k, 1, 1 << 20)

    # Saturated: the measurement says "this `k` is infeasible" and nothing more.
    # Refine unconditionally rather than extrapolate a ratio from decorrelation.
    rel >= STRANG_DECORR && return clamp(2k, 1, 1 << 20)

    # `r == 0` is an exactly reproduced step, not a failure -- it happens whenever
    # the split flows commute (pure dephasing: `D(h)∘D(h) = D(2h)`). There is no
    # error to control, so coarsen by the standard bound. Doubling `k` here
    # instead runs the pure-dephasing regression in 81 s against 5 s.
    r <= 0 && return max(cld(4k, 5), 1)

    ε = 1 / r
    # Proportional term alone on a rejection: the previous ratio came from a
    # different `k` and an accepted state, so it says nothing about the step that
    # just failed. Standard practice (Hairer & Wanner II.4).
    q = if r > 1 || ctl.e1 <= 0
            STRANG_SAFETY * ε^(1 / 3)
        else
            STRANG_SAFETY * ε^STRANG_BETA1 * ctl.e1^STRANG_BETA2
        end
    q = clamp(q, STRANG_QMIN, STRANG_QMAX)

    # `h <- h*q` and `h = dt/2k`, so `k <- k/q`.
    knew = r > 1 ? ceil(Int, k / q) : floor(Int, k / q)
    knew = max(knew, 1)
    # Coarsen by at most a factor 4/5 per accepted step; refine as far as asked.
    #
    # The directions are not symmetric. Refinement is verified immediately -- the
    # step is retaken and re-estimated. Coarsening is an EXTRAPOLATION: nothing
    # checks it until the next step is already spent, and it fails near the
    # stability wall where the error stops following the order-3 law (measured: a
    # factor 77 for a 2x step increase against the 8x predicted, giving a 50%
    # rejection rate in a 2 -> 1 -> reject -> 2 cycle).
    #
    # Bounding the RATIO rather than the integer delta matters: one pair out of
    # k=2 is 50% of `h`, one out of k=32 is 3%, so a fixed one-pair limit stalls
    # at large `k` -- measured 20-30% slower at tight tolerances. The exact ratio
    # does not: 4/5, 3/4 and 2/3 all measure within noise.
    r <= 1 && (knew = max(knew, cld(4k, 5)))
    # Never coarsen to a `k` whose predicted error reaches the decorrelation
    # floor, where the estimator stops working.
    if r <= 1 && rel > 0 && knew < k
        while knew < k && rel * (k / knew)^3 >= STRANG_DECORR
            knew += 1
        end
    end
    # A rejection must refine and an acceptance must not: otherwise the integer
    # floor/ceil can return the same `k` and the retry loop spins.
    r > 1 && knew <= k && (knew = k + 1)
    r <= 1 && knew > k && (knew = k)
    return clamp(knew, 1, 1 << 20)
end

