# Integrator selection and propagator plans.
#
# `AbstractIntegrator` picks a scheme by dispatch; `IntegratorPlan` holds the
# per-step data that scheme needs (Chebyshev coefficients, Taylor scratch) so the
# step loop allocates nothing. Also holds `ThreadCache`, the per-thread scratch
# used by the kernels that have no plan in scope.

#------------------------------------------------------------------------------
# Integrator selection
#------------------------------------------------------------------------------

"""
    AbstractIntegrator

Supertype for the scheme used to apply `exp(-iH dt)` to a state.

Selected by dispatch rather than a runtime branch, so adding one costs nothing in
the hot loop. To add one, define a subtype and one `propagate!` method:

```julia
struct MyScheme <: AbstractIntegrator end

Dynamiq.propagate!(::MyScheme, ψ, terms, dt, spec; kwargs...) = ...
```

`spec` carries whatever the scheme needs that is expensive to recompute per step
(spectral bounds, for instance); see [`SpectralSpec`](@ref).
"""
abstract type AbstractIntegrator end

"""
    Chebyshev() <: AbstractIntegrator

Chebyshev-expansion propagator -- the default.

No stability limit (verified to machine precision at `‖H‖dt = 50`, 18x past the
Taylor-4 limit), `O(d)` workspace independent of degree, and a degree chosen
automatically from the spectrum and tolerance.
"""
struct Chebyshev <: AbstractIntegrator end

"""
    Taylor(order = 4) <: AbstractIntegrator

Truncated-Taylor propagator, kept selectable:

    play(sys, seq; integrator = AtomTwin.Dynamiq.Taylor(8))

Fixed cost per step against Chebyshev's tolerance-dependent degree, so cheaper
when `ΔE·dt` is small -- but with a hard stability limit at `θ = ‖H‖·dt = 2√2`,
beyond which it diverges. Being a structurally different scheme, it is also a
cheap cross-check when a result looks wrong.

`order` is the number of Taylor terms kept, and belongs to this propagator alone
(Chebyshev sets its degree from `ΔE·dt` and `tol`). Valid orders are 3, 4, 7, 8;
the constructor rejects the rest -- see [`stability_limit`](@ref).
"""
struct Taylor <: AbstractIntegrator
    order::Int
    Taylor(order::Int = 4) = (check_order(order); new(order))
end

"""
    SpectralSpec(Emin, Emax)

Per-instruction data an integrator may need but should not recompute per step.
For [`Chebyshev`](@ref), bounds on the spectrum of `H` from
[`spectral_spec`](@ref). Schemes that need nothing may ignore it.
"""
struct SpectralSpec
    Emin::Float64
    Emax::Float64
    Iradius::Float64      # Gershgorin radius of the ANTI-Hermitian part
end

# Back-compat: a spec built without an imaginary radius is treated as Hermitian.
SpectralSpec(Emin::Float64, Emax::Float64) = SpectralSpec(Emin, Emax, 0.0)

"""
    spectral_spec(terms; peak = true) -> SpectralSpec

Bound the spectrum of `Σⱼ cⱼHⱼ`, plus the radius of its anti-Hermitian part, in
O(nnz) with no matrix assembled. Conservative by construction: a looser interval
costs a slightly higher Chebyshev degree, never correctness.

`peak = true` bounds over the whole sequence rather than the current instant,
since couplings are built inactive and switched on later.
"""
function spectral_spec(terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}};
                       peak::Bool = true)
    lo, hi = gershgorin_interval(terms; peak = peak)
    SpectralSpec(lo, hi, antihermitian_bound(terms; peak = peak))
end

"""
    gershgorin_interval(terms; peak = false) -> (Emin, Emax)

Gershgorin bounds on the spectrum of `Σⱼ cⱼHⱼ` as an interval, each disc centred
on its own diagonal entry: `Emin = minᵢ(Hᵢᵢ - rᵢ)`, `Emax = maxᵢ(Hᵢᵢ + rᵢ)` with
`rᵢ` the off-diagonal row sum.

Tighter than a symmetric `±‖H‖` bound whenever the spectrum is offset from zero
-- a far-detuned drive, a hyperfine splitting -- which halves `ΔE` and so the
Chebyshev degree. Shifting `H` by a constant changes no eigenvalue difference, so
no observable moves.

With `peak = true` the radius uses the peak coupling amplitude; the disc centre
always uses the actual coefficient.
"""
function gershgorin_interval(terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}};
                             peak::Bool = false)
    isempty(terms) && return (0.0, 0.0)
    dim = terms[1][2].dim
    ws = _gersh_interval_ws(dim)
    diag, rad = ws.diag, ws.rad
    fill!(diag, 0.0); fill!(rad, 0.0)
    @inbounds for (coeff_ref, op) in terms
        c = coeff_ref[]
        # The RADIUS uses the peak amplitude for the same reason `gershgorin_bound`
        # does, but the CENTRE must use the actual coefficient: a diagonal term
        # contributes its value, not its magnitude, so inflating it would MOVE
        # the interval rather than widen it.
        cp = peak ? max(abs(c), 1.0) : abs(c)
        for (i, j, v) in op.forward
            i == j ? (diag[i] += real(c * v)) : (rad[i] += cp * abs(v))
        end
        for (i, j, u) in op.reverse
            i == j ? (diag[i] += real(conj(c) * u)) : (rad[i] += cp * abs(u))
        end
    end
    lo = Inf; hi = -Inf
    @inbounds for i in 1:dim
        lo = min(lo, diag[i] - rad[i])
        hi = max(hi, diag[i] + rad[i])
    end
    return (lo, hi)
end

"""
    antihermitian_bound(terms; peak = false) -> Float64

Gershgorin radius of the anti-Hermitian part of `Σⱼ cⱼHⱼ`. Distinguishes a
Hermitian propagation from an MCWF effective Hamiltonian, which is what bounds
the Chebyshev step (see [`propagator_theta`](@ref)).

Entries are accumulated per `(i,j)` before differencing: `|aᵢⱼ - conj(aⱼᵢ)|/2`
needs the values, and a row sum over `|entry|` would discard the phase and return
the full radius for any operator.
"""
function antihermitian_bound(terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}};
                             peak::Bool = false)
    isempty(terms) && return 0.0
    dim = terms[1][2].dim
    acc = Dict{Tuple{Int,Int},ComplexF64}()
    @inbounds for (coeff_ref, op) in terms
        c = peak ? ComplexF64(max(abs(coeff_ref[]), 1.0)) : coeff_ref[]
        for (i, j, v) in op.forward
            acc[(i, j)] = get(acc, (i, j), 0.0im) + c * v
        end
        for (i, j, u) in op.reverse
            acc[(i, j)] = get(acc, (i, j), 0.0im) + conj(c) * u
        end
    end
    rows = zeros(Float64, dim)
    @inbounds for ((i, j), v) in acc
        w = conj(get(acc, (j, i), 0.0im))
        rows[i] += abs(v - w) / 2
    end
    maximum(rows)
end

"""
    ChebyshevWorkspace(dim)

Scratch for [`chebyshev!`](@ref): three state vectors for the three-term
recurrence plus the Bessel coefficient table, allocated once per problem size.

Memory is `O(d)` regardless of degree -- the structural advantage over a Krylov
method, which must retain `m` basis vectors (~105 MB at `d`=2¹⁶, `m`=150,
against ~4 MB here).
"""
struct ChebyshevWorkspace
    ϕkm1::Vector{ComplexF64}
    ϕk::Vector{ComplexF64}
    ϕkp1::Vector{ComplexF64}
    coef::Vector{Float64}
end
ChebyshevWorkspace(dim::Int; maxdeg::Int = 4096) =
    ChebyshevWorkspace(zeros(ComplexF64, dim), zeros(ComplexF64, dim),
                       zeros(ComplexF64, dim), zeros(Float64, maxdeg))

"""
    ensure_degree!(ws, need)

Grow the Bessel coefficient buffer so a degree of `need` fits.

The degree is set by `ΔE·dt`, unknown when the workspace is built -- a stiff
Hamiltonian over a long step can need tens of thousands of terms. Growing is
safer than capping: a silently truncated series returns a WRONG state with no
error raised and no response to `tol`.
"""
function ensure_degree!(ws::ChebyshevWorkspace, need::Int)
    length(ws.coef) >= need && return ws
    resize!(ws.coef, nextpow(2, need))
    return ws
end

"""
    ThreadCache{V}(build)

Per-thread scratch store: one `Dict{Int,V}` keyed by dimension per thread, held
in a vector indexed by `threadid()`. `build(dim)` makes a missing entry.

`play` runs shots under `Threads.@threads`, so global scratch would let
concurrent shots overwrite each other. One shared `Dict` keyed by
`(threadid(), dim)` is not enough either -- concurrent `get!` inserts corrupt it
during `rehash!`. Indexing a pre-sized outer vector means threads never write to
shared structure at all.

Sized by `maxthreadid()` and grown if the pool grows.
"""
struct ThreadCache{V,F}
    slots::Vector{Dict{Int,V}}
    build::F
    lock::ReentrantLock
end
ThreadCache{V}(build::F) where {V,F} =
    ThreadCache{V,F}([Dict{Int,V}() for _ in 1:Threads.maxthreadid()],
                     build, ReentrantLock())

"""
    reset!(c::ThreadCache)

Drop every cached workspace and resize the slot vector to the current thread
count.

A `const` `ThreadCache` sizes itself at package build time, so this is called
from `__init__` to give the loading session slots for its own threads, and after
a precompile workload so no build-time state is serialised into the image.
"""
function reset!(c::ThreadCache{V}) where {V}
    @lock c.lock begin
        resize!(c.slots, Threads.maxthreadid())
        for i in eachindex(c.slots)
            c.slots[i] = Dict{Int,V}()
        end
    end
    return c
end

@inline function get_ws!(c::ThreadCache{V}, dim::Int) where {V}
    tid = Threads.threadid()
    # Fast path: this thread's own `Dict`, touched by no one else.
    tid <= length(c.slots) && return get!(() -> c.build(dim), c.slots[tid], dim)
    # Slow path: the pool grew past our vector (an interactive-pool `@spawn`).
    # Growing *is* shared mutation, so it takes the lock -- once per new thread,
    # never in the stepping loop.
    @lock c.lock begin
        if tid > length(c.slots)
            old = length(c.slots)
            resize!(c.slots, max(tid, Threads.maxthreadid()))
            for i in (old+1):length(c.slots)
                c.slots[i] = Dict{Int,V}()
            end
        end
    end
    return get!(() -> c.build(dim), c.slots[tid], dim)
end



"""
    besselj_series!(c, x, tol) -> n

Fill `c` with `J₀(x), J₁(x), …` by Miller's downward recurrence, stopping once a
coefficient falls below `tol`. Returns the number of usable coefficients.

Hand-rolled rather than taken from SpecialFunctions.jl so `Dynamiq` gains no
dependency (CONTRIBUTING asks that it stay extractable as a standalone engine).
The coefficients depend only on `ΔE·t`, so a constant-`H` instruction computes
them once -- this never runs in the hot loop.

Downward recurrence is the stable direction: `J_{k-1} = (2k/x)J_k − J_{k+1}`
damps error going down, whereas the upward form amplifies it catastrophically
once `k > x`.
"""
function besselj_series!(c::Vector{Float64}, x::Float64, tol::Float64)
    n = length(c)
    if x == 0
        c[1] = 1.0
        return 1
    end

    # Miller: recur DOWNWARD from a start well above both n and x, seeded
    # arbitrarily, then fix the scale by `J₀(x) + 2·Σ_{k≥1} J_{2k}(x) = 1`.
    # Downward is the stable direction; upward amplifies error once k > x.
    #
    # The start must be generous -- the recurrence forgets its seed only after
    # enough steps -- and is set by `x`, not the buffer capacity: tying it to `n`
    # would run a full-length recurrence for a degree of 5. Holds machine
    # precision to at least x = 2000.
    m = ceil(Int, 2 * abs(x) + 64 + 12 * sqrt(abs(x)))
    nwork = min(n, m)
    @inbounds for j in 1:nwork; c[j] = 0.0; end
    jkp1 = 0.0      # J_{k+1}
    jk   = 1e-300   # J_k  (unnormalised seed)
    total = 0.0
    @inbounds for k in m:-1:1
        jkm1 = (2k / x) * jk - jkp1         # J_{k-1}
        jkp1, jk = jk, jkm1                 # now jk == J_{k-1}
        k - 1 < nwork && (c[k] = jk)        # store J_{k-1} at index k
        iseven(k - 1) && k - 1 > 0 && (total += 2jk)
        if abs(jk) > 1e250                  # rescale before overflow
            jk *= 1e-250; jkp1 *= 1e-250; total *= 1e-250
            for j in 1:nwork; c[j] *= 1e-250; end
        end
    end
    total += c[1]                           # the J₀ term
    s = 1.0 / total
    @inbounds for j in 1:nwork; c[j] *= s; end

    # Truncation. J_k(x) OSCILLATES for k < x and decays monotonically only past
    # it, so "first coefficient below tol" fires on a zero crossing and halves
    # the series -- at x ~ 1.4e4, tol = 1e-6 gave an O(1) error while tol = 1e-9
    # looked fine. Test only past the turning point, and require a RUN of small
    # coefficients.
    start = max(3, ceil(Int, abs(x)) + 1)
    start > n && return n
    last = n
    runlen = 0
    @inbounds for j in start:n
        runlen = abs(c[j]) < tol ? runlen + 1 : 0
        if runlen >= 4
            # Trim the confirming tail: convergence was at the FIRST of the four,
            # and keeping all four costs three extra matvecs per step (degree 7
            # instead of 4 on a well-resolved step).
            last = j - 3
            break
        end
    end
    return max(last, 2)
end


"""
    IntegratorPlan

Per-instruction state for a propagator: everything derivable from `dt` and the
spectrum, hoisted out of the step loop by `plan_step`.

Each integrator defines its own subtype. The plan also owns that propagator's
scratch: built once per instruction and used by one task, so it is private by
construction with no global cache shared between concurrent shots.
"""
abstract type IntegratorPlan end

"""
    ChebyshevPlan(dt, Emin, Emax, tol, ws)

The shifted/scaled spectrum, Bessel coefficients and truncation degree -- all
fixed across an instruction, since `x = ΔE·dt` and `tol` are. Miller's recurrence
dominates a small step, so hoisting it is what makes short steps cheap.

The plan owns its workspace; stepping through a built plan allocates nothing.
"""
mutable struct ChebyshevPlan <: IntegratorPlan
    const ws::ChebyshevWorkspace
    const dt::Float64
    const ΔE::Float64
    Ē::Float64                  # see `recenter!`: refreshed per step, not per plan
    const deg::Int
    const degenerate::Bool
end

function ChebyshevPlan(dt::Float64, Emin::Float64, Emax::Float64,
                       tol::Float64, ws::ChebyshevWorkspace)
    ΔE = (Emax - Emin) / 2
    Ē  = (Emax + Emin) / 2
    # Degenerate spectrum: H is a multiple of the identity over this subspace,
    # so the step is a global phase and no expansion is needed.
    ΔE <= 0 && return ChebyshevPlan(ws, dt, ΔE, Ē, 0, true)

    x = ΔE * dt
    ensure_degree!(ws, ceil(Int, x) + 128 + 8 * ceil(Int, sqrt(max(x, 1.0))))
    deg = besselj_series!(ws.coef, x, tol)
    if deg >= length(ws.coef) && abs(ws.coef[end]) > tol
        error("Chebyshev expansion did not converge: ΔE·dt = $x needs a degree " *
              "beyond the coefficient buffer (got $deg, last coefficient " *
              "$(ws.coef[end])). This should not happen -- ensure_degree! is " *
              "meant to size the buffer first.")
    end
    return ChebyshevPlan(ws, dt, ΔE, Ē, deg, false)
end

"""
    recenter!(plan::ChebyshevPlan, terms) -> plan

Move the plan's expansion centre `Ē` to the current spectrum of `terms`, keeping
its Bessel coefficients.

# Why this exists

The Chebyshev expansion is built about a centre `Ē` and a half-width `ΔE`:

    exp(-iH dt)ψ = e^{-iĒdt} Σ_k c_k(ΔE·dt) T_k((H - Ē)/ΔE) ψ

and it converges only while the spectrum of `H` stays inside `[Ē-ΔE, Ē+ΔE]`. A
plan is built once per instruction because `ΔE·dt` and `tol` are fixed, and the
coefficients `c_k` -- Miller's recurrence, the expensive part -- depend on
nothing else.

`Ē`, though, is not fixed when the Hamiltonian follows the atom. A trapped atom
crossing its tweezer sweeps the trap light shift over its whole range: for a
50 mW, 1 µm tweezer on Yb-171 the centre moves by 1.8e8 rad/s -- 176 radians per
step -- while the half-width, set by the much smaller *differential* shift,
barely moves at all. The expansion is then evaluated far outside its domain,
where Chebyshev polynomials grow exponentially, and the trajectory diverges.

The two fixes that do not work are worth recording. Bounding the light shift
conservatively over its whole range takes `ΔE·dt` from 0.53 to 88.8 and the
degree from 9 to 169, penalising every trapped simulation. Rebuilding the plan
each step costs 7.3 µs against a 95 ns propagate, because it redoes the Bessel
series.

Refreshing `Ē` alone is neither: it enters only as a scalar in the recurrence
and as the global phase `e^{-iĒdt}`, never inside `c_k`, so it costs one
`gershgorin_interval` sweep -- O(nnz), no allocation -- and leaves the expensive
work hoisted.

Uses the instantaneous coefficients (`peak = false`): the centre must sit where
the spectrum actually is, not where a switched-on coupling would put it.
"""
function recenter!(plan::ChebyshevPlan,
                   terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}})
    lo, hi = gershgorin_interval(terms; peak = false)
    plan.Ē = (lo + hi) / 2
    return plan
end

"""
    recenter!(plan, terms) -> plan

No-op for schemes that need no spectral centre. See the `ChebyshevPlan` method
for what recentering is and why it is needed.
"""
recenter!(plan::IntegratorPlan, ::Any) = plan

"""
    chebyshev!(ψ, terms, dt, Emin, Emax; tol = 1e-12, ws = nothing)

Propagate `ψ ← exp(-i H dt) ψ` by a Chebyshev expansion, where
`H = Σⱼ cⱼHⱼ` is given by `terms` and its spectrum lies in `[Emin, Emax]`.

Writing `ΔE = (Emax-Emin)/2`, `Ē = (Emax+Emin)/2` and `H̃ = (H - Ē)/ΔE` (so
`σ(H̃) ⊆ [-1,1]`),

    exp(-iH dt) ψ = e^{-iĒdt} [ J₀(ΔE·dt) φ₀ + 2 Σ_{k≥1} (-i)ᵏ J_k(ΔE·dt) φ_k ]

with the Chebyshev vectors from the three-term recurrence
`φ₀ = ψ`, `φ₁ = H̃ψ`, `φ_{k+1} = 2H̃φ_k − φ_{k-1}`.

The Bessel coefficients decay superexponentially past `k > ΔE·dt`, so the degree
follows from the spectrum and tolerance rather than being chosen in advance, and
there is no stability limit on `‖H‖dt`. Taylor-4 by contrast contracts only below
`‖H‖dt = 2√2`, forcing a step-count floor however little accuracy is wanted.

One matrix-vector product per degree, three state vectors of workspace. The
caller supplies the spectral bounds, which are computed once per instruction.
"""
function chebyshev!(ψ::Vector{ComplexF64},
                    terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                    dt::Float64, Emin::Float64, Emax::Float64;
                    tol::Float64 = 1e-12,
                    ws::Union{Nothing,ChebyshevWorkspace} = nothing)
    W = ws === nothing ? ChebyshevWorkspace(length(ψ)) : ws
    return chebyshev!(ψ, terms, ChebyshevPlan(dt, Emin, Emax, tol, W))
end

"""
    chebyshev!(ψ, terms, plan)

Apply one Chebyshev step using a prebuilt [`ChebyshevPlan`](@ref) -- the hot-loop
form, doing no coefficient work and touching no global state.
"""
function chebyshev!(ψ::Vector{ComplexF64},
                    terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                    plan::ChebyshevPlan)
    W  = plan.ws
    ΔE = plan.ΔE
    Ē  = plan.Ē
    dt = plan.dt

    if plan.degenerate
        phase = cis(-Ē * dt)
        @inbounds for i in eachindex(ψ); ψ[i] *= phase; end
        return ψ
    end
    deg = plan.deg

    ϕkm1, ϕk, ϕkp1 = W.ϕkm1, W.ϕk, W.ϕkp1

    # φ₀ = ψ ; accumulate J₀φ₀ into ψ after saving φ₀.
    copyto!(ϕkm1, ψ)

    # φ₁ = H̃ φ₀ = (Hφ₀ - Ē φ₀)/ΔE
    fill!(ϕk, 0)
    apply!(ϕk, terms, ϕkm1, ComplexF64(1.0))
    @inbounds for i in eachindex(ϕk)
        ϕk[i] = (ϕk[i] - Ē * ϕkm1[i]) / ΔE
    end

    # ψ = J₀φ₀ + 2(-i)J₁φ₁
    @inbounds for i in eachindex(ψ)
        ψ[i] = W.coef[1] * ϕkm1[i] + 2 * (-1im) * W.coef[2] * ϕk[i]
    end

    # φ_{k+1} = 2H̃φ_k - φ_{k-1}, accumulating 2(-i)^k J_k φ_k
    ik = ComplexF64(-1im)                 # (-i)^1, updated per term
    @inbounds for k in 2:deg-1
        fill!(ϕkp1, 0)
        apply!(ϕkp1, terms, ϕk, ComplexF64(1.0))
        for i in eachindex(ϕkp1)
            ϕkp1[i] = 2 * (ϕkp1[i] - Ē * ϕk[i]) / ΔE - ϕkm1[i]
        end
        ik *= -1im                        # (-i)^k
        ck = 2 * W.coef[k+1] * ik
        for i in eachindex(ψ)
            ψ[i] += ck * ϕkp1[i]
        end
        ϕkm1, ϕk, ϕkp1 = ϕk, ϕkp1, ϕkm1   # rotate buffers
    end

    phase = cis(-Ē * dt)
    @inbounds for i in eachindex(ψ); ψ[i] *= phase; end
    return ψ
end

"""
    propagate!(integrator, ψ, terms, dt, spec; tol)

Apply `exp(-iH dt)` to `ψ` in place using `integrator`.
"""
propagate!(::Chebyshev, ψ::Vector{ComplexF64},
           terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
           dt::Float64, spec::SpectralSpec; tol::Float64 = 1e-12) =
    chebyshev!(ψ, terms, dt, spec.Emin, spec.Emax; tol = tol)

# Hot-loop form: the plan carries the Bessel coefficients.
propagate!(::Chebyshev, ψ::Vector{ComplexF64},
           terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
           plan::ChebyshevPlan) = chebyshev!(ψ, terms, plan)

"""
    plan_step(integrator, ψ, dt, spec; tol)

Precompute whatever `integrator` can reuse across the steps of one instruction.
`dt` and `spec` are both constant there, so everything derived from them belongs
here rather than in the loop.

The workspace is allocated here and reached only through the returned plan, so
each task owns its own. Amortised over thousands of steps that is free: ~1 us
becomes ~0.3 ns/call.
"""
plan_step(::Chebyshev, ψ::Vector{ComplexF64}, dt::Float64, spec::SpectralSpec;
          tol::Float64 = 1e-12) =
    ChebyshevPlan(dt, spec.Emin, spec.Emax, tol, ChebyshevWorkspace(length(ψ)))

"""
    TaylorPlan(dt)

Per-instruction plan for [`Taylor`](@ref): the step, the truncation order, and
the two scratch vectors `fquantum!` needs. No coefficient precomputation.
"""
struct TaylorPlan <: IntegratorPlan
    dt::Float64
    order::Int
    q1::Vector{ComplexF64}      # scratch; carried here because `propagate!` is
    q2::Vector{ComplexF64}      # the hot loop and must not allocate
end

plan_step(t::Taylor, ψ::Vector{ComplexF64}, dt::Float64, spec::SpectralSpec;
          tol::Float64 = 1e-12) =
    TaylorPlan(dt, t.order,
               zeros(ComplexF64, length(ψ)), zeros(ComplexF64, length(ψ)))

propagate!(::Taylor, ψ::Vector{ComplexF64},
           terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
           plan::TaylorPlan) =
    fquantum!(plan.dt, ψ, terms, plan.q1, plan.q2; order = plan.order)

