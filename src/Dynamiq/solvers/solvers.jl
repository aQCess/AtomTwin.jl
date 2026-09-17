# The six solver front-ends and the classical motion they share.
#
# `evolve!` dispatches on state type to `tdse`, `wfmc` or `qme`, each with a
# semiclassical twin that also advances atomic motion. The six step loops are
# written out rather than unified: profiling puts the dispatch a generic loop
# would remove below 1.2% of any run, which does not justify a type-parameterised
# abstraction in the hottest loop in the engine.

#------------------------------------------------------------------------------
# High-level evolve! front-ends
#------------------------------------------------------------------------------

"""
    evolve!(atoms, tspan; beams, kwargs...)

Pure classical motion: every atom is put in its ground level and the Newtonian
equations of motion are integrated by `newton`.
"""
function evolve!(atoms::Vector{<:NLevelAtom},
                 tspan::AbstractVector{Float64};
                 beams::Vector{<:AbstractBeam}=AbstractBeam[],
                 frozen = false,
                 kwargs...)

    for atom in atoms
        fill!(atom._P, 0.0)
        atom._P[1] = 1.0
    end

    newton(atoms, tspan; beams = beams, kwargs...)
end

"""
    evolve!((rho, atoms), tspan; fields, jumps, beams, frozen, kwargs...)

Evolve a density matrix over `tspan` by [`qme`](@ref) with frozen atoms, or
[`qme_semiclassical`](@ref) with classical trajectories.
"""
function evolve!(state::Tuple{Matrix{ComplexF64},Vector{<:NLevelAtom}},
                 tspan::AbstractVector{Float64};
                 fields::Vector{<:AbstractField} = AbstractField[],
                 jumps::Vector{Jump}              = Jump[],
                 beams::Vector{<:AbstractBeam}    = AbstractBeam[],
                 frozen::Bool                     = true,
                 rng = nothing,                   # unused: the master equation
                 kwargs...)                       # is deterministic
    ρ, atoms = state
    for j in jumps
        if isnothing(j.LdagL_diag)
            precompute!(j, Matrix)
        end
    end

    L = Tuple{Base.RefValue{ComplexF64},Op}[(d._coeff, d.H) for d in fields]
    J = Tuple{Base.RefValue{ComplexF64},Op,Vector{Float64}}[
        (j._coeff, j.J, j.LdagL_diag) for j in jumps
    ]

    warn_if_step_too_large(L, length(tspan) > 1 ? tspan[2] - tspan[1] : 0.0,
                           get(kwargs, :integrator, Chebyshev()))
    if frozen || isempty(beams)
        qme(ρ, L, J, tspan; fields = fields, kwargs...)
    else
        qme_semiclassical(ρ, atoms, L, J, tspan; beams = beams, fields = fields, kwargs...)
    end
end

"""
    evolve!((psi, atoms), tspan; fields, jumps, beams, frozen, kwargs...)

Evolve a state vector over `tspan` by `tdse` when there are no jumps, or
one [`wfmc`](@ref) trajectory when there are -- each with a semiclassical twin
that also advances atomic motion.
"""
function evolve!(state::Tuple{Vector{ComplexF64},Vector{<:NLevelAtom}},
                 tspan::AbstractVector{Float64};
                 fields::Vector{<:AbstractField}  = AbstractField[],
                 jumps::Vector{Jump}              = Jump[],
                 beams::Vector{<:AbstractBeam}    = AbstractBeam[],
                 frozen::Bool                     = false,
                 rng = Random.Xoshiro(),
                 kwargs...)
    for j in jumps
        if isnothing(j.Hnh)
            precompute!(j, Vector)
        end
    end

    psi, atoms = state
    _dt_probe = length(tspan) > 1 ? tspan[2] - tspan[1] : 0.0
    H = Tuple{Base.RefValue{ComplexF64},Op}[(d._coeff, d.H) for d in fields]
    warn_if_step_too_large(H, _dt_probe, get(kwargs, :integrator, Chebyshev()))

    if isempty(jumps)
        if frozen
            tdse(psi, H, tspan; beams = beams, fields = fields, kwargs...)
        else
            tdse_semiclassical(psi, atoms, H, tspan; beams = beams, fields = fields,
                               kwargs...)
        end
    else
        Hnh = Tuple{Base.RefValue{ComplexF64},Op}[(j._coeff, j.Hnh) for j in jumps]
        if frozen
            wfmc(psi, H, Hnh, jumps, tspan; fields = fields, beams = beams,
                 rng = rng, kwargs...)
        else
            wfmc_semiclassical(psi, atoms, H, Hnh, jumps, tspan; fields = fields,
                               beams = beams, rng = rng, kwargs...)
        end
    end
    return
end

#------------------------------------------------------------------------------
# Schrödinger evolution
#------------------------------------------------------------------------------

"""
    write_detectors!(detectors, i, steps, downsample)

Record every detector, if step `i` falls on a downsample boundary.

Writes on each boundary AND on the final step, so the endpoint is recorded even
when `downsample` does not divide the step count. `PopulationDetector`s are
prepped first: they read a cached population vector the step does not otherwise
refresh.
"""
@inline function write_detectors!(detectors, i::Int, steps::Int, downsample::Int)
    (i % downsample == 0 || i == steps) || return
    i_out = i == steps ? cld(steps, downsample) : i ÷ downsample
    @inbounds for d in detectors
        d isa PopulationDetector && prep!(d)
        write!(d, i_out)
    end
    return
end

"""
    _step_of(tspan, dt) -> Float64

The solver's output step. `tspan` holds ABSOLUTE times and may hold a single
entry -- an instruction recording only its final state -- so the spacing cannot
always be recovered from it. `compile` always passes `dt`; the
`tspan[2] - tspan[1]` fallback serves direct callers.
"""
@inline function _step_of(tspan::AbstractVector{Float64}, dt)
    dt === nothing || return Float64(dt)
    length(tspan) ≥ 2 && return tspan[2] - tspan[1]
    length(tspan) == 1 &&
        throw(ArgumentError("single-sample tspan needs an explicit `dt`: the " *
                            "entry is an absolute time, not a step"))
    return 0.0
end

"""
    tdse(psi, H, tspan; kwargs...)

Schrödinger solver: evolves the state vector `psi` over the equidistant grid
`tspan` under the Hamiltonian terms `H`, with frozen atoms.
"""
function tdse(psi::Vector{ComplexF64},
              H::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
              tspan::AbstractVector{Float64};
              beams::Vector{<:AbstractBeam}     = AbstractBeam[],
              fields::Vector{<:AbstractField}   = AbstractField[],
              modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
              detectors::Vector{<:AbstractDetector} = AbstractDetector[],
              integrator::AbstractIntegrator     = Chebyshev(),
              tol::Float64                       = 1e-12,
              steptol::Float64                   = 1e-4,
              jtol::Float64                      = 1e-4,
              downsample::Int                    = 1,
              dt                                 = nothing)

    steps = length(tspan)
    dt    = _step_of(tspan, dt)
    # Once per instruction: `spectral_spec` bounds over the whole sequence
    # (`peak = true`), so the interval stays valid for every step even as a
    # shaped pulse changes the coefficients.
    spec = spectral_spec(H)
    plan = plan_step(integrator, psi, dt, spec; tol = tol)

    has_modifiers = !isempty(modifiers)
    has_fields    = !isempty(fields)
    has_detectors = !isempty(detectors)

    @inbounds for i in 1:steps
        if has_modifiers
            for m in modifiers
                update!(m, _modifier_time(m, (i - 0.5) * dt, dt))
            end
        end
        if has_fields
            for f in fields
                update!(f, i)
            end
        end
        propagate!(integrator, psi, H, plan)
        has_detectors && write_detectors!(detectors, i, steps, downsample)
    end
end

"""
    tdse_semiclassical(psi, atoms, H, tspan; kwargs...)

Time-dependent Schrödinger solver with semiclassical atomic motion.

Quantum dynamics is driven by `H` while classical trajectories evolve
under the optical forces from `beams`. `tspan` is assumed equidistant.
"""
function tdse_semiclassical(psi::Vector{ComplexF64},
                            atoms::Vector{A},
                            H::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                            tspan::AbstractVector{Float64};
                            beams::Vector{<:AbstractBeam}      = AbstractBeam[],
                            fields::Vector{<:AbstractField}    = AbstractField[],
                            modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
                            detectors::Vector{<:AbstractDetector} = AbstractDetector[],
                            integrator::AbstractIntegrator      = Chebyshev(),
                            tol::Float64                        = 1e-12,
                            steptol::Float64                    = 1e-4,
                            jtol::Float64                       = 1e-4,
                            downsample::Int                     = 1,
                            dt                                  = nothing) where {A}

    steps = length(tspan)
    dt    = _step_of(tspan, dt)
    spec  = spectral_spec(H)
    plan  = plan_step(integrator, psi, dt, spec; tol = tol)

    has_modifiers = !isempty(modifiers)
    has_fields    = !isempty(fields)
    has_detectors = !isempty(detectors)

    @inbounds for i in 1:steps
        if has_modifiers
            for m in modifiers
                update!(m, _modifier_time(m, (i - 0.5) * dt, dt))
            end
        end
        for atom in atoms
            updatepop!(atom, psi)
            fclassical!(dt, atom, beams)
        end
        if has_fields
            for f in fields
                update!(f, i)
            end
        end
        propagate!(integrator, psi, H, plan)
        has_detectors && write_detectors!(detectors, i, steps, downsample)
    end
end

#------------------------------------------------------------------------------
# Classical updates (forces and Newton solver)
#------------------------------------------------------------------------------

"""
    force(atom, beams)

Total optical dipole force on `atom` at `atom.x`, as `(Fx, Fy, Fz)`.

Beams of the SAME wavelength are summed coherently in the complex field, so
interference is included; different wavelengths add incoherently. Per wavelength

    F = -2 (Σᵢ α[λ][i]·P[i]) Re(E* ∇E)

with `α[λ][i]` the level-`i` polarizability at `λ`, `P` the level populations and
`E` the total complex field.
"""
function force(atom::A, beams::Vector{<:AbstractBeam}) where {A}
    Fx, Fy, Fz = 0.0, 0.0, 0.0
    isempty(beams) && return (Fx, Fy, Fz)
    x_pos = atom.x
    ws = _force_ws(length(beams))
    seen = ws.seen
    nseen = 0

    # One pass over wavelengths, not one per level: the field sum depends only on
    # the wavelength and position, and the level sum `Σᵢ α[λ][i]·P[i]` is a
    # scalar.
    @inbounds for b_ref in beams
        λ = getwavelength(b_ref)

        # Exact equality throughout, matching `atom.alpha`, which is a
        # `Dict{Float64,...}` keyed by wavelength. Deduplicating exactly while
        # summing with `isapprox` double-counts beams differing at round-off.
        dup = false
        for k in 1:nseen
            if seen[k] == λ
                dup = true
                break
            end
        end
        dup && continue
        nseen += 1
        seen[nseen] = λ

        Etot = 0.0 + 0.0im
        dEdr_x = 0.0 + 0.0im
        dEdr_y = 0.0 + 0.0im
        dEdr_z = 0.0 + 0.0im
        for b in beams
            if getwavelength(b) == λ
                Etot += efield_scalar(b, x_pos)
                dE = dEdr(b, x_pos)
                dEdr_x += dE[1]
                dEdr_y += dE[2]
                dEdr_z += dE[3]
            end
        end

        # Σᵢ α[λ][i]·P[i] -- the only level-dependent part.
        αλ = atom.alpha[λ]
        c = 0.0
        for i in 1:atom.n
            c += αλ[i] * atom._P[i]
        end
        c == 0.0 && continue

        Etot_conj = conj(Etot)
        Fx += c * real(Etot_conj * dEdr_x)
        Fy += c * real(Etot_conj * dEdr_y)
        Fz += c * real(Etot_conj * dEdr_z)
    end
    return Fx, Fy, Fz
end

"""
    fclassical!(dt, atom, beams)

One classical step of `atom` under the forces from `beams`, by velocity Verlet:

    x ← x + v dt + (F_old/2m) dt²
    F_new ← F(x)
    v ← v + ((F_old + F_new)/2m) dt

Second order at exactly ONE force evaluation per step -- the same cost as
first-order symplectic Euler -- because `F_new` is cached on `atom._F` and reused
as the next step's `F_old`. Measured against symplectic Euler in a Gaussian trap:
order 1 → 2.000, position error ~1000x smaller, energy drift ~12000x smaller.

`atom._Fvalid` guards the first step of an instruction, where no previous force
exists; `reset_force!` clears it.
"""
function fclassical!(dt::Float64, atom::A, beams::Vector{<:AbstractBeam}) where {A}
    inv_m  = 1.0 / atom.m
    half   = 0.5 * dt * inv_m

    @inbounds begin
        # First step: F_old must correspond to the current x.
        if !atom._Fvalid
            f0x, f0y, f0z = force(atom, beams)
            atom._F[1] = f0x; atom._F[2] = f0y; atom._F[3] = f0z
            atom._Fvalid = true
        end

        # Drift: x ← x + v dt + (F_old / 2m) dt²
        hdt2 = 0.5 * dt * dt * inv_m
        atom.x[1] = muladd(atom.v[1], dt, muladd(atom._F[1], hdt2, atom.x[1]))
        atom.x[2] = muladd(atom.v[2], dt, muladd(atom._F[2], hdt2, atom.x[2]))
        atom.x[3] = muladd(atom.v[3], dt, muladd(atom._F[3], hdt2, atom.x[3]))

        # One force evaluation at the NEW position.
        Fx, Fy, Fz = force(atom, beams)

        # Kick: v ← v + ((F_old + F_new) / 2m) dt
        atom.v[1] = muladd(atom._F[1] + Fx, half, atom.v[1])
        atom.v[2] = muladd(atom._F[2] + Fy, half, atom.v[2])
        atom.v[3] = muladd(atom._F[3] + Fz, half, atom.v[3])

        # Carry F_new forward as the next step's F_old.
        atom._F[1] = Fx; atom._F[2] = Fy; atom._F[3] = Fz
    end
end

"""
    reset_force!(atom)

Invalidate the cached velocity-Verlet force. Call whenever `atom.x` moves outside
the integrator -- a new instruction, a repositioned tweezer -- so the next
[`fclassical!`](@ref) re-primes `F_old` at the current position.
"""
reset_force!(atom) = (atom._Fvalid = false; nothing)

"""
    newton(atoms, tspan; beams, modifiers, detectors)

Purely classical motion of `atoms` over the equidistant grid `tspan`. The force
sweep is parallelised with `@batch` when `length(atoms) > 2`.
"""
function newton(atoms::Vector{A},
                tspan::AbstractVector{Float64};
                beams::Vector{<:AbstractBeam}     = AbstractBeam[],
                modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
                detectors::Vector{<:AbstractDetector} = AbstractDetector[],
                downsample::Int                   = 1,
                kwargs...) where {A}

    steps = length(tspan)
    dt    = steps > 1 ? tspan[2] - tspan[1] : 0.0

    # Measured on the `fclassical!` sweep at 10 threads: @batch is 1.28x at N=4,
    # 2.8x at 16, 4.7x at 42, 5.9x at 256, and loses only at N=2 (0.80x). So the
    # crossover is just above 2. `qme_semiclassical` uses the same threshold.
    parallel = length(atoms) > 2

    has_modifiers = !isempty(modifiers)
    has_detectors = !isempty(detectors)

    @inbounds for i in 1:steps
        if has_modifiers
            for m in modifiers
                update!(m, _modifier_time(m, (i - 0.5) * dt, dt))
            end
        end
        if parallel
            @batch for atom in atoms
                fclassical!(dt, atom, beams)
            end
        else
            for atom in atoms
                fclassical!(dt, atom, beams)
            end
        end
        has_detectors && write_detectors!(detectors, i, steps, downsample)
    end
end

"""
    wfmc(psi, H, Hnh, jumps, tspan; kwargs...)

Wavefunction Monte Carlo solver: one trajectory of `psi` over the equidistant
grid `tspan`.

`H` holds the Hermitian terms and `Hnh` the anti-Hermitian ones that make up the
effective Hamiltonian; `jumps` are applied stochastically. Each step propagates
under `H + Hnh` and then tests for a jump ([`quantum_jump!`](@ref)), sub-stepping
as [`jump_substeps`](@ref) requires.
"""
function wfmc(psi::Vector{ComplexF64},
              H::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
              Hnh::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
              jumps::Vector{Jump},
              tspan::AbstractVector{Float64};
              beams::Vector{<:AbstractBeam}          = AbstractBeam[],
              fields::Vector{<:AbstractField}        = AbstractField[],
              modifiers::Vector{<:AbstractModifier}   = AbstractModifier[],
              detectors::Vector{<:AbstractDetector}   = AbstractDetector[],
              integrator::AbstractIntegrator               = Chebyshev(),
              tol::Float64                             = 1e-12,
              steptol::Float64                   = 1e-4,
              jtol::Float64                      = 1e-4,
              downsample::Int                          = 1,
              dt                                       = nothing,
              _q1::Vector{ComplexF64}                  = copy(psi),
              _q2::Vector{ComplexF64}                  = copy(psi),
              rng = Random.Xoshiro())

    steps = length(tspan)
    dt    = _step_of(tspan, dt)

    _prob = Weights(zeros(length(jumps)))

    has_modifiers  = !isempty(modifiers)
    has_fields     = !isempty(fields)
    # PhotoDetectors are written when their bound jump fires, not every step.
    photo_detectors = PhotoDetector[d for d in detectors if d isa PhotoDetector]
    state_detectors = [d for d in detectors if !(d isa PhotoDetector)]
    has_detectors  = !isempty(state_detectors)

    # The deterministic step propagates the NON-HERMITIAN `H_eff = H - (i/2)Σγ L†L`,
    # which is `H` and `Hnh` summed. Chebyshev handles it -- the expansion is
    # valid over the complex plane -- and holds to machine precision far past
    # where Taylor-4 diverges, out to γ·dt = 5 for purely dissipative evolution.
    #
    # Built once, not per step: `vcat` in the loop would allocate every step.
    Heff_terms = vcat(H, Hnh)
    # `dt` is the SAMPLING resolution; the jump test needs its own, possibly
    # finer, step, bounded BOTH by `jtol` (two-jump omission) and by the
    # spectral width of H_eff (jump-time resolution) -- see `jump_substeps`.
    spec = spectral_spec(Heff_terms)
    ΔE   = (spec.Emax - spec.Emin) / 2
    nsub = jump_substeps(dt, jump_rate_bound(jumps), jtol, ΔE)
    h    = dt / nsub
    plan = plan_step(integrator, psi, h, spec; tol = tol)

    @inbounds for i in 1:steps
        if has_modifiers
            for m in modifiers
                update!(m, _modifier_time(m, (i - 0.5) * dt, dt))
            end
        end
        if has_fields
            for f in fields
                update!(f, i)
            end
        end
        t0 = (i - 1) * dt
        for q in 1:nsub
            # Drives are read at each sub-step's own midpoint, not frozen across
            # the sampling step -- see `strang_substeps!` for why.
            has_modifiers && _resample!(modifiers, t0 + (q - 0.5) * h, h)
            propagate!(integrator, psi, Heff_terms, plan)
            quantum_jump!(psi, jumps, photo_detectors, i, steps, downsample,
                          _prob, _q1, _q2, rng)
        end
        has_detectors && write_detectors!(state_detectors, i, steps, downsample)
    end
end


"""
    updatepop!(atom, psi)

Refresh the level populations `atom._P` from the many-body state vector `psi`,
using the precomputed index lists `atom._pidx`.
"""
function updatepop!(atom::AbstractAtom, psi::Vector{ComplexF64})
    for lvl in 1:atom.n
        a = 0.0
        for j in atom._pidx[lvl]
            a += abs2(psi[j[1]])
        end
        atom._P[lvl] = a
    end
end

"""
    wfmc_semiclassical(psi, atoms, H, Hnh, jumps, tspan; kwargs...)

[`wfmc`](@ref) with semiclassical atomic motion: each step also advances the
atoms under [`fclassical!`](@ref) and applies the radiation-pressure impulse from
[`fdipole!`](@ref).
"""
function wfmc_semiclassical(psi::Vector{ComplexF64},
                            atoms::Vector{A},
                            H::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                            Hnh::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                            jumps::Vector{},
                            tspan::AbstractVector{Float64};
                            beams::Vector{<:AbstractBeam}        = AbstractBeam[],
                            modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
                            fields::Vector{<:AbstractField}       = AbstractField[],
                            detectors::Vector{<:AbstractDetector} = AbstractDetector[],
                            integrator::AbstractIntegrator             = Chebyshev(),
                            tol::Float64                           = 1e-12,
                            steptol::Float64                    = 1e-4,
                            jtol::Float64                       = 1e-4,
                            downsample::Int                        = 1,
                            dt                                     = nothing,
                            _q1::Vector{ComplexF64}                = copy(psi),
                            _q2::Vector{ComplexF64}                = copy(psi),
                            rng = Random.Xoshiro()) where {A}

    steps = length(tspan)
    dt    = _step_of(tspan, dt)

    _prob = Weights(zeros(length(jumps)))

    # PhotoDetectors are written on jump firing, not in the per-step sweep.
    photo_detectors = PhotoDetector[d for d in detectors if d isa PhotoDetector]
    state_detectors = [d for d in detectors if !(d isa PhotoDetector)]
    has_modifiers   = !isempty(modifiers)
    has_fields      = !isempty(fields)
    has_detectors   = !isempty(state_detectors)

    Heff_terms = vcat(H, Hnh)          # see `wfmc`
    spec = spectral_spec(Heff_terms)
    ΔE   = (spec.Emax - spec.Emin) / 2
    nsub = jump_substeps(dt, jump_rate_bound(jumps), jtol, ΔE)
    h    = dt / nsub
    plan = plan_step(integrator, psi, h, spec; tol = tol)

    @inbounds for i in 1:steps
        if has_modifiers
            for m in modifiers
                update!(m, _modifier_time(m, (i - 0.5) * dt, dt))
            end
        end
        for atom in atoms
            updatepop!(atom, psi)
            fclassical!(dt, atom, beams)
            fdipole!(dt, psi, atom, fields, jumps, _q1)
        end
        if has_fields
            for f in fields
                update!(f, i)
            end
        end
        t0 = (i - 1) * dt
        for q in 1:nsub
            # Drives are read at each sub-step's own midpoint, not frozen across
            # the sampling step -- see `strang_substeps!` for why.
            has_modifiers && _resample!(modifiers, t0 + (q - 0.5) * h, h)
            propagate!(integrator, psi, Heff_terms, plan)
            quantum_jump!(psi, jumps, photo_detectors, i, steps, downsample,
                          _prob, _q1, _q2, rng)
        end
        has_detectors && write_detectors!(state_detectors, i, steps, downsample)
    end
end

"""
    qme(rho, L, J, tspan; kwargs...)

Quantum master equation solver: evolves the density matrix `rho` over the
equidistant grid `tspan` with frozen atoms.

`L` holds the Hamiltonian terms `(coeff, Op)` and `J` the jump data
`(coeff, L_op, LdagL_diag)`. Each user step is divided adaptively by
[`strang_substeps!`](@ref) to meet `steptol`.
"""
function qme(rho::Matrix{ComplexF64},
             L::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
             J::Vector{Tuple{Base.RefValue{ComplexF64},Op,Vector{Float64}}},
             tspan::AbstractVector{Float64};
             beams::Vector{<:AbstractBeam}        = AbstractBeam[],
             fields::Vector{<:AbstractField}      = AbstractField[],
             modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
             detectors::Vector{<:AbstractDetector} = AbstractDetector[],
             integrator::AbstractIntegrator             = Chebyshev(),
             tol::Float64                           = 1e-12,
             steptol::Float64                       = 1e-4,
             jtol::Float64                          = 1e-4,
             downsample::Int                        = 1,
             dt                                     = nothing,
             _q1::Matrix{ComplexF64}                = copy(rho),
             _q2::Matrix{ComplexF64}                = copy(rho))

    steps = length(tspan)
    dt    = _step_of(tspan, dt)

    has_modifiers  = !isempty(modifiers)
    has_fields     = !isempty(fields)
    has_detectors  = !isempty(detectors)

    # The integration step adapts within `dt`; `dt` itself stays the control and
    # output grid, so modifiers and detectors fire exactly where compiled.
    ctl = strang_control(size(rho, 1))
    ctl.k = 1
    ctl.seeded = false
    strang_reset!(ctl)

    @inbounds for i in 1:steps
        if has_modifiers
            for m in modifiers
                update!(m, _modifier_time(m, (i - 0.5) * dt, dt))
            end
        end
        if has_fields
            for f in fields
                update!(f, i)
            end
        end
        strang_substeps!(dt, rho, L, J, steptol, ctl, _q1, _q2,
                         taylor_order(integrator), modifiers, (i - 1) * dt)
        has_detectors && write_detectors!(detectors, i, steps, downsample)
    end
end

"""
    qme_semiclassical(rho, atoms, L, J, tspan; kwargs...)

[`qme`](@ref) with semiclassical atomic motion: each step also advances the atoms
under [`fclassical!`](@ref), in parallel when `length(atoms) > 2`.
"""
function qme_semiclassical(rho::Matrix{ComplexF64},
                           atoms::Vector{A},
                           L::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                           J::Vector{Tuple{Base.RefValue{ComplexF64},Op,Vector{Float64}}},
                           tspan::AbstractVector{Float64};
                           beams::Vector{<:AbstractBeam}        = AbstractBeam[],
                           fields::Vector{<:AbstractField}      = AbstractField[],
                           modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
                           detectors::Vector{<:AbstractDetector} = AbstractDetector[],
                           integrator::AbstractIntegrator             = Chebyshev(),
                           tol::Float64                           = 1e-12,
                           steptol::Float64                       = 1e-4,
                           jtol::Float64                          = 1e-4,
                           downsample::Int                        = 1,
                           dt                                     = nothing,
                           _q1::Matrix{ComplexF64}                = copy(rho),
                           _q2::Matrix{ComplexF64}                = copy(rho)) where {A}

    steps = length(tspan)
    dt    = _step_of(tspan, dt)

    parallel      = length(atoms) > 2     # see `newton` for the measurement
    has_modifiers = !isempty(modifiers)
    has_fields    = !isempty(fields)
    has_detectors = !isempty(detectors)

    ctl = strang_control(size(rho, 1))    # see `qme`
    ctl.k = 1
    ctl.seeded = false
    strang_reset!(ctl)

    @inbounds for i in 1:steps
        if has_modifiers
            for m in modifiers
                update!(m, _modifier_time(m, (i - 0.5) * dt, dt))
            end
        end
        if parallel
            @batch for atom in atoms
                fclassical!(dt, atom, beams)
            end
        else
            for atom in atoms
                fclassical!(dt, atom, beams)
            end
        end
        if has_fields
            for f in fields
                update!(f, i)
            end
        end
        strang_substeps!(dt, rho, L, J, steptol, ctl, _q1, _q2,
                         taylor_order(integrator), modifiers, (i - 1) * dt)
        has_detectors && write_detectors!(detectors, i, steps, downsample)
    end
end
