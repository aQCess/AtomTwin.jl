using Base.Threads

const PARALLEL_THRESH::Int = 4

# Whether the calling task runs inside a `Threads.@threads` loop. This is the
# test `@threads :static` itself makes before throwing "cannot be used
# concurrently or nested"; Base offers no public equivalent.
_in_threaded_region() = ccall(:jl_in_threaded_region, Cint, ()) != 0

"""
    play(sys::System, seq::Sequence; 
            initial_state=sys.initial_state, 
            rng=Random.default_rng(), kwargs...) -> NamedTuple

Execute a quantum simulation by compiling and running a pulse sequence on a system.

This is the high-level entry point for running simulations. It automatically handles 
compilation, state initialization, and execution in a single call. For performance-critical 
workflows with repeated executions, consider using [`compile`](@ref) followed by 
[`play(::SimulationJob, ::System)`](@ref) to avoid recompilation overhead.

# Arguments
- `sys::System`: System specification containing atoms, beams, operators, and detector configurations
- `seq::Sequence`: Time-ordered instruction sequence (pulses, moves, ramps, waits) with timestep `dt`
- `initial_state`: Initial quantum state specification (required for quantum systems). Can be:
  - `AbstractVector`: Basis-ordered state vector or density matrix
  - `Tuple`: Collection of basis levels, e.g., `(g, g, e)` for three atoms
  - `AbstractLevel`: Single level for uniform initialization
  - Default: `sys.initial_state`

# Keyword Arguments
- `shots::Int = 1`: Number of Monte Carlo trajectory shots to execute
- `density_matrix::Bool = false`: Use density matrix formalism if `true`, statevector if `false`
- `savefinalstate::Bool = false`: Include final quantum states in output (increases memory usage)
- `rng::AbstractRNG = Random.default_rng()`: Random number generator for reproducible simulations
- `shot_callback::Union{Nothing,Function} = nothing`: Optional callback invoked after each completed
  trajectory as `shot_callback(shot, shots)`, where `shot` is the 1-based index and `shots` is the
  total. Useful for progress reporting (e.g. `shot_callback = (s,n) -> @printf "shot %d/%d\\n" s n`).
  In multithreaded runs the callback is still called per shot but invocation order is non-deterministic.
- `frozen::Bool = false`: Atoms move by default — trap forces, recoil (`add_decay!(…; λ)`)
  and the radiation pressure of plane-wave drives. `frozen = true` holds every atom at its
  initial position (runs with a quantum state; a purely classical run always moves its
  atoms). A run in which nothing can move an atom (at rest, no recoil, no plane-wave
  drive, no moving beam, no dipole force that would move it by a picometre over the
  run) takes the frozen solvers automatically; the result is the same.
- Additional `kwargs` are treated as parameter values for resolving `Parameter`s and
  other parametric components in the system and sequence

# Returns
Returns a `NamedTuple` with the following fields:

- `detectors::Dict{String, Array}`: Detector measurement outputs. Format depends on `shots`:
  - Single shot (`shots=1`): `Dict{String, Vector}` for 1D detectors, `Dict{String, Matrix}` for multi-dimensional
  - Multiple shots (`shots>1`): `[n_times × shots]` for 1D detectors, `[n_times × n_dims × shots]` for multi-dimensional
- `times::Vector{Float64}`: Global time points at which measurements were recorded (starts at `dt`, not zero)
- `final_states::Vector`: Final quantum state after evolution for each shot (only if `savefinalstate=true`)

# Notes
- For quantum systems, `initial_state` must be specified in `sys.initial_state` or overridden via this argument 
- Each shot reinitializes atomic velocities/positions with fresh randomness
- Detector measurements occur at the **end** of each timestep, not the beginning
- Multi-shot simulations use parallel execution when `shots ≥ 4` and `Threads.nthreads() > 1`,
  and run serially when `play` is itself called inside a `Threads.@threads` loop
- Classical systems (no quantum state) skip quantum evolution and only simulate atomic motion

# Examples

```julia
using AtomTwin

g, e = Level("g"), Level("e")
atom = Atom(; levels = [g, e])
sys  = System(atom)
Ω    = Parameter(:Ω, 2π * 1e6)                    # 1 MHz Rabi frequency
c    = add_coupling!(sys, atom, g => e, Ω; active = false)
add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))

seq = Sequence(1e-9)
@sequence seq begin
    Pulse(c, 0.5e-6)
end

out = play(sys, seq; initial_state = g)
out.detectors["P_e"]                              # Vector over out.times

# With dissipation, average over wavefunction Monte Carlo trajectories...
add_decay!(sys, atom, e => g, 2π * 1e5)
out = play(sys, seq; initial_state = g, shots = 100)
out.detectors["P_e"]                              # Matrix: [n_times × 100]

# ...or integrate the master equation directly.
out = play(sys, seq; initial_state = g, density_matrix = true)

# Parameters are `play` keywords: no rebuild.
out = play(sys, seq; initial_state = g, Ω = 2π * 2e6)
```
"""
function play(sys::System, seq::Sequence;
                initial_state=sys.initial_state,
                density_matrix=false,
                rng=Random.default_rng(),
                shots::Int = 1,
                shot_callback::Union{Nothing,Function}=nothing,
                frozen::Bool = false,
                kwargs...)

    # Sanitize initial_state to a vector
    s = _tovector(initial_state)
    if isempty(s)
        @warn "Initial state not specified. Defaulting to classical dynamics." maxlog=1
    end

    # `shots` reaches `compile` because the derived step depends on it: an unset
    # `jtol` resolves to `1/sqrt(shots)` (see `resolve_jtol`), which feeds
    # `_derive_dt`.
    job = compile(sys, seq; initial_state = s, density_matrix=density_matrix, rng=rng,
                  shots=shots, kwargs...)
    return play(job, sys; initial_state = s, density_matrix=density_matrix, rng=rng,
                shots=shots, shot_callback=shot_callback, frozen=frozen, kwargs...)
end

function _execute_shot!(shot, local_job, sys, shot_rng, all_outputs_vec,
                        det_names, n_detectors, final_states, savefinalstate;
                        frozen::Bool = false, kwargs...)

    result = _play(local_job; rng=shot_rng, savefinalstate=savefinalstate, frozen=frozen)
    
    @inbounds for j in 1:n_detectors
        if ndims(all_outputs_vec[j]) == 2
            all_outputs_vec[j][:, shot] .= result.detectors[det_names[j]]
        else
            all_outputs_vec[j][:, :, shot] .= result.detectors[det_names[j]]
        end
    end
    
    savefinalstate && (final_states[shot] = result.final_state)
end

"""
    play(job::SimulationJob, sys::System; initial_state = nothing, shots = 1, kwargs...)

Run a compiled job -- the "compile once, play many" form of
`play(sys, seq)`, taking the same keywords.

Every run starts clean, whatever ran before: every shot from `initial_state` if
given (for this run only) or else the state the job was compiled with, with the
detectors zeroed and every field in its declared switch state.
"""
function play(job::SimulationJob, sys::System; initial_state = nothing, kwargs...)
    job.state === nothing && return _play_shots(job, sys; kwargs...)
    # Every shot starts from `job.initial_state` -- shot 1 via `_reset_run_state!`,
    # the rest via `recompile!` -- so a requested state goes there, restored
    # afterwards. It used to reach only `sys.state[]`, which the solvers never
    # read: a reused job started from the previous run's FINAL state, and process
    # tomography ran every input from the first.
    compiled = copy(job.initial_state)
    if initial_state !== nothing && !isempty(_tovector(initial_state))
        s0 = getqstate(sys, _tovector(initial_state);
                       density_matrix = job.state isa AbstractMatrix)
        size(s0) == size(compiled) || throw(ArgumentError(
            "initial_state gives a state of size $(size(s0)), but the job was compiled " *
            "for $(size(compiled)); compile with the same `density_matrix`"))
        job.initial_state .= s0
    end
    sys.state[] = copy(job.initial_state)
    try
        return _play_shots(job, sys; kwargs...)
    finally
        job.initial_state .= compiled
    end
end

# `density_matrix` is accepted with the rest of `play`'s keywords but not used:
# the job knows what it was compiled as.
function _play_shots(job::SimulationJob, sys::System;
                     savefinalstate::Bool=false,
                     shots::Int = 1,
                     frozen::Bool = false,
                     density_matrix = nothing,
                     parallel_thresh = PARALLEL_THRESH,
                     rng = Random.default_rng(),
                     shot_callback::Union{Nothing,Function}=nothing,
                     kwargs...)

    @assert shots > 0 "shots must be positive"
    density_matrix = job.state isa AbstractMatrix

    if density_matrix && isempty(job.jumps)
        @warn """
        `density_matrix = true` on a system with no dissipation.

        Without jump operators the master equation and the Schrodinger equation
        give the same answer, but the density matrix is d x d where the state
        vector is d — typically one to two orders of magnitude slower.

        Drop `density_matrix = true` unless you need the density matrix itself
        (a reduced state, a purity, a coherence between subsystems).
        """ maxlog = 1
    end

    if shots == 1 && !density_matrix && !isempty(job.jumps)
        @warn """
        `shots = 1` on a system with dissipation.

        A single wavefunction Monte Carlo trajectory is one stochastic sample,
        not the ensemble average: it contains discrete quantum jumps and can
        differ from the mean by O(1).

        Use `shots = N` and average, or `density_matrix = true` for the ensemble
        directly.
        """ maxlog = 1
    end

    # Restore the job's trapping beams to their compile-time state before the first
    # shot. Move/Position/Amplitude modifiers mutate beam.r0/_coeff in place, so
    # replaying a job that was already run (or whose beams a prior shot moved) must
    # start from the snapshot. recompile! repeats this for each subsequent shot.
    for k in eachindex(job.initial_beams)
        restore_beam!(job.beams[k], job.initial_beams[k])
    end
    _restore_atoms!(job)
    # ...and the rest of what a shot starts from, as `recompile!` does for every
    # later shot: a reused job otherwise kept its final state, its accumulated
    # clicks, and any coupling the last run left switched on.
    _reset_run_state!(job)
    _foreach_job_output((node, f) -> _reset_activity!(node, f), job,
                        _topological_sort(sys.nodes))

    # Single-shot fast path
    if shots == 1
        shot_seed = rand(rng, UInt)
        shot_rng = Random.Xoshiro(shot_seed)
        result = _play(job; rng=shot_rng, savefinalstate=savefinalstate, frozen=frozen)
        final_states = savefinalstate ? [result.final_state] : typeof(job.state)[]
        return (
            detectors = result.detectors,
            times = result.times,
            final_states = final_states
        )
    end
    
    # Multi-shot handling
    n_times = length(job.times)
    n_detectors = length(job.detectors[1])
    det_names = [job.detectors[1][j].name for j in 1:n_detectors]
    
    # Allocate output storage sized from the full detector_outputs (spans all instructions)
    all_outputs_vec = [
        let full_vals = job.detector_outputs[det_names[j]]
            ndims(full_vals) == 1 ?
                zeros(eltype(full_vals), length(full_vals), shots) :
                zeros(eltype(full_vals), size(full_vals, 1), size(full_vals, 2), shots)
        end
        for j in 1:n_detectors
    ]
    
    final_states = savefinalstate ? Vector{typeof(job.state)}(undef, shots) : typeof(job.state)[]
    
    # Determine execution mode. Inside someone else's threaded region -- a
    # parameter sweep under `Threads.@threads` -- run the shots serially: the
    # sweep already occupies the threads, and `@threads :static` below cannot
    # nest (it throws rather than falling back).
    use_parallel = shots ≥ parallel_thresh && Threads.nthreads() > 1 &&
                   !_in_threaded_region()
    
    if use_parallel
        # Memory check
        job_size = Base.summarysize(job)
        memory_required = job_size * Threads.maxthreadid()
        memory_available = Sys.total_memory() - Base.gc_live_bytes()
        
        if memory_required > 0.8 * memory_available
            @warn """Insufficient memory for multithreading.
                    Required: $(round(memory_required / 1e9, digits=2)) GB
                    Available: $(round(memory_available / 1e9, digits=2)) GB
                    Falling back to serial execution."""
            use_parallel = false
        end
    end
    
    # Generate seeds and RNGs.
    #
    # One generator is built per shot, and `Xoshiro` constructs ~45x faster than
    # `MersenneTwister`. The type is internal: `rng` supplies the seeds, so a
    # caller-supplied generator of any type still seeds the run reproducibly.
    shot_seeds = [rand(rng, UInt) for _ in 1:shots]
    shot_rngs = [Random.Xoshiro(shot_seeds[i]) for i in 1:shots]
    
    # Execute
    if use_parallel
        # Pre-allocate one job copy per thread (reused across shots)
        thread_jobs = [deepcopy(job) for _ in 1:Threads.maxthreadid()]
        
        # `:static` pins each task to one thread for the whole body, so
        # `threadid()` is a stable identity. Bare `@threads` is `:dynamic` on
        # Julia >= 1.12, where a task may migrate mid-body and two tasks then
        # share one `thread_jobs` entry -- and likewise one solver workspace
        # (see `ThreadCache` in Dynamiq/solvers.jl). Migration is observable
        # even on 1.11.
        Threads.@threads :static for shot in 1:shots
            tid = Threads.threadid()
            if shot != 1
                recompile!(thread_jobs[tid], sys;
                                rng=shot_rngs[shot],
                                kwargs...)
            end
            _execute_shot!(shot, thread_jobs[tid], sys, shot_rngs[shot],
                        all_outputs_vec, det_names, n_detectors, final_states,
                        savefinalstate; frozen = frozen, kwargs...)
            shot_callback !== nothing && shot_callback(shot, shots)
        end
    else
        for shot in 1:shots
            if shot != 1
                recompile!(job, sys;
                                rng=shot_rngs[shot],
                                kwargs...)
            end
            _execute_shot!(shot, job, sys, shot_rngs[shot],
                        all_outputs_vec, det_names, n_detectors, final_states,
                        savefinalstate; frozen = frozen, kwargs...)
            shot_callback !== nothing && shot_callback(shot, shots)
        end
    end

    return (
        detectors = Dict(det_names[j] => all_outputs_vec[j] for j in 1:n_detectors),
        times = job.times,
        final_states = final_states
    )
end


"""
    _propagator_tol(seq_tol) -> Float64

Map a sequence-level `tol` onto the propagator's local-error target.

`Sequence(; tol)` bounds the error of one integration step; the default is 1e-4.
The exponential propagator's `tol` is a different quantity: it sets the
Chebyshev degree within a step. That cost is essentially flat in tolerance, so
there is no reason to propagate a loose value — doing so buys nothing and
silently caps accuracy.

So `tol` may only *tighten* the propagator beyond its machine-precision default:
a user asking for `tol = 1e-14` gets it; the default still integrates each step
as exactly as it would have.
"""
_propagator_tol(seq_tol::Float64) = min(seq_tol, 1e-12)

# The two tolerances are different quantities and are passed separately:
# `tol` (above) bounds the exponential propagator's truncation within a step,
# while `steptol` -- the user's `tol`, unmodified -- bounds the Strang splitting
# error of one step and drives the sub-step controller in `strang_substeps!`.
# Collapsing them would either force the splitting to machine precision (absurd
# cost) or cap the propagator at a loose value (silent accuracy loss).

"""
    _play(job::SimulationJob; savefinalstate = false, frozen = false, rng) -> NamedTuple

Execute a compiled simulation job for a single quantum trajectory shot, drawing its
jumps from `rng`. `frozen` holds the atoms (see [`play`](@ref)); a run that
[`_is_static`](@ref) takes the frozen solvers anyway.

Returns a NamedTuple with:
- `detectors`: Dict{String, Array} of detector outputs
- `times`: Vector{Float64} of time points
- `final_state`: Copy of final quantum state (only if savefinalstate=true)
"""
function _play(job::SimulationJob;
                savefinalstate::Bool=false,
                frozen::Bool=false,
                rng=Random.Xoshiro())

    n_instructions = length(job.modifiers)
    if job.state === nothing
        # Classical evolution
        @inbounds for i in 1:n_instructions
            isempty(job.local_tspans[i]) && continue
            for m in job.boundary_modifiers[i]; begin_instruction!(m); end
            for m in job.modifiers[i]; begin_instruction!(m); end
            evolve!(job.atoms, job.local_tspans[i];
                    beams=job.beams, modifiers=job.modifiers[i],
                    detectors=job.detectors[i], rng=rng, frozen=false,
                    downsample=job.downsamples[i], dt=job.inst_dts[i])
            for m in job.boundary_modifiers[i]; end_instruction!(m); end
        end
    else
        # Atoms move unless the caller freezes them (`play(…; frozen = true)`).
        # A run in which nothing can move them takes the frozen solvers instead —
        # same answer, less work (see `_is_static`).
        frozen = frozen || _is_static(job)
        @inbounds for i in 1:n_instructions
            isempty(job.local_tspans[i]) && continue
            for m in job.boundary_modifiers[i]; begin_instruction!(m); end
            for m in job.modifiers[i]; begin_instruction!(m); end
            evolve!((job.state, job.atoms), job.local_tspans[i];
                    fields=job.fields, beams=job.beams, jumps=job.jumps,
                    modifiers=job.modifiers[i], detectors=job.detectors[i],
                    rng=rng, frozen=frozen, downsample=job.downsamples[i],
                    dt=job.inst_dts[i],
                    tol=_propagator_tol(job.tol), steptol=job.tol, jtol=job.jtol,
                    integrator=job.integrator)
            for m in job.boundary_modifiers[i]; end_instruction!(m); end
        end
    end

    final_state = savefinalstate ? copy(job.state) : nothing
    return (detectors = job.detector_outputs, times = job.times, final_state = final_state)
end

"""
    _is_static(job) -> Bool

True when no atom can move during the run, so that the frozen solvers give the same
trajectory as the semiclassical ones for less work: every atom at rest, no recoil
(`add_decay!(…; λ)`), no plane-wave drive (radiation pressure), no instruction that
moves, ramps or switches a beam, and, for every internal state, a dipole force at
the starting position too weak to matter over the run `T`: a displacement
`δx = |F|T²/2m` below 1 pm and a light-shift phase `|F|δx T/ħ` below 1e-6. (The
tails of neighbouring tweezers never give exactly zero.) Checked per shot (positions
and velocities are resampled).
"""
function _is_static(job::SimulationJob)
    atoms = job.atoms
    for a in atoms
        (any(!iszero, a.v) || !isempty(a.lambda)) && return false
    end
    any(f -> f isa PlanarCoupling, job.fields) && return false
    for ms in (job.modifiers, job.boundary_modifiers), mi in ms, m in mi
        _acts_on_beam(m) && return false
    end
    isempty(job.beams) && return true
    T = isempty(job.times) ? 0.0 : job.times[end]
    for a in atoms
        P = copy(a._P)
        try
            for l in 1:length(P)
                fill!(a._P, 0.0); a._P[l] = 1.0
                F  = norm(Dynamiq.force(a, job.beams))
                δx = F * T^2 / (2a.m)
                (δx > 1e-12 || F * δx * T / Units.hbar > 1e-6) && return false
            end
        finally
            copyto!(a._P, P)
        end
    end
    return true
end

_acts_on_beam(m) = m isa Dynamiq.PositionModifier || m isa Dynamiq.MoveModifier ||
                   (hasproperty(m, :field) && getproperty(m, :field) isa Dynamiq.AbstractBeam)
