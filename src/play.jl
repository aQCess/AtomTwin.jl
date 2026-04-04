using Base.Threads

const PARALLEL_THRESH::Int = 4

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
- Additional `kwargs` are treated as parameter values for resolving `Deferred` objects and 
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
- Multi-shot simulations use parallel execution when `shots ≥ 4` and `Threads.nthreads() > 1`
- Classical systems (no quantum state) skip quantum evolution and only simulate atomic motion

# Examples

## Single-shot quantum simulation
```julia
using AtomTwin

# Define system with ground and excited states
g, e = AtomTwin.Level(:g), AtomTwin.Level(:e)
atoms = [Atom(position=[0.0, 0.0, 0.0], levels=[g, e])]
basis = Basis([g, e])

# Create Rabi pulse and detector
Ω = RabiField(amplitude=2π*1.0, detuning=0.0)
detector = PopulationDetector("P_g", level=g)

sys = System(atoms, [], [g], basis, [Ω], [detector])
seq = Sequence(dt=0.01)
push!(seq, Pulse(Ω, duration=1.0))

# Run simulation
result = play(sys, seq; initial_state=g, shots=1)
result.detectors["P_g"]  # Vector of ground state populations vs time

## Multi-shot Monte Carlo simulation
# Same system as above, run 100 trajectories with quantum jumps
jump = Jump(rate=0.1, source=e, target=g)
sys_jumps = System(atoms, [], [g], basis, [Ω, jump], [detector])

result = play(sys_jumps, seq; initial_state=g, shots=100, savefinalstate=true)
result.detectors["P_g"]  # Matrix: [n_times × 100]
mean_population = mean(result.detectors["P_g"], dims=2)  # Average over shots

## Parametric simulation with deferred values
# Define amplitude as a parameter
Ω_param = Deferred(:amplitude)
pulse = Pulse(RabiField(amplitude=Ω_param, detuning=0.0), duration=1.0)

# Run with specific parameter value
result = play(sys, seq; initial_state=g, amplitude=2π*2.0)
"""
function play(sys::System, seq::Sequence; 
                initial_state=sys.initial_state,
                density_matrix=false,
                rng=Random.default_rng(),
                kwargs...)

    # Sanitize initial_state to a vector
    s = _tovector(initial_state)
    if isempty(s)
        @warn "Initial state not specified. Defaulting to classical dynamics."
    end
    
    job = compile(sys, seq; initial_state = s, density_matrix=density_matrix, rng=rng, kwargs...)
    return play(job, sys; initial_state = s, density_matrix=density_matrix, rng=rng, kwargs...)
end

function _execute_shot!(shot, local_job, sys, shot_rng, initial_state, all_outputs_vec, 
                        det_names, n_detectors, final_states, savefinalstate; kwargs...)

    result = _play(local_job; rng=shot_rng, savefinalstate=savefinalstate)
    
    @inbounds for j in 1:n_detectors
        if ndims(all_outputs_vec[j]) == 2
            all_outputs_vec[j][:, shot] .= result.detectors[det_names[j]]
        else
            all_outputs_vec[j][:, :, shot] .= result.detectors[det_names[j]]
        end
    end
    
    savefinalstate && (final_states[shot] = result.final_state)
end

function play(job::SimulationJob, sys::System;
              savefinalstate::Bool=false,
              shots::Int = 1,
              density_matrix = false,
              initial_state = nothing,
              parallel_thresh = PARALLEL_THRESH,
              rng = Random.default_rng(),
              kwargs...)

    # Optional override: update sys.state[] so recompile! picks it up for all shots
    if initial_state !== nothing && !isempty(_tovector(initial_state)) && job.state !== nothing
        sys.state[] = getqstate(sys, _tovector(initial_state); density_matrix=density_matrix)
    end

    @assert shots > 0 "shots must be positive"
    
    # Single-shot fast path
    if shots == 1
        shot_seed = rand(rng, UInt)
        shot_rng = Random.MersenneTwister(shot_seed)
        result = _play(job; rng=shot_rng, savefinalstate=savefinalstate)
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
    
    # Allocate output storage
    all_outputs_vec = [
        let single_shot_vals = job.detectors[1][j].vals
            ndims(single_shot_vals) == 1 ?
                zeros(eltype(single_shot_vals), n_times, shots) :
                zeros(eltype(single_shot_vals), n_times, size(single_shot_vals, 2), shots)
        end
        for j in 1:n_detectors
    ]
    
    final_states = savefinalstate ? Vector{typeof(job.state)}(undef, shots) : typeof(job.state)[]
    
    # Determine execution mode
    use_parallel = shots ≥ parallel_thresh && Threads.nthreads() > 1
    
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
    
    # Generate seeds and RNGs
    shot_seeds = [rand(rng, UInt) for _ in 1:shots]
    shot_rngs = [Random.MersenneTwister(shot_seeds[i]) for i in 1:shots]
    
    # Execute
    if use_parallel
        # Pre-allocate one job copy per thread (reused across shots)
        thread_jobs = [deepcopy(job) for _ in 1:Threads.maxthreadid()]
        
        Threads.@threads for shot in 1:shots
            tid = Threads.threadid()
            if shot != 1
                recompile!(thread_jobs[tid], sys;
                                rng=shot_rngs[shot],
                                kwargs...)
            end
            _execute_shot!(shot, thread_jobs[tid], sys, shot_rngs[shot], initial_state, 
                        all_outputs_vec, det_names, n_detectors, final_states, 
                        savefinalstate; kwargs...)
        end
    else
        for shot in 1:shots
            if shot != 1
                recompile!(job, sys;
                                rng=shot_rngs[shot],
                                kwargs...)
            end
            _execute_shot!(shot, job, sys, shot_rngs[shot], initial_state, 
                        all_outputs_vec, det_names, n_detectors, final_states, 
                        savefinalstate; kwargs...)
        end
    end

    return (
        detectors = Dict(det_names[j] => all_outputs_vec[j] for j in 1:n_detectors),
        times = job.times,
        final_states = final_states
    )
end


"""
    _play(job::SimulationJob; savefinalstate::Bool=false) -> NamedTuple

Execute a compiled simulation job for a single quantum trajectory shot.

Returns a NamedTuple with:
- `detectors`: Dict{String, Array} of detector outputs
- `times`: Vector{Float64} of time points
- `final_state`: Copy of final quantum state (only if savefinalstate=true)
"""
function _play(job::SimulationJob; 
                savefinalstate::Bool=false,
                rng=Random.MersenneTwister())

    n_instructions = length(job.modifiers)
    if job.state === nothing
        # Classical evolution
        @inbounds for i in 1:n_instructions
            evolve!(job.atoms, job.local_tspans[i];
                    beams=job.beams, modifiers=job.modifiers[i], 
                    detectors=job.detectors[i], rng=rng, frozen=false)
        end
    else
        # Quantum/semiclassical evolution
        frozen = isempty(job.beams) && all(
            isapprox(sum(abs2, atom.v), 0.0; atol=1e-14)
            for atom in job.atoms
        )
        @inbounds for i in 1:n_instructions
            evolve!((job.state, job.atoms), job.local_tspans[i];
                    fields=job.fields, beams=job.beams, jumps=job.jumps,
                    modifiers=job.modifiers[i], detectors=job.detectors[i], 
                    rng=rng, frozen=frozen)
        end
    end
    
    final_state = savefinalstate ? copy(job.state) : nothing
    return (detectors = job.detector_outputs, times = job.times, final_state = final_state)
end