"""
    SimulationJob{S}

Compiled simulation execution structure containing all runtime state and operators.

This object should not be constructed directly by users. Instead, use
`compile(system, sequence; shots=1)` which handles optimization and memory preallocation.

# Structure
- **Runtime state** (reset between shots): `state`, `atoms`, `beams`
- **Execution structures** (shared across shots): `fields`, `jumps`, `modifiers`
- **Detectors**: `detectors` (per-instruction), `detector_outputs` (views to results)
- **Time grids**: `times` (global downsampled), `local_tspans` (per-instruction solver time grids)
- **Configuration**: `downsamples` (per-instruction downsample factors)

# Per-Instruction Customization

When the `Sequence` has per-instruction overrides (e.g., `push!(seq, Pulse(...); dt=1e-9)`),
the `SimulationJob` builds heterogeneous time grids:
- Each instruction gets its own `local_tspans[i]` with the appropriate timestep
- Downsampling is applied per-instruction via `downsamples[i]`
- The returned `times` array is non-uniform (concatenation of per-instruction downsampled grids)

# Performance Notes
- Detector types are automatically concretized for optimal performance
- Multiple shots write directly to preallocated matrix columns (zero-copy)
- Output views avoid allocations when accessing results
- Per-instruction customization has negligible overhead (computed at compile time)
"""
struct SimulationJob{S}
    state::S
    initial_state::S            # copy of state at compile time; used by recompile! to reset
    atoms::Vector{NLevelAtom}
    beams::Vector{AbstractBeam}
    fields::Vector{<:Dynamiq.AbstractField}
    jumps::Vector{Jump}
    modifiers::Vector{Any}          # Vector{Vector{AbstractModifier}} — inner-loop, passed to evolve!
    boundary_modifiers::Vector{Any} # Vector{Vector{AbstractBoundaryModifier}} — called at instruction boundaries only
    detectors::Vector{Any}          # Vector{Vector{<:AbstractDetector}} — element type varies per instruction
    local_tspans::Vector     # Vector of SubArray views into full time grid, one per instruction
    detector_outputs::Dict{String, Any}
    times::Vector{Float64}   # downsampled time grid (length = sum(t_steps[i] ÷ downsamples[i]))
    downsamples::Vector{Int} # per-instruction downsample factors
end


_tovector(state::AbstractLevel) = [state]
_tovector(state::Tuple) = collect(state)
_tovector(state::AbstractVector) = state
_tovector(state) = [state]

"""
    compile(system::System, sequence::Sequence; initial_state=nothing, density_matrix=false) -> SimulationJob

Compile a System and Sequence into an executable SimulationJob with preallocated single-shot storage.

Builds heterogeneous time grids when the Sequence has per-instruction `dt` or `downsample`
overrides, enabling cost optimization across protocol phases.

# Arguments
- `system`: System specification (atoms, beams, nodes, detectors)
- `sequence`: Pulse sequence to execute (instruction list with base timestep `dt`).
  Individual instructions can override `dt` and `downsample` via `push!` keyword arguments.
- `initial_state`: Initial quantum state (required for quantum systems)
- `density_matrix`: Use density matrix formalism if `true` (default: `false`)
- Additional keyword arguments are treated as parameter overrides (e.g. `Ω = 2π*1e6`)

# Returns
- `SimulationJob` ready for execution, containing single-shot detector output buffers

# Per-Instruction Customization

```julia
seq = Sequence(1e-8; downsample=1)
push!(seq, Pulse(...); dt=1e-9)      # fine timestep for accuracy
push!(seq, Wait(1e-6); downsample=10) # coarse output for efficiency
job = compile(sys, seq)  # builds per-instruction time grids automatically
```

# Notes
- Compiles all DAG nodes: samples parameter values and updates fields in-place
- Detectors are automatically type-specialized to avoid dynamic dispatch
- Detector outputs are preallocated views into storage, avoiding allocations during simulation
- Time grids are heterogeneous when per-instruction overrides are present
- Each call to `compile()` creates a single-shot job. Multi-shot execution in `play()`
  uses thread-local copies of this job, with results aggregated into output matrices.
"""
function compile(sys::System, seq::Sequence;
    initial_state = sys.initial_state,
    density_matrix = false,
    rng = Random.MersenneTwister(),
    kwargs...)
      
    param_values = Dict{Symbol,Any}(kwargs)

    cache = IdDict{Any, Any}()

    sorted_nodes = _topological_sort(sys.nodes)

    # === PHASE 1: COMPILE BEAM NODES FIRST ===
    # BeamNodes must be compiled before CouplingNodes (which read beam_node._compiled[])
    # and before atom initialization (which uses beams for polarizability computation).
    for node in sorted_nodes
        node isa BeamNode && compile_node!(node, sys.basis, rng, param_values)
    end

    # Collect all beams: trapping beams from sys.beams + coupling beams from BeamNodes
    resolved_trapping = AbstractBeam[resolve(b, param_values; cache=cache) for b in sys.beams]
    resolved_coupling = AbstractBeam[n._compiled[] for n in sorted_nodes if n isa BeamNode]
    resolved_beams    = vcat(resolved_trapping, resolved_coupling)

    # === PHASE 2: INITIALIZE ATOMS (uses resolved beams, may sample positions/velocities) ===
    atoms = [initialize!(sys.atoms[i], sys.atoms[i].inner;
                         beams=resolved_beams, rng=rng, param_values=param_values)
             for i in 1:length(sys.atoms)]

    # === PHASE 3: COMPILE REMAINING NODES (CouplingNode, DetuningNode, etc.) ===
    # Atom positions are now set; BeamNodes already compiled.
    resolved_fields = AtomTwin.Dynamiq.AbstractField[]
    resolved_jumps  = Jump[]

    for node in sorted_nodes
        node isa BeamNode && continue  # already compiled
        obj = compile_node!(node, sys.basis, rng, param_values)
        if obj isa AtomTwin.Dynamiq.AbstractField
            push!(resolved_fields, obj)
        elseif obj isa Jump
            AtomTwin.Dynamiq.precompute!(obj, Vector)
            AtomTwin.Dynamiq.precompute!(obj, Matrix)
            push!(resolved_jumps, obj)
        end
    end
    resolved_fields = [obj for obj in resolved_fields]   # eltype inferred from content

    # Create global time reference for noisy fields
    global_time_ref = Ref(0.0)

    # Resolver function that uses the SAME cache for pointer sharing
    resolve_target = obj -> begin
        resolved_obj = resolve(obj, param_values; cache=cache)
        return update_noisy_field_time_refs!(resolved_obj, global_time_ref)
    end

    # Initialize quantum state
    if !isempty(initial_state)
        qstate = getqstate(sys, initial_state; density_matrix=density_matrix)
        sys.state[] = qstate
    else
        qstate = nothing
    end

    # === PHASE 4: COMPILE INSTRUCTIONS WITH RESOLVED SYSTEM ===

    n_instructions = length(seq)
    modifiers = Vector{Any}(undef, n_instructions)
    boundary_modifiers = Vector{Any}(undef, n_instructions)
    step_counts = Vector{Int}(undef, n_instructions)
    total_tspan_size = 0

    for (i, inst) in enumerate(seq)
        # Resolve instruction if it contains deferred objects (using same cache)
        resolved_inst = resolve(inst, param_values; cache=cache)

        # Resolve per-instruction dt: use instruction's dt if specified, else sequence default
        dt_i = something(resolved_inst.dt, seq.dt)

        # Compile and resolve_target (which uses same cache)
        mods, bmods, n_steps = compile(atoms, resolved_inst, dt_i; resolve_target=resolve_target)
        modifiers[i] = mods
        boundary_modifiers[i] = bmods
        step_counts[i] = n_steps
        total_tspan_size += n_steps
    end

    # === PHASE 5: BUILD DETECTORS AND OUTPUT STORAGE ===

    offsets  = cumsum([0; step_counts])

    # Resolve per-instruction dt and downsample from instructions, with sequence defaults
    inst_dts = [something(seq[i].dt, seq.dt) for i in 1:n_instructions]
    inst_ds  = [something(seq[i].downsample, seq.downsample) for i in 1:n_instructions]

    ds_counts  = [step_counts[i] ÷ inst_ds[i] for i in 1:n_instructions]
    ds_offsets = cumsum([0; ds_counts])
    ds_total   = ds_offsets[end]

    # Full solver-level time grid (heterogeneous: each instruction may have different dt)
    full_times = Vector{Float64}(undef, total_tspan_size)
    abs_start  = 0.0
    for i in 1:n_instructions
        dt_i = inst_dts[i]
        seg  = offsets[i]+1:offsets[i+1]
        full_times[seg] .= range(abs_start + dt_i, step=dt_i, length=step_counts[i])
        abs_start += step_counts[i] * dt_i
    end
    local_tspans = [view(full_times, offsets[i]+1:offsets[i+1]) for i in 1:n_instructions]

    # Downsampled time grid for user output (heterogeneous: each instruction may have different dt and ds)
    times = Vector{Float64}(undef, ds_total)
    abs_start = 0.0
    for i in 1:n_instructions
        dt_i  = inst_dts[i]
        ds_i  = inst_ds[i]
        step  = ds_i * dt_i
        seg   = ds_offsets[i]+1:ds_offsets[i+1]
        times[seg] .= range(abs_start + step, step=step, length=ds_counts[i])
        abs_start += step_counts[i] * dt_i
    end

    n_detectors = length(sys.detector_specs)

    detector_vals = Vector{Any}(undef, n_detectors)
    for j in 1:n_detectors
        spec = sys.detector_specs[j]
        detector_vals[j] = spec.ndims == 1 ?
            zeros(spec.eltype, ds_total) :
            zeros(spec.eltype, ds_total, spec.ndims)
    end

    # Build detectors using downsampled tspan/vals views
    detectors = Vector{Any}(undef, n_instructions)
    for i in 1:n_instructions
        ds_tspan = view(times, ds_offsets[i]+1:ds_offsets[i+1])
        detectors[i] = Vector{AbstractDetector}(map(1:n_detectors) do j
            vals_slice = ds_offsets[i]+1:ds_offsets[i+1]
            vals_view  = ndims(detector_vals[j]) == 1 ?
                view(detector_vals[j], vals_slice) :
                view(detector_vals[j], vals_slice, :)
            build_detector(sys.detector_specs[j], ds_tspan, vals_view, resolve_target, sys)
        end)
    end

    detector_outputs = Dict{String, Any}(
        sys.detector_specs[j].params.name => detector_vals[j]
        for j in 1:n_detectors
    )

    return SimulationJob(qstate, qstate === nothing ? nothing : copy(qstate),
                        atoms, resolved_beams, resolved_fields, resolved_jumps,
                        modifiers, boundary_modifiers, detectors, local_tspans,
                        detector_outputs, times, inst_ds)
end


"""
    recompile!(job::SimulationJob, sys::System; kwargs...)

Reinitialize a `SimulationJob` for a new Monte Carlo trajectory.

Updates all DAG node outputs in-place (re-sampling parameter values and noise),
reinitializes atom velocities, resets the quantum state, and zeroes detector outputs.

Additional keyword arguments override parameter values (same as `compile`).

# Thread Safety
Safe to call on thread-local job copies (`deepcopy(job)`). MUST NOT be called on
shared job objects. The `sys` argument may be shared across threads.
"""
function recompile!(job::SimulationJob, sys::System;
                    rng = Random.MersenneTwister(),
                    kwargs...)

    param_values = Dict{Symbol, Any}(kwargs)

    sorted_nodes = _topological_sort(sys.nodes)

    # Phase 1: recompile BeamNodes first
    for node in sorted_nodes
        node isa BeamNode && recompile_node!(node, nothing, rng, param_values)
    end

    # Collect beams for atom reinitialization
    resolved_coupling = AbstractBeam[n._compiled[] for n in sorted_nodes if n isa BeamNode]
    all_beams = vcat(job.beams, resolved_coupling)  # job.beams holds trapping beams

    # Phase 2: reinitialize atoms
    for i in 1:length(sys.atoms)
        initialize!(sys.atoms[i], job.atoms[i]; beams=all_beams, rng=rng, param_values=param_values)
    end

    # Phase 3: recompile remaining nodes
    field_counter = 0
    jump_counter  = 0

    for node in sorted_nodes
        node isa BeamNode && continue  # already recompiled
        obj = node_output(node)
        if obj isa AtomTwin.Dynamiq.AbstractField
            field_counter += 1
            recompile_node!(node, job.fields[field_counter], rng, param_values)
        elseif obj isa Jump
            jump_counter += 1
            # recompiling jumps is expensive
            #recompile_node!(node, job.jumps[jump_counter], rng, param_values)
            #AtomTwin.Dynamiq.precompute!(obj, Vector)
            #AtomTwin.Dynamiq.precompute!(obj, Matrix)
        end
    end

    # Regenerate noise in modifiers
    for modifier_list in job.modifiers, mod in modifier_list
        if mod isa AmplitudeModifier && mod.field isa NoisyField
            Random.seed!(mod.field.rng, rand(rng, UInt))
            update_noise!(mod.vals, job.times, 0.0, mod.field)
        end
    end
    
    # Reset quantum state from the compile-time copy
    if job.state !== nothing && job.initial_state !== nothing
        job.state .= job.initial_state
    end
    
    # Zero detector outputs
    for vals in values(job.detector_outputs)
        fill!(vals, 0.0)
    end
    
    return job
end
