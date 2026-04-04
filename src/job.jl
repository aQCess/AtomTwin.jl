"""
    SimulationJob{S}

Compiled simulation execution structure containing all runtime state and operators.

This object should not be constructed directly by users. Instead, use 
`compile(system, sequence; shots=1)` which handles optimization and memory preallocation.

# Structure
- **Runtime state** (reset between shots): `state`, `atoms`, `beams`
- **Execution structures** (shared across shots): `fields`, `jumps`, `modifiers`
- **Detectors**: `detectors` (per-instruction), `detector_outputs` (views to results)
- **Time grids**: `times` (global), `local_tspans` (per-instruction views)
- **Storage**: `_output_storage` (preallocated vectors or matrices)
- **Configuration**: `shots` (number of Monte Carlo runs)

# Performance Notes
- Detector types are automatically concretized for optimal performance
- Multiple shots write directly to preallocated matrix columns (zero-copy)
- Output views avoid allocations when accessing results
"""
struct SimulationJob{S}
    state::S
    atoms::Vector{NLevelAtom}
    beams::Vector{AbstractBeam}
    fields::Vector{Dynamiq.AbstractField}
    jumps::Vector{Jump}
    modifiers::Vector{Any}   # Vector{Vector{AbstractModifier}} — element type varies per instruction
    detectors::Vector{Any}   # Vector{Vector{<:AbstractDetector}} — element type varies per instruction
    local_tspans::Vector     # Vector of SubArray views into times, one per instruction
    detector_outputs::Dict{String, Any}
    times::Vector{Float64}
end


_tovector(state::AbstractLevel) = [state]
_tovector(state::Tuple) = collect(state)
_tovector(state::AbstractVector) = state
_tovector(state) = [state]

"""
    compile(system::System, sequence::Sequence; initial_state=nothing, density_matrix=false) -> SimulationJob

Compile a System and Sequence into an executable SimulationJob with preallocated single-shot storage.

# Arguments
- `system`: System specification (atoms, beams, nodes, detectors)
- `sequence`: Pulse sequence to execute (instruction list with timestep `dt`)
- `initial_state`: Initial quantum state (required for quantum systems)
- `density_matrix`: Use density matrix formalism if `true` (default: `false`)
- Additional keyword arguments are treated as parameter overrides (e.g. `Ω = 2π*1e6`)

# Returns
- `SimulationJob` ready for execution, containing single-shot detector output buffers

# Notes
- Compiles all DAG nodes: samples parameter values and updates fields in-place
- Detectors are automatically type-specialized to avoid dynamic dispatch
- Detector outputs are preallocated views into storage, avoiding allocations during simulation
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

    # === PHASE 1: COMPILE BEAM NODES FIRST ===
    # BeamNodes must be compiled before CouplingNodes (which read beam_node._compiled[])
    # and before atom initialization (which uses beams for polarizability computation).
    for node in sys.nodes
        node isa BeamNode && compile_node!(node, sys.basis, rng, param_values)
    end

    # Collect all beams: trapping beams from sys.beams + coupling beams from BeamNodes
    resolved_trapping = AbstractBeam[resolve(b, param_values; cache=cache) for b in sys.beams]
    resolved_coupling = AbstractBeam[n._compiled[] for n in sys.nodes if n isa BeamNode]
    resolved_beams    = vcat(resolved_trapping, resolved_coupling)

    # === PHASE 2: INITIALIZE ATOMS (uses resolved beams, may sample positions/velocities) ===
    atoms = [initialize!(sys.atoms[i], sys.atoms[i].inner;
                         beams=resolved_beams, rng=rng, param_values=param_values)
             for i in 1:length(sys.atoms)]

    # === PHASE 3: COMPILE REMAINING NODES (CouplingNode, DetuningNode, etc.) ===
    # Atom positions are now set; BeamNodes already compiled.
    resolved_fields = AtomTwin.Dynamiq.AbstractField[]
    resolved_jumps  = Jump[]

    for node in sys.nodes
        node isa BeamNode && continue  # already compiled
        obj = compile_node!(node, sys.basis, rng, param_values)
        if obj isa AtomTwin.Dynamiq.AbstractField
            push!(resolved_fields, obj)
        elseif obj isa Jump
            push!(resolved_jumps, obj)
        end
    end

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

    dt = seq.dt

    # === PHASE 4: COMPILE INSTRUCTIONS WITH RESOLVED SYSTEM ===
    
    n_instructions = length(seq)
    modifiers = Vector{Any}(undef, n_instructions)
    step_counts = Vector{Int}(undef, n_instructions)
    total_tspan_size = 0
    
    for (i, inst) in enumerate(seq)
        # Resolve instruction if it contains deferred objects (using same cache)
        resolved_inst = resolve(inst, param_values; cache=cache)
        
        # Compile and resolve_target (which uses same cache)
        mods, n_steps = compile(atoms, resolved_inst, dt; resolve_target=resolve_target)
        modifiers[i] = mods
        step_counts[i] = n_steps
        total_tspan_size += n_steps
    end

    # === PHASE 5: BUILD DETECTORS AND OUTPUT STORAGE ===

    offsets = cumsum([0; step_counts])

    # Preallocate outputs for single shot based on detector dimensions
    times = collect(range(dt, step=dt, length=total_tspan_size))
    n_detectors = length(sys.detector_specs)
    
    detector_vals = Vector{Any}(undef, n_detectors)

    for j in 1:n_detectors
        spec = sys.detector_specs[j]
        detector_vals[j] = spec.ndims == 1 ? 
            zeros(spec.eltype, total_tspan_size) : 
            zeros(spec.eltype, total_tspan_size, spec.ndims)
    end

    # Create views for local tspans
    local_tspans = [view(times, offsets[i]+1:offsets[i+1]) for i in 1:n_instructions]

    # Build detectors for single shot with resolve_target
    detectors = Vector{Any}(undef, n_instructions)

    for i in 1:n_instructions
        _detectors = [
            begin
                # Create appropriate view based on dimensionality
                if ndims(detector_vals[j]) == 1
                    vals_view = view(detector_vals[j], offsets[i]+1:offsets[i+1])
                else
                    vals_view = view(detector_vals[j], offsets[i]+1:offsets[i+1], :)
                end
                
                build_detector(
                    sys.detector_specs[j],
                    local_tspans[i],
                    vals_view,
                    resolve_target,  # Use our resolver with shared cache
                    sys
                )
            end
            for j in 1:n_detectors
        ]
        
        detectors[i] = _detectors
    end

    # Create output dict with views
    detector_outputs = Dict{String, Any}(
        sys.detector_specs[j].params.name => detector_vals[j] 
        for j in 1:n_detectors
    )
    
    return SimulationJob(qstate, atoms, resolved_beams, resolved_fields, resolved_jumps,
                        modifiers, detectors, local_tspans,
                        detector_outputs, times)
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

    # Phase 1: recompile BeamNodes first
    for node in sys.nodes
        node isa BeamNode && recompile_node!(node, nothing, rng, param_values)
    end

    # Collect beams for atom reinitialization
    resolved_coupling = AbstractBeam[n._compiled[] for n in sys.nodes if n isa BeamNode]
    all_beams = vcat(job.beams, resolved_coupling)  # job.beams holds trapping beams

    # Phase 2: reinitialize atoms
    for i in 1:length(sys.atoms)
        initialize!(sys.atoms[i], job.atoms[i]; beams=all_beams, rng=rng, param_values=param_values)
    end

    # Phase 3: recompile remaining nodes
    field_counter = 0
    jump_counter  = 0

    for node in sys.nodes
        node isa BeamNode && continue  # already recompiled
        obj = node_output(node)
        if obj isa AtomTwin.Dynamiq.AbstractField
            field_counter += 1
            recompile_node!(node, job.fields[field_counter], rng, param_values)
        elseif obj isa Jump
            jump_counter += 1
            recompile_node!(node, job.jumps[jump_counter], rng, param_values)
        end
    end

    # Regenerate noise in modifiers
    for modifier_list in job.modifiers, mod in modifier_list
        if mod isa AmplitudeModifier && mod.field isa NoisyField
            Random.seed!(mod.field.rng, rand(rng, UInt))
            update_noise!(mod.vals, job.times, 0.0, mod.field)
        end
    end
    
    # Reset quantum state from sys.state[] (set by compile or caller before recompile!)
    if job.state !== nothing && sys.state[] !== nothing
        job.state .= sys.state[]
    end
    
    # Zero detector outputs
    for vals in values(job.detector_outputs)
        fill!(vals, 0.0)
    end
    
    return job
end
