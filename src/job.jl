"""
    SimulationJob{S}

Compiled simulation execution structure containing all runtime state and operators.

This object should not be constructed directly by users. Instead, use
`compile(system, sequence; shots=1)` which handles optimization and memory preallocation.

# Structure
- **Runtime state** (reset between shots): `state`, `atoms`, `beams` (restored from
  `initial_state` / `initial_beams`)
- **Execution structures** (shared across shots): `fields`, `jumps`, `modifiers`
- **Detectors**: `detectors` (per-instruction), `detector_outputs` (views to results)
- **Time grids**: `times` (global downsampled), `local_tspans` (per-instruction solver time grids)
- **Configuration**: `downsamples` (per-instruction downsample factors), `tol`
  (target local relative error per step, forwarded to the propagator)

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
    initial_beams::Vector{AbstractBeam}  # copies of trapping beams at compile time; restore moved/ramped beam state between shots
    fields::Vector{<:Dynamiq.AbstractField}
    jumps::Vector{Jump}
    modifiers::Vector{Any}          # Vector{Vector{AbstractModifier}} — inner-loop, passed to evolve!
    boundary_modifiers::Vector{Any} # Vector{Vector{AbstractBoundaryModifier}} — called at instruction boundaries only
    detectors::Vector{Any}          # Vector{Vector{<:AbstractDetector}} — element type varies per instruction
    local_tspans::Vector     # Vector of SubArray views into full time grid, one per instruction
    detector_outputs::Dict{String, Any}
    times::Vector{Float64}   # downsampled time grid (length = sum(t_steps[i] ÷ downsamples[i]))
    downsamples::Vector{Int} # per-instruction downsample factors
    inst_dts::Vector{Float64}# per-instruction OUTPUT step; the solver needs it
                             # explicitly because `local_tspans[i]` may hold a
                             # single absolute time, from which no spacing can
                             # be recovered
    tol::Float64             # target local relative error, forwarded to the propagator
    jtol::Float64            # target MCWF jump-omission probability
    integrator::Dynamiq.AbstractIntegrator  # propagator backend, forwarded to the solver
end


_tovector(state::AbstractLevel) = [state]
_tovector(state::Tuple) = collect(state)
_tovector(state::AbstractVector) = state
_tovector(state) = [state]

"""
    _derive_dt(seq, fields) -> Float64

Pick a solver step for a `Sequence` built without an explicit `dt`.

Returns the **output** resolution, not the integration step. The solver
sub-divides this as its own error estimators require (`strang_substeps!` for the
splitting error, `jump_substeps` for the MCWF jump test), so `tol` and `jtol`
govern accuracy while this governs only how often results are recorded.

With no `dt` or `steps` on the `Sequence`, the default is one sample per
instruction: a user who wrote only `Sequence(; tol)` is asking for an accurate
final state, and one who wants a trace says so with `dt`, `steps` or
`downsample`. QuTiP and QuantumOptics.jl both make output times a required
argument for the same reason.

The accuracy-derived step from `Dynamiq.suggested_dt` still applies where it is
*smaller*, since recording more often than the solver steps would be misleading.

The step need not divide any instruction's duration: `stepgrid` adjusts it per
instruction so every duration is realised exactly.
"""
# The field that actually carries `H`. Every `AbstractField` is a leaf except
# `NoisyField`, which wraps one; unwrapping here keeps the step estimator from
# silently losing a noisy system's Hamiltonian.
_hamiltonian_source(f) = hasproperty(f, :coupling) ? f.coupling : f

"""
    resolve_jtol(jtol, shots) -> Float64

The MCWF jump-omission tolerance to use. An explicit `jtol` is honoured
unchanged; `nothing` — the default — derives one from the shot count as
`clamp(1/sqrt(shots), 1e-4, 1e-2)`.

`jtol` bounds a sampling error, the probability of missing a second jump within
one sub-step, not an integrator error. Tightening it past the statistical noise
floor costs sub-steps as `1/sqrt(jtol)` and buys nothing, so it is matched to the
`1/sqrt(N)` noise of an `N`-shot trajectory mean. The clamps bound the ends,
where a single trajectory has no ensemble to average a bias into and where the
bound would otherwise cost more than the statistics it protects.
"""
resolve_jtol(jtol::Float64, ::Integer) = jtol
resolve_jtol(::Nothing, shots::Integer) = clamp(1 / sqrt(max(shots, 1)), 1e-4, 1e-2)

function _derive_dt(seq::Sequence, fields, jumps = (), qstate = nothing;
                    integrator::Dynamiq.AbstractIntegrator = Dynamiq.Chebyshev(),
                    shots::Integer = 1)
    # `NoisyField` WRAPS a coupling rather than carrying `H` itself, so a plain
    # `hasproperty` filter drops it and the Hamiltonian never reaches the
    # accuracy bound: `suggested_dt` then returns `Inf` and the step falls
    # through to the control-grid floor, with `tol` having no effect at all.
    terms = Tuple{Base.RefValue{ComplexF64},Dynamiq.Op}[]
    for f in fields
        ff = _hamiltonian_source(f)
        (hasproperty(ff, :H) && hasproperty(ff, :_coeff)) || continue
        push!(terms, (ff._coeff, ff.H))
    end

    durations = Float64[inst.duration for inst in seq
                        if hasproperty(inst, :duration) && inst.duration > 0]
    longest   = isempty(durations) ? 0.0 : maximum(durations)

    # `dt` is the OUTPUT resolution, and the user did not ask for one, so give
    # the minimum that is still meaningful: one sample per instruction, at its
    # end. Someone who wanted a trace would have said so with `dt`, `steps` or
    # `downsample`; someone who only wrote `Sequence(; tol)` is asking for an
    # accurate final state, which is what they get.
    #
    # This is the convention every comparable package follows -- QuTiP's `tlist`
    # and QuantumOptics.jl's `tspan` are both REQUIRED arguments, and neither
    # invents output times.
    #
    # No floor on the output grid: resolving pulse envelopes is `sample_at`'s
    # job, since envelopes are read at whatever time the solver asks for. A
    # `shortest/100` floor here bound on six of nine examples --
    # over-resolving `gateX_tomography` by 148x -- and that capped `tol`: the
    # delivered error sat at 1.08e-5 for every `tol` from 1e-3 to 1e-6, because
    # the floor, not the tolerance, was choosing the step.
    #
    # The integration step is NOT this: the solver sub-divides `dt` as its own
    # error estimator requires (`strang_substeps!`, `jump_substeps`).
    dt = min(Dynamiq.suggested_dt(terms, seq.tol; integrator = integrator),
             longest > 0 ? longest : Inf)

    # MCWF tests `‖ψ‖² < rand()` ONCE per step, so at most one jump can fire per
    # step. When the norm drops appreciably within a single step the extra jumps
    # are lost and population is stranded in the decaying level -- a SILENT
    # failure, no NaN and no warning. Measured on a driven two-level atom
    # (steady state Ω²/(2Ω²+γ²) = 0.3333): γ·dt = 2.5 gave 0.2269 and γ·dt = 4.0
    # gave 0.1528, 32% and 54% low.
    #
    # It is not a propagator error. Chebyshev integrates the deterministic part
    # correctly, which is exactly why it made this worse: a more accurate
    # propagator licenses larger steps.
    #
    # The fix is the standard MCWF Δp-criterion (Dörner et al.,
    # Comput. Phys. Commun. 234 (2019) 44, arXiv:1803.08589): bound the total
    # jump probability per step by `Δp`, whereupon the probability of TWO jumps
    # in one step -- the event the one-jump-per-step scheme omits -- is `Δp²`.
    # Setting `Δp = sqrt(tol)` therefore bounds the omitted probability by `tol`,
    # with no fitted constant.
    #
    # `Δp` converts to a step through a rigorous, state-independent bound. With
    # `D = Σⱼ γⱼ Lⱼ†Lⱼ`, `d‖ψ‖²/dt = −⟨ψ|D|ψ⟩ ≥ −λ_max(D)·‖ψ‖²`, so the jump
    # probability over one step is at most `1 − exp(−Γ_max·dt)` for ANY state.
    # `D` is diagonal because AtomTwin's jumps are single-transition operators
    # (verified on k39: max |offdiag| = 0), so `Γ_max` is the largest entry of the
    # cached `LdagL_diag` sums -- no eigendecomposition. Checked on k39 (d=24,
    # 54 jumps): the worst ratio of actual to bounded jump probability over 2000
    # random states is 0.911, so the bound holds and is tight.
    # MCWF ONLY. The bound corrects a sampling error of the stochastic unravelling
    # -- omitted second jumps -- which the master equation does not have: it
    # propagates the ensemble directly and never draws a jump. Measured on the ME
    # path, `jtol` moves nothing but the step count (identical P_e to 8 digits at
    # `tol = 1e-6`, 469 steps either way), so applying it there buys no accuracy
    # and costs up to 67x (eit_with_dissipation 0.18 s -> 12.0 s).
    #
    # The ME and MCWF grids may therefore differ. Code that compares the two
    # elementwise must put them on a common grid explicitly, as
    # `rabi_with_dissipation` now does, rather than relying on both being clamped
    # by a bound only one of them needs.
    if qstate isa Vector{ComplexF64} && !isempty(jumps)
        Γmax = 0.0
        for j in jumps
            (hasproperty(j, :_coeff) && hasproperty(j, :LdagL_diag)) || continue
            j.LdagL_diag === nothing && continue
            γ = abs2(j._coeff[])
            for x in j.LdagL_diag
                Γmax = max(Γmax, γ * x)
            end
        end
        jt = resolve_jtol(seq.jtol, shots)
        if Γmax > 0
            Δp = sqrt(min(jt, 0.25))          # two-jump probability ≈ Δp² ≤ jtol
            dt = min(dt, -log1p(-Δp) / Γmax)
        end
    end

    # The Strang splitting error is NOT bounded here. It is measured and
    # corrected inside the solver: `strang_substeps!` divides each `dt` into
    # however many equal sub-steps the step tolerance needs, re-estimating every
    # step from a step-doubling comparison. A compile-time bound cannot do this
    # job -- it would have to predict, from the initial state, an error that
    # depends on the trajectory and on drive amplitudes that are still zero when
    # the sequence is built.

    (isfinite(dt) && dt > 0) || throw(ArgumentError(
        "cannot derive a time step: the sequence has no Hamiltonian terms and no " *
        "instruction durations. Pass an explicit step, e.g. Sequence(1e-9)."))
    return dt
end

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
    integrator::Dynamiq.AbstractIntegrator = Dynamiq.Chebyshev(),
    rng = Random.default_rng(),
    shots::Integer = 1,
    kwargs...)
      
    param_values = Dict{Symbol,Any}(kwargs)

    # Photo (click) detectors count individual quantum jumps, which only exist in
    # the statevector / wavefunction Monte Carlo solver. The density-matrix solver
    # integrates the smooth master equation with no discrete jumps, so a photo
    # detector would silently record nothing. Fail loudly instead.
    if density_matrix && any(s -> s.kind === Dynamiq.PhotoDetector, sys.detector_specs)
        error("PhotoDetector (photon clicks) requires the statevector solver: call " *
              "play(...; density_matrix = false). The density-matrix solver has no " *
              "discrete quantum jumps to count.")
    end

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
    clicks_jumps    = Dict{String,Jump}()   # PhotoDetector name -> the jump it counts

    for node in sorted_nodes
        node isa BeamNode && continue  # already compiled
        obj = compile_node!(node, sys.basis, rng, param_values)
        if obj isa AtomTwin.Dynamiq.AbstractField
            push!(resolved_fields, obj)
        elseif obj isa Jump
            AtomTwin.Dynamiq.precompute!(obj, Vector)
            AtomTwin.Dynamiq.precompute!(obj, Matrix)
            push!(resolved_jumps, obj)
            if node isa DecayNode && node.clicks !== nothing
                clicks_jumps[node.clicks] = obj
            end
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

    # When the sequence carries no explicit dt, derive one from the Hamiltonian
    # that is actually present. `Dynamiq.suggested_dt` uses a Gershgorin upper
    # bound on ‖H‖ (O(nnz), no matrix assembled), so the result is conservative.
    derived_dt = seq.dt === nothing ? _derive_dt(seq, resolved_fields, resolved_jumps, qstate; integrator = integrator, shots = shots) : nothing

    for (i, inst) in enumerate(seq)
        # Resolve instruction if it contains deferred objects (using same cache)
        resolved_inst = resolve(inst, param_values; cache=cache)

        # Per-instruction dt: instruction's own, else the sequence's, else derived.
        dt_i = something(resolved_inst.dt, seq.dt, derived_dt)

        # Detectors record every `downsample`-th step, so `dt` is also refined so
        # that `downsample` divides the step count. Otherwise the final group is
        # partial and `out.times` is non-uniform in exactly one interval -- a grid
        # that looks uniform but breaks `diff(t)` at the boundary. `stepgrid`
        # rounds the count UP, so this only ever makes the step smaller.
        ds_i = something(resolved_inst.downsample, seq.downsample)
        if ds_i > 1 && hasproperty(resolved_inst, :duration) && resolved_inst.duration > 0
            _, dt_i = stepgrid(resolved_inst.duration, dt_i, ds_i)
        end

        # Compile and resolve_target (which uses same cache)
        mods, bmods, n_steps = compile(atoms, resolved_inst, dt_i; resolve_target=resolve_target)
        # Narrow the element type. `compile` returns `Vector{AbstractModifier}`,
        # and iterating an abstractly-typed vector is a dynamic dispatch: the
        # solver calls `update!` on every modifier at every sub-step, and the
        # boxing costs 48 B per call. `identity.(...)` re-infers the eltype from
        # the contents, which for the usual single-modifier case is concrete and
        # allocation-free. (Same reason `resolved_fields` is narrowed above.)
        modifiers[i] = isempty(mods) ? mods : identity.(mods)
        boundary_modifiers[i] = bmods
        step_counts[i] = n_steps
        total_tspan_size += n_steps
    end

    # === PHASE 5: BUILD DETECTORS AND OUTPUT STORAGE ===

    offsets  = cumsum([0; step_counts])

    # Per-instruction dt must be the step `compile` ACTUALLY used, not the one
    # requested. `stepgrid` adjusts dt so the instruction's duration is realised
    # exactly (duration is physical; dt is a discretisation choice), so recompute
    # it from the realised step count — otherwise the time grid built here drifts
    # from the grid the solver steps on.
    inst_dts = Float64[]
    for i in 1:n_instructions
        req = something(seq[i].dt, seq.dt, derived_dt)
        dur = hasproperty(seq[i], :duration) ? Float64(seq[i].duration) : 0.0
        push!(inst_dts, (dur > 0 && step_counts[i] > 0) ? dur / step_counts[i] : req)
    end
    inst_ds  = [something(seq[i].downsample, seq.downsample) for i in 1:n_instructions]

    # `stepgrid` has already refined `dt` so `downsample` divides the step count,
    # so this divides exactly. `cld` rather than `÷` regardless, so that any path
    # which bypasses that refinement still sizes the buffer for the final partial
    # group instead of silently dropping the endpoint.
    ds_counts  = [max(1, cld(step_counts[i], inst_ds[i])) for i in 1:n_instructions]
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
        # The last sample is the instruction's final step, which is only a whole
        # `step` from the previous one when `ds_i` divides `step_counts[i]`.
        times[ds_offsets[i+1]] = abs_start + step_counts[i] * dt_i
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
        # Narrow the element type, as for `modifiers` above: `write_detectors!`
        # runs once per output step, and iterating an abstractly-typed vector
        # boxes.
        detectors[i] = identity.(map(1:n_detectors) do j
            vals_slice = ds_offsets[i]+1:ds_offsets[i+1]
            vals_view  = ndims(detector_vals[j]) == 1 ?
                view(detector_vals[j], vals_slice) :
                view(detector_vals[j], vals_slice, :)
            build_detector(sys.detector_specs[j], ds_tspan, vals_view, resolve_target, sys)
        end)
    end

    # Bind each PhotoDetector to the jump it counts (from add_decay!(...; clicks=…)).
    # Each per-instruction PhotoDetector holds a view into its own time segment and
    # a direct ref to the shared jump; the solver, which already receives both the
    # per-instruction detectors and the jumps, increments a detector only when its
    # bound jump fires. deepcopy for the parallel path preserves this shared ref.
    if !isempty(clicks_jumps)
        for i in 1:n_instructions, d in detectors[i]
            d isa Dynamiq.PhotoDetector || continue
            j = get(clicks_jumps, d.name, nothing)
            j === nothing && error("PhotoDetector \"$(d.name)\" is attached but not " *
                "bound to a decay; pass it to add_decay!(...; clicks = spec).")
            d.jump = j
        end
    end

    detector_outputs = Dict{String, Any}(
        sys.detector_specs[j].params.name => detector_vals[j]
        for j in 1:n_detectors
    )

    # Snapshot the trapping beams' compile-time state (positions/amplitudes) so
    # recompile! can restore them between shots: modifiers (Move/Position/Amplitude)
    # mutate beam.r0 / beam._coeff in place, which would otherwise accumulate across
    # trajectories. Coupling beams are rebuilt fresh by recompile! and need no snapshot.
    initial_beams = AbstractBeam[copy(b) for b in resolved_trapping]

    return SimulationJob(qstate, qstate === nothing ? nothing : copy(qstate),
                        atoms, resolved_beams, initial_beams, resolved_fields, resolved_jumps,
                        modifiers, boundary_modifiers, detectors, local_tspans,
                        detector_outputs, times, inst_ds, inst_dts, seq.tol,
                        resolve_jtol(seq.jtol, shots),
                        integrator)
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
                    rng = Random.default_rng(),
                    kwargs...)

    param_values = Dict{Symbol, Any}(kwargs)

    # Phase 0: restore trapping beams to their compile-time state. Inner-loop
    # modifiers (Move/Position/Amplitude) mutate beam.r0/_coeff in place; without
    # this the displacement/amplitude from one shot carries into the next (e.g. a
    # MoveCol would step the tweezer an extra `delta` every trajectory).
    for k in eachindex(job.initial_beams)
        restore_beam!(job.beams[k], job.initial_beams[k])
    end

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
