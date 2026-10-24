"""
    SimulationJob{S}

Compiled simulation execution structure containing all runtime state and operators.

This object should not be constructed directly by users. Instead, use
`compile(system, sequence)` which handles optimization and memory preallocation.

A job is a snapshot: it owns a private copy of the `System` it was compiled from
(`system`), and its atoms, fields and jumps are that copy's. Nothing a run does
reaches the user's `System`, two jobs compiled from one `System` are independent,
and a change to the `System` after `compile` takes effect at the next `compile`.

# Structure
- **Runtime state** (reset between shots): `state`, `atoms`, `beams` (restored from
  `initial_state` / `initial_atoms` / `initial_beams`)
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
    initial_atoms::Vector{NLevelAtom}    # copies of the atoms at compile time; shot 1 of a replay starts from them
    fields::Vector{<:Dynamiq.AbstractField}
    n_auto_fields::Int       # leading entries of `fields` with no DAG node behind
                             # them (automatic light shifts). `recompile!` walks
                             # nodes, so it must skip these or it updates the
                             # wrong field -- see the offset in `recompile!`.
    jumps::Vector{Jump}
    modifiers::Vector{Any}          # Vector{Vector{AbstractModifier}} — inner-loop, passed to evolve!
    boundary_modifiers::Vector{Any} # Vector{Vector{AbstractBoundaryModifier}} — called at instruction boundaries only
    detectors::Vector{Any}          # Vector{Vector{<:AbstractDetector}} — element type varies per instruction
    local_tspans::Vector     # Vector of SubArray views into full time grid, one per instruction
    detector_outputs::Dict{String, Any}
    detector_grids::Dict{String, Any}  # detectors with their own `tspan`: see `_detector_grid`
    times::Vector{Float64}   # downsampled time grid (length = sum(t_steps[i] ÷ downsamples[i]))
    downsamples::Vector{Int} # per-instruction downsample factors
    inst_dts::Vector{Float64}# per-instruction OUTPUT step; the solver needs it
                             # explicitly because `local_tspans[i]` may hold a
                             # single absolute time, from which no spacing can
                             # be recovered
    tol::Float64             # target local relative error, forwarded to the propagator
    integrator::Dynamiq.AbstractIntegrator  # propagator backend, forwarded to the solver
    system::System           # the job's private copy of the System (nodes, atoms, specs)
    parameter_names::Set{Symbol}  # every `Parameter` a `play` keyword may set
    param_values::Dict{Symbol, Any}  # the values the job was last given (compile, recompile!, play)
end

# `play` keywords that reach `compile`/`recompile!` through `kwargs` and are not
# parameter overrides.
const _PLAY_KEYWORDS = (:shots, :savefinalstate, :frozen, :density_matrix, :parallel_thresh,
                        :shot_callback, :initial_state, :rng, :integrator)

_parameter_values(kwargs) = Dict{Symbol,Any}(k => v for (k, v) in kwargs if !(k in _PLAY_KEYWORDS))

# Every `Parameter` reachable from the system and the sequence, by name: the DAG
# nodes' knobs (a `BeamRabiFrequency` reaches its parametric beam), the atoms'
# `x_init`/`v_init`, the trapping beams, the quantization axis, the instructions.
function _parameter_names(sys::System, seq::Sequence)
    names, seen = Set{Symbol}(), Base.IdSet{Any}()
    _collect_parameters!(names, sys, seen)
    _collect_parameters!(names, seq, seen)
    return names
end

_collect_parameters!(names, p::Parameter, seen) = (push!(names, p.name); nothing)
_collect_parameters!(names, e::ParametricExpression, seen) =
    (foreach(a -> _collect_parameters!(names, a, seen), e.args); nothing)
# Leaves: values, compiled objects (whose knobs live on their nodes), and types
_collect_parameters!(names, ::Union{Number, AbstractString, Symbol, Nothing, Function, Type,
                                   Module, Random.AbstractRNG, AbstractArray{<:Number}, IO,
                                   Basis, NLevelAtom, Dynamiq.AbstractField, Jump,
                                   Dynamiq.Op}, seen) = nothing
_collect_parameters!(names, x::Union{AbstractArray, Tuple, Pair, NamedTuple}, seen) =
    (foreach(e -> _collect_parameters!(names, e, seen), x); nothing)
_collect_parameters!(names, x::AbstractDict, seen) =
    (foreach(v -> _collect_parameters!(names, v, seen), values(x)); nothing)
_collect_parameters!(names, x::Base.RefValue, seen) = _collect_parameters!(names, x[], seen)
function _collect_parameters!(names, x, seen)      # any other struct: its fields
    if ismutable(x)
        x in seen && return nothing
        push!(seen, x)
    end
    for i in 1:nfields(x)
        isdefined(x, i) && _collect_parameters!(names, getfield(x, i), seen)
    end
    return nothing
end

# A keyword that names no `Parameter` is a typo (`Omeg = …`, `delta` for `δ`): it used
# to run the default silently, a flat line that looks like physics.
function _check_parameter_names(known::Set{Symbol}, param_values)
    unknown = setdiff(keys(param_values), known)
    isempty(unknown) && return nothing
    throw(ArgumentError("unknown parameter(s) $(join(sort!(collect(unknown)), ", ")); " *
                        "the system and sequence define " *
                        (isempty(known) ? "none" : join(sort!(collect(known)), ", "))))
end


# A detector given its own `tspan` reports on that grid instead of `out.times`. It
# records on the output grid like any other; each run's record is then mapped onto
# the detector's: a `PhotoDetector` sums its clicks over (tspan[k-1], tspan[k]] (from
# t = 0), any other detector is read at the output time nearest each tspan[k].
function _detector_grid(spec, times)
    name, t = spec.params.name, spec.tspan
    T = isempty(times) ? 0.0 : times[end]
    tol = 1e-9 * T
    issorted(t) || throw(ArgumentError("detector \"$name\": tspan must be sorted"))
    (isempty(t) || (t[1] > 0 && t[end] <= T + tol)) ||
        throw(ArgumentError("detector \"$name\": tspan must lie within the run, (0, $T] s"))
    return spec.kind === Dynamiq.PhotoDetector ?
        (bins = [searchsortedfirst(t, x - tol) for x in times], n = length(t)) :
        (rows = [argmin(abs.(times .- x)) for x in t],)
end

_on_detector_grid(vals, g::NamedTuple{(:rows,)}) = ndims(vals) == 1 ? vals[g.rows] : vals[g.rows, :]
function _on_detector_grid(vals, g::NamedTuple{(:bins, :n)})
    out = zeros(eltype(vals), g.n)
    for (k, b) in enumerate(g.bins)
        b <= g.n && (out[b] += vals[k])
    end
    return out
end

# The run's detector outputs, each on its own grid.
_detector_outputs(job) = Dict{String, Any}(
    name => haskey(job.detector_grids, name) ? _on_detector_grid(vals, job.detector_grids[name]) : vals
    for (name, vals) in job.detector_outputs)

_tovector(state::AbstractLevel) = [state]
_tovector(state::Tuple) = collect(state)
_tovector(state::AbstractVector) = state
_tovector(state::AbstractMatrix) = state     # a density matrix in the basis
_tovector(state) = [state]

# The field that actually carries `H`. Every `AbstractField` is a leaf except
# `NoisyField`, which wraps one; unwrapping here keeps the step estimator from
# silently losing a noisy system's Hamiltonian.
_hamiltonian_source(f) = hasproperty(f, :coupling) ? f.coupling : f

"""
    _derive_dt(seq, fields) -> Float64

Pick a solver step for a `Sequence` built without an explicit `dt`.

Returns the **output** resolution, not the integration step. The solver
sub-divides this as its own error estimator requires (`strang_substeps!` for the
splitting error; MCWF places each jump exactly and needs no division), so `tol`
governs accuracy while this governs only how often results are recorded.

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

function _derive_dt(seq::Sequence, fields;
                    integrator::Dynamiq.AbstractIntegrator = Dynamiq.Chebyshev())
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

    durations = filter(>(0), Float64[instruction_duration(inst) for inst in seq])
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
    # error estimator requires (`strang_substeps!`).
    dt = min(Dynamiq.suggested_dt(terms, seq.tol; integrator = integrator),
             longest > 0 ? longest : Inf)

    # MCWF needs no bound here: every quantum jump is placed at the time its norm
    # threshold is crossed (`Dynamiq.mcwf_advance!`), however many fall in one step.

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
    compile(system::System, sequence::Sequence; initial_state = system.initial_state,
            density_matrix = false, integrator = Chebyshev(), rng, kwargs...) -> SimulationJob

Compile a System and Sequence into an executable SimulationJob with preallocated single-shot storage.

The job works on a **private copy** of `system`: compiling and playing it never changes
the system's fields or atoms, and two jobs compiled from one system are independent.
A change to the system after `compile` takes effect at the next `compile`.

Builds heterogeneous time grids when the Sequence has per-instruction `dt` or `downsample`
overrides, enabling cost optimization across protocol phases.

# Arguments
- `system`: System specification (atoms, beams, nodes, detectors)
- `sequence`: Pulse sequence to execute (instruction list with base timestep `dt`).
  Individual instructions can override `dt` and `downsample` via `push!` keyword arguments.
- `initial_state`: Initial quantum state (default: the system's; empty for a classical run)
- `density_matrix`: Use density matrix formalism if `true` (default: `false`)
- Additional keyword arguments are treated as parameter overrides (e.g. `Ω = 2π*1e6`).
  A keyword that names no `Parameter` of the system or sequence is an error; the
  keywords of `play` (`shots`, `savefinalstate`, …) are passed through.

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
function compile(user_sys::System, seq::Sequence;
    initial_state = user_sys.initial_state,
    density_matrix = false,
    integrator::Dynamiq.AbstractIntegrator = Dynamiq.Chebyshev(),
    rng = Random.default_rng(),
    kwargs...)

    isempty(seq) && throw(ArgumentError("compile: the sequence has no instructions"))
    param_values = _parameter_values(kwargs)
    parameter_names = _parameter_names(user_sys, seq)
    _check_parameter_names(parameter_names, param_values)

    # The job's own copy of the system. Every node's field, every atom and every
    # detector spec below is the copy's, so the shot-by-shot values written into
    # them (a sampled Ω, a thermal position) never reach `user_sys`, and
    # `gethamiltonian(user_sys)` keeps reporting the system as declared.
    copies = IdDict{Any, Any}()
    sys = Base.deepcopy_internal(user_sys, copies)::System
    _bind_quantization_axis!(sys)
    # The user's handles -- the coupling a `Pulse` names, the array a `MoveCol`
    # moves -- must resolve to the copy's objects, or the instruction would drive
    # the user's field while the solver evolves the job's. `resolve` consults
    # `cache` first, so it is seeded with the copy map.
    # The copy's own objects resolve to themselves (`resolve` would otherwise copy a
    # beam it does not know, and a detector would watch a beam no instruction moves).
    cache = IdDict{Any, Any}()
    for (k, v) in copies
        if k isa Union{AtomTwin.Dynamiq.AbstractField, AbstractBeam, TweezerArray,
                       AbstractAtom, NLevelAtom, Jump}
            cache[k] = v
            cache[v] = v
        end
    end

    # Photo (click) detectors count individual quantum jumps, which only exist in
    # the statevector / wavefunction Monte Carlo solver. The density-matrix solver
    # integrates the smooth master equation with no discrete jumps, so a photo
    # detector would silently record nothing. Fail loudly instead.
    if density_matrix && any(s -> s.kind === Dynamiq.PhotoDetector, sys.detector_specs)
        error("PhotoDetector (photon clicks) requires the statevector solver: call " *
              "play(...; density_matrix = false). The density-matrix solver has no " *
              "discrete quantum jumps to count.")
    end

    sorted_nodes = _topological_sort(sys.nodes)

    # === PHASE 1: COMPILE BEAM NODES FIRST ===
    # BeamNodes must be compiled before CouplingNodes (which read beam_node._compiled[])
    # and before atom initialization (which uses beams for polarizability computation).
    # The quantization axis joins this phase for the same reason beams do: the
    # tensor light shift needs it, and atom polarizabilities are computed in
    # Phase 2.
    for node in sorted_nodes
        (node isa BeamNode || node isa QuantizationAxisNode) &&
            compile_node!(node, sys.basis, rng, param_values)
    end

    # Collect all beams: trapping beams from sys.beams + coupling beams from BeamNodes
    resolved_trapping = AbstractBeam[resolve(b, param_values; cache=cache, rng=rng) for b in sys.beams]
    resolved_coupling = _coupling_beams(sorted_nodes)
    resolved_beams    = vcat(resolved_trapping, resolved_coupling)

    # === PHASE 2: INITIALIZE ATOMS (uses resolved beams, may sample positions/velocities) ===
    q_axis = getquantizationaxis(sys)
    _set_photon_dipoles!(sys, q_axis)
    tensor_pairs = _tensor_may_vary(sys) ? :all : :fit
    atoms = [initialize!(sys.atoms[i], sys.atoms[i].inner;
                         beams=resolved_beams, rng=rng, param_values=param_values,
                         q_axis=q_axis, tensor_pairs=tensor_pairs)
             for i in 1:length(sys.atoms)]

    # === PHASE 3: COMPILE REMAINING NODES (CouplingNode, DetuningNode, etc.) ===
    # Atom positions are now set; BeamNodes already compiled.
    resolved_fields = AtomTwin.Dynamiq.AbstractField[]
    resolved_jumps  = Jump[]

    # A trapping beam shifts the levels it traps. That is physics, not an option,
    # so it needs no `add_*!` call: every atom level with a polarizability at a
    # trapping beam's wavelength gets its AC Stark shift automatically.
    #
    # Only `sys.beams` — the beams the user handed to `System` — are included.
    # A coupling beam's effect on the atom is already the coupling term; adding a
    # Stark shift for it too would double-count.
    append!(resolved_fields,
            _auto_light_shifts(sys, atoms, resolved_trapping))
    n_auto_fields = length(resolved_fields)   # these have no node; see `recompile!`
    # PhotoDetector name -> every jump it counts. A manifold decay expands into one
    # jump per sublevel channel and they all share the detector, so this must
    # accumulate; keying a single Jump per name silently counted only the last.
    clicks_jumps    = Dict{String,Vector{Jump}}()

    for node in sorted_nodes
        (node isa BeamNode || node isa QuantizationAxisNode) && continue  # Phase 1
        obj = compile_node!(node, sys.basis, rng, param_values)
        (obj isa AtomTwin.Dynamiq.AbstractField || obj isa Jump) && _reset_activity!(node, obj)
        if obj isa AtomTwin.Dynamiq.AbstractField
            push!(resolved_fields, obj)
        elseif obj isa Jump
            AtomTwin.Dynamiq.precompute!(obj, Vector)
            AtomTwin.Dynamiq.precompute!(obj, Matrix)
            push!(resolved_jumps, obj)
            if node isa DecayNode && node.clicks !== nothing
                push!(get!(clicks_jumps, node.clicks, Jump[]), obj)
            end
        end
    end
    resolved_fields = [obj for obj in resolved_fields]   # eltype inferred from content

    # Create global time reference for noisy fields
    global_time_ref = Ref(0.0)

    # Resolver function that uses the SAME cache for pointer sharing
    resolve_target = obj -> begin
        resolved_obj = resolve(obj, param_values; cache=cache, rng=rng)
        return update_noisy_field_time_refs!(resolved_obj, global_time_ref)
    end

    # Initialize quantum state
    if !isempty(initial_state)
        qstate = getqstate(sys, initial_state; density_matrix=density_matrix)
        # The detectors built below bind to this array (the job's live state); the
        # user's system keeps a handle on it too, as `getqstate(sys)` documents.
        sys.state[] = qstate
        user_sys.state[] = qstate
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
    # Bound ‖H‖ from the fields as they stand at the atoms' initial positions --
    # a `VdWInteraction` has no strength until its first update.
    Dynamiq.refresh_fields!(resolved_fields)
    derived_dt = seq.dt === nothing ? _derive_dt(seq, resolved_fields; integrator = integrator) : nothing

    for (i, inst) in enumerate(seq)
        # Resolve instruction if it contains deferred objects (using same cache)
        resolved_inst = resolve(inst, param_values; cache=cache, rng=rng)

        # Per-instruction dt: instruction's own, else the sequence's, else derived.
        dt_i = something(resolved_inst.dt, seq.dt, derived_dt)

        # Detectors record every `downsample`-th step, so `dt` is also refined so
        # that `downsample` divides the step count. Otherwise the final group is
        # partial and `out.times` is non-uniform in exactly one interval -- a grid
        # that looks uniform but breaks `diff(t)` at the boundary. `stepgrid`
        # rounds the count UP, so this only ever makes the step smaller.
        ds_i = something(resolved_inst.downsample, seq.downsample)
        dur_i = instruction_duration(resolved_inst)
        if ds_i > 1 && dur_i > 0
            _, dt_i = stepgrid(dur_i, dt_i, ds_i)
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

    # The statevector solvers read a time-dependent drive once per step and do
    # not sub-divide, so on a derived grid `tol` has no say in how well a shaped
    # pulse, ramp or move is resolved -- measured, a Gaussian pulse was 14% off at
    # every `tol` from 1e-3 to 1e-9. The density-matrix solver does sub-divide.
    if qstate isa Vector && seq.dt === nothing &&
       any(i -> seq[i].dt === nothing && !isempty(modifiers[i]), 1:n_instructions)
        @warn """
        `Sequence(; tol)` with the statevector solver and a time-dependent drive.

        A shaped pulse, ramp or move is read once per step, and a sequence with
        no `dt` has as few steps as the Hamiltonian allows -- typically one per
        instruction. `tol` does not refine that, so the drive may be badly
        under-resolved. Give the sequence (or the instruction) a `dt` that
        resolves the drive, or use `density_matrix = true`, which sub-divides to
        meet `tol`.
        """ maxlog = 1
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
        dur = instruction_duration(seq[i])
        push!(inst_dts, (dur > 0 && step_counts[i] > 0) ? dur / step_counts[i] : req)
    end
    inst_ds  = Int[something(seq[i].downsample, seq.downsample) for i in 1:n_instructions]

    # `stepgrid` has already refined `dt` so `downsample` divides the step count,
    # so this divides exactly. `cld` rather than `÷` regardless, so that any path
    # which bypasses that refinement still sizes the buffer for the final partial
    # group instead of silently dropping the endpoint.
    # An instruction with no steps (`On`, `Off`, a retune, `Wait(0)`) has no sample:
    # one would duplicate the previous instruction's final time.
    ds_counts  = [step_counts[i] == 0 ? 0 : max(1, cld(step_counts[i], inst_ds[i]))
                  for i in 1:n_instructions]
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
        ds_counts[i] > 0 && (times[ds_offsets[i+1]] = abs_start + step_counts[i] * dt_i)
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
        # boxes. With no detectors `map` gives a `Vector{Any}`, which the
        # solvers' `detectors::Vector{<:AbstractDetector}` rejects.
        dets = map(1:n_detectors) do j
            vals_slice = ds_offsets[i]+1:ds_offsets[i+1]
            vals_view  = ndims(detector_vals[j]) == 1 ?
                view(detector_vals[j], vals_slice) :
                view(detector_vals[j], vals_slice, :)
            build_detector(sys.detector_specs[j], ds_tspan, vals_view, resolve_target, sys)
        end
        detectors[i] = isempty(dets) ? AbstractDetector[] : identity.(dets)
    end

    # Bind each PhotoDetector to the jump it counts (from add_decay!(...; clicks=…)).
    # Each per-instruction PhotoDetector holds a view into its own time segment and
    # a direct ref to the shared jump; the solver, which already receives both the
    # per-instruction detectors and the jumps, increments a detector only when its
    # bound jump fires. deepcopy for the parallel path preserves this shared ref.
    if !isempty(clicks_jumps)
        for i in 1:n_instructions, d in detectors[i]
            d isa Dynamiq.PhotoDetector || continue
            js = get(clicks_jumps, d.name, nothing)
            js === nothing && error("PhotoDetector \"$(d.name)\" is attached but not " *
                "bound to a decay; pass it to add_decay!(...; clicks = spec).")
            d.jumps = js
        end
    end

    detector_outputs = Dict{String, Any}(
        sys.detector_specs[j].params.name => detector_vals[j]
        for j in 1:n_detectors
    )
    detector_grids = Dict{String, Any}(spec.params.name => _detector_grid(spec, times)
                                       for spec in sys.detector_specs if spec.tspan !== nothing)

    # Snapshot the trapping beams' compile-time state (positions/amplitudes) so
    # recompile! can restore them between shots: modifiers (Move/Position/Amplitude)
    # mutate beam.r0 / beam._coeff in place, which would otherwise accumulate across
    # trajectories. Coupling beams are rebuilt fresh by recompile! and need no snapshot.
    initial_beams = AbstractBeam[copy(b) for b in resolved_trapping]
    # ...and the atoms, for the same reason: shot 1 of a replayed job starts from
    # them (later shots re-`initialize!`).
    initial_atoms = [copy(a) for a in atoms]

    return SimulationJob(qstate, qstate === nothing ? nothing : copy(qstate),
                        atoms, resolved_beams, initial_beams, initial_atoms, resolved_fields,
                        n_auto_fields, resolved_jumps,
                        modifiers, boundary_modifiers, detectors, local_tspans,
                        detector_outputs, detector_grids, times, inst_ds, inst_dts, seq.tol,
                        integrator, sys, parameter_names, param_values)
end


"""
    recompile!(job::SimulationJob; rng = Random.default_rng(), kwargs...)
    recompile!(job::SimulationJob, sys::System; kwargs...)

Reinitialize a `SimulationJob` for a new Monte Carlo trajectory.

Updates all DAG node outputs in-place (re-sampling parameter values and noise),
reinitializes atom velocities, resets the quantum state, and zeroes detector outputs.
Works on the job's own copy of the system (`job.system`); the `sys` argument of the
two-argument form is accepted for compatibility and not read.

Additional keyword arguments set parameter values (same as `compile`) and the job
remembers them: a parameter not named keeps the value it was last given, by
`compile`, `recompile!` or `play`, else its default. So `recompile!(job; Δ = x);
play(job, sys)` and `play(job, sys; Δ = x)` are the same sweep step.

# Thread Safety
Safe to call on thread-local job copies (`deepcopy(job)`, which copies the job's
system with it). MUST NOT be called on a job another thread is running.
"""
recompile!(job::SimulationJob, ::System; kwargs...) = recompile!(job; kwargs...)

function recompile!(job::SimulationJob;
                    rng = Random.default_rng(),
                    kwargs...)

    new_values = _parameter_values(kwargs)
    _check_parameter_names(job.parameter_names, new_values)
    param_values = merge!(job.param_values, new_values)
    sys = job.system

    # Phase 0: restore trapping beams to their compile-time state. Inner-loop
    # modifiers (Move/Position/Amplitude) mutate beam.r0/_coeff in place; without
    # this the displacement/amplitude from one shot carries into the next (e.g. a
    # MoveCol would step the tweezer an extra `delta` every trajectory).
    for k in eachindex(job.initial_beams)
        restore_beam!(job.beams[k], job.initial_beams[k])
    end

    sorted_nodes = _topological_sort(sys.nodes)

    # Phase 1: recompile BeamNodes and the quantization axis first (a sampled axis
    # reaches the beam-derived couplings through the axis node, see
    # `_bind_quantization_axis!`)
    for node in sorted_nodes
        (node isa BeamNode || node isa QuantizationAxisNode) &&
            recompile_node!(node, nothing, rng, param_values)
    end

    # Collect beams for atom reinitialization. `job.beams` is the trapping beams
    # followed by the compile-time coupling beams (see `compile`); take only the
    # former, since the coupling beams were just re-resolved.
    resolved_coupling = _coupling_beams(sorted_nodes)
    all_beams = vcat(job.beams[1:length(job.initial_beams)], resolved_coupling)

    # Phase 2: reinitialize atoms, with the same quantization axis `compile`
    # used. The tensor polarizability depends on it, so omitting it recomputes α
    # against ẑ from the second shot on.
    q_axis = getquantizationaxis(sys)
    _set_photon_dipoles!(sys, q_axis)
    for i in 1:length(sys.atoms)
        initialize!(sys.atoms[i], job.atoms[i]; beams=all_beams, rng=rng,
                    param_values=param_values, q_axis=q_axis)
    end

    # The automatic light shifts carry α in their operator (they have no node,
    # so the walk below does not reach them). Keep them in step with the α just
    # recomputed; a no-op unless α changed.
    for k in 1:job.n_auto_fields
        _refresh_auto_shift!(job.fields[k])
    end

    # Phase 3: recompile remaining nodes, each in its declared switch state.
    _foreach_job_output(job, sorted_nodes) do node, f
        if node isa DecayNode
            recompile_node!(node, f, rng, param_values; basis = sys.basis)   # cheap unless resampled
        else
            recompile_node!(node, f, rng, param_values)
        end
        _reset_activity!(node, f)
    end

    # Regenerate noise in modifiers
    for modifier_list in job.modifiers, mod in modifier_list
        if mod isa AmplitudeModifier && mod.field isa NoisyField
            Random.seed!(mod.field.rng, rand(rng, UInt))
            update_noise!(mod.vals, job.times, 0.0, mod.field)
        end
    end
    
    _reset_run_state!(job)
    return job
end

# The coupling beams the atoms see (polarizability, dipole force): every `BeamNode`'s
# resolved beam except a plane wave, which has no profile to trap with (its
# radiation pressure is the `PlanarCoupling`'s own).
_coupling_beams(sorted_nodes) = AbstractBeam[n._compiled[] for n in sorted_nodes
                                             if n isa BeamNode && !(n._compiled[] isa PlanarBeam)]

# Put the atoms back where `compile` left them: position, velocity, and the motion
# caches of the previous run. Shot 1 of every `play` needs it; later shots are
# re-initialised by `recompile!`, which must not undo their resampling.
function _restore_atoms!(job::SimulationJob)
    for (a, a0) in zip(job.atoms, job.initial_atoms)
        copyto!(a.x, a0.x); copyto!(a.v, a0.v)
        Dynamiq.reset_force!(a)
        fill!(a._Eb, 0.0)
    end
    return job
end

# An automatic light shift's α, from its atom's as `initialize!` just left it.
function _refresh_auto_shift!(f::StarkShiftAC)
    α = f.atom.alpha[getwavelength(f.beam)][f.level]
    α == f.alpha || Dynamiq.set_alpha!(f, α)
end
function _refresh_auto_shift!(f::Dynamiq.StarkShiftACTensor)
    α = f.atom.alpha_off[getwavelength(f.beam)]
    α == f.alpha || Dynamiq.set_alpha!(f, α)
end

# Reset what a shot starts from: the quantum state (from `job.initial_state`)
# and the detector outputs, several of which accumulate (a `PhotoDetector`
# counts clicks).
function _reset_run_state!(job::SimulationJob)
    job.state === nothing || (job.state .= job.initial_state)
    for vals in values(job.detector_outputs)
        fill!(vals, 0.0)
    end
    return job
end

# Call `fn(node, output)` for every DAG node that produced a field or jump, with
# the JOB's own copy of that output. `job.fields` begins with `n_auto_fields`
# automatic light shifts that have no node, so the walk starts past them --
# without that offset every node reached the field `n_auto_fields` before its
# own, a DetuningNode writing into a StarkShiftAC. Jumps are in node order too.
function _foreach_job_output(fn, job::SimulationJob, sorted_nodes)
    fi, ji = job.n_auto_fields, 0
    for node in sorted_nodes
        node isa BeamNode && continue
        obj = node_output(node)
        if obj isa AtomTwin.Dynamiq.AbstractField
            fn(node, job.fields[fi += 1])
        elseif obj isa Jump
            fn(node, job.jumps[ji += 1])
        end
    end
    return nothing
end
