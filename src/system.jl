"""
    System

Container for the full physical model of an atomic quantum processor.

A `System` bundles the atoms, beams (or tweezer arrays), basis, static fields,
and detector specifications required to simulate dynamics with `play`
and related routines. The quantum state is stored internally in `state`
and is typically initialized lazily by the simulator.

# Fields

- `atoms::Vector{AbstractAtom}` – list of atoms in the register
- `beams::Vector{AbstractBeam}` – list of beams acting on the atoms
- `initial_state::Vector{AbstractLevel}` – per-atom logical/physical levels
  used to prepare the initial many-body state
- `state::Ref{Union{Nothing,Array{ComplexF64}}}` – current state (statevector
  or density matrix) in the chosen basis; `nothing` if uninitialized
- `basis::Union{Nothing,Basis}` – Hilbert-space basis associated with `atoms`
- `nodes::Vector{AbstractNode}` – DAG nodes encoding Hamiltonian terms,
  jump operators, and other system components. Each node holds its compiled
  output in `node._field`. Built by `add_coupling!`, `add_detuning!`, etc.
- `detector_specs::Vector{Dynamiq.DetectorSpec}` – detector configuration
  used for measurement modelling
"""
struct System
    atoms::Vector{AbstractAtom}
    beams::Vector{AbstractBeam}
    initial_state::Vector{AbstractLevel}
    state::Ref{Union{Nothing, Array{ComplexF64}}}
    basis::Union{Nothing,Basis}
    nodes::Vector{AbstractNode}
    detector_specs::Vector{Dynamiq.DetectorSpec}
end

"""
    System(atoms::Vector{<:AbstractAtom};
           initial_state = AbstractLevel[],
           basis = nothing,
           beams::Vector{AbstractBeam} = AbstractBeam[],
           detector_specs::Vector{Dynamiq.DetectorSpec} = Dynamiq.DetectorSpec[],
           maxoccupations = [])

Main constructor for a `System` from atoms and optional beams, basis,
and detector specifications.

If `basis` is not supplied it is constructed as `Basis(atoms; maxoccupations=...)`,
where `maxoccupations` is a list of `(level, occ)` pairs specifying
maximum local occupations per atomic level.

`initial_state` specifies the desired per-atom starting levels or
superpositions; it is later converted to a full many-body state.

The quantum state is stored in `state`, initially `nothing` and filled
by the simulator. Hamiltonian terms and dissipators are added via
`add_coupling!`, `add_detuning!`, `add_decay!`, etc., which register
DAG nodes in `sys.nodes`.
"""
function System(atoms::Vector{<:AbstractAtom};
                initial_state=AbstractLevel[],
                basis=nothing,
                beams::Vector{<:AbstractBeam}=AbstractBeam[],
                detector_specs::Vector{Dynamiq.DetectorSpec}=Dynamiq.DetectorSpec[],
                maxoccupations=[])

    @assert !isempty(atoms) "System requires at least one atom"

    # Convert maxoccupations from (Level, Int) to (Int, Int) per atom
    maxocc_tuples = Tuple[]
    for atom in atoms
        for (lvl, occ) in maxoccupations
            idx = get(atom.level_indices, lvl, nothing)
            idx === nothing && error("Level $lvl not found in atom.")
            push!(maxocc_tuples, (idx, occ))
        end
    end

    basis_val = isnothing(basis) ? Basis(atoms; maxoccupations=maxocc_tuples) : basis
    state_ref = Ref{Union{Nothing, Array{ComplexF64}}}(nothing)

    return System(atoms, beams, initial_state, state_ref, basis_val, AbstractNode[], detector_specs)
end

"""
    System(atoms, beam_sources; kwargs...)

Convenience constructor that accepts beam sources (individual beams,
tweezers, or a mixed vector). All sources are flattened into beams
via `getbeams` and passed to the main constructor.

# Examples
System(yb, beam) # single beam
System(yb, tweezer) # single tweezer
System(yb, [beam1, tweezer, beam2]) # mixed vector
System([yb1, yb2], tweezers) # multiple atoms
"""
function System(atoms::Vector{<:AbstractAtom}, beam_sources; kwargs...)
    sources_vec = beam_sources isa AbstractVector ? beam_sources : [beam_sources]
    all_beams = mapreduce(getbeams, vcat, sources_vec; init=AbstractBeam[])
    return System(atoms; beams=all_beams, kwargs...)
end

"""
    System(atom::AbstractAtom; kwargs...)
    System(atom::AbstractAtom, beam_sources; kwargs...)

Convenience constructors for single-atom systems. Wraps `atom` 
in a vector and delegates to the main constructor.
"""
System(atom::AbstractAtom; kwargs...) = System([atom]; kwargs...)

System(atom::AbstractAtom, beam_sources; kwargs...) = 
    System([atom], beam_sources; kwargs...)

"""
    per_atom_indices(atom::AbstractAtom, s::AbstractLevel)

Return a list of `(index, coefficient)` pairs corresponding to placing
`atom` in the pure level `s`.

This looks up the basis index of `s` in `atom.level_indices` and returns
a single entry `[(idx, 1.0)]`. It is used internally by [`getqstate`](@ref)
when building product states across multiple atoms.
"""
function per_atom_indices(atom::AbstractAtom, s::AbstractLevel)
    idx = get(atom.level_indices, s, nothing)
    idx === nothing && error("Level $s not found.")
    [(idx, 1.0)]
end

"""
    per_atom_indices(atom::AbstractAtom, s::Superposition)

Return a list of `(index, coefficient)` pairs corresponding to the
superposition `s` for a single `atom`.

For each level–amplitude pair in `s.coeffs`, the corresponding basis
index is extracted from `atom.level_indices`. The result is a list that
encodes the local state of the atom and is used by [`getqstate`](@ref)
to assemble many-body product states.
"""
function per_atom_indices(atom::AbstractAtom, s::Superposition)
    out = []
    for (lvl, coeff) in s.coeffs
        idx = get(atom.level_indices, lvl, nothing)
        idx === nothing && error("Level $lvl not found.")
        push!(out, (idx, coeff))
    end
    out
end

"""
    getqstate(system::System, state::AbstractVector; density_matrix = false)

Construct a many-body quantum state for `system` from a per-atom
specification `state`.

Each entry of `state` is either an `AbstractLevel` or a `Superposition`
for the corresponding atom; a single entry is taken for every atom. The
amplitude of every basis element is the product of the per-atom amplitudes
(one pass over the basis, so an all-atoms superposition costs the size of the
state, not its square), the statevector is normalized, and the result is either:

- the normalized statevector (default, `density_matrix = false`), or
- the corresponding pure-state density matrix (`density_matrix = true`).

A numeric `state` is taken as given: a vector of length `basis.dim` is a
statevector (normalized), a `dim × dim` matrix a density matrix (needs
`density_matrix = true`).

This is typically used to prepare initial states for simulations and
analysis routines such as tomography.
"""
function getqstate(system::System, state::AbstractVector; density_matrix=false, kwargs...)
    atoms = system.atoms
    N = length(atoms)
    b = system.basis
    length(state) == 1 && N > 1 && (state = fill(state[1], N))   # one level for every atom
    length(state) == N || throw(ArgumentError(
        "initial_state has $(length(state)) entries for $N atoms"))
    # Each atom's amplitude on each of its levels; a basis element's amplitude is
    # their product. One pass over the basis, O(dim · N), no temporaries.
    amps = [_level_amplitudes(atoms[i], state[i]) for i in 1:N]
    psi = _product_amplitudes(b, amps)
    n = norm(psi)
    n == 0 && throw(ArgumentError(
        "initial_state has no weight in the basis (its product states were removed by `maxoccupations`?)"))
    psi ./= n
    return density_matrix ? psi * psi' : psi
end

# A function barrier: `sys.basis` is `Union{Nothing, Basis}`, and the loop must
# know the element tuples' type to run without allocating.
function _product_amplitudes(b::Basis{N}, amps::Vector{Vector{ComplexF64}}) where {N}
    psi = zeros(ComplexF64, b.dim)
    @inbounds for (k, el) in enumerate(b.elements)
        c = one(ComplexF64)
        for i in 1:N
            c *= amps[i][el[i]]
            c == 0 && break
        end
        psi[k] = c
    end
    return psi
end

function _level_amplitudes(atom, s)
    v = zeros(ComplexF64, length(atom.levels))
    for (i, c) in per_atom_indices(atom, s)
        v[i] += c
    end
    return v
end
getqstate(system::System, state::S; kwargs...) where {S<:AbstractLevel} = getqstate(system, [state]; kwargs...)

function getqstate(system::System, ψ::AbstractVector{<:Number}; density_matrix=false, kwargs...)
    length(ψ) == system.basis.dim || throw(ArgumentError(
        "initial_state: a statevector of length $(length(ψ)) for a basis of dimension $(system.basis.dim)"))
    psi = ComplexF64.(ψ) ./ norm(ψ)
    return density_matrix ? psi * psi' : psi
end

function getqstate(system::System, ρ::AbstractMatrix{<:Number}; density_matrix=false, kwargs...)
    d = system.basis.dim
    size(ρ) == (d, d) || throw(ArgumentError(
        "initial_state: a density matrix of size $(size(ρ)) for a basis of dimension $d"))
    density_matrix || throw(ArgumentError(
        "initial_state is a density matrix; pass `density_matrix = true`"))
    return ComplexF64.(ρ)
end

"""
    getqstate(sys::System)

Return the current quantum state stored in `sys`.

The state is populated by `play`. If the system has not been
simulated yet (i.e. `sys.state[]` is `nothing`), an error is thrown to
signal that no state is available.
"""
function getqstate(sys::System)
    if isnothing(sys.state[])
        error("System state not initialized. Call play to compute state.")
    end
    return sys.state[]  # Return dereferenced value
end

"""
    getmatrix(field::<:AbstractField)

Return the operator matrix for a field
"""
function getmatrix(field::F) where F<:Dynamiq.AbstractField
    if hasproperty(field, :H)
        return Matrix(field.H)
    else
        return nothing
    end
end

getmatrix(::Any) = nothing


"""
    gethamiltonian(sys::System)

Return the system Hamiltonian (rad/s). Couplings enter at full amplitude, and the
light shifts of the trapping beams (those passed to `System`, scalar and tensor)
at the beams' peak intensity, as for [`add_light_shift!`](@ref).
"""
function gethamiltonian(sys::System)
    m = zeros(ComplexF64, sys.basis.dim, sys.basis.dim)
    # `compile` adds the trapping beams' light shifts itself, so they have no node.
    # Build them as it does. That refreshes the atoms' polarizabilities in place for
    # the current quantization axis, as `compile` would; positions are left alone.
    # An explicit `add_light_shift!` node caches α from the axis in force when it
    # was added, so it is refreshed the same way before it is read.
    explicit = LightShiftNode[n for n in sys.nodes if n isa LightShiftNode]
    trapping = AbstractBeam[resolve(b, Dict{Symbol,Any}()) for b in sys.beams]
    if !isempty(trapping) || !isempty(explicit)
        beams = copy(trapping)
        for n in explicit
            b = _lightshift_beam(n)
            b === nothing || any(x -> x === b, beams) || push!(beams, b)
        end
        tensor_pairs = _tensor_may_vary(sys) ? :all : :fit
        for a in sys.atoms
            _init_species_data!(a, a.inner, beams; q_axis = getquantizationaxis(sys), tensor_pairs)
        end
        for n in explicit
            n._field === nothing || recompile_node!(n, n._field, nothing, Dict{Symbol,Any}())
        end
    end
    # A beam-derived coupling follows the system's axis; built before the axis was
    # set, its default operator still has ẑ. Refresh it at the parameter defaults.
    _bind_quantization_axis!(sys)
    for n in sys.nodes
        f = node_output(n)
        f === nothing && continue
        if n isa GaussianCouplingNode
            Ωref = n.Ω0_override === nothing ? nothing : _resolve_node_default(n.Ω0_override)
            x_ref = hasproperty(n.beam, :r0) ? Vector{Float64}(n.beam.r0) : n.atom.x
            update!(f, Val(:_), _gaussian_Ω0(n, x_ref, Ωref))
        elseif n isa Union{CouplingNode, PlanarCouplingNode} && n.Ω isa BeamRabiFrequency
            update!(f, Val(:_), ComplexF64(_resolve_node_default(n.Ω)))
        end
    end
    for node in sys.nodes
        obj = node_output(node)
        obj === nothing && continue
        M = getmatrix(obj)
        M === nothing && continue
        m .+= M
    end
    for f in _auto_light_shifts(sys, [a.inner for a in sys.atoms], trapping)
        m .+= f._coeff[] .* getmatrix(f)
    end
    return m
end

"""
    getbasis(sys:System)
"""
function getbasis(sys::System)
    [[sys.basis.atoms[i].levels[j].label for (i,j) in enumerate(b)] for b in sys.basis.elements]
end


"""
    Base.push!(sys::System, node::AbstractNode)

Append a DAG node to `sys.nodes`.

Typically called by `add_coupling!`, `add_detuning!`, `add_decay!`, etc.
after constructing and building the node.
"""
Base.push!(sys::System, node::AbstractNode) = push!(sys.nodes, node)

"""
    Base.getindex(sys::System, idx::Int)

Return the `idx`-th DAG node in `sys.nodes`.
"""
Base.getindex(sys::System, idx::Int) = sys.nodes[idx]

"""
    is_quantum(sys::System) -> Bool

Return `true` if `sys` contains any quantum field objects (instances of
`Dynamiq.AbstractField`) in its node outputs.

This is used to distinguish purely classical configurations from
fully quantum or semiclassical systems.
"""
is_quantum(sys::System) = any(n -> node_output(n) isa Dynamiq.AbstractField, sys.nodes)

"""
    is_classical(sys::System) -> Bool

Return `true` if `sys` has no initial quantum state (`initial_state` empty), so
that `play` without one simulates the atomic motion only.
"""
is_classical(sys::System) = isempty(sys.initial_state)

"""
    has_beams(sys::System) -> Bool

Return `true` if `sys` has any associated `AbstractBeam` objects.
"""
has_beams(sys::System) = !isempty(sys.beams)

"""
    is_semiclassical(sys::System) -> Bool

Return `true` if `sys` combines quantum fields with optical tweezers.

A semiclassical system is one where `is_quantum(sys)` is true and
`beams` is non-empty, corresponding to quantum internal dynamics
coupled to classical external potentials.
"""
is_semiclassical(sys::System) = is_quantum(sys) && has_beams(sys)

"""
    Base.copy(sys::System) -> System

An independent copy of `sys`: its atoms, beams, nodes (with their compiled
fields), basis, state and detector specifications are all copied, so the two
systems can be built on and played separately. (The former shallow copy shared the
nodes and always threw.)
"""
Base.copy(sys::System) = deepcopy(sys)

Base.show(io::IO, sys::System) = print(io, "AtomTwin.System($(length(sys.atoms)) atom(s))")

function Base.show(io::IO, ::MIME"text/plain", sys::System)
    n_atoms   = length(sys.atoms)
    n_twz     = length(sys.beams)
    n_det     = length(sys.detector_specs)
    has_basis = sys.basis !== nothing
    has_state = sys.state[] !== nothing

    # Atom summaries via your helper
    atom_summaries = String[atom_summary(a) for a in sys.atoms]

    n_nodes   = length(sys.nodes)

    # Beams summary
    twz_types = unique(nameof.(typeof.(sys.beams)))

    # Detector summary: type + name if field exists
    det_summaries = String[]
    for d in sys.detector_specs
        ty = string(d.kind)
        name_str = ""
        if hasfield(typeof(d), :kwargs)
            kw = getfield(d, :kwargs)
            if haskey(kw, :name)
                name_str = string(" \"", kw[:name], "\"")
            end
        end
        push!(det_summaries, ty * name_str)
    end

    basis_str = has_basis ? "dim = $(sys.basis.dim)" : "none"

    state_str = has_state ? "initialized" : "empty"
    if has_state && sys.basis !== nothing
        len = length(sys.state[])
        if len == sys.basis.dim
            state_str *= " (statevector)"
        elseif len == sys.basis.dim^2
            state_str *= " (density matrix)"
        end
    end

    println(io, "AtomTwin.System")
    println(io, "├─ Atoms ($(n_atoms))",
             n_atoms > 0 ? ": " * join(atom_summaries, ", ") : "")
    println(io, "├─ Beams ($(n_twz))",
             n_twz > 0 ? ": " * join(twz_types, ", ") : "")
    println(io, "├─ Nodes ($(n_nodes))")
    println(io, "├─ Detectors ($(n_det))",
             n_det > 0 ? ": " * join(det_summaries, ", ") : "")
    println(io, "├─ Basis: ", basis_str)
    println(io, "└─ State: ", state_str)
end


