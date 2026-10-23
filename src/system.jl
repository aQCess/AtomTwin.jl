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
    getqstate(system::System, state::Vector; density_matrix = false)

Construct a many-body quantum state for `system` from a per-atom
specification `state`.

Each entry of `state` is either an `AbstractLevel` or a `Superposition`
for the corresponding atom. The function expands these into all product
basis components using `productstate(system.basis, ...)`, normalizes
the resulting statevector, and returns either:

- the normalized statevector (default, `density_matrix = false`), or
- the corresponding pure-state density matrix (`density_matrix = true`).

This is typically used to prepare initial states for simulations and
analysis routines such as tomography.
"""
function getqstate(system::System, state::Vector; density_matrix=false, kwargs...)
    state === nothing && return nothing
    atoms = system.atoms
    N = length(atoms)
    # For each atom, collect all possible (idx, coeff) pairs
    idx_coeff_lists = [
        per_atom_indices(atoms[i], state[i])
        for i in 1:N
    ]
    # Enumerate all tuples
    psi = zeros(ComplexF64, system.basis.dim)
    for idx_coeff_tuple in Iterators.product(idx_coeff_lists...)
        idx_tuple = ntuple(i -> idx_coeff_tuple[i][1], N)
        coeff = prod(idx_coeff_tuple[i][2] for i in 1:N)
        psi += coeff * productstate(system.basis, idx_tuple...; rho=false)
    end

    psi /= sqrt(psi'*psi)
    if density_matrix
        return psi * psi'
    else
        return psi
    end
end
getqstate(system::System, state::S; kwargs...) where {S<:AbstractLevel} = getqstate(system, [state]; kwargs...)

"""
    getqstate(sys::System)

Return the current quantum state stored in `sys`.

The state is populated by `play` / `play!`. If the system has not been
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
    for node in sys.nodes
        obj = node_output(node)
        obj === nothing && continue
        M = getmatrix(obj)
        M === nothing && continue
        m .+= M
    end
    # `compile` adds the trapping beams' light shifts itself, so they have no node.
    # Build them as it does. That refreshes the atoms' polarizabilities in place for
    # the current quantization axis, as `compile` would; positions are left alone.
    if !isempty(sys.beams)
        beams = AbstractBeam[resolve(b, Dict{Symbol,Any}()) for b in sys.beams]
        tensor_pairs = _tensor_may_vary(sys) ? :all : :fit
        for a in sys.atoms
            _init_species_data!(a, a.inner, beams; q_axis = getquantizationaxis(sys), tensor_pairs)
        end
        for f in _auto_light_shifts(sys, [a.inner for a in sys.atoms], beams)
            m .+= f._coeff[] .* getmatrix(f)
        end
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

Return `true` if `sys` has no explicit initial quantum state.

A classical system is characterized here by `initial_state === nothing`,
indicating that only classical degrees of freedom are relevant.
"""
is_classical(sys::System) = sys.initial_state === nothing

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

Create a shallow copy of `sys` with copied atoms, beams, basis,
nodes, and detector specifications.

Atoms and beams are duplicated via `copy` per element; `basis`,
`nodes`, and `detector_specs` are copied container-wise; and the
state reference is copied if present. Nodes are shared (they hold
compiled field references), so this is safe for concurrent reads
but not for concurrent writes to node state.
"""
function Base.copy(sys::System)
    atoms_copy = [copy(a) for a in sys.atoms]
    beams_copy = [copy(b) for b in sys.beams]
    state_copy = sys.state === nothing ? nothing : copy(sys.state)
    basis_copy = sys.basis === nothing ? nothing : copy(sys.basis)
    nodes_copy = copy(sys.nodes)  # shallow — nodes are shared (they hold compiled refs)
    detector_specs_copy = copy(sys.detector_specs)
    System(atoms_copy, beams_copy, sys.initial_state, state_copy, basis_copy, nodes_copy, detector_specs_copy)
end

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


