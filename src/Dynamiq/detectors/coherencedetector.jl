"""
    CoherenceDetector{A,S} <: AbstractDetector{A}

Detector that measures the coherence between two specific quantum levels
of an atom over time.

Detectors trigger at the end of each time step, so `tspan[i]`  corresponds
to the state after evolving to time `tspan[i]`.

# Fields

- `qstate::S`: Reference to the quantum system (Vector{ComplexF64} or Matrix{ComplexF64}).
- `atom::A`: Atom being observed.
- `levels::Pair{Int,Int}`: Pair of level indices \\((i, j)\\) that define the coherence.
- `vals::Vector{ComplexF64}`: Recorded coherence values as a function of time.
- `tspan::Vector{Float64}`: Time vector.
- `O::Op`: Operator that projects onto the \\(|i\\rangle\\langle j|\\) coherence.
- `name::String`: Optional detector name.

# Constructor

- `CoherenceDetector(qstate, basis, atom, levels, tspan; name = \"\")`
"""
struct CoherenceDetector{S,V,T} <: AbstractDetector
    qstate::S
    atom::NLevelAtom
    levels::Pair{Int,Int}
    vals::V
    tspan::T
    O::Op
    name::String

    function CoherenceDetector(qstate::Array{ComplexF64},
                               basis::Basis,
                               atom::AbstractAtom,
                               levels::Pair{Int,Int},
                               tspan::Vector{Float64};
                               name::AbstractString = "")
        O = Op(basis, atom, levels[1] => levels[2], 1.0)
        new{typeof(qstate), Vector{ComplexF64}, Vector{Float64}}(qstate, atom, levels,
                                         zeros(length(tspan)),
                                         tspan, O, name)
    end
        # Constructor with preallocated views)
    function CoherenceDetector(qstate::Array{ComplexF64},
                                basis::Basis,
                                atom::NLevelAtom,
                                levels::Pair{Int,Int},
                                tspan::AbstractVector{Float64},
                                vals::AbstractVector{ComplexF64};
                                name::AbstractString = "")
        @assert length(tspan) == length(vals) "tspan and vals must have same length"
        O = Op(basis, atom, levels[1] => levels[2], 1.0)
        new{typeof(qstate), typeof(vals), typeof(tspan)}(
            qstate, atom, levels,
            vals,    # Use provided view
            tspan,   # Use provided view
            O, name)
    end
end

"""
    CoherenceDetectorSpec(atom; levels = 1=>2, name = \"\", tspan = nothing) -> DetectorSpec

Create a detector specification for a coherence detector.

Only the atom is required here; the quantum state and basis will be
retrieved from the system at build time via `build_detectors`.

# Arguments

- `atom::AbstractAtom`: Atom to observe (will be resolved when building).

# Keywords

- `levels::Pair{Int,Int}`: Pair of levels whose coherence is monitored.
- `name::AbstractString`: Optional detector name.
- `tspan::Union{Nothing,Vector{Float64}}`: Optional time vector. If `nothing`,
  the time grid is taken from the simulation.
"""
CoherenceDetectorSpec(atom::AbstractAtom;
                      levels::Pair{Int,Int} = 1 => 2,
                      name::AbstractString   = "",
                      tspan::Union{Nothing,Vector{Float64}} = nothing) =
    DetectorSpec{typeof(CoherenceDetector)}(
        CoherenceDetector,
        atom,                       # atom goes in obj (will be resolved)
        (levels = levels, name = name),  # detector-specific parameters
        tspan,
        ComplexF64,
        1
    )

"""
    write!(d::CoherenceDetector, i)

Record the current coherence value at time step `i`.

For a pure state \\(\\vert\\psi\\rangle\\), the coherence \\(\\rho_{ij}\\)
is computed as \\(\\psi_j \\psi_i^{\\ast}\\) (= ρ_{ji}, the documented convention) summed over all basis elements
corresponding to \\(|i\\rangle\\langle j|\\). For a density matrix
\\(\\rho\\), the corresponding matrix elements \\(\\rho_{ij}\\) are summed.
"""
function write!(d::CoherenceDetector{Vector{ComplexF64},V,T}, i::Int) where {V,T}
    stateq = d.qstate
    # ⟨i|ρ|j⟩ = ψ_i ψ*_j for pure state
    d.vals[i] = sum(stateq[i_] * conj(stateq[j_]) for (i_, j_, _) in d.O.forward)
end

function write!(d::CoherenceDetector{Matrix{ComplexF64},V,T}, i::Int) where {V,T}
    stateq = d.qstate
    # Direct ρ_{ij} from density matrix
    d.vals[i] = sum(stateq[i_, j_] for (i_, j_, _) in d.O.forward)
end
