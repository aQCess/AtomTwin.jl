"""
    PopulationDetector{S,V,T} <: AbstractDetector{A,S}

Detector that measures the population of a specific quantum level over time.

Detectors trigger at the end of each time step, so `tspan[i]`  corresponds
to the state after evolving to time `tspan[i]`.

# Fields

- `qstate::S`: Quantum state representation (Vector{ComplexF64} or Matrix{ComplexF64}).
- `atom::NLevelAtom`: Atom being observed.
- `level::Int`: Level index \\(i\\) to monitor.
- `vals::V`: Recorded population values \\(P_i(t)\\).
- `tspan::T`: Time vector, Vector{Float} or SubArray.
- `O::Op`: Projection operator onto the chosen level.
- `name::String`: Optional detector name.

# Constructor

- `PopulationDetector(qstate, basis, atom, level, tspan; name = \"\")`
"""
mutable struct PopulationDetector{S,V,T} <: AbstractDetector
    qstate::S
    atom::NLevelAtom
    level::Int
    vals::V  # can be Vector{Float64} or SubArray
    tspan::T # can be Vector{Float64} or SubArray
    O::Op
    name::String

    # Original constructor (allocates new arrays)
    function PopulationDetector(qstate::Array{ComplexF64},
                                basis::Basis,
                                atom::NLevelAtom,
                                level::Int,
                                tspan::Vector{Float64};
                                name::AbstractString = "")
        O = Op(basis, atom, level => level, 1.0)
        new{typeof(qstate), Vector{Float64}, Vector{Float64}}(
            qstate, atom, level,
            zeros(length(tspan)),
            tspan, O, name)
    end

    # Constructor with preallocated views
    function PopulationDetector(qstate::Array{ComplexF64},
                                basis::Basis,
                                atom::NLevelAtom,
                                level::Int,
                                tspan::AbstractVector{Float64},
                                vals::AbstractVector{Float64};
                                name::AbstractString = "")
        @assert length(tspan) == length(vals) "tspan and vals must have same length"
        O = Op(basis, atom, level => level, 1.0)
        new{typeof(qstate), typeof(vals), typeof(tspan)}(
            qstate, atom, level,
            vals,    # Use provided view
            tspan,   # Use provided view
            O, name)
    end
end

"""
    PopulationDetectorSpec(atom; level = 1, name = \"\", tspan = nothing) -> DetectorSpec

Create a detector specification for a population detector.

Only the atom is required here; the quantum state and basis are retrieved
from the system at build time via `build_detectors`.

# Arguments

- `atom::AbstractAtom`: Atom to observe (will be resolved when building).

# Keywords

- `level::Int = 1`: Level index to monitor.
- `name::AbstractString = \"\"`: Optional detector name.
- `tspan::Union{Nothing,Vector{Float64}} = nothing`: Optional time vector.
  If `nothing`, the time grid is supplied by the simulation.
"""
PopulationDetectorSpec(atom::AbstractAtom;
                       level::Int                         = 1,
                       name::AbstractString               = "",
                       tspan::Union{Nothing,Vector{Float64}} = nothing) =
    DetectorSpec{typeof(PopulationDetector)}(
        PopulationDetector,
        atom,                                # atom goes in obj (will be resolved)
        (level = level, name = name),
        tspan,
        Float64,
        1
    )

"""
    prep!(d::PopulationDetector)

Update the atom's internal population `d.atom._P[d.level]` by computing the
current population of the monitored level from the system's quantum state.

For a pure state \\(\\vert\\psi\\rangle\\), the population is

\\[
P_i = \\sum_{k \\in \\mathcal{S}_i} |\\psi_k|^2,
\\]

where \\(\\mathcal{S}_i\\) is the support of the projector for level `i`.
For a density matrix \\(\\rho\\), the population is

\\[
P_i = \\sum_{k \\in \\mathcal{S}_i} \\rho_{kk}.
\\]
"""
function prep!(d::PopulationDetector{Vector{ComplexF64},V,T}) where {V,T}
    stateq = d.qstate

    d.atom._P[d.level] =
        sum(abs2(stateq[i]) for (i, _, _) in d.O.forward)
end

function prep!(d::PopulationDetector{Matrix{ComplexF64}, V, T}) where {V,T}
    stateq = d.qstate

    # Population from density matrix: sum Re(ρ_{ii}) over projector support.
    # Diagonal elements of a physical density matrix are real and non-negative;
    # taking real() avoids spurious contributions from numerical imaginary parts.
    d.atom._P[d.level] =
        sum(real(stateq[i, i]) for (i, _, _) in d.O.forward)
end

"""
    write!(d::PopulationDetector, i)

Record the current population value at time step `i`.

Assumes `prep!(d)` has been called beforehand so that `d.atom._P[d.level]`
reflects the latest quantum state.
"""

function write!(d::PopulationDetector{S,V,T}, i::Int) where {S,V,T}
    d.vals[i] = d.atom._P[d.level]
end