"""
    ExpectationDetector{S,V,T} <: AbstractDetector

Detector that records the expectation value ⟨O⟩ of an arbitrary operator `O`
over time.

Detectors trigger at the end of each time step, so `tspan[i]` corresponds to
the state after evolving to time `tspan[i]`.

For a pure state \\(\\vert\\psi\\rangle\\) the recorded value is
\\(\\langle\\psi\\vert O\\vert\\psi\\rangle\\); for a density matrix \\(\\rho\\)
it is \\(\\mathrm{Tr}(O\\rho)\\). No assumption of Hermiticity is made, so the
stored value is complex in general; for a Hermitian `O` the imaginary part is
zero up to round-off.

# Fields

- `qstate::S`: Reference to the quantum state (`Vector{ComplexF64}` or
  `Matrix{ComplexF64}`).
- `O::Op`: Operator whose expectation value is monitored.
- `vals::V`: Recorded expectation values as a function of time.
- `tspan::T`: Time vector.
- `name::String`: Optional detector name.

# Constructor

- `ExpectationDetector(qstate, O, tspan; name = "")`
"""
struct ExpectationDetector{S,V,T} <: AbstractDetector
    qstate::S
    O::Op
    vals::V
    tspan::T
    name::String

    function ExpectationDetector(qstate::Array{ComplexF64},
                                 O::Op,
                                 tspan::Vector{Float64};
                                 name::AbstractString = "")
        new{typeof(qstate),Vector{ComplexF64},Vector{Float64}}(
            qstate, O, zeros(ComplexF64, length(tspan)), tspan, name)
    end

    # Constructor with preallocated views
    function ExpectationDetector(qstate::Array{ComplexF64},
                                 O::Op,
                                 tspan::AbstractVector{Float64},
                                 vals::AbstractVector{ComplexF64};
                                 name::AbstractString = "")
        @assert length(tspan) == length(vals) "tspan and vals must have same length"
        new{typeof(qstate),typeof(vals),typeof(tspan)}(
            qstate, O, vals, tspan, name)
    end
end

"""
    ExpectationDetectorSpec(op::Op; name = "", tspan = nothing) -> DetectorSpec

Create a detector specification for an expectation-value detector monitoring the
operator `op` (an `Op` in the system basis, e.g. built with [`levelop`](@ref)).

Only the operator is required here; the quantum state is retrieved from the
system at build time via `build_detectors`. Unlike the population/coherence
specs, the primary observable is the operator itself, so `op` is carried in the
spec's `obj` slot and is *not* resolved through `resolve_target` (operators are
basis-global and unaffected by per-shot atom copies).

# Arguments

- `op::Op`: Operator to observe, dimensioned to the system basis.

# Keywords

- `name::AbstractString`: Optional detector name.
- `tspan::Union{Nothing,Vector{Float64}}`: Optional time vector. If `nothing`,
  the time grid is taken from the simulation.
"""
ExpectationDetectorSpec(op::Op;
                        name::AbstractString = "",
                        tspan::Union{Nothing,Vector{Float64}} = nothing) =
    DetectorSpec{typeof(ExpectationDetector)}(
        ExpectationDetector,
        op,                          # operator is the primary observable (obj slot)
        (name = name,),
        tspan,
        ComplexF64,
        1
    )

"""
    write!(d::ExpectationDetector, i)

Record the current expectation value at time step `i`.

For a pure state \\(\\vert\\psi\\rangle\\), \\(\\langle O\\rangle =
\\langle\\psi\\vert O\\vert\\psi\\rangle\\) is accumulated directly from the
sparse `O` triplets. For a density matrix \\(\\rho\\), \\(\\langle O\\rangle =
\\mathrm{Tr}(O\\rho) = \\sum_{ij} O_{ij}\\,\\rho_{ji}\\).
"""
function write!(d::ExpectationDetector{Vector{ComplexF64},V,T}, i::Int) where {V,T}
    ψ = d.qstate
    a = 0.0 + 0.0im
    @inbounds for (r, c, v) in d.O.forward
        a += conj(ψ[r]) * v * ψ[c]
    end
    @inbounds for (r, c, u) in d.O.reverse
        a += conj(ψ[r]) * u * ψ[c]
    end
    d.vals[i] = a
end

function write!(d::ExpectationDetector{Matrix{ComplexF64},V,T}, i::Int) where {V,T}
    ρ = d.qstate
    a = 0.0 + 0.0im
    @inbounds for (r, c, v) in d.O.forward
        a += v * ρ[c, r]
    end
    @inbounds for (r, c, u) in d.O.reverse
        a += u * ρ[c, r]
    end
    d.vals[i] = a
end

"""
    reset!(d::ExpectationDetector)

Reset all recorded samples to zero while preserving the length and `tspan`.
"""
function reset!(d::ExpectationDetector)
    fill!(d.vals, 0.0im)
end
