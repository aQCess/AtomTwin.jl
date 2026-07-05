"""
    PhotoDetector{V,T} <: AbstractDetector

Photon-counting detector that registers integer click counts per time step.

Used together with the stochastic quantum-jump (wavefunction Monte Carlo)
solver: each time the associated jump fires, `write!(d, i)` increments the count
in the current time bin. Over many shots the per-bin counts recover the emission
rate, and their inter-click structure gives antibunching / g²(τ).

Unlike the other detectors, the recorded value is an integer *count of jump
events* in each bin, not a state expectation value. Clicks are written at the
raw solver step at which the jump fires (they are not downsampled), then binned
onto the detector's `tspan`.

# Fields

- `vals::V`: Per-timestep integer count of detected events.
- `tspan::T`: Time vector for the bins in `vals`.
- `name::String`: Optional detector name.
- `jump::Union{Nothing,Jump}`: The jump this detector counts. Set at build time
  from the `add_decay!(...; clicks = spec)` binding; the solver increments the
  detector only when *this* jump fires. `nothing` until bound.

# Constructors

- `PhotoDetector(tspan; name = "")`
- `PhotoDetector(tspan, vals; name = "")`   (preallocated view)
"""
mutable struct PhotoDetector{V,T} <: AbstractDetector
    vals::V
    tspan::T
    name::String
    jump::Union{Nothing,Jump}

    function PhotoDetector(tspan::Vector{Float64}; name::AbstractString = "")
        new{Vector{Int},Vector{Float64}}(zeros(Int, length(tspan)), tspan, name, nothing)
    end

    function PhotoDetector(tspan::AbstractVector{Float64},
                           vals::AbstractVector{<:Integer};
                           name::AbstractString = "")
        @assert length(tspan) == length(vals) "tspan and vals must have same length"
        new{typeof(vals),typeof(tspan)}(vals, tspan, name, nothing)
    end
end

"""
    PhotoDetectorSpec(; name = "", tspan = nothing) -> DetectorSpec

Create a detector specification for a photon-counting detector. Attach it to a
system with `add_detector!`, then bind it to a decay channel with `add_decay!`'s
`clicks = spec` keyword so its jump events are counted:

```julia
pd = PhotoDetectorSpec(name = "clicks")
add_detector!(sys, pd)
add_decay!(sys, atom, e => g, Γ; clicks = pd)
```

The output has the same shape as any other detector: a `Vector{Int}` of per-bin
counts for `shots = 1`, or an `[n_times × shots]` `Matrix{Int}` for `shots > 1`.
Photo detectors only register events under the statevector (wavefunction
Monte Carlo) solver — i.e. `play(...; density_matrix = false)`.

# Keywords

- `name::AbstractString`: Optional detector name.
- `tspan::Union{Nothing,Vector{Float64}}`: Optional time vector. If `nothing`,
  the time grid is taken from the simulation.
"""
PhotoDetectorSpec(; name::AbstractString = "",
                    tspan::Union{Nothing,Vector{Float64}} = nothing) =
    DetectorSpec{typeof(PhotoDetector)}(
        PhotoDetector,
        nothing,              # no resolvable target object; bound to a jump via add_decay!
        (name = name,),
        tspan,
        Int,
        1
    )

"""
    write!(d::PhotoDetector, i)

Register a single detection event in time bin `i` by incrementing `d.vals[i]`.
"""
function write!(d::PhotoDetector, i::Int)
    @inbounds d.vals[i] += 1
end

"""
    reset!(d::PhotoDetector)

Reset all recorded counts to zero while preserving the length and `tspan`.
"""
function reset!(d::PhotoDetector)
    fill!(d.vals, zero(eltype(d.vals)))
end
