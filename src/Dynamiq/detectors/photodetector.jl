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
- `jumps::Vector{Jump}`: The jumps this detector counts. Filled at build time from
  the `add_decay!(...; clicks = spec)` bindings; the solver increments the detector
  whenever *any* of them fires. Empty until bound.

  A manifold decay expands into one jump per sublevel channel, and a detector bound
  to it must count them all -- a detector holding a single jump silently reports the
  photons of one channel and discards the rest.

# Constructors

- `PhotoDetector(tspan; name = "")`
- `PhotoDetector(tspan, vals; name = "")`   (preallocated view)
"""
mutable struct PhotoDetector{V,T} <: AbstractDetector
    vals::V
    tspan::T
    name::String
    jumps::Vector{Jump}
    axis::NTuple{3,Float64}    # aperture: photons with n̂·axis ≥ cosθ are collected
    cosθ::Float64              # −1: every direction
    efficiency::Float64        # probability that a collected photon is counted

    function PhotoDetector(tspan::Vector{Float64}; name::AbstractString = "",
                           axis = (0.0, 0.0, 1.0), cosθ = -1.0, efficiency = 1.0)
        new{Vector{Int},Vector{Float64}}(zeros(Int, length(tspan)), tspan, name, Jump[],
                                         Tuple(axis), cosθ, efficiency)
    end

    function PhotoDetector(tspan::AbstractVector{Float64},
                           vals::AbstractVector{<:Integer};
                           name::AbstractString = "",
                           axis = (0.0, 0.0, 1.0), cosθ = -1.0, efficiency = 1.0)
        @assert length(tspan) == length(vals) "tspan and vals must have same length"
        new{typeof(vals),typeof(tspan)}(vals, tspan, name, Jump[], Tuple(axis), cosθ, efficiency)
    end
end

has_aperture(d::PhotoDetector) = d.cosθ > -1

"""
    detects(d::PhotoDetector, n, rng) -> Bool

Whether `d` counts a photon emitted along the unit vector `n` (not read by a detector
without an aperture): inside its aperture, `n̂·axis ≥ cos θ`, and then with probability
`efficiency`.
"""
function detects(d::PhotoDetector, n, rng)
    has_aperture(d) && n[1] * d.axis[1] + n[2] * d.axis[2] + n[3] * d.axis[3] < d.cosθ && return false
    return d.efficiency >= 1 || rand(rng) < d.efficiency
end

"""
    PhotoDetectorSpec(; name = "", tspan = nothing, aperture = nothing, efficiency = 1.0) -> DetectorSpec

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
- `tspan::Union{Nothing,Vector{Float64}}`: Bin ends, in place of `out.times`: row
  `k` counts the clicks in `(tspan[k-1], tspan[k]]` (from `t = 0`), resolved to the
  output grid. Clicks after `tspan[end]` are dropped. `tspan = [t₁, t₂]` gives the
  counts of two exposure windows ending at `t₁` and `t₂`.
- `aperture`: `(axis = [0, 0, 1], NA = 0.7)` collects only photons emitted into the
  cone of half-angle `asin(NA)` about `axis` (from the atom towards the lens) — an
  objective. The direction is the photon's own, drawn from its jump's dipole pattern
  and shared with its recoil, so the polarization dependence of the collection follows
  from the decay's physics. `nothing` (default): every direction.
- `efficiency`: probability that a collected photon is counted (optics, detector).

```julia
cam = PhotoDetectorSpec(name = "camera", aperture = (axis = [0, 0, 1], NA = 0.7), efficiency = 0.7)
```
"""
function PhotoDetectorSpec(; name::AbstractString = "",
                             tspan::Union{Nothing,Vector{Float64}} = nothing,
                             aperture = nothing, efficiency::Real = 1.0)
    0 <= efficiency <= 1 || throw(ArgumentError("PhotoDetectorSpec: efficiency must lie in [0, 1]"))
    axis, cosθ = (0.0, 0.0, 1.0), -1.0
    if aperture !== nothing
        0 < aperture.NA <= 1 || throw(ArgumentError("PhotoDetectorSpec: aperture NA must lie in (0, 1]"))
        a = collect(Float64, aperture.axis)
        axis, cosθ = Tuple(a ./ sqrt(sum(abs2, a))), sqrt(1 - aperture.NA^2)
    end
    return DetectorSpec{typeof(PhotoDetector)}(
        PhotoDetector,
        nothing,              # no resolvable target object; bound to a jump via add_decay!
        (name = name, axis = axis, cosθ = cosθ, efficiency = Float64(efficiency)),
        tspan,
        Int,
        1
    )
end

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
