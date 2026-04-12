"""
    PlanarBeam <: AbstractBeam

Plane-wave beam with fixed intensity, propagation direction, and polarization.

Fields:
- `λ::Float64`: Wavelength in meters.
- `I::Float64`: Intensity in \\(\\mathrm{W/m^2}\\).
- `E_field::Float64`: Electric-field amplitude in V/m (`sqrt(2I/(c*ε₀))`).
- `k::NTuple{3,Float64}`: Wavevector components in \\(\\mathrm{m^{-1}}\\).
- `unit_k::NTuple{3,Float64}`: Normalized propagation direction.
- `pol::NTuple{3,ComplexF64}`: Normalized Cartesian polarization vector (⟂ k).
- `_coeff::Base.RefValue{ComplexF64}`: Complex amplitude envelope used by
  time-dependent modifiers.

The constructor accepts a Cartesian `direction` and Jones `polarization` vector.
The polarization is normalized and its component along `k` is projected out,
matching the `GeneralGaussianBeam` convention.
"""
struct PlanarBeam <: AbstractBeam
    λ::Float64                     # Wavelength [m]
    I::Float64                     # Intensity [W/m^2]
    E_field::Float64               # |E| amplitude (scaled by 1/ħ)
    k::NTuple{3,Float64}           # wavevector [1/m]
    unit_k::NTuple{3,Float64}      # normalized propagation direction
    pol::NTuple{3,ComplexF64}      # normalized Cartesian polarization
    _coeff::Base.RefValue{ComplexF64}

    function PlanarBeam(λ, I, direction, polarization)
        unit_k = normalize(collect(Float64, direction))

        p = ComplexF64.(polarization)
        p ./= sqrt(sum(abs2, p))
        # project out component along k
        pk = sum(p[i] * unit_k[i] for i in 1:3)
        p .-= pk .* ComplexF64.(unit_k)
        p_norm = sqrt(sum(abs2, p))
        p_norm == 0 && error("PlanarBeam: polarization must not be parallel to k")
        p ./= p_norm

        e_field = sqrt(2 * I / (c * ε0))
        k_vec   = (2π / λ) .* unit_k

        new(λ, I, e_field, Tuple(k_vec), Tuple(unit_k), Tuple(p),
            Ref(ComplexF64(1.0)))
    end
end

"""
    Efield(b::PlanarBeam, r::Vector{Float64}) -> Vector{ComplexF64}

Cartesian electric field of the plane wave at position `r`.

Returns `E_field * pol * exp(ik·r)` as a 3-component complex vector,
matching the `GeneralGaussianBeam` convention used by `Efield_spherical`.
`_coeff[]` is intentionally excluded here; it is the pulse-amplitude envelope
applied externally by `AmplitudeModifier` via `GaussianCoupling._amplitude`.
"""
@inline function Efield(b::PlanarBeam, r::Vector{Float64})
    phase = b.k[1]*r[1] + b.k[2]*r[2] + b.k[3]*r[3]
    amp   = b.E_field * cis(phase)
    return ComplexF64[amp * b.pol[1],
                      amp * b.pol[2],
                      amp * b.pol[3]]
end

"""
    efield_scalar(b::PlanarBeam, r::Vector{Float64}) -> ComplexF64

Scalar field amplitude of the plane wave at position `r`, including the
propagation phase `exp(ik·r)` but excluding `_coeff[]`.
Used by `GaussianCoupling` to compute the motional phase ratio
`efield_scalar(r) / E0`; the pulse-amplitude envelope is handled separately.
"""
@inline function efield_scalar(b::PlanarBeam, r::Vector{Float64})
    phase = b.k[1]*r[1] + b.k[2]*r[2] + b.k[3]*r[3]
    return b.E_field * cis(phase)
end

"""
    getbeams(pb::PlanarBeam) -> Vector{AbstractBeam}

Return a vector containing the beam pb
"""
getbeams(pb::PlanarBeam) = AbstractBeam[pb]
