#####################################################################################
# Atoms
#####################################################################################

"""
    Atom(x, v, m, λs, γs, ls)

Simple atomic model with position, velocity, and scalar polarizability.

Fields:
- `x::Vector{Float64}`: atomic position in real space.
- `v::Vector{Float64}`: atomic velocity.
- `m::Float64`: atomic mass.
- `λ::NTuple{N,Float64}`: wavelengths of relevant transitions (in meters).
- `γ::NTuple{N,Float64}`: spontaneous emission rates (in Hz).
- `ls::NTuple{N,Int}`: line-strength factors for each transition.
- `alpha::F`: polarizability as a function of probe wavelength.

The constructor builds a dispersion-like polarizability

\\[
\\alpha(\\omega) \\propto \\sum_i \\frac{\\text{ls}_i \\gamma_i}{\\omega - \\omega_i}
\\]

from the transition data, where `ω(λ) = 2πc/λ`.
"""
struct Atom{N,F}
    x::Vector{Float64}
    v::Vector{Float64}
    m::Float64
    λ::NTuple{N,Float64}  # wavelength of relevant transitions in m
    γ::NTuple{N,Float64}  # spontaneous emission rates Hz
    ls::NTuple{N,Int}     # linestrength factors
    alpha::F              # polarizability as a function of wavelength

    function Atom(x::Vector{Float64}, v::Vector{Float64}, m::Float64, λs, γs, ls)
        N = length(λs)

        w(x) = 2 * pi * c / x
        d(λ, x) = w(x) - w(λ)

        alpha(x) = pi / 2 * c^2 / w(x)^3 * sum(ls .* γs ./ d.(λs, x))

        new{N,typeof(alpha)}(x, v, m, Tuple(λs), Tuple(γs), Tuple(ls), alpha)
    end
end

#####################################################################################
# Lasers
#####################################################################################

# (`BasisPolarization`, a lab-to-q-basis Jones conversion that lost the sign of
# k_y and had no caller, was removed; `Efield_spherical` in physics/couplings.jl
# does this against the quantization axis.)
function _removed_basis_polarization()
    return nothing
end
