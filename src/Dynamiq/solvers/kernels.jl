"""
Features:

- Semiclassical simulation of atomic dynamics including motion.
- Accurate treatment of multilevel effects and light shifts.
- Efficient sparse representation of operators.
- Efficient simulation of closed and open system dynamics using high-order
  Trotter-like or Magnus-like decompositions.
- Support for both global and local Hamiltonian terms.
"""

using StatsBase   # sample, Weights
using Polyester   # @batch
# using ExponentialUtilities

# The two hot kernels -- sparse operator application and the density-matrix
# commutator -- and the radiation forces.

#------------------------------------------------------------------------------
# Operator application (apply!)
#------------------------------------------------------------------------------

"""
    apply!(w, terms, v, scale; hermitian_pairs = true)

Accumulate `scale * Σⱼ (cⱼ Hⱼ.forward + conj(cⱼ) Hⱼ.reverse) * v` into `w`.
`w` is added to, not overwritten.

With `hermitian_pairs = false` only the `forward` part is applied, for
non-Hermitian terms such as a Monte Carlo effective Hamiltonian.
"""
Base.@propagate_inbounds function apply!(w::Vector{ComplexF64},
                                        terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                                        v::Vector{ComplexF64},
                                        scale::ComplexF64;
                                        hermitian_pairs::Bool = true)
    @inbounds for (coeff_ref, op) in terms
        coeff = coeff_ref[]
        iszero(coeff) && continue           # switched-off coupling
        ac = scale * coeff
        _accum!(w, op.forward, v, ac)
        if hermitian_pairs
            _accum!(w, op.reverse, v, scale * conj(coeff))
        end
    end
    return w
end

# The COO walk, unrolled two at a time: both `v[j]` and `w[i]` are indirect, so
# two independent chains per iteration hide the dependent loads. 1.15x at
# d = 4096, neutral below. `@simd` does not help -- the scatter defeats it.
@inline function _accum!(w::Vector{ComplexF64}, coo::Vector{Tuple{Int,Int,ComplexF64}},
                         v::Vector{ComplexF64}, ac::ComplexF64)
    n = length(coo)
    k = 1
    @inbounds while k + 1 <= n
        (i1, j1, x1) = coo[k]
        (i2, j2, x2) = coo[k+1]
        @fastmath w[i1] += ac * x1 * v[j1]
        @fastmath w[i2] += ac * x2 * v[j2]
        k += 2
    end
    @inbounds while k <= n
        (i, j, x) = coo[k]
        @fastmath w[i] += ac * x * v[j]
        k += 1
    end
    return w
end

"""
    apply_commutator!(W, terms, R, scale)

Accumulate `scale * Σⱼ [cⱼ Hⱼ, R]` into `W`, where each `Hⱼ` is stored as an
`Op` in split forward/reverse form. `W` is added to, not overwritten.

**`R` must be Hermitian**: only the `HR` half is computed and `RH` is taken as
its adjoint. The sole caller ([`fquantum!`](@ref)) satisfies this. A
non-Hermitian `R` gives a silently wrong result.
"""
Base.@propagate_inbounds function apply_commutator!(W::Matrix{ComplexF64},
                                                   terms::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                                                   R::Matrix{ComplexF64},
                                                   scale::ComplexF64)
    n = size(R, 1)
    # Accumulate (HR)^T contiguously, then reflect. Written directly, `HR` is
    # `W[i, col] += a * R[j, col]` -- `col` walks across a row, jumping `n`
    # complex numbers per iteration on a column-major layout. Transposed,
    # `(HR)^T[col, i] = Σⱼ a * conj(R[col, j])` reads and writes down columns.
    # ~15x at d = 256, for identical flops.
    ws = _commutator_ws(n)
    Wt = ws.Wt
    fill!(Wt, 0)
    @inbounds for (coeff_ref, op) in terms
        c  = coeff_ref[]
        # A sequence carries every term it will ever use from the first step,
        # most switched off at any instant (k39: 79 terms, 24 driven). 5.4x.
        iszero(c) && continue
        cc = conj(c)
        # `view`, not 2D indexing: the compiler sees a contiguous vector. ~1.5x.
        for (i, j, v) in op.forward
            α = c * v                      # unscaled -- see the reflect below
            wcol = view(Wt, :, i); rcol = view(R, :, j)
            @simd for col in 1:n
                wcol[col] += α * conj(rcol[col])
            end
        end
        for (i, j, u) in op.reverse
            β = cc * u
            wcol = view(Wt, :, i); rcol = view(R, :, j)
            @simd for col in 1:n
                wcol[col] += β * conj(rcol[col])
            end
        end
    end
    # W += scale * (HR - (HR)^dagger), reading (HR)^T from `Wt` columnwise.
    #
    # `scale` is applied HERE, not folded into the accumulation: `[H,R] =
    # HR - (HR)^dagger` holds for the UNSCALED product. Carried inside, an
    # imaginary `scale` gives `scale*(HR + (HR)^dagger)` -- the anticommutator.
    @inbounds for j in 1:n
        @simd for i in 1:n
            hr = Wt[j, i]              # (HR)^T[j,i] = HR[i,j]
            W[i, j] += scale * (hr - conj(Wt[i, j]))
        end
    end
    return W
end


#------------------------------------------------------------------------------
# Core propagators and forces
#------------------------------------------------------------------------------

"""
    fdipole!(dt, psi, atom, drives, jumps, _dpsi)

Add the radiation-pressure impulse from `drives` to `atom.v` over `dt`, using
the trajectory state `psi`.

The scattering rate `R_tot = Σⱼ ⟨Lⱼ† Lⱼ⟩` comes from the jump operators. Each
planar drive `b` gets a weight `w_b = |ψ[i]* (H_b ψ)[j]|²` over its own
transition `i => j`, and the mean force is

    F = ħ R_tot Σ_b (w_b / Σ_b′ w_b′) k_b

`_dpsi` is caller-supplied scratch of length `d`.
"""
Base.@inline function fdipole!(dt::Float64,
                               psi::Vector{ComplexF64},
                               atom::NLevelAtom,
                               drives::Vector{<:AbstractField},
                               jumps::Vector{<:Jump},
                               _dpsi::Vector{ComplexF64})

    # Total scattering rate over all jumps.
    Rtot = 0.0
    for j in jumps
        mul!(_dpsi, j._coeff[], j.J, psi)              # _dpsi = Lⱼ ψ
        @inbounds @fastmath for i in eachindex(_dpsi)
            Rtot += abs2(_dpsi[i])                     # += ⟨Lⱼ† Lⱼ⟩
        end
    end

    # Per-beam weights from the coherent couplings.
    wtot = 0.0
    Fx = Fy = Fz = 0.0
    for d in drives
        if d isa PlanarCoupling
            k = d.beam.k
            mul!(_dpsi, d._coeff[], d.H, psi)              # H_b ψ
            Jb = conj(psi[d.transition[1]]) * _dpsi[d.transition[2]]
            w  = abs2(Jb)
            wtot += w
            Fx += w * k[1]
            Fy += w * k[2]
            Fz += w * k[3]
        end
    end

    if wtot > 0.0 && Rtot > 0.0
        norm = hbar * Rtot * dt / (atom.m * wtot)
        atom.v[1] += Fx * norm
        atom.v[2] += Fy * norm
        atom.v[3] += Fz * norm
    end
end



"""
    recoil!(J, rng)

Apply an isotropic recoil kick of magnitude `ħk` to `J`'s atom, with
`k = 2π / J.atom.lambda[J.transition]`.
"""
Base.@inline function recoil!(J::Jump, rng::AbstractRNG)
    _x, _y, _z = randn(rng), randn(rng), randn(rng)
    _norm = sqrt(_x * _x + _y * _y + _z * _z)

    k = 2 * pi / J.atom.lambda[J.transition]
    c = hbar * k / J.atom.m / _norm

    J.atom.v[1] += c * _x
    J.atom.v[2] += c * _y
    J.atom.v[3] += c * _z
end

