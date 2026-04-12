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

#------------------------------------------------------------------------------
# Core propagators and forces
#------------------------------------------------------------------------------

"""
    fdipole!(dt, psi, atom, drives, jumps, _dpsi)

Update the atomic velocity due to radiation-pressure forces from a set of
classical driving fields, using Monte Carlo wavefunction data.

The total scattering rate is obtained from the jump operators as
R_tot = Σ_j ⟨L_j† L_j⟩. For each planar drive, a driven component
|dψ_b⟩ = H_b |ψ⟩ is computed into `_dpsi`, and a dimensionless weight
w_b ∝ |ψ_g* (dψ_b)_e|^2 is formed to quantify how strongly that beam
drives the atom. The mean force

    F = ħ R_tot Σ_b (w_b / Σ_b' w_b') k_b

is then used to update `atom.v` over the time step `dt`.
"""
Base.@inline function fdipole!(dt::Float64,
                               psi::Vector{ComplexF64},
                               atom::NLevelAtom,
                               drives::Vector{<:AbstractField},
                               jumps::Vector{<:Jump},
                               _dpsi::Vector{ComplexF64})

    # 1. Total scattering rate from all relevant jumps
    Rtot = 0.0
    for j in jumps
        mul!(_dpsi, j._coeff[], j.J, psi)              # _dpsi = L_j ψ
        @inbounds @fastmath for i in eachindex(_dpsi)
            Rtot += abs2(_dpsi[i])         # += ⟨L_j† L_j⟩
        end
    end

    # 2. Beam weights from coherent couplings
    wtot = 0.0
    Fx = Fy = Fz = 0.0
    for d in drives
        if d isa PlanarCoupling
            k = d.beam.k
            mul!(_dpsi, d._coeff[], d.H, psi)   # H_b ψ
            Jb = conj(psi[d.transition[1]]) * _dpsi[d.transition[2]]        # proxy for Ω_b ⟨σ_ge⟩_b
            w  = abs2(Jb)
            wtot += w
            Fx += w * k[1]
            Fy += w * k[2]
            Fz += w * k[3]
        end
    end

    # 3. Apply mean force
    if wtot > 0.0 && Rtot > 0.0
        norm = hbar * Rtot * dt / (atom.m * wtot)
        atom.v[1] += Fx * norm
        atom.v[2] += Fy * norm
        atom.v[3] += Fz * norm
    end
end



"""
    recoil!(J, rng)

Apply a random recoil kick to the atom associated with jump process `J`.

A random direction is sampled isotropically, and a momentum kick of magnitude
\\(\\hbar k\\) is applied, where \\(k = 2\\pi / \\lambda\\) is set by the
transition wavelength stored in `J.atom.lambda[J.transition]`.
"""
Base.@inline function recoil!(J::Jump, rng::AbstractRNG)
    _x, _y, _z = randn(rng), randn(rng), randn(rng)
    _norm = sqrt(_x * _x + _y * _y + _z * _z)

    k = 2 * pi / J.atom.lambda[J.transition]
    c = hbar * k / J.atom.m / _norm

    J.atom.v[1] += c * _x  # random kick
    J.atom.v[2] += c * _y
    J.atom.v[3] += c * _z
end

#------------------------------------------------------------------------------
# Core quantum propagators (fquantum!)
#------------------------------------------------------------------------------

"""
    fquantum!(dt, qstate, Hlist, _q1, _q2; order = 4)

Evolve a pure state vector `qstate` forward by a time step `dt` under a
sum of Hamiltonian terms using a multi-stage Magnus-like scheme.

# Arguments

- `dt::Float64`: Time step size.
- `qstate::Vector{ComplexF64}`: State vector, updated in-place.
- `Hlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}}`: Hamiltonian terms.
  Each entry is `(coeff, H)` where `coeff[]::ComplexF64` is a (possibly
  time-dependent) scalar prefactor and `H::Op` stores the operator via
  `H.forward` and `H.reverse`.
- `_q1, _q2::Vector{ComplexF64}`: Work buffers with the same length as `qstate`.
- `order::Int = 4`: Number of internal stages used for the time-stepping scheme.

# Details

For each term `(coeff, H)` in `Hlist`, the Hermitian contribution is applied as

    -i dt ( coeff[] * H.forward + conj(coeff[]) * H.reverse ) * qstate

in a multi-stage expansion, accumulating the effect of all terms into `qstate`.
The split storage of `H` avoids reconstructing full matrices while keeping the
Hermitian structure explicit through the forward/reverse parts.
"""
function fquantum!(dt::Float64,
                   qstate::Vector{ComplexF64},
                   Hlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                   _q1::Vector{ComplexF64},
                   _q2::Vector{ComplexF64};
                   order::Int = 4)

    _q1 .= qstate
    for k in 1:order
        fill!(_q2, 0.0)
        a = ComplexF64(-1.0im / k * dt)
        for (coeff_ref, op) in Hlist
            ac  = a * coeff_ref[]
            acc = a * conj(coeff_ref[])

            @inbounds for (i, j, v) in op.forward
                @fastmath _q2[i] += ac * v * _q1[j]
            end
            @inbounds for (i, j, u) in op.reverse
                @fastmath _q2[i] += acc * u * _q1[j]
            end
        end
        _q1, _q2 = _q2, _q1
        qstate .+= _q1
    end
end

"""
    fquantum!(dt, qstate, Hlist, Hnhlist, _q1, _q2; order = 4)

Evolve a pure state vector with both Hermitian and non-Hermitian
contributions, e.g. for effective non-Hermitian Hamiltonians in
wavefunction Monte Carlo.

- `Hlist`: Hermitian terms, applied as
  `coeff[] * H.forward + conj(coeff[]) * H.reverse`.
- `Hnhlist`: Non-Hermitian terms, applied as `coeff[] * H.forward` only.
"""
function fquantum!(dt::Float64,
                   qstate::Vector{ComplexF64},
                   Hlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                   Hnhlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                   _q1::Vector{ComplexF64},
                   _q2::Vector{ComplexF64};
                   order::Int = 4)

    _q1 .= qstate

    for k in 1:order
        fill!(_q2, 0.0)
        a = ComplexF64(-1.0im * dt / k)

        # Hermitian part
        @inbounds for (coeff_ref, op) in Hlist
            coeff = coeff_ref[]  # Load once
            ac  = a * coeff
            acc = a * conj(coeff)

            for (i, j, v) in op.forward
                @fastmath _q2[i] += ac * v * _q1[j]
            end
            for (i, j, u) in op.reverse
                @fastmath _q2[i] += acc * u * _q1[j]
            end
        end

        # Non-Hermitian part
        @inbounds for (coeff_ref, op) in Hnhlist
            ac = a * coeff_ref[]
            for (i, j, v) in op.forward
                @fastmath _q2[i] += ac * v * _q1[j]
            end
        end
        
        _q1, _q2 = _q2, _q1
        qstate .+= _q1
    end
end


"""
    fquantum!(dt, ρ, Hlist, _ρ1, _ρ2; order = 4)

Evolve a density matrix `ρ` under unitary dynamics generated by a sum
of Hamiltonian terms.

The commutator \\(-i[H, \\rho]\\) is applied using the `Op` split representation
of each term in `Hlist`, without constructing full dense matrices.
"""
function fquantum!(dt::Float64,
                   ρ::Matrix{ComplexF64},
                   Hlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                   _ρ1::Matrix{ComplexF64},
                   _ρ2::Matrix{ComplexF64};
                   order::Int = 4)

    _ρ1 .= ρ
    n = size(ρ, 1)  # assuming square

    for k in 1:order
        fill!(_ρ2, 0.0)
        a = ComplexF64(-1im * dt / k)

        for (coeff_ref, op) in Hlist
            c  = coeff_ref[]
            cc = conj(c)

            # forward part: c * H₀
            @inbounds for (i, j, v) in op.forward
                α = a * (c * v)
                for col in 1:n
                    _ρ2[i, col] += α * _ρ1[j, col]
                end
                for row in 1:n
                    _ρ2[row, j] -= α * _ρ1[row, i]
                end
            end

            # reverse part: c* * H₀†
            @inbounds for (i, j, u) in op.reverse
                β = a * (cc * u)
                for col in 1:n
                    _ρ2[i, col] += β * _ρ1[j, col]
                end
                for row in 1:n
                    _ρ2[row, j] -= β * _ρ1[row, i]
                end
            end
        end

        _ρ1, _ρ2 = _ρ2, _ρ1
        ρ .+= _ρ1
    end
end

"""
    fquantum!(dt, ρ, Hlist, Jlist, _ρ1, _ρ2; order = 4)

Evolve a density matrix `ρ` by a time step `dt` under combined Hamiltonian
and Lindblad evolution.

# Arguments

- `dt::Float64`: Time step size.
- `ρ::Matrix{ComplexF64}`: Density matrix, updated in-place.
- `Hlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}}`:
  Hamiltonian terms \\(H_j\\), each stored as `(coeff, H)`.
- `Jlist::Vector{Tuple{Base.RefValue{ComplexF64},Op,Vector{Float64}}}`:
  Lindblad jump data. Each entry is `(coeff, L, LdagL_diag)` where:
  - `coeff[]::ComplexF64`: Jump amplitude (physical rate \\(\\gamma = |\\text{coeff}|^2\\)).
  - `L::Op`: Jump operator \\(L\\); only `L.forward` is used.
  - `LdagL_diag::Vector{Float64}`: Diagonal of \\(L^\\dagger L\\) for the
    anticommutator part.
- `_ρ1, _ρ2::Matrix{ComplexF64}`: Workspace matrices.

# Physics

Integrates the Lindblad master equation

\\[
\\frac{\\mathrm{d}\\rho}{\\mathrm{d}t}
= -i[H, \\rho] + \\sum_j \\gamma_j
\\Big( L_j \\rho L_j^{\\dagger}
      - \\tfrac{1}{2} \\{ L_j^{\\dagger} L_j, \\rho \\} \\Big)
\\]

with \\(\\gamma_j = |\\text{coeff}_j|^2\\) and
\\(H = \\sum_j \\text{coeff}_j H_j\\).
"""
function fquantum!(dt::Float64,
                   ρ::Matrix{ComplexF64},
                   Hlist::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                   Jlist::Vector{Tuple{Base.RefValue{ComplexF64},Op,Vector{Float64}}},
                   _ρ1::Matrix{ComplexF64},
                   _ρ2::Matrix{ComplexF64};
                   order::Int = 4)

    # Hamiltonian commutator part
    fquantum!(dt, ρ, Hlist, _ρ1, _ρ2; order = order)

    n = size(ρ, 1)
    for (coeff, L, LdagL_diag) in Jlist
        γ = dt * abs2(coeff[])

        # L ρ L† term
        @inbounds for (a, j, vL) in L.forward
            for (b, k, vR) in L.forward
                ρ[a, b] += γ * vL * ρ[j, k] * conj(vR)
            end
        end

        # -1/2 {L†L, ρ} term via precomputed diagonal of L†L
        @inbounds for i in 1:n, j in 1:n
            ρ[i, j] -= 0.5 * γ * (LdagL_diag[i] + LdagL_diag[j]) * ρ[i, j]
        end
    end
end

#------------------------------------------------------------------------------
# Jump selection
#------------------------------------------------------------------------------

"""
    jump!(psi, J, _prob, _psi1, _psi2, rng)

Apply a randomly selected jump operator from the list `J` to the state vector
`psi` and return the chosen `Jump`.

- For a single jump in `J`, that jump is always applied.
- For multiple jumps, the probability of each process is estimated from
  the norm of `J[k].J * psi`, and one is sampled accordingly.

The state vector is projected and renormalized, with a safeguard to
avoid division by zero if the jump has vanishing norm.
"""
Base.@inline function jump!(psi::Vector{ComplexF64},
                            J::Vector{Jump},
                            _prob::Weights,
                            _psi1::Vector{ComplexF64},
                            _psi2::Vector{ComplexF64},
                            rng::AbstractRNG)
    if length(J) == 1
        jump = J[1]
    else
        for (j, jump) in enumerate(J)
            _psi1 .= psi
            mul!(jump.J, _psi1, _psi2)
            _prob[j] = norm(_psi1)^2
        end
        jump = sample(rng, J, _prob)
    end
    _psi1 .= psi
    mul!(jump.J, _psi1, _psi2)
    if norm(_psi1) != 0.0   # dirty fix to avoid divide by zero
        psi .= _psi1
    end
    return jump
end

#------------------------------------------------------------------------------
# High-level evolve! front-ends
#------------------------------------------------------------------------------

"""
    evolve!(atoms, system, tspan; frozen = false, kwargs...)

High-level entry point for pure classical motion of atoms `atoms` in a
composite `system` containing beams and possibly other components.

All atoms are initialized in their ground level population, and Newtonian
equations of motion are integrated by `newton`.
"""
function evolve!(atoms::Vector{<:NLevelAtom},
                 tspan::AbstractVector{Float64};
                 beams::Vector{<:AbstractBeam}=AbstractBeam[],
                 frozen = false,
                 kwargs...)

    # Pure classical motion: initialize populations of each atom
    for atom in atoms
        fill!(atom._P, 0.0)
        atom._P[1] = 1.0
    end

    newton(atoms, tspan; beams = beams, kwargs...)
end

"""
    evolve!((rho, atoms), tspan; fields, jumps, beams, frozen, kwargs...)

Evolve a density matrix `rho` and atomic positions `atoms` over `tspan`
using either a frozen-atom quantum master equation or a semiclassical
master equation with classical trajectories.
"""
function evolve!(state::Tuple{Matrix{ComplexF64},Vector{<:NLevelAtom}},
                 tspan::AbstractVector{Float64};
                 fields::Vector{<:AbstractField} = AbstractField[],
                 jumps::Vector{Jump}              = Jump[],
                 beams::Vector{<:AbstractBeam}    = AbstractBeam[],
                 frozen::Bool                     = true,
                 kwargs...)
    ρ, atoms = state
    for j in jumps
        if isnothing(j.LdagL_diag)
            precompute!(j, Matrix)
        end
    end

    # Build Hamiltonian and Lindblad structures
    L = Tuple{Base.RefValue{ComplexF64},Op}[(d._coeff, d.H) for d in fields]
    J = Tuple{Base.RefValue{ComplexF64},Op,Vector{Float64}}[
        (j._coeff, j.J, j.LdagL_diag) for j in jumps
    ]

    if frozen || isempty(beams)
        # Pure quantum, frozen atoms
        qme(ρ, L, J, tspan; fields = fields, kwargs...)
    else
        # Semiclassical with atomic motion
        qme_semiclassical(ρ, atoms, L, J, tspan; beams = beams, fields = fields, kwargs...)
    end
end

"""
    evolve!((psi, atoms), tspan; fields, jumps, beams, frozen, kwargs...)

Evolve a state vector `psi` and atomic positions `atoms` over `tspan` using
either deterministic Schrödinger dynamics or a wavefunction Monte Carlo
trajectory, with optional semiclassical motion.
"""
function evolve!(state::Tuple{Vector{ComplexF64},Vector{<:NLevelAtom}},
                 tspan::AbstractVector{Float64};
                 fields::Vector{<:AbstractField}  = AbstractField[],
                 jumps::Vector{Jump}              = Jump[],
                 beams::Vector{<:AbstractBeam}    = AbstractBeam[],
                 frozen::Bool                     = false,
                 rng = Random.MersenneTwister(),
                 kwargs...)
    
    # Precompute jump helpers for state-vector propagation
    for j in jumps
        if isnothing(j.Hnh)
            precompute!(j, Vector)
        end
    end
    
    psi, atoms = state
    if length(jumps) == 0
        # Schrödinger dynamics
        H = Tuple{Base.RefValue{ComplexF64},Op}[(d._coeff, d.H) for d in fields]
        if frozen
            tdse(psi, H, tspan; beams = beams, fields = fields, kwargs...)
        else
            tdse_semiclassical(psi, atoms, H, tspan; beams = beams, fields = fields, kwargs...)
        end
        return
    end

    # Wavefunction Monte Carlo method
    if length(jumps) > 0
        H   = Tuple{Base.RefValue{ComplexF64},Op}[(d._coeff, d.H) for d in fields]
        Hnh = Tuple{Base.RefValue{ComplexF64},Op}[(j._coeff, j.Hnh) for j in jumps]

        if frozen
            wfmc(psi, H, Hnh, jumps, tspan; beams = beams, rng=rng, kwargs...)
            return
        else
            wfmc_semiclassical(psi, atoms, H, Hnh, jumps, tspan; fields = fields, beams = beams, rng=rng, kwargs...)
            return
        end
    end
    
end

#------------------------------------------------------------------------------
# Schrödinger evolution
#------------------------------------------------------------------------------

"""
    tdse(psi, H, tspan; kwargs...)

Time-dependent Schrödinger equation solver for pure-state evolution.

- `tspan::Vector{Float64}` is currently assumed to be an equidistant grid.
- `H` is a vector of `(coeff, Op)` pairs representing Hamiltonian terms.
"""
function tdse(psi::Vector{ComplexF64},
              H::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
              tspan::AbstractVector{Float64};
              beams::Vector{<:AbstractBeam}     = AbstractBeam[],
              fields::Vector{<:AbstractField}   = AbstractField[],
              modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
              detectors::Vector{<:AbstractDetector} = AbstractDetector[],
              order::Int                         = 4,
              downsample::Int                    = 1,
              _q1::Vector{ComplexF64}            = copy(psi),
              _q2::Vector{ComplexF64}            = copy(psi))

    steps = length(tspan)
    dt    = tspan[2] - tspan[1]
    for i in 1:steps
        for m in modifiers
            update!(m, i)
        end
        for f in fields
            update!(f, i)
        end
        fquantum!(dt, psi, H, _q1, _q2; order = order)
        if i % downsample == 0
            i_out = i ÷ downsample
            for d in detectors
                if d isa PopulationDetector
                    prep!(d)
                end
                write!(d, i_out)
            end
        end
    end
end

"""
    tdse_semiclassical(psi, atoms, H, tspan; kwargs...)

Time-dependent Schrödinger solver with semiclassical atomic motion.

Quantum dynamics is driven by `H` while classical trajectories evolve
under the optical forces from `beams`. `tspan` is assumed equidistant.
"""
function tdse_semiclassical(psi::Vector{ComplexF64},
                            atoms::Vector{A},
                            H::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                            tspan::AbstractVector{Float64};
                            beams::Vector{<:AbstractBeam}      = AbstractBeam[],
                            fields::Vector{<:AbstractField}    = AbstractField[],
                            modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
                            detectors::Vector{<:AbstractDetector} = AbstractDetector[],
                            order::Int                          = 4,
                            downsample::Int                     = 1,
                            _q1::Vector{ComplexF64}             = copy(psi),
                            _q2::Vector{ComplexF64}             = copy(psi)) where {A}

    steps = length(tspan)
    dt    = tspan[2] - tspan[1]

    for i in 1:steps
        for m in modifiers
            update!(m, i)
        end
        for atom in atoms
            updatepop!(atom, psi)
            fclassical!(dt, atom, beams)
        end
        for f in fields
            update!(f, i)
        end
        fquantum!(dt, psi, H, _q1, _q2; order = order)
        if i % downsample == 0
            i_out = i ÷ downsample
            for d in detectors
                if d isa PopulationDetector
                    prep!(d)
                end
                write!(d, i_out)
            end
        end
    end
end

#------------------------------------------------------------------------------
# Classical updates (forces and Newton solver)
#------------------------------------------------------------------------------

"""
    force(atom, beams)

Compute the total optical dipole force on `atom` at position `atom.x`
from a set of beams.

- Beams with the same wavelength are summed coherently in the complex field
  (interference included).
- Beams with different wavelengths contribute incoherently (no interference).

The dipole force is computed as

\\[
\\mathbf{F} = -2 \\alpha(\\lambda) P \\operatorname{Re}\\big( E^{\\ast} \\nabla E \\big),
\\]

where \\(\\alpha(\\lambda)\\) is the polarizability at wavelength \\(\\lambda\\),
\\(P\\) is the population of the level, and \\(E\\) is the total complex field.
"""
function force(atom::A, beams::Vector{<:AbstractBeam}) where {A}
    Fx, Fy, Fz = 0.0, 0.0, 0.0
    x_pos = atom.x  
    @inbounds for i in 1:atom.n
        P_val = atom._P[i]  
        used_λ = Float64[]
        
        for b_ref in beams
            λ = getwavelength(b_ref)
            if λ in used_λ
                continue
            end
            push!(used_λ, λ)
            
            Etot = 0.0 + 0.0im
            dEdr_x = 0.0 + 0.0im
            dEdr_y = 0.0 + 0.0im
            dEdr_z = 0.0 + 0.0im
            
            for b in beams
                if isapprox(getwavelength(b), λ)
                    Etot += efield_scalar(b, x_pos)
                    dE = dEdr(b, x_pos)
                    dEdr_x += dE[1]
                    dEdr_y += dE[2]
                    dEdr_z += dE[3]
                end
            end
            
            α = atom.alpha[λ][i]
            c = α * P_val
            Etot_conj = conj(Etot)
            Fx += c * real(Etot_conj * dEdr_x)
            Fy += c * real(Etot_conj * dEdr_y)
            Fz += c * real(Etot_conj * dEdr_z)
        end
    end
    return Fx, Fy, Fz
end

"""
    fclassical!(dt, atom, beams)

Single-step classical update of position and velocity for `atom` under
forces from `beams`, using a simple Euler integrator.

- First updates position via \\(\\dot{\\mathbf{x}} = \\mathbf{v}\\),
- then updates velocity via \\(\\dot{\\mathbf{v}} = \\mathbf{F} / m\\).
"""
function fclassical!(dt::Float64, atom::A, beams::Vector{<:AbstractBeam}) where {A}
    dt_m = dt / atom.m  # precompute division
    
    @inbounds begin
        # Update position
        atom.x[1] = muladd(atom.v[1], dt, atom.x[1])
        atom.x[2] = muladd(atom.v[2], dt, atom.x[2])
        atom.x[3] = muladd(atom.v[3], dt, atom.x[3])
        
        # Compute force and update velocity
        Fx, Fy, Fz = force(atom, beams)
        atom.v[1] = muladd(Fx, dt_m, atom.v[1])
        atom.v[2] = muladd(Fy, dt_m, atom.v[2])
        atom.v[3] = muladd(Fz, dt_m, atom.v[3])
    end
end

"""
    newton(atoms, tspan; beams, modifiers, detectors)

Newton-equation solver for purely classical atomic motion.

- `tspan::Vector{Float64}` is assumed to be an equidistant time grid.
- If `length(atoms) > 42`, the update loop is parallelized with `@batch`.
"""
function newton(atoms::Vector{A},
                tspan::AbstractVector{Float64};
                beams::Vector{<:AbstractBeam}     = AbstractBeam[],
                modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
                detectors::Vector{<:AbstractDetector} = AbstractDetector[],
                downsample::Int                   = 1,
                kwargs...) where {A}

    steps = length(tspan)
    dt    = steps > 1 ? tspan[2] - tspan[1] : 0.0

    parallel = length(atoms) > 42

    for i in 1:steps
        for m in modifiers
            update!(m, i)
        end
        if parallel
            @batch for atom in atoms
                fclassical!(dt, atom, beams)
            end
        else
            for atom in atoms
                fclassical!(dt, atom, beams)
            end
        end
        if i % downsample == 0
            i_out = i ÷ downsample
            for d in detectors
                write!(d, i_out)
            end
        end
    end
end

#------------------------------------------------------------------------------
# Wavefunction Monte Carlo evolution
#------------------------------------------------------------------------------

"""
    wfmc(psi, H, Hnh, jumps, tspan; kwargs...)

Time-dependent wavefunction Monte Carlo solver.

- `H` contains Hermitian Hamiltonian terms.
- `Hnh` contains effective non-Hermitian contributions.
- `jumps` is a list of `Jump` processes applied stochastically.
- `tspan` is an equidistant time grid.

At each step, the state is propagated by `fquantum!` and a jump is applied
with probability set by the norm loss.
"""
function wfmc(psi::Vector{ComplexF64},
              H::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
              Hnh::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
              jumps::Vector{Jump},
              tspan::AbstractVector{Float64};
              beams::Vector{<:AbstractBeam}          = AbstractBeam[],
              fields::Vector{<:AbstractField}        = AbstractField[],
              modifiers::Vector{<:AbstractModifier}   = AbstractModifier[],
              detectors::Vector{<:AbstractDetector}   = AbstractDetector[],
              order::Int                               = 4,
              downsample::Int                          = 1,
              _q1::Vector{ComplexF64}                  = copy(psi),
              _q2::Vector{ComplexF64}                  = copy(psi),
              rng = Random.MersenneTwister())

    steps = length(tspan)
    dt    = tspan[2] - tspan[1]

    _prob = Weights(zeros(length(jumps)))

    has_modifiers  = !isempty(modifiers)
    has_fields     = !isempty(fields)
    has_detectors  = !isempty(detectors)
    prep_detectors = [d for d in detectors if d isa PopulationDetector]
    has_prep       = !isempty(prep_detectors)

    @inbounds for i in 1:steps
        if has_modifiers
            for m in modifiers
                update!(m, i)
            end
        end
        if has_fields
            for f in fields
                update!(f, i)
            end
        end
        fquantum!(dt, psi, H, Hnh, _q1, _q2; order = order)
        n = norm(psi)
        if n^2 < rand(rng)
            jump!(psi, jumps, _prob, _q1, _q2, rng)
            n = norm(psi)
        end
        inv_n = 1.0 / n
        @simd for k in eachindex(psi)
            psi[k] *= inv_n
        end
        if has_detectors && i % downsample == 0
            i_out = i ÷ downsample
            if has_prep
                for d in prep_detectors
                    prep!(d)
                end
            end
            for d in detectors
                write!(d, i_out)
            end
        end
    end
end


"""
    updatepop!(atom, psi)

Update the internal level populations `atom._P` from the many-body
state vector `psi`, using the precomputed index lists `atom._pidx`.
"""
function updatepop!(atom::AbstractAtom, psi::Vector{ComplexF64})
    for lvl in 1:atom.n
        a = 0.0
        for j in atom._pidx[lvl]
            a += abs2(psi[j[1]])
        end
        atom._P[lvl] = a
    end
end

"""
    wfmc_semiclassical(psi, atoms, H, Hnh, jumps, tspan; kwargs...)

Wavefunction Monte Carlo solver with semiclassical atomic motion.

Combines:
- Quantum trajectory evolution under `H` and `Hnh`.
- Classical motion updated by `fclassical!`.
- Additional dipole-force-induced velocity changes via `fdipole!`.
"""
function wfmc_semiclassical(psi::Vector{ComplexF64},
                            atoms::Vector{A},
                            H::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                            Hnh::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                            jumps::Vector{},
                            tspan::AbstractVector{Float64};
                            beams::Vector{<:AbstractBeam}        = AbstractBeam[],
                            modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
                            fields::Vector{<:AbstractField}       = AbstractField[],
                            detectors::Vector{<:AbstractDetector} = AbstractDetector[],
                            order::Int                             = 4,
                            downsample::Int                        = 1,
                            _q1::Vector{ComplexF64}                = copy(psi),
                            _q2::Vector{ComplexF64}                = copy(psi),
                            rng=Random.MersenneTwister()) where {A}

    steps = length(tspan)
    dt    = tspan[2] - tspan[1]

    _prob = Weights(zeros(length(jumps)))

    for i in 1:steps
        for m in modifiers
            update!(m, i)
        end
        for atom in atoms
            updatepop!(atom, psi)
            fclassical!(dt, atom, beams)
            fdipole!(dt, psi, atom, fields, jumps, _q1)
        end
        for f in fields
            update!(f, i)
        end
        fquantum!(dt, psi, H, Hnh, _q1, _q2; order = order)
        n = norm(psi)
        if n^2 < rand(rng)
            jump = jump!(psi, jumps, _prob, _q1, _q2, rng)
            #recoil!(jump, rng)
            for d in jump.detectors
                write!(d, i)  # photo events are not downsampled
            end
            n = norm(psi)
        end
        psi ./= n
        if i % downsample == 0
            i_out = i ÷ downsample
            for d in detectors
                if d isa PopulationDetector
                    prep!(d)
                end
                write!(d, i_out)
            end
        end
    end
end

#------------------------------------------------------------------------------
# Master equation evolution (QME)
#------------------------------------------------------------------------------

"""
    qme(rho, L, J, tspan; kwargs...)

Quantum master equation solver for density matrices with frozen atoms.

- `L`: Hamiltonian terms `(coeff, Op)`.
- `J`: Jump data `(coeff, L_op, LdagL_diag)`.

The evolution is implemented by repeated calls to `fquantum!` for the
combined Hamiltonian and dissipative contributions.
"""
function qme(rho::Matrix{ComplexF64},
             L::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
             J::Vector{Tuple{Base.RefValue{ComplexF64},Op,Vector{Float64}}},
             tspan::AbstractVector{Float64};
             beams::Vector{<:AbstractBeam}        = AbstractBeam[],
             fields::Vector{<:AbstractField}      = AbstractField[],
             modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
             detectors::Vector{<:AbstractDetector} = AbstractDetector[],
             order::Int                             = 4,
             downsample::Int                        = 1,
             _q1::Matrix{ComplexF64}                = copy(rho),
             _q2::Matrix{ComplexF64}                = copy(rho),
             rng = Random.MersenneTwister())

    steps = length(tspan)
    dt    = tspan[2] - tspan[1]

    for i in 1:steps
        for m in modifiers
            update!(m, i)
        end
        for f in fields
            update!(f, i)
        end
        fquantum!(dt, rho, L, J, _q1, _q2; order = order)
        if i % downsample == 0
            i_out = i ÷ downsample
            for d in detectors
                if d isa PopulationDetector
                    prep!(d)
                end
                write!(d, i_out)
            end
        end
    end
end

"""
    qme_semiclassical(rho, atoms, L, J, tspan; kwargs...)

Quantum master equation solver with semiclassical atomic motion.

Combines:
- Density-matrix evolution under Hamiltonian and Lindblad terms.
- Classical motion updated in parallel by `fclassical!`.
"""
function qme_semiclassical(rho::Matrix{ComplexF64},
                           atoms::Vector{A},
                           L::Vector{Tuple{Base.RefValue{ComplexF64},Op}},
                           J::Vector{Tuple{Base.RefValue{ComplexF64},Op,Vector{Float64}}},
                           tspan::AbstractVector{Float64};
                           beams::Vector{<:AbstractBeam}        = AbstractBeam[],
                           fields::Vector{<:AbstractField}      = AbstractField[],
                           modifiers::Vector{<:AbstractModifier} = AbstractModifier[],
                           detectors::Vector{<:AbstractDetector} = AbstractDetector[],
                           order::Int                             = 4,
                           downsample::Int                        = 1,
                           _q1::Matrix{ComplexF64}                = copy(rho),
                           _q2::Matrix{ComplexF64}                = copy(rho),
                           rng=Random.MersenneTwister()) where {A}

    steps = length(tspan)
    dt    = tspan[2] - tspan[1]

    for i in 1:steps
        for m in modifiers
            update!(m, i)
        end
        @batch for atom in atoms
            fclassical!(dt, atom, beams)
        end
        for f in fields
            update!(f, i)
        end
        fquantum!(dt, rho, L, J, _q1, _q2; order = order)
        if i % downsample == 0
            i_out = i ÷ downsample
            for d in detectors
                if d isa PopulationDetector
                    prep!(d)
                end
                write!(d, i_out)
            end
        end
    end
end
