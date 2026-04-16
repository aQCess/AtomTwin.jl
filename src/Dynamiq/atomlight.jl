#------------------------------------------------------------------------------
# Jump operators and dissipators
#------------------------------------------------------------------------------

"""
    Jump(b, atom, transition, rate; detectors = AbstractDetector[], blockade = 0)

Generic quantum jump process for a single atomic transition with decay rate `rate`.

The constructor builds the collapse operator `L` as an `Op` in basis `b`,
optionally with a Rydberg blockade constraint on a given level. The same
`Jump` object can be reused across different solvers, which may fill the
cached non-Hermitian Hamiltonian or Lindblad diagonals for performance.
"""
mutable struct Jump
    atom::AbstractAtom
    transition::Pair{Int,Int}
    J::Op                         # collapse operator L
    _rate::Float64                # physical decay rate (stored for parameter updates)

    # Optional helpers, filled depending on simulation type
    Hnh::Union{Nothing,Op}        # non-Hermitian -i/2 L†L term (WFMC)
    LdagL_diag::Union{Nothing,Vector{Float64}}  # diag(L†L) (QME)

    _coeff::Base.RefValue{ComplexF64}
    detectors::Vector{AbstractDetector}

    function Jump(b, atom, transition, rate;
                  detectors = AbstractDetector[], blockade = 0)

        # Always build the collapse operator L
        J = Op(b, atom, transition, sqrt(rate); jump = true, blockade = blockade)

        # Helpers are left uninitialized here; evolve! will fill them
        Hnh        = nothing
        LdagL_diag = nothing

        return new{}(atom, transition, J, Float64(rate),
                     Hnh, LdagL_diag,
                     Ref(ComplexF64(1.0)), detectors)
    end
end

"""
    precompute!(j::Jump, ::Type{<:AbstractVector})

Precompute and cache the non-Hermitian contribution \\(-\\mathrm{i}/2 L^\\dagger L\\)
for wavefunction Monte Carlo propagation. The result is stored as an `Op`
in `j.Hnh` and reused during time evolution.
"""
function precompute!(j::Jump, ::Type{<:AbstractVector})
    if j.Hnh === nothing
        L  = sparse(j.J)
        Hm = -0.5im * (L' * L)
        j.Hnh = Op(Hm)
    end
    return j
end

"""
    precompute!(j::Jump, ::Type{<:AbstractMatrix})

Precompute and cache data required for Lindblad master equation propagation.

The diagonal of \\((L^\\dagger L)\\) is stored in `j.LdagL_diag` and reused in
the anticommutator part of the Liouvillian.

# Returns

- `j::Jump` with `LdagL_diag::Vector{Float64}` containing
  `LdagL_diag[j] = ∑ₘ |Lₘⱼ|²`.
"""
function precompute!(j::Jump, ::Type{<:AbstractMatrix})
    if j.LdagL_diag === nothing
        L = sparse(j.J)
        n = size(L, 2)
        d = zeros(Float64, n)

        vals = nonzeros(L)
        rows = rowvals(L)
        @inbounds for col in 1:n
            for p in nzrange(L, col)
                v = vals[p]
                d[col] += abs2(v)
            end
        end

        j.LdagL_diag = d
    end
    return j
end

#------------------------------------------------------------------------------
# Coherent couplings and detunings
#------------------------------------------------------------------------------

"""
    PlanarCoupling(b, atom, transition, rate, beam)

Planar laser coupling between two internal levels with a spatially dependent
phase and amplitude given by `beam`.

The underlying `Op` `H` encodes the bare coupling in basis `b`, while
`update!(::PlanarCoupling, ::Int)` updates the complex coefficient using
the instantaneous position of `atom` and the beam wavevector.
"""
mutable struct PlanarCoupling{A} <: AbstractField
    atom::A
    transition::Pair{Int,Int}
    beam::PlanarBeam
    H::Op
    _coeff::Base.RefValue{ComplexF64}

    function PlanarCoupling(b, atom, transition, rate, beam)
        H = Op(b, atom, transition, rate / 2)
        new{typeof(atom)}(atom, transition, beam, H, Ref(Complex(1.0)))
    end
end

"""
    update!(drive::PlanarCoupling, step)

Update the complex amplitude of a planar coupling using the current atomic
position and beam wavevector. This is typically called by the time integrator.
"""
function update!(drive::PlanarCoupling{A}, i::Int) where A
    k = drive.beam.k
    r = drive.atom.x
    drive._coeff[] = cis(k[1] * r[1] + k[2] * r[2] + k[3] * r[3])
    return nothing
end

"""
    GlobalCoupling(b, atom, transition, rate)

Spatially uniform laser coupling between two internal levels with complex
Rabi rate `rate`.

The internal operator `H` stores the forward and reverse parts of the
interaction in basis `b`, while the `_coeff` field acts as a time-dependent
envelope (default `1`).
"""
mutable struct GlobalCoupling{A} <: AbstractField
    atom::A
    transition::Pair{Int,Int}
    H::Op                           # base operator with forward/reverse parts
    rate::ComplexF64                # physical complex Rabi rate Ω
    _coeff::Base.RefValue{ComplexF64}  # time-dependent scalar envelope (default 1)

    function GlobalCoupling(b, atom, transition, rate::ComplexF64)
        # Build H0 with explicit forward and reverse parts; operator1/Op
        # should already split into forward/reverse for jump = false.
        H = Op(b, atom, transition, rate / 2; jump = false)
        return new{typeof(atom)}(atom, transition, H, rate, Ref(ComplexF64(1.0)))
    end
end

GlobalCoupling(b, atom, transition, rate::Float64) =
    GlobalCoupling(b, atom, transition, Complex(rate))

"""
    update!(d::GlobalCoupling, step)

No-op update for global couplings. The coefficient is assumed to be
handled externally or remain constant in time.
"""
update!(d::GlobalCoupling, i::Int) = nothing

"""
    GaussianCoupling

Coupling with spatially-varying Rabi rate for a `GaussianBeam` or `GeneralGaussianBeam`.

At build time, Ω₀ (peak Rabi frequency at the atom's position) and E₀ = `efield_scalar(beam, atom.x)`
are computed once. Each solver timestep `update!` recomputes only the scalar field envelope
(~5 FP ops) and scales: `_coeff[] = Ω₀ * efield_scalar(beam, atom.x) / E₀`.

!!! note
    Spatially dependent polarization is not supported. The polarization direction is
    evaluated once at the atom's build-time position and held fixed. Effects such as
    tight-focusing polarization gradients require a different approach.
"""
mutable struct GaussianCoupling{A,B<:AbstractBeam} <: AbstractField
    atom::A
    transition::Pair{Int,Int}
    H::Op                                       # Ω0/2 baked in (same convention as GlobalCoupling)
    _coeff::Base.RefValue{ComplexF64}           # = _amplitude[] × efield/E0 (written by update!)
    _amplitude::Base.RefValue{ComplexF64}       # dimensionless pulse amplitude (written by AmplitudeModifier)
    beam::B
    Ω0::ComplexF64                              # peak Ω at reference position (baked into H)
    E0::ComplexF64                              # efield scalar at reference position
end

function GaussianCoupling(b::Basis, atom, transition::Pair{Int,Int},
                          beam::AbstractBeam, Ω0::ComplexF64)
    E0 = ComplexF64(efield_scalar(beam, atom.x))
    H  = Op(b, atom, transition, Ω0 / 2; jump = false)   # Ω0 baked in, like GlobalCoupling
    GaussianCoupling{typeof(atom),typeof(beam)}(
        atom, transition, H,
        Ref(ComplexF64(1.0)),   # _coeff — overwritten by update! before first fquantum!
        Ref(ComplexF64(1.0)),   # _amplitude — set by AmplitudeModifier each step
        beam, Ω0, E0)
end

"""
    update!(f::GaussianCoupling, step)

Compute `_coeff[] = _amplitude[] × efield_scalar(beam, x) / E₀`.

`_amplitude[]` is set by `AmplitudeModifier` (from a `Pulse`/`On`/`Off` instruction)
before this method is called each timestep.  Multiplying by the spatial envelope ensures
both the commanded pulse amplitude and the atom's position scale the instantaneous Ω.

Cost: one `efield_scalar` evaluation + one complex multiply + one divide — no alloc.
"""
function update!(f::GaussianCoupling, ::Int)
    f._coeff[] = f._amplitude[] * efield_scalar(f.beam, f.atom.x) / f.E0
    return nothing
end

"""
    BlockadeCoupling(b, atom, transition, rate)

Laser coupling that includes a Rydberg blockade shift on the excited state.

The constructor builds an `Op` with a `blockade` flag set to the upper level
index, enabling state-dependent suppression of population.
"""
mutable struct BlockadeCoupling{A} <: AbstractField
    atom::A
    transition::Pair{Int,Int}
    H::Op
    _coeff::Base.RefValue{ComplexF64}

    function BlockadeCoupling(b, atom, transition, rate)
        H = Op(b, atom, transition, rate / 2; blockade = transition[2])
        new{typeof(atom)}(atom, transition, H, Ref(Complex(1.0)))
    end
end

"""
    update!(d::BlockadeCoupling, step)

No-op update for blockade couplings. The blockade effect is encoded in
the static operator `H`.
"""
update!(d::BlockadeCoupling, i::Int) = nothing

"""
    Detuning(b, atom, level, value)

Single-level detuning term acting on `level` of `atom` with energy shift
`value`.

The corresponding diagonal operator is stored as an `Op` in basis `b`,
and contributes an on-site phase evolution to the Hamiltonian.
"""
struct Detuning{A} <: AbstractField
    atom::A
    level::Int
    H::Op
    _coeff::Base.RefValue{ComplexF64}
    function Detuning(b, atom, lvl, value)
        H = Op(b, atom, lvl => lvl, value)
        new{typeof(atom)}(atom, lvl, H, Ref(Complex(1.0)))
    end
end

"""
    update!(::Detuning, step)

No-op update for static detuning terms. The coefficient remains fixed.
"""
update!(::Detuning, ::Int) = nothing

"""
    StarkShiftAC(b, atom, level, beam)

AC Stark shift on a single internal level induced by an optical `beam`.

The constructor computes the differential polarizability of the chosen
`level` from `atom.alpha` at the beam wavelength and stores it in `alpha`.
The `update!` method evaluates the local intensity and updates the
time-dependent energy shift.
"""
struct StarkShiftAC{A} <: AbstractField
    atom::A
    level::Int
    H::Op
    beam::AbstractBeam
    alpha::Float64
    _coeff::Base.RefValue{ComplexF64}

    function StarkShiftAC(b, atom, lvl, beam)
        H = Op(b, atom, lvl => lvl, 1.0)
        alphas = atom.alpha[getwavelength(beam)]
        alpha = alphas[lvl] - mean(alphas)
        new{typeof(atom)}(atom, lvl, H, beam, alpha, Ref(Complex(0.0)))
    end
end

"""
    update!(f::StarkShiftAC, step)

Update the AC Stark shift coefficient from the instantaneous beam
intensity at the atomic position. The stored coefficient is
\\(\alpha I / \\hbar\\) in angular-frequency units.
"""
function update!(f::StarkShiftAC{A}, i::Int) where A
    f._coeff[] = 1 / hbar * f.alpha * intensity(f.beam, f.atom.x)
    return nothing
end

"""
    Interaction(b, atoms::Pair, transition1, transition2, value)

Two-atom interaction coupling specified by transitions `transition1`
and `transition2` on a pair of atoms.

The underlying `Op` encodes the interaction matrix elements in basis `b`,
while `_coeff` enables a scalar time-dependent prefactor.
"""
mutable struct Interaction{A} <: AbstractField
    atom1::A
    atom2::A
    H::Op
    _coeff::Base.RefValue{ComplexF64}
    function Interaction(b, atoms::Pair, transition1, transition2, value)
        H = Op(b, atoms, transition1, transition2, value)
        new{typeof(atoms[1])}(atoms..., H, Ref(Complex(1.0)))
    end
end

"""
    update!(d::Interaction, step)

No-op update for static pairwise interactions. The operator is fixed
and its coefficient is assumed constant unless modified externally.
"""
function update!(d::Interaction{A}, i::Int) where A
end

"""
    VdWInteraction(b, atoms, transition1, transition2, C6; V_cap=Inf)

Distance-dependent van der Waals interaction V(r) = C6 / r⁶ between two atoms.

The underlying `Op` is built with a unit coefficient (1.0); the scalar `_coeff`
is updated every solver timestep from the instantaneous inter-atom separation.
`C6` is in rad/s·m⁶ (ħ = 1 units).

If `V_cap` is finite, the interaction is clamped: `V = min(C6/r⁶, V_cap)`.
"""
mutable struct VdWInteraction{A} <: AbstractField
    atom1::A
    atom2::A
    H::Op               # unit projector |rr⟩⟨rr|; _coeff carries C6/r^6
    _coeff::Base.RefValue{ComplexF64}
    C6::Float64         # rad/s·m^6
    V_cap::Float64      # maximum interaction strength (rad/s); Inf = no cap
    function VdWInteraction(b, atoms::Pair, transition1, transition2, C6::Float64;
                            V_cap::Float64 = Inf)
        H = Op(b, atoms, transition1, transition2, 1.0)
        return new{typeof(atoms[1])}(atoms[1], atoms[2], H, Ref(ComplexF64(C6)), C6, V_cap)
    end
end

"""
    update!(d::VdWInteraction, ::Int)

Recompute the van der Waals coefficient from the current inter-atom separation:

    V = C6 / r⁶

clamped to `V_cap` when finite. Called each solver timestep after `fclassical!`
has updated positions.
"""
function update!(d::VdWInteraction, ::Int)
    x1 = d.atom1.x;  x2 = d.atom2.x
    dx = x2[1] - x1[1];  dy = x2[2] - x1[2];  dz = x2[3] - x1[3]
    r2 = dx*dx + dy*dy + dz*dz
    r6 = r2 * r2 * r2
    V  = d.C6 / r6
    d._coeff[] = ComplexF64(isfinite(d.V_cap) ? min(V, d.V_cap) : V)
    return nothing
end

#------------------------------------------------------------------------------
# N-level atom model
#------------------------------------------------------------------------------

"""
    NLevelAtom(n; x = [0, 0, 0], v = [0, 0, 0],
                  m = 1amu, alphas = Dict(), lambdas = Dict())

Minimal `n`-level atomic model with classical center-of-mass motion.

- `x`, `v`: position and velocity vectors in real space.
- `m`: atomic mass.
- `alphas`: dictionary of scalar or tensor polarizabilities keyed by wavelength.
- `lambdas`: dictionary of transition wavelengths keyed by level pairs.

Additional internal fields `_P` and `_pidx` are used for caching populations
and basis-dependent index mappings during simulations.
"""
mutable struct NLevelAtom <: AbstractAtom
    n::Int
    x::Vector{Float64}
    v::Vector{Float64}
    m::Float64
    alpha::Dict{Float64,Vector{Float64}}
    lambda::Dict{Pair{Int,Int},Float64}

    _P::Vector{Float64}           # populations, used for intermediate computations
    _pidx::Vector{Vector{Int}}    # updated when a basis is constructed

    function NLevelAtom(n;
                        x       = [0.0, 0.0, 0.0],
                        v       = [0.0, 0.0, 0.0],
                        m       = 1amu,
                        alphas  = Dict(),
                        lambdas = Dict())
        new(n, x, v, m, alphas, lambdas, zeros(n), [[0]])
    end
end

"""
    copy(a::NLevelAtom)

Create a deep copy of an `NLevelAtom`, including position, velocity,
polarizabilities, transition wavelengths, and internal cache fields.
"""
function copy(a::NLevelAtom)
    # Recreate via public API
    b = NLevelAtom(
        a.n;
        x       = copy(a.x),
        v       = copy(a.v),
        m       = a.m,
        alphas  = Dict(k => copy(v) for (k, v) in a.alpha),
        lambdas = copy(a.lambda),
    )
    # Copy internal/derived state
    b._P    = copy(a._P)
    b._pidx = [copy(idx) for idx in a._pidx]
    return b
end

#------------------------------------------------------------------------------
# Display helpers for fields and dissipators
#------------------------------------------------------------------------------

# One-line summary used by `show(io, x)`
function Base.show(io::IO, f::AbstractField)
    T = typeof(f)
    print(io, "AtomTwin.$(nameof(T))(")

    if hasfield(T, :transition)
        i, j = f.transition
        print(io, "$(i)→$(j)")
    elseif hasfield(T, :level)
        print(io, "level=$(f.level)")
    end

    if hasfield(T, :rate)
        print(io, ", Ω=$(getfield(f, :rate))")
    end

    print(io, ")")
end

# Multiline text/plain
function Base.show(io::IO, ::MIME"text/plain", f::AbstractField)
    println(io, sprint(show, f))

    T = typeof(f)

    if hasfield(T, :transition)
        i, j = f.transition
        println(io, "├─ Transition: $(i) → $(j)")
    elseif hasfield(T, :level)
        println(io, "├─ Level:      ", f.level)
    end

    println(io, "├─ Coefficient: ", f._coeff[])

    H = f.H
    println(io, "└─ H:          ",
            H.dim, "×", H.dim,
            " operator (", length(H.forward) + length(H.reverse), " nonzero elements")
end

function Base.show(io::IO, d::AbstractDissipator)
    T = typeof(d)
    print(io, "AtomTwin.$(nameof(T))(")

    if hasfield(T, :transition)
        tr = getfield(d, :transition)
        i, j = tr isa Pair ? (tr.first, tr.second) : tr
        print(io, "$(i)→$(j)")
    elseif hasfield(T, :level)
        print(io, "level=$(getfield(d, :level))")
    end

    if hasfield(T, :rate)
        print(io, ", Ω=$(getfield(d, :rate))")
    end

    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", d::AbstractDissipator)
    println(io, sprint(show, d))

    T = typeof(d)

    if hasfield(T, :transition)
        tr = getfield(d, :transition)
        i, j = tr isa Pair ? (tr.first, tr.second) : tr
        println(io, "├─ Transition: $(i) → $(j)")
    elseif hasfield(T, :level)
        println(io, "├─ Level:      ", getfield(d, :level))
    end

    println(io, "├─ Coefficient: ", getfield(d, :_coeff)[])

    J = getfield(d, :J)
    println(io, "└─ J:        ",
            J.dim, "×", J.dim,
            " operator (", length(J.forward) + length(J.reverse), " nonzero elements")
end
