#------------------------------------------------------------------------------
# Jump operators and dissipators
#------------------------------------------------------------------------------

"""
    Jump(b, atom, transition, rate; blockade = 0)

Generic quantum jump process for a single atomic transition with decay rate `rate`.

The constructor builds the collapse operator `L` as an `Op` in basis `b`,
optionally with a Rydberg blockade constraint on a given level. The same
`Jump` object can be reused across different solvers, which may fill the
cached non-Hermitian Hamiltonian or Lindblad diagonals for performance.

Photon-click detection is handled from the detector side: a `PhotoDetector`
holds a reference to the jump it counts (see `add_decay!(...; clicks = spec)`),
so the jump itself carries no detector list.
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

    function Jump(b, atom, transition, rate; blockade = 0)

        # Always build the collapse operator L
        J = Op(b, atom, transition, sqrt(rate); jump = true, blockade = blockade)

        # Helpers are left uninitialized here; evolve! will fill them
        Hnh        = nothing
        LdagL_diag = nothing

        return new{}(atom, transition, J, Float64(rate),
                     Hnh, LdagL_diag,
                     Ref(ComplexF64(1.0)))
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
`update!(::PlanarCoupling, t)` sets the coefficient to the commanded amplitude
times the plane-wave phase at the atom's instantaneous position (see
`envelope`).
"""
mutable struct PlanarCoupling{A} <: AbstractField
    atom::A
    transition::Pair{Int,Int}
    beam::PlanarBeam
    H::Op
    rate::ComplexF64                        # Rabi rate baked into H, as GlobalCoupling
    _coeff::Base.RefValue{ComplexF64}       # = _amplitude[] × cis(k·x), by update!
    _amplitude::Base.RefValue{ComplexF64}   # commanded amplitude (Pulse/On/Off)

    function PlanarCoupling(b, atom, transition, rate, beam)
        H = Op(b, atom, transition, rate / 2)
        new{typeof(atom)}(atom, transition, beam, H, ComplexF64(rate),
                          Ref(Complex(1.0)), Ref(Complex(1.0)))
    end
end

"""
    update!(drive::PlanarCoupling, step)

Update the complex amplitude of a planar coupling using the current atomic
position and beam wavevector. This is typically called by the time integrator.
"""
function update!(drive::PlanarCoupling{A}, ::Real) where A
    k = drive.beam.k
    r = drive.atom.x
    drive._coeff[] = drive._amplitude[] * cis(k[1] * r[1] + k[2] * r[2] + k[3] * r[3])
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
update!(d::GlobalCoupling, ::Real) = nothing

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
function update!(f::GaussianCoupling, ::Real)
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
update!(d::BlockadeCoupling, ::Real) = nothing

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
update!(::Detuning, ::Real) = nothing

"""
    Hamiltonian(H::Op)

A Hamiltonian term supplied directly as an operator `H`, with no atom/transition
structure of its own. Produced by `add_hamiltonian!` when a Hamiltonian is
written as an explicit `Operator` (built from levels) or handed as a
dense/sparse matrix.

Unlike the physical fields (couplings, detunings, Stark shifts), a `Hamiltonian`
carries no beam/atom recipe; it is just an operator that enters the dynamics.
Structurally it participates in the field-evaluation path so the solver applies
it as `_coeff[] * H.forward + conj(_coeff[]) * H.reverse` (see `fquantum!`),
with `_coeff` a time-dependent scalar envelope (default `1`) that makes the term
`Switchable`/pulse-drivable.

!!! warning
    `H` must be Hermitian for unitary/trace-preserving dynamics; `add_hamiltonian!`
    checks this at build time. The term is applied as
    `c * forward + conj(c) * reverse`, so a real, Hermitian `H` stays Hermitian
    under a real `_coeff` but not under a complex one.
"""
mutable struct Hamiltonian <: AbstractField
    H::Op
    _coeff::Base.RefValue{ComplexF64}

    Hamiltonian(H::Op) = new(H, Ref(ComplexF64(1.0)))
end

"""
    update!(::Hamiltonian, step)

No-op update: the operator is static and its coefficient is handled by the
instruction layer (constant `1` unless pulsed).
"""
update!(::Hamiltonian, ::Real) = nothing

"""
    StarkShiftAC(b, atom, level, beam; reference = nothing)

AC Stark shift on a single internal level induced by an optical `beam`.

The stored `alpha` is read from `atom.alpha` at the beam's wavelength — the same
array the dipole force uses, so the shift an atom feels and the force that moves it
come from one number.

`update!` evaluates the local intensity at the atom's position and sets the
coefficient to `α I / ħ`, an angular frequency.

# Reference level

By default the shift is **absolute**: level `lvl` is shifted by its own `α I/ħ`.
Pass `reference = i` to shift by `(α[lvl] − α[i]) I/ħ` instead, which puts level
`i` at zero — convenient when only a transition frequency matters.

!!! note "Changed behaviour"
    This previously subtracted `mean(alphas)`, the mean over *every* level in the
    basis. That made the shift on one level depend on which other levels happened
    to be present, so adding a leakage level silently changed every shift in the
    system. Use `reference` to name the level you actually mean.
"""
mutable struct StarkShiftAC{A} <: AbstractField
    const atom::A
    const level::Int
    const H::Op
    const beam::AbstractBeam
    alpha::Float64                  # refreshed per shot by `set_alpha!`
    const _coeff::Base.RefValue{ComplexF64}      # = _amplitude[] × I(x)/I₀, by update!
    const _amplitude::Base.RefValue{ComplexF64}  # commanded amplitude (Pulse/On/Off)

    function StarkShiftAC(b, atom, lvl, beam; reference = nothing)
        alphas = atom.alpha[getwavelength(beam)]
        alpha = reference === nothing ? alphas[lvl] :
                                        alphas[lvl] - alphas[reference]
        # The PEAK shift goes into `H`; `_coeff` carries only the intensity
        # envelope, which is in [0,1]. That is the `Detuning` convention, and
        # `spectral_spec` depends on it: it bounds the spectrum once per
        # instruction from the operator values and the coefficients as they
        # stand, inflating only the OFF-DIAGONAL radius for later switching
        # (`peak = true`). A diagonal term hiding its magnitude in `_coeff`
        # is invisible to that bound, so the Chebyshev plan is built for an
        # interval the Hamiltonian then leaves -- and the expansion is
        # evaluated far outside its domain, where it diverges.
        H = Op(b, atom, lvl => lvl, _peak_shift(alpha, beam))
        new{typeof(atom)}(atom, lvl, H, beam, alpha, Ref(Complex(0.0)), Ref(Complex(1.0)))
    end
end

_peak_shift(α, beam) = α * peak_intensity(beam) / (c * ε0 * hbar)

"""
    set_alpha!(f::StarkShiftAC, α) -> f

Refresh the polarizability in place. Used per shot, when `initialize!` has
recomputed `atom.alpha` but nothing structural has changed.

`H` is `peak·|l⟩⟨l|` on the many-body basis: one diagonal entry per basis state
with the atom in `l`, each equal to the peak shift (`operator1` stores them even
when the shift is zero). So the entries are overwritten, not rescaled -- dividing
by the previous α would give NaN whenever it was zero. In place, because the
field is shared by reference with the job that runs it.
"""
function set_alpha!(f::StarkShiftAC, α::Float64)
    f.alpha = α
    peak = ComplexF64(_peak_shift(α, f.beam))
    fw = f.H.forward
    @inbounds for k in eachindex(fw)
        i, j, _ = fw[k]
        fw[k] = (i, j, peak)
    end
    return f
end

"""
    update!(f::StarkShiftAC, step)

Update the AC Stark shift coefficient from the instantaneous beam
intensity at the atomic position. The stored coefficient is
\\(\alpha I / \\hbar\\) in angular-frequency units.
"""
function update!(f::StarkShiftAC{A}, ::Real) where A
    # U = -α I/(c ε₀) is the convention `polarizability_si` sets, so the angular
    # frequency is α I/(c ε₀ ħ) -- NOT α I/ħ. The missing 1/(c ε₀) is the vacuum
    # impedance, 376.73, and it made every trap light shift ~380x too small.
    # This was unexercised until `add_light_shift!` existed: StarkShiftAC shipped
    # in v0.1.0 but was never constructed anywhere.
    #
    # The magnitude lives in `H` (see the constructor); what varies per step is
    # the intensity envelope at the atom's position, which is in [0,1]. Keeping
    # the coefficient O(1) is what lets `spectral_spec` bound this term.
    I0 = peak_intensity(f.beam)
    f._coeff[] = I0 == 0 ? 0.0 : f._amplitude[] * intensity(f.beam, f.atom.x) / I0
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
    rate::ComplexF64                  # strength baked into H, as GlobalCoupling
    _coeff::Base.RefValue{ComplexF64}
    function Interaction(b, atoms::Pair, transition1, transition2, value)
        H = Op(b, atoms, transition1, transition2, value)
        new{typeof(atoms[1])}(atoms..., H, ComplexF64(value), Ref(Complex(1.0)))
    end
end

"""
    update!(d::Interaction, step)

No-op update for static pairwise interactions. The operator is fixed
and its coefficient is assumed constant unless modified externally.
"""
function update!(d::Interaction{A}, ::Real) where A
end

"""
    VdWInteraction(b, atoms, transition1, transition2, C6; V_cap=Inf)

Distance-dependent van der Waals interaction V(r) = C6 / r⁶ between two atoms.

The underlying `Op` is built with a unit coefficient (1.0); the scalar `_coeff`
is updated every solver timestep from the instantaneous inter-atom separation.
`C6` is in rad/s·m⁶ (ħ = 1 units).

If `V_cap` is finite, the interaction's magnitude is clamped to `|V_cap|`,
for either sign of `C6`.
"""
mutable struct VdWInteraction{A} <: AbstractField
    atom1::A
    atom2::A
    H::Op               # unit projector |rr⟩⟨rr|; _coeff carries C6/r^6
    _coeff::Base.RefValue{ComplexF64}
    C6::Float64         # rad/s·m^6
    V_cap::Float64      # maximum interaction strength (rad/s); Inf = no cap
    _amplitude::Base.RefValue{ComplexF64}   # commanded amplitude (Pulse/On/Off)
    function VdWInteraction(b, atoms::Pair, transition1, transition2, C6::Float64;
                            V_cap::Float64 = Inf)
        H = Op(b, atoms, transition1, transition2, 1.0)
        return new{typeof(atoms[1])}(atoms[1], atoms[2], H, Ref(ComplexF64(C6)), C6, V_cap,
                                     Ref(ComplexF64(1.0)))
    end
end

"""
    update!(d::VdWInteraction, t)

Recompute the van der Waals coefficient from the current inter-atom separation:

    V = C6 / r⁶

with its magnitude clamped to `|V_cap|` when finite. Called each solver
timestep after `fclassical!` has updated positions.
"""
function update!(d::VdWInteraction, ::Real)
    x1 = d.atom1.x;  x2 = d.atom2.x
    dx = x2[1] - x1[1];  dy = x2[2] - x1[2];  dz = x2[3] - x1[3]
    r2 = dx*dx + dy*dy + dz*dz
    r6 = r2 * r2 * r2
    V  = d.C6 / r6
    # Clamp the MAGNITUDE. `min(V, V_cap)` pinned an attractive interaction
    # (C6 < 0, so a negative default cap) at the cap for every separation.
    cap = abs(d.V_cap)
    d._coeff[] = d._amplitude[] * (isfinite(cap) ? clamp(V, -cap, cap) : V)
    return nothing
end

"""
    envelope(f) -> Base.RefValue{ComplexF64}

Where an instruction -- `Pulse`, `On`, `Off`, a shaped amplitude -- writes the
commanded amplitude of field `f`.

For most fields that is `_coeff` itself. A field whose `update!` recomputes
`_coeff` from geometry every step -- a position-dependent Rabi rate, a local
trap intensity, a distance-dependent interaction -- would overwrite anything
written there, leaving it permanently on. Those keep the commanded amplitude in
`_amplitude`, and `update!` multiplies the two.
"""
envelope(f) = f._coeff
envelope(f::Union{PlanarCoupling, GaussianCoupling, StarkShiftAC, VdWInteraction}) =
    f._amplitude

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

Internal fields: `_P` and `_pidx` cache populations and basis-dependent index
mappings; `_F`/`_Fvalid` the velocity-Verlet force of the previous step; `_Eb` the
MCWF radiation-pressure excited-branch momentum (see `RadiationPressure`), which
carries across instructions and is reset per shot.
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
    _F::Vector{Float64}           # force cached from the previous step (velocity Verlet)
    _Fvalid::Bool                 # false until _F holds a force for the current x
    _Eb::Vector{Float64}          # MCWF radiation pressure: excited-branch momentum/ħ
                                  # × population, m⁻¹ (see `RadiationPressure`)

    function NLevelAtom(n;
                        x       = [0.0, 0.0, 0.0],
                        v       = [0.0, 0.0, 0.0],
                        m       = 1amu,
                        alphas  = Dict(),
                        lambdas = Dict())
        new(n, x, v, m, alphas, lambdas, zeros(n), [[0]], zeros(3), false, zeros(3))
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
    b._P      = copy(a._P)
    b._pidx   = [copy(idx) for idx in a._pidx]
    b._F      = copy(a._F)
    b._Fvalid = a._Fvalid
    b._Eb     = copy(a._Eb)
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
