# # [[4,2,2]] Shared Setup — Ytterbium-171 2×2 Tweezer Array
#
# Shared parameters, levels, atoms, and system builder used by:
#   422_CZ_gate.jl    — CZ gate characterisation and Rydberg population figure
#   422_Bell_state.jl — logical Bell state encoding and fidelity analysis
#
# ## Qubit encoding
#   |0⟩ = |mF = −½⟩,  |1⟩ = |mF = +½⟩  in the ³P₀ nuclear spin manifold (F=1/2)
#
# ## Rydberg gate mechanism
# σ⁻ laser at 302 nm drives |1, mF=+½⟩ → |r, mF=−½⟩ (resonant).
# σ⁺ drives |0, mF=−½⟩ → |r, mF=+½⟩, suppressed by Zeeman offset.
# |0⟩ is therefore effectively dark to the Rydberg laser.
#
# ## Array layout
#   atom 3 ── atom 4     (y = +ysep/2, fixed)
#      │          │
#   atom 1 ── atom 2     (y = −ysep/2; shuttled to y = ysep/2−ysep_gate for gate)
#
# Blockaded pairs: (1,3) and (2,4) — ysep_gate separation, V ≫ Ω_R
# Crosstalk pairs: (1,2),(3,4),(1,4),(2,3) — xsep apart, V ≪ Ω_R

using AtomTwin
using AtomTwin.Dynamiq.Units
using Printf
using LinearAlgebra
using Statistics
using Plots
using JSON3

# ── Physical constants ────────────────────────────────────────────────────────
# From M. Peper et al., arXiv:2406.01482

const g_eff = 2.357                    # effective Rydberg g-factor
const d_eff = 0.001071 * e * a0        # effective dipole moment
const C6    = 2π * 34 * (GHz * µm^6)  # van der Waals coefficient
const k_ryd = [1, 0, 0]               # Rydberg beam along x (parallel to B)

# ── Level definitions ─────────────────────────────────────────────────────────

# ³P₀ nuclear-spin qubit (F=1/2, J=0 so electronic Zeeman vanishes)
const metastable = HyperfineManifold(1//2, 0; label = "3P0",     g_F = -0.00067875)
const qubit0     = metastable[-1//2]   # |0⟩ = |mF = −½⟩
const qubit1     = metastable[+1//2]   # |1⟩ = |mF = +½⟩

# Effective 54.28 ³S₁ Rydberg manifold (F=1/2, g_F = g_eff)
const rydberg = HyperfineManifold(1//2, 0; label = "54.28S1", g_F = g_eff)
const r_level = rydberg[-1//2]         # |r⟩: resonantly driven sublevel
const leakage = rydberg[1//2]

# Display labels for basis indices (used in state printouts)
const level_label = Dict(1 => '0', 2 => '1', 3 => 'r', 4 => 'R')

# ── Single-qubit gate parameters ──────────────────────────────────────────────

const Ω_sq  = 2π * 1.0e6   # single-qubit Rabi frequency
const T_pi  = π / Ω_sq     # π-pulse duration
const Δ_det = 2π * 1.0e6   # virtual-Z detuning rate
const θ_cz  = 2.1663       # single-atom CZ phase correction (rad)

# ── Time-optimal CZ pulse envelope ───────────────────────────────────────────
# Jandura & Pupillo, Quantum 6, 712 (2022);
# Figshare data repository https://doi.org/10.6084/m9.figshare.19658427
# ryd_amplitudes[k] = exp(-i·φ_k); interpolated over T_gate.

const ryd_amplitudes = cis.(-[
 0.0,                     -0.005206856121121928,  -0.0155559081588863,
-0.030828542869261066,   -0.051042137300428614,  -0.07574482851642761,
-0.1048009448116215,     -0.13794753952123806,   -0.17488957360068502,
-0.21514299761372846,    -0.25829462382360313,   -0.30417810115918875,
-0.35201942855829677,    -0.4015948481500137,    -0.4522888352104099,
-0.5037067145561759,     -0.5552900188064229,    -0.6065589530465751,
-0.6568415861004193,     -0.7060185052363823,    -0.7531910691546025,
-0.7981445740157893,     -0.8403461724409429,    -0.8794553029025127,
-0.9150662040602274,     -0.9468446189385619,    -0.9743284440729487,
-0.9975489872184375,     -1.0160987089453475,    -1.029642700606746,
-1.0383062473987315,     -1.0417006803754092,    -1.0399118930114524,
-1.0327347627560062,     -1.0203361779097824,    -1.002686005093267,
-0.9798861336804422,     -0.9519629010811237,    -0.9191038043524484,
-0.8814984866779377,     -0.8395190381737427,    -0.7932981489076697,
-0.7432407243784729,     -0.6897398457666474,    -0.6331560654109526,
-0.5739816958670872,     -0.5126482772453329,    -0.4497300920383065,
-0.38568509830202335,    -0.3210642997403219,    -0.2564694519674281,
-0.1924873610272202,     -0.12944107809028071,   -0.06830197102324309,
-0.009260420418868875,    0.047524267108279505,   0.100836667323695,
 0.1510009334288679,      0.19719138507393674,    0.23933227088605313,
 0.2766423479946386,      0.3095309185833254,     0.33761622888204756,
 0.36045944587159584,     0.37800058135263825,    0.3904139328835502,
 0.3974168865616853,      0.3995241922589122,     0.3959956408371662,
 0.38727900721675057,     0.373673929127344,      0.3552545364430505,
 0.3321644791769166,      0.3046526162740927,     0.2728567208477478,
 0.23723319128393816,     0.19814116654220848,    0.1560092595645891,
 0.1109898953575309,      0.06374067196625166,    0.014789610048091206,
-0.03570939807395257,    -0.08696066452576412,   -0.1385803264865273,
-0.18993522897035753,    -0.24072419789183208,   -0.2903863490741848,
-0.338279019260814,      -0.38413205260111827,   -0.42727249780106685,
-0.4675547727608495,     -0.5044079984494114,    -0.5374066870356393,
-0.5666190497649339,     -0.591244574420264,     -0.6112628636344262,
-0.6266833782262359,     -0.6369880510447442,    -0.6421179524296483
])

# ── Experimental parameters ───────────────────────────────────────────────────

const temperature = 3µK
const xsep        = 10µm
const ysep        = 10µm   # storage row separation (initial geometry)
const ysep_gate   = 3µm    # gate separation (blockade regime)
const P_ryd       = 20mW
const w_ryd       = 12µm
const Γ_ryd       = 1/(2π * 56µs)
const pol_ryd     = [0.0, 1.0, 0.0]   # y-linear: equal σ⁺/σ⁻
const B_vec       = [4.88G, 0.0, 0.0]
const dt          = 0.05e-9            # simulation timestep

# ── Derived quantities ────────────────────────────────────────────────────────

const Δ_ryd  = -1//2 * g_eff * µB * norm(B_vec) / hbar
const T_gate = 7.612 / (2π * 2.501MHz)
const T_move = 10µs/µm * (ysep - ysep_gate) + 1µs

# ── Parameter summary ─────────────────────────────────────────────────────────

let B_MHz      = g_eff * µB/h * norm(B_vec) * 1e-6
    V_blockade = C6 / ysep_gate^6
    println("─────────────────────────────────────────────────────────────")
    println("[[4,2,2]] simulation — experimental parameters")
    println("─────────────────────────────────────────────────────────────")
    @printf "  Temperature            : %.0f uK\n"             temperature/µK
    @printf "  Array spacing (x)      : %.0f um\n"             xsep/µm
    @printf "  Row sep (storage/gate) : %.0f um / %.0f um\n"   ysep/µm ysep_gate/µm
    @printf "  Move duration          : %.1f us\n"             T_move/µs
    @printf "  Rydberg beam           : 302 nm, w=%.0f um, P=%.0f mW\n" w_ryd/µm P_ryd/mW
    @printf "  Zeeman splitting       : %.1f MHz  (B = %.2f G)\n" B_MHz norm(B_vec)*1e4
    @printf "  Laser detuning D_ryd   : %.1f MHz\n"            Δ_ryd/(2π*1e6)
    @printf "  VdW at gate sep        : %.0f MHz\n"            abs(V_blockade)/(2π*1e6)
    @printf "  Gate time T_gate       : %.2f us\n"             T_gate/µs
    println("─────────────────────────────────────────────────────────────")
end

# ── System builder ────────────────────────────────────────────────────────────
#
# Builds the full four-atom system from the global experimental parameters.
# Error sources are independently switchable for error budget analysis:
#
#   doppler_dephasing    — finite temperature: Maxwell-Boltzmann velocity sampling
#   beam_inhomogeneity   — position-dependent Rabi freq (GaussianBeam vs PlanarBeam)
#   spontaneous_decay    — Rydberg spontaneous emission at rate Γ_ryd
#   finite_blockade      — distance-dependent VdW C6/r^6 (vs constant at ysep_gate)
#   interpair_crosstalk  — include inter-column VdW interactions
#   zeeman_dephasing     — Zeeman detunings on metastable and Rydberg manifolds
#   sigma_p_coupling     — include σ⁺ and π Rabi components (off-resonant |0⟩ coupling)
#
# y_start sets the initial row separation; use ysep for Bell state (atoms start
# at storage positions) or ysep_gate for CZ gate (atoms start at gate position).
#
# Returns: (sys, ryd, atoms, tweezer, T_gate, T_move)

function build_system(;
    doppler_dephasing   = true,
    beam_inhomogeneity  = true,
    spontaneous_decay   = true,
    finite_blockade     = true,
    interpair_crosstalk = true,
    zeeman_dephasing    = true,
    sigma_p_coupling    = true,
    y_start             = ysep_gate,
)
    # ── Step 1: atoms and tweezer ────────────────────────────────────────────
    tweezer = TweezerArray(
        λ         = 759nm,
        w0        = 1µm,
        P_total   = 100mW,
        row_freqs = y_start/µm  * [-0.5, +0.5] * MHz,
        col_freqs = xsep/µm     * [-0.5, +0.5] * MHz,
        dx        = 1µm/MHz,
        dy        = 1µm/MHz,
    )
    _temperature = doppler_dephasing ? temperature : 0µK
    atoms = [Ytterbium171Atom(;
                 levels = [metastable..., rydberg...],
                 x_init = copy(getposition(t)),
                 v_init = maxwellboltzmann(T = _temperature))
             for t in tweezer]

    # ── Step 2: system assembly ──────────────────────────────────────────────
    sys = System(atoms, [tweezer])

    for atom in atoms
        if zeeman_dephasing
            add_zeeman_detunings!(sys, atom, metastable, B = norm(B_vec))
        end
        add_zeeman_detunings!(sys, atom, rydberg, B = norm(B_vec), delta = Δ_ryd)
    end

    if finite_blockade
        add_vdwinteraction!(sys, (atoms[1], atoms[3]),
                            (r_level, r_level) => (r_level, r_level), C6)
        add_vdwinteraction!(sys, (atoms[2], atoms[4]),
                            (r_level, r_level) => (r_level, r_level), C6)
    else
        # Constant blockade: fix strength at the gate separation
        V0 = 64 * C6 / ysep_gate^6
        add_interaction!(sys, (atoms[1], atoms[3]),
                         (r_level, r_level) => (r_level, r_level), V0)
        add_interaction!(sys, (atoms[2], atoms[4]),
                         (r_level, r_level) => (r_level, r_level), V0)
    end

    if interpair_crosstalk
        add_vdwinteraction!(sys, (atoms[1], atoms[2]),
                            (r_level, r_level) => (r_level, r_level), C6)
        add_vdwinteraction!(sys, (atoms[3], atoms[2]),
                            (r_level, r_level) => (r_level, r_level), C6)
        add_vdwinteraction!(sys, (atoms[1], atoms[4]),
                            (r_level, r_level) => (r_level, r_level), C6)
        add_vdwinteraction!(sys, (atoms[3], atoms[4]),
                            (r_level, r_level) => (r_level, r_level), C6)
    end

    beam = if beam_inhomogeneity
        GeneralGaussianBeam(302nm, w_ryd, w_ryd, P_ryd, k_ryd, pol_ryd)
    else
        PlanarBeam(302nm, 2*P_ryd/(π*w_ryd^2), k_ryd, pol_ryd)
    end

    Ω_π, Ω_p, Ω_m = rabi_frequencies(beam; q_axis = B_vec, d_red = d_eff)
    _Ω_π, _Ω_p, _Ω_m = sigma_p_coupling ? (Ω_π, Ω_p, Ω_m) : (0.0, 0.0, Ω_m)

    ryd = vcat([add_coupling!(sys, atom, metastable => rydberg;
                              beam = beam, Ω_π = _Ω_π, Ω_p = _Ω_p, Ω_m = _Ω_m,
                              active = false)
                for atom in atoms]...)

    if spontaneous_decay
        for atom in atoms
            add_decay!(sys, atom, rydberg => metastable, Γ_ryd)
        end
    end

    return (sys = sys, ryd = ryd, atoms = atoms, tweezer = tweezer,
            T_gate = T_gate, T_move = T_move)
end
