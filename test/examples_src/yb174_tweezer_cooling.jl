# # Laser cooling in a tweezer: Doppler and Sisyphus
#
# A single \\(^{174}\\)Yb atom in an optical tweezer is laser-cooled on the narrow
# \\(^1S_0 \\leftrightarrow {}^3P_1\\) line (556 nm, \\(\\Gamma = 2\\pi \\times
# 182\\,\\mathrm{kHz}\\)) by a pair of counter-propagating \\(\\sigma^+/\\sigma^-\\)
# beams along \\(x\\). The atom's motion is semiclassical; every absorbed and
# emitted photon kicks it (radiation pressure and recoil), so the temperature it
# settles at is set by the competition between friction and photon-recoil
# heating.
#
# Two traps are compared:
#
# 1. **Magic.** Ground and excited state feel the same trap,
#    \\(\\beta = \\alpha(^3P_1)/\\alpha(^1S_0) = 1\\). The transition frequency
#    does not depend on where the atom is, and the atom cools to the **Doppler
#    limit**. For this one-dimensional geometry, with the \\(\\sigma\\) emission
#    pattern putting 2/5 of the recoil along \\(x\\), it is
#    \\(k_B T_x = \\tfrac{7}{10}\\,\\hbar\\Gamma/2\\) at \\(\\delta = -\\Gamma/2\\).
# 2. **State-dependent, \\(\\beta > 1\\).** The excited state is trapped more
#    deeply, so the transition is red-shifted most at the trap centre. With the
#    laser tuned just red of that trap-bottom resonance, the atom is excited
#    near the centre and decays further out, where the transition is bluer: each
#    photon it scatters carries away potential energy. This **Sisyphus**
#    mechanism adds to Doppler friction, and the atom gets colder than the
#    Doppler limit.
#
# The trap is \\(\\approx 1\\) mK deep (\\(U_0/h = 20\\) MHz), so the radial
# trap frequency, \\(2\\pi \\times 68\\) kHz, is a third of \\(\\Gamma\\):
# narrow-line cooling, but not yet resolved sidebands.

# Internal notes for test scripts (not included in docs/examples):         #src
# - Files in `test/examples_src/` are run as tests and also used to        #src
#   generate docs and runnable examples.                                   #src
# - Lines containing `#src` are removed by `make.jl` when generating       #src
#   docs/examples, but are present when running tests.                     #src

using AtomTwin
using AtomTwin.Units: MHz, kHz, µs, ms, nm, µm, µK, hbar, kb, amu
using Printf
using Random
using Statistics
if false      #src
using Plots
end           #src

# ## Parameters

Γ      = 2π * 182.4kHz       # ³P₁ linewidth
λ_cool = 556nm
λ_trap = 532nm
w0     = 1.0µm               # tweezer waist
U0     = 20MHz               # trap depth of ¹S₀, in Hz (h × 20 MHz ≈ 1 mK)
s0     = 0.05                # saturation parameter of each cooling beam
T_run  = 0.8ms
shots  = 192

m_Yb = 174amu
T_D  = hbar * Γ / (2kb)      # the textbook Doppler limit, ħΓ/2k_B = 4.38 µK

descriptor = "Yb-174 tweezer cooling: U0/h = $(U0/1e6) MHz, s = $s0, $shots shots" #src

# ## Trap
#
# The tweezer light shift of ¹S₀ comes from its polarizability model. The
# excited state is given a polarizability `β` times larger, through a line-free
# model: a measured (or here, chosen) light shift per intensity.

ls_g = light_shift_coeff_Hz_per_Wcm2(Ytterbium174Atom(; levels = [Level("1S0")]),
                                     l"1S0", λ_trap / nm) * 1e-4      # Hz/(W/m²), < 0
I0   = U0 / abs(ls_g)                                                  # peak intensity, W/m²
P    = I0 * π * w0^2 / 2
ωx   = sqrt(4 * U0 * 2π * hbar / (m_Yb * w0^2))                        # radial trap frequency
@printf("trap: P = %.1f mW, ω_x/2π = %.1f kHz\n", P * 1e3, ωx / 2π / 1e3)

# ## System
#
# `δ` is the laser detuning from the **trap-bottom** resonance. The trap shifts
# that resonance by \\((\\beta - 1)\\,U_g(0)\\); the detuning term adds it back so
# that `δ` keeps its meaning for every β.

function cool(β, δ; seed = 1)
    gm = HyperfineManifold(0//1, 0; label = "¹S₀", term = l"1S0", g_F = 0.0)
    em = HyperfineManifold(1//1, 1; label = "³P₁", term = l"3P1", g_F = 1.5)
    g  = gm[0]
    e_model = PolarizabilityModel("3P1", NamedTuple[]; J = 1//1, offset_Hz_per_Wm2 = β * ls_g)
    yb = Ytterbium174Atom(; levels = [gm..., em...], v_init = maxwellboltzmann(T = 6µK),
                          polarizabilities = Dict("3P1" => e_model))

    sys = System(yb, GaussianBeam(λ = λ_trap, w0 = w0, P = P))
    add_quantization_axis!(sys, [1.0, 0.0, 0.0])          # along the cooling beams

    ## σ⁺ travelling +x drives m = +1, σ⁻ travelling −x drives m = −1.
    Ω  = Γ * sqrt(s0 / 2)
    σp = add_coupling!(sys, yb, gm => em, 0.0, Ω, 0.0;
                       beam = PlanarBeam(λ_cool, 1.0, [1.0, 0, 0], [0.0, 1, 1im]), active = false)
    σm = add_coupling!(sys, yb, gm => em, 0.0, 0.0, Ω;
                       beam = PlanarBeam(λ_cool, 1.0, [-1.0, 0, 0], [0.0, 1, -1im]), active = false)

    Δ_trap = 2π * (β - 1) * ls_g * I0                     # trap-bottom shift (rad/s)
    for lv in em.levels
        add_detuning!(sys, yb, lv, δ + Δ_trap)
    end

    ## Spontaneous emission with recoil along each transition's dipole pattern.
    ## A photon counter bound to all three channels records every scattered photon.
    pd = PhotoDetectorSpec(name = "photons")
    add_detector!(sys, pd)
    add_decay!(sys, yb, em[1]  => g, Γ; λ = λ_cool, dipole = [0.0, 1, 1im], clicks = pd)
    add_decay!(sys, yb, em[-1] => g, Γ; λ = λ_cool, dipole = [0.0, 1, -1im], clicks = pd)
    add_decay!(sys, yb, em[0]  => g, Γ; λ = λ_cool, dipole = [1.0, 0, 0], clicks = pd)

    add_detector!(sys, MotionDetectorSpec(yb; dims = [1, 2, 3], name = "r"))
    seq = Sequence(1µs; downsample = 10)
    @sequence seq begin
        Parallel([Pulse(σp, T_run), Pulse(σm, T_run)])
    end
    return play(sys, seq; initial_state = g, shots = shots, rng = MersenneTwister(seed))
end

# ## Temperature
#
# Equipartition in the harmonic trap: \\(k_B T_x = m\\,\\omega_x^2 \\langle x^2
# \\rangle\\), over the atoms still in the tweezer, averaged over the second half
# of the run. Only \\(x\\) is cooled; \\(y\\) and \\(z\\) slowly heat from
# emission recoil, which the Gaussian trap feeds back into \\(x\\), so the run is
# kept short.

function T_x(out; from = 0.5)
    r  = out.detectors["r"]
    ts = out.times
    in_trap = [hypot(r[end, 1, k], r[end, 2, k]) < 2w0 && abs(r[end, 3, k]) < 10µm
               for k in axes(r, 3)]
    rows = findall(≥(from * ts[end]), ts)
    return m_Yb * ωx^2 * mean(abs2, r[rows, 1, in_trap]) / kb
end

function T_trace(out)
    r = out.detectors["r"]
    [m_Yb * ωx^2 * mean(abs2, r[k, 1, :]) / kb for k in axes(r, 1)]
end

# ## Run
#
# Magic trap at the Doppler optimum \\(\\delta = -\\Gamma/2\\), and a
# state-dependent trap with \\(\\beta = 2\\) and 3 (Yb's ³P₁ \\(m = \\pm1\\)
# states have \\(\\beta \\approx 1.8\\) in a 767 nm tweezer), tuned
# \\(0.3\\,\\Gamma\\) red of the trap-bottom resonance.

cases = [(label = "magic, β = 1, δ = −Γ/2", β = 1.0, δ = -0.5Γ),
         (label = "β = 2,     δ = −0.3Γ",  β = 2.0, δ = -0.3Γ),
         (label = "β = 3,     δ = −0.3Γ",  β = 3.0, δ = -0.3Γ)]

runtime = @elapsed begin                                                  #src
outs = [cool(c.β, c.δ) for c in cases]
end                                                                       #src

Ts = [T_x(o) for o in outs]
## Photons scattered per atom over the run, from the photon counter.
Rs = [mean(sum(o.detectors["photons"]; dims = 1)) / T_run for o in outs]
for (c, T, R) in zip(cases, Ts, Rs)
    @printf("%-24s  T_x = %.2f µK = %.2f T_D   (k_B T_x / ħω_x = %.1f)   scatters %.1f kHz\n",
            c.label, T / µK, T / T_D, kb * T / (hbar * ωx), R / 1e3)
end
@printf("Doppler theory for this geometry: %.2f µK; low-saturation scattering rate at δ = −Γ/2: %.1f kHz\n",
        0.7T_D / µK, (Γ / 2) * 2s0 / (1 + 2s0 + 1) / 1e3)
@printf("colder than the magic trap by %.2f× (β = 2) and %.2f× (β = 3)\n", Ts[1]/Ts[2], Ts[1]/Ts[3])

checksum_data = Ts ./ µK                                                  #src

# The magic trap settles near the one-dimensional Doppler limit; the residual
# excess is the heat leaking in from the uncooled axes. The state-dependent
# traps go below it: about 1.5× colder at β = 2 and nearly 2× at β = 3.
#
# The photon counter shows what that costs. The magic trap scatters at the
# low-saturation rate, \\(R = \\tfrac{\\Gamma}{2}\\,s/(1 + s + 4\\delta^2/\\Gamma^2)\\)
# with \\(s = 2 s_0\\) for the two beams. The Sisyphus-cooled atoms are colder
# *and* scatter fewer photons: off-centre the differential light shift detunes
# the transition from the laser. For imaging that means more cooling per
# scattered photon.

# !!! note "Semiclassical motion"
#     The atom's centre of mass is a classical particle; photon recoil is
#     included, motional quantisation is not. That is accurate while
#     \\(k_B T \\gg \\hbar\\omega\\). In this deep trap even the Doppler limit
#     is \\(k_B T \\approx \\hbar\\omega_x\\), and the Sisyphus-cooled atoms go
#     below it, where real atoms occupy only the lowest few motional states and
#     sideband structure matters. The comparison between the traps is the
#     result; the absolute temperatures there are indicative.

if false                                                    #src
plt = Plots.plot(xlabel = "time (ms)", ylabel = "T_x (µK)", legend = :topright,
                 title = "¹⁷⁴Yb on 556 nm in a 1 mK tweezer")
for (c, o) in zip(cases, outs)
    Plots.plot!(plt, o.times ./ ms, T_trace(o) ./ µK; lw = 2, label = c.label)
end
Plots.hline!(plt, [0.7T_D / µK]; ls = :dash, color = :gray, label = "1D Doppler limit")
plt
end                                                         #src
