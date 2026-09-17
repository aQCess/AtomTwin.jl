# # Light-shift spectroscopy of Yb-174 in a non-magic tweezer
#
# A single \\(^{174}\\)Yb atom is held in a 767 nm optical tweezer and probed on
# the narrow \\(^1S_0 \\leftrightarrow {}^3P_1\\) intercombination line (556 nm,
# \\(\\Gamma = 2\\pi \\times 182\\,\\mathrm{kHz}\\)).
#
# 767 nm is **not** magic for this transition, so the trap shifts the excited
# state relative to the ground state and the resonance moves with trap depth.
# Scanning the probe detuning maps out that shift — light-shift spectroscopy.
#
# Three features appear in one scan:
#
# 1. **Three resonances.** \\(^3P_1\\) has \\(J = 1\\) and \\(^{174}\\)Yb is
#    spin-zero, so \\(F = J = 1\\) and the excited state is a triplet
#    \\(m_F = 0, \\pm 1\\). A magnetic field splits them by
#    \\(g_J \\mu_B B / h = 2.1\\,\\mathrm{MHz/G}\\).
# 2. **A common light shift.** All three sit several MHz from the bare
#    resonance, because the trap shifts \\(^3P_1\\) and \\(^1S_0\\) differently.
# 3. **Thermal broadening.** An atom with finite temperature samples a range of
#    trap intensities, so each line is inhomogeneously broadened well beyond its
#    182 kHz natural width.
#
# The polarisation is set to the geometric magic angle
# \\(\\arccos(1/\\sqrt 3) \\approx 54.74^\\circ\\), where the **tensor** light
# shift vanishes. That is deliberate: it leaves the three sublevels degenerate in
# the trap so the splitting seen is purely Zeeman. Tilting away from it — see the
# sweep at the end — reintroduces a tensor splitting of order 10 MHz.

# Internal notes for test scripts (not included in docs/examples):         #src
# - Files in `test/examples_src/` are run as tests and also used to        #src
#   generate docs and runnable examples.                                   #src
# - Lines containing `#src` are removed by `make.jl` when generating       #src
#   docs/examples, but are present when running tests.                     #src

using AtomTwin
using AtomTwin.Units: MHz, kHz, GHz, Hz, s, ms, µs, ns, m, cm, mm, µm, nm, mW, µW, W, µK, mK, nK, K, hbar, kb, c, a0, amu
using Printf
using Statistics
if false      #src
using Plots
end           #src

# ## Parameters

λ_trap   = 767nm        # tweezer wavelength — NOT magic for ¹S₀–³P₁
w0       = 1.0µm        # tweezer waist
P_trap   = 50mW         # deep enough that a hot atom stays bound
θ_pol    = 54.7356      # polarisation angle from B: the tensor-free magic angle

B_field  = 2.0Units.G   # splits the m_F triplet by ≈4.2 MHz
T_atom   = 20µK         # 1.5% of the trap depth

Ω_probe  = 2π * 50kHz   # < Γ, so the lines are not badly power broadened
t_probe  = 100µs
shots    = 60           # thermal average over the position spread

descriptor = "Yb-174 light-shift spectroscopy: λ=$(λ_trap*1e9) nm, B=$(B_field*1e4) G, T=$(T_atom*1e6) µK" #src

# ## System
#
# The ground state is a single level; the excited state is a hyperfine manifold
# with `F = 1`, which supplies the three `m_F` sublevels. The `term` arguments
# bind each to its polarizability model — that is what makes the trap shift the
# two states differently, and it is the only place the atomic data enters.

# ¹S₀ is F = 0 (a single state); ³P₁ is F = 1 (the m_F triplet). Both are
# manifolds so the coupling carries the proper dipole matrix elements.
gm = HyperfineManifold(0//1, 0; label = "¹S₀", term = l"1S0", g_F = 0.0)
e  = HyperfineManifold(1//1, 1; label = "³P₁", term = l"3P1", g_F = 1.5)
g  = gm[0]                      # the single m_F = 0 ground state

yb = Ytterbium174Atom(; levels = [gm..., e...],
                      v_init  = maxwellboltzmann(T = T_atom))

# Linear polarisation at `θ_pol` to the quantisation axis, which is set by B.
trap = GaussianBeam(λ = λ_trap, w0 = w0, P = P_trap,
                    pol = [sind(θ_pol), 0.0, cosd(θ_pol)])

# ## Trap depth and the shift it produces
#
# The light shift is included automatically: the trap is part of the system and
# the atom has polarizability data, so no extra call is needed.

I0    = 2 * P_trap / (π * w0^2)
α_g   = polarizability_si(yb, l"1S0", λ_trap * 1e9)
U0    = α_g * I0 / (c * Units.ε0)                     # trap depth (J)
ω_r   = sqrt(4 * α_g * I0 / (174amu * c * Units.ε0 * w0^2))

@printf("trap depth      = %.0f µK  (%.2f MHz)\n", U0 / kb * 1e6, U0 / (2π * hbar) / 1e6)
@printf("radial freq     = %.1f kHz\n", ω_r / 2π / 1e3)
@printf("T / U₀          = %.3f\n", T_atom * kb / U0)

# The differential shift each sublevel sees at the trap centre. At the magic
# angle these coincide, which is the point.
Δ_centre = [ (light_shift_coeff_Hz_per_Wcm2(yb, l"1S0", λ_trap * 1e9) -
              light_shift_coeff_Hz_per_Wcm2(yb, l"3P1", λ_trap * 1e9;
                                            F = 1//1, mF = mF, I = 0//1,
                                            ε_z = cosd(θ_pol))) * I0 * 1e-4
             for mF in (-1//1, 0//1, 1//1) ]
@printf("light shift     = %.2f MHz (m_F = 0, ±1 degenerate at the magic angle)\n",
        Δ_centre[2] / 1e6)

# ## Spectroscopy
#
# Scan the probe detuning and record the excited-state population. Each point is
# an independent set of shots, so the thermal distribution is resampled — the
# resulting lineshape is inhomogeneously broadened, not just power broadened.

function scan_point(δ)
    sys = System(yb, trap)
    add_quantization_axis!(sys, [0.0, 0.0, 1.0])

    # Zeeman splitting of the excited triplet, plus the scanned probe detuning.
    add_zeeman_detunings!(sys, yb, e; B = B_field, delta = δ)

    # The probe carries π and σ± components, so it addresses all three m_F
    # sublevels — a π-only probe would show the m_F = 0 line alone.
    c_probe = add_coupling!(sys, yb, gm => e, Ω_probe, Ω_probe, Ω_probe;
                            active = false)
    add_decay!(sys, yb, e => gm, 2π * 182kHz)
    for (k, lv) in enumerate(e.levels)
        add_detector!(sys, PopulationDetectorSpec(yb, lv; name = "P_e$k"))
    end

    seq = Sequence(200e-9)
    @sequence seq begin
        Pulse(c_probe, t_probe)
    end
    out = play(sys, seq; initial_state = g, shots = shots)
    # total excited population = sum over the three m_F sublevels
    sum(mean(out.detectors["P_e$k"][end, :]) for k in 1:length(e.levels))
end

# `add_zeeman_detunings!`'s `delta` is measured from the **bare** (untrapped)
# transition, so the scan axis needs no offset and two reference positions fall
# on it naturally:
#
#   * **0 MHz** — where the line would sit with no trap at all.
#   * **Δ_peak** — the shift at the trap centre, the largest an atom can feel.
#
# A thermal atom spends its time away from the centre, where the intensity is
# lower, so its resonance sits *below* Δ_peak. The Zeeman triplet straddles it.

Δ_peak   = Δ_centre[2] / 1e6                     # MHz, at the trap centre
δ_span   = 2π * 7MHz                             # ≈1.7 Zeeman splittings either side
δs       = range(2π * Δ_peak * 1e6 - δ_span,
                 2π * Δ_peak * 1e6 + δ_span; length = 61)   #src
if false                                                    #src
δs       = range(2π * Δ_peak * 1e6 - δ_span,
                 2π * Δ_peak * 1e6 + δ_span; length = 161)
end                                                         #src

runtime = @elapsed begin                          #src
spectrum = [scan_point(δ) for δ in δs]
end                                               #src
checksum_data = spectrum                          #src

# `δ` already reads as detuning from the bare transition.
ν_abs = δs ./ 2π ./ 1e6

# ## Results

peak = maximum(spectrum)
@printf("\npeak excitation  = %.3f\n", peak)
@printf("\nreference positions (detuning from the BARE transition):\n")
@printf("   no light shift        = %+8.3f MHz\n", 0.0)
@printf("   peak (trap centre)    = %+8.3f MHz\n", Δ_peak)

# Locate the three resonances: local maxima above half the peak.
peaks = Int[]
for i in 2:length(spectrum)-1
    if spectrum[i] > spectrum[i-1] && spectrum[i] >= spectrum[i+1] &&
       spectrum[i] > 0.4 * peak
        push!(peaks, i)
    end
end
# Parabolic interpolation on each peak, so the line centre is not quantised to
# the scan grid (0.35 MHz here).
function refine(i)
    (i == 1 || i == length(spectrum)) && return ν_abs[i]
    y1, y2, y3 = spectrum[i-1], spectrum[i], spectrum[i+1]
    d = y1 - 2y2 + y3
    d == 0 && return ν_abs[i]
    ν_abs[i] + 0.5 * (y1 - y3) / d * (ν_abs[2] - ν_abs[1])
end
centres = [refine(i) for i in peaks]

@printf("\nresonances found = %d\n", length(peaks))
for (i, ν) in zip(peaks, centres)
    @printf("   %+8.3f MHz   P_e = %.3f   (%.2f MHz from the peak shift)\n",
            ν, spectrum[i], ν - Δ_peak)
end

# The splitting between adjacent lines is the Zeeman shift, g_J µ_B B / h.
if length(peaks) >= 2
    splits = diff(centres)
    @printf("\nmean splitting   = %.3f MHz   (expected %.3f MHz at %.1f G)\n",
            mean(abs.(splits)), 1.5 * 1.39962e10 * B_field / 1e6, B_field * 1e4)
end

# A compact text rendering of the spectrum with both markers.
@printf("\n%10s  %s\n", "ν (MHz)", "excitation")
for (ν, v) in zip(ν_abs, spectrum)
    mark = abs(ν) < 0.15 ? " <- no shift" :
           abs(ν - Δ_peak) < 0.15 ? " <- peak shift" : ""
    v > 0.15 * peak &&
        @printf("%+10.2f  %-28s %.3f%s\n", ν, "#"^round(Int, 28v / peak), v, mark)
end

# ## The tensor shift, for contrast
#
# At the magic angle the three sublevels are degenerate in the trap. Away from
# it, the tensor light shift splits them by several MHz on its own — which is
# what makes a non-magic tweezer awkward for narrow-line imaging, and why the
# magic angle is worth finding.

@printf("\ntensor splitting m_F=±1 vs m_F=0, by polarisation angle:\n")
for θ in (0.0, 30.0, 54.7356, 80.0)
    d0 = light_shift_coeff_Hz_per_Wcm2(yb, l"3P1", λ_trap * 1e9;
                                       F = 1//1, mF = 0//1, I = 0//1, ε_z = cosd(θ))
    d1 = light_shift_coeff_Hz_per_Wcm2(yb, l"3P1", λ_trap * 1e9;
                                       F = 1//1, mF = 1//1, I = 0//1, ε_z = cosd(θ))
    @printf("   θ = %6.2f°  ->  %+7.2f MHz\n", θ, (d1 - d0) * I0 * 1e-4 / 1e6)
end

# !!! note "Accuracy"
#     The ³P₁ polarizability model is empirical and good to a few nm in magic
#     wavelength (a few degrees in magic angle) — see the docstring for
#     [`YB174_POLARIZABILITY_3P1`](@ref). The structure shown here is right; do
#     not use it to set a waveplate.

# The spectrum, with both reference positions marked: the bare transition at
# 0 MHz, and the peak (trap-centre) light shift. The Zeeman triplet straddles a
# centre that sits just inside the peak shift, because a thermal atom samples
# intensities below the maximum.
if false                                                    #src
plt = Plots.plot(
    ν_abs, spectrum;
    xlabel = "detuning from the bare ¹S₀–³P₁ transition (MHz)",
    ylabel = "excited population",
    title  = "Yb-174 in a 767 nm tweezer, T = $(T_atom*1e6) µK, B = $(B_field*1e4) G",
    label  = "spectrum",
    lw = 2, marker = :circle, ms = 2.5, legend = :topleft,
)

Plots.vline!(plt, [0.0];
             ls = :dash, color = :gray, lw = 2, label = "no light shift")
Plots.vline!(plt, [Δ_peak];
             ls = :dash, color = :red, lw = 2, label = "peak shift (trap centre)")

# The fitted line centres.
Plots.scatter!(plt, centres, spectrum[peaks];
               color = :black, ms = 6, marker = :vline, label = "line centres")

plt
end                                                         #src
