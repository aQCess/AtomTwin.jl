# Physics: Rabi and Lindblad dynamics, selection rules, beams, parametric values,
# polarizability models, light shifts, couplings.

import AtomTwin.Dynamiq

# ── A. Rabi oscillations ──────────────────────────────────────────────────────

@testset "Rabi: a π pulse flips, a 2π pulse returns" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    Ω    = 2π * 1e6
    coupling = add_coupling!(sys, atom, g => e, Ω; active = false)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))
    for (T, P) in ((π / Ω, 1.0), (2π / Ω, 0.0))
        seq = Sequence(1e-9)
        @sequence seq begin
            Pulse(coupling, T)
        end
        @test play(sys, seq; initial_state = g).detectors["P_e"][end] ≈ P atol = 0.01
    end
end

@testset "Rabi: off-resonance drive reduces contrast" begin
    # δ = Ω → Ω_eff = √2·Ω → max population = Ω²/Ω_eff² = 0.5
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    Ω = 2π * 1e6
    δ = 2π * 1e6
    coupling = add_coupling!(sys, atom, g => e, Ω; active = false)
    add_detuning!(sys, atom, e, δ)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))
    # simulate 3 full Rabi cycles at Ω_eff = √2·Ω so we capture the peak
    T_total = 3 * 2π / (√2 * Ω)
    seq = Sequence(1e-9)
    @sequence seq begin
        Pulse(coupling, T_total)
    end
    out = play(sys, seq; initial_state = g)
    @test maximum(out.detectors["P_e"]) < 0.6
end

# ── B. Lindblad dissipation ───────────────────────────────────────────────────

@testset "Lindblad: steady-state population ≈ Ω²/(Γ²+2Ω²) for Ω=Γ" begin
    # Analytical result for driven two-level system with decay Γ and no dephasing:
    # ρ_ee(∞) = Ω² / (Γ² + 2Ω²) = 1/3 when Ω = Γ
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    Ω = 2π * 1e6
    Γ = 2π * 1e6
    coupling = add_coupling!(sys, atom, g => e, Ω; active = false)
    add_decay!(sys, atom, e => g, Γ)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))
    T_total = 10 / Γ   # ~10 decay times, enough to reach steady state
    seq = Sequence(1e-9)
    @sequence seq begin
        Pulse(coupling, T_total)
    end
    out = play(sys, seq; initial_state = g, density_matrix = true)
    @test isapprox(out.detectors["P_e"][end], 1/3, atol = 0.02)
    # Steady-state population must stay in [0,1]
    @test all(x -> 0.0 ≤ x ≤ 1.0 + 1e-8, out.detectors["P_e"])
end

@testset "Lindblad: pure dephasing is trace-preserving (DM dissipator)" begin
    # Pure dephasing L = √γ|e⟩⟨e| leaves populations and trace exactly fixed and
    # damps coherences at γ/2.
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    γ = 2π * 1e6
    add_dephasing!(sys, atom, e, γ)
    add_detector!(sys, PopulationDetectorSpec(atom, g; name = "P_g"))
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))
    add_detector!(sys, CoherenceDetectorSpec(atom, g => e; name = "coh"))
    seq = Sequence(1e-9; downsample = 100)
    @sequence seq begin
        Wait(1e-6)
    end

    # |+⟩ = (|g⟩+|e⟩)/√2: population on the dephased level and a coherence to damp.
    out = play(sys, seq; initial_state = (1/√2)*g + (1/√2)*e, density_matrix = true)

    trace = out.detectors["P_g"] .+ out.detectors["P_e"]
    @test all(isapprox.(trace, 1.0; atol = 1e-9))
    # Pure dephasing leaves populations fixed (P_g = P_e = 1/2)
    @test all(isapprox.(out.detectors["P_e"], 0.5; atol = 1e-9))
    @test all(isapprox.(out.detectors["P_g"], 0.5; atol = 1e-9))
    # Coherence damps at γ/2 (standard L=√γ|e⟩⟨e| convention), not γ
    @test isapprox(abs(out.detectors["coh"][end]), 0.5 * exp(-γ * 1e-6 / 2); rtol = 1e-2)
end

# ── C. Selection rules ────────────────────────────────────────────────────────

@testset "Selection rules: |Δm| ≤ 1, and each polarization drives its own Δm" begin
    g  = HyperfineLevel(1//1, 0//1, 0//1, 1.0, "g")
    e1 = HyperfineLevel(2//1, 1//1, 1//1, 1.0, "e1")
    e2 = HyperfineLevel(2//1, 1//1, 2//1, 1.0, "e2")
    @test AtomTwin.compute_coupling_strength(g, e2, nothing, 1.0, 1.0, 1.0) == 0.0   # Δm = 2
    @test AtomTwin.compute_coupling_strength(g, e1, nothing, 0.0, 1.0, 0.0) != 0.0   # σ⁺
    @test AtomTwin.compute_coupling_strength(g, e1, nothing, 1.0, 0.0, 0.0) == 0.0   # π
end

@testset "rabi_frequency: light polarised along ε excites the dipole along ε" begin
    # J = 0 → J′ = 1: the excited superposition Σ_q Ω_q |1 q⟩ ↔ Σ_q Ω_q ê_q must be ∥ ε,
    # for π, σ± and their mixtures alike.
    yb = Ytterbium174Atom()
    g  = HyperfineLevel(0//1, 0//1, 0//1, 0.0, "g")
    es = [HyperfineLevel(1//1, 1//1, q//1, 1.0, "e$q") for q in (-1, 0, 1)]
    ê  = Dict(1 => -ComplexF64[1, im, 0] / √2, 0 => ComplexF64[0, 0, 1], -1 => ComplexF64[1, -im, 0] / √2)
    for (pol, k) in (([1.0, 0, 1] / √2, [0.0, 1, 0]), ([1.0, im, 0] / √2, [0.0, 0, 1]),
                     ([1.0, -im, 0] / √2, [0.0, 0, 1]), ([0.0, 1, 1] / √2, [1.0, 0, 0]))
        b = Dynamiq.PlanarBeam(556e-9, 1.0, k, ComplexF64.(pol))
        Ω = [AtomTwin.rabi_frequency(yb, g, e, b, zeros(3); d_red = 1e-29) for e in es]
        v = sum(Ω[i] * ê[q] for (i, q) in enumerate((-1, 0, 1)))
        @test abs(sum(conj.(pol) .* v)) / sqrt(sum(abs2, v)) ≈ 1 rtol = 1e-12
    end
end

@testset "Fine → hyperfine: σ⁺ raises m, with the ground-to-excited dipole coefficient" begin
    # J = 1/2, m_J = +1/2 → (J′ = 3/2, I = 3/2) F′ = 3, m_F′ = 3: stretched, only σ⁺
    # reaches it, with amplitude 1.
    atom = (I = 3//2,)
    g = FineLevel(1//2, 1//2)
    e = HyperfineLevel(3//1, 3//2, 3//1, 1.0, "e")
    @test AtomTwin.compute_coupling_strength(g, e, atom, 0.0, 1.0, 0.0) ≈ 1.0
    @test AtomTwin.compute_coupling_strength(g, e, atom, 0.0, 0.0, 1.0) == 0.0
    @test AtomTwin.compute_coupling_strength(g, e, atom, 1.0, 0.0, 0.0) == 0.0
    # m_J = −1/2 → F′ = 1, m_F′ = 1: the components m_J′ = −1/2 (π) and +1/2 (σ⁺) are reachable; σ⁺ alone
    # gives the m_J′ = +1/2 term, and σ⁻ nothing
    g2 = FineLevel(1//2, -1//2)
    e2 = HyperfineLevel(1//1, 3//2, 1//1, 1.0, "e2")
    expect = AtomTwin.clebschgordan(3//2, 1//2, 3//2, 1//2, 1//1, 1//1) * AtomTwin.clebschgordan(1//2, -1//2, 1, 1, 3//2, 1//2)
    @test AtomTwin.compute_coupling_strength(g2, e2, atom, 0.0, 1.0, 0.0) ≈ expect
    @test AtomTwin.compute_coupling_strength(g2, e2, atom, 0.0, 0.0, 1.0) == 0.0
end

# ── D. GaussianBeam analytical checks ────────────────────────────────────────

@testset "GaussianBeam: peak field and intensity, and the 8-waist cutoff" begin
    λ, w0, P = 1e-6, 50e-6, 1e-3
    b  = GaussianBeam(λ, w0, P)
    I0 = 2P / (π * w0^2)
    cε = AtomTwin.Units.c * AtomTwin.Units.ε0
    @test Dynamiq.intensity(b, zeros(3)) ≈ I0 rtol = 1e-10
    @test abs(Dynamiq.Efield(b, zeros(3))) ≈ sqrt(2I0 / cε) rtol = 1e-10
    @test Dynamiq.Efield(b, [10w0, 0.0, 0.0]) == 0.0 + 0.0im     # (10 w0)²/w0² ≫ 16
    @test Dynamiq.intensity(b, [10w0, 0.0, 0.0]) == 0.0
end

# ── E. Parametric values end to end (resolution itself: test_parameters.jl) ──

function _run_pi_pulse(Ω_expr)
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    coupling = add_coupling!(sys, atom, g => e, Ω_expr; active = false)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))
    Ω_val = 2π * 1e6
    T_pi  = π / Ω_val
    seq = Sequence(1e-9)
    @sequence seq begin
        Pulse(coupling, T_pi)
    end
    return play(sys, seq; initial_state = g)
end

@testset "DAG: scalar parameter override changes dynamics" begin
    Ω_param = Parameter(:Omega, 2π * 1e6)
    out_default = _run_pi_pulse(Ω_param)
    # Run with halved Ω via override — result should differ (not a π-pulse any more)
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    coupling = add_coupling!(sys, atom, g => e, Ω_param; active = false)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))
    T_pi = π / (2π * 1e6)
    seq  = Sequence(1e-9)
    @sequence seq begin
        Pulse(coupling, T_pi)
    end
    out_override = play(sys, seq; initial_state = g, Omega = 2π * 0.5e6)
    @test out_default.detectors["P_e"][end] > 0.99           # default is a π-pulse
    @test out_override.detectors["P_e"][end] < out_default.detectors["P_e"][end]  # halved Ω ≠ π-pulse
end

# ── F. GaussianPosition and MaxwellBoltzmann with parametric T ────────────────

@testset "GaussianPosition: parametric σ samples nonzero positions" begin
    using Random
    rng = Xoshiro(1)
    σ = Parameter(:σ_pos, 1e-6)
    gp = GaussianPosition(σ, σ, 0)
    # Default (build time) → zeros
    @test AtomTwin._resolve_node_default(gp) == zeros(3)
    # Sampled (compile time) → nonzero in x and y, zero in z (σz = 0)
    samples = [AtomTwin._resolve_node_value(gp, Dict{Symbol,Any}(), rng) for _ in 1:100]
    xs = [s[1] for s in samples]
    ys = [s[2] for s in samples]
    zs = [s[3] for s in samples]
    @test abs(mean(xs)) < 3e-7          # mean ≈ 0
    @test 0.5e-6 < std(xs) < 2.0e-6    # std ≈ σ = 1e-6
    @test all(iszero, zs)               # σz = 0 → always zero
    # Parameter override: σ = 2e-6 → std doubles
    samples2 = [AtomTwin._resolve_node_value(gp, Dict(:σ_pos => 2e-6), rng) for _ in 1:100]
    @test std([s[1] for s in samples2]) > std(xs)
end

@testset "MaxwellBoltzmann: parametric T samples velocities with correct σ" begin
    using Random
    rng = Xoshiro(2)
    kb_v = 1.380_649e-23
    amu  = 1.660_539_066_60e-27
    m    = 171 * amu
    T    = 10e-6   # 10 µK
    σ_v  = sqrt(kb_v * T / m)   # expected per-axis velocity std

    mb = MaxwellBoltzmann(Parameter(:T_mb, T))
    # Default → zeros
    @test AtomTwin._resolve_node_default(mb) == zeros(3)
    # Sampled velocities via initialize! on a real atom
    atom = Ytterbium171Atom(; levels = [Level("1S0")], v_init = mb)
    vs = Float64[]
    for _ in 1:500
        initialize!(atom; rng = rng)
        push!(vs, atom.inner.v[1])
    end
    @test abs(mean(vs)) < 0.3 * σ_v      # mean ≈ 0
    @test isapprox(std(vs), σ_v, rtol = 0.15)  # std ≈ √(kB T/m)
end

# ── F2. GaussianCoupling: position-dependent Rabi frequency ──────────────────

@testset "GaussianCoupling: _coeff[] = _amplitude[] × spatial_envelope" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)

    λ  = 1e-6
    w0 = 50e-6
    P  = 1e-3
    Ω0 = ComplexF64(2π * 1e6)
    beam = GaussianBeam(λ, w0, P)

    # Build GaussianCoupling directly (atom starts at origin)
    idx_g = atom.level_indices[g]
    idx_e = atom.level_indices[e]
    gc = Dynamiq.GaussianCoupling(sys.basis, atom.inner, idx_g => idx_e, beam, Ω0)

    # Set pulse amplitude to 0.5 (e.g. half-π) — verifies both factors compose
    gc._amplitude[] = ComplexF64(0.5)

    # At beam center: spatial scale = 1, so _coeff = 0.5
    Dynamiq.update!(gc, 1)
    @test isapprox(abs(gc._coeff[]), 0.5, rtol = 1e-8)

    # Move atom to 1 waist off-axis; amplitude drops by exp(-1)
    atom.inner.x .= [w0, 0.0, 0.0]
    E0  = Dynamiq.efield_scalar(beam, [0.0, 0.0, 0.0])
    E1  = Dynamiq.efield_scalar(beam, [w0,  0.0, 0.0])
    Dynamiq.update!(gc, 2)
    @test isapprox(abs(gc._coeff[]), 0.5 * abs(E1 / E0), rtol = 1e-6)

    # Off → spatial scale irrelevant; _coeff must be zero
    gc._amplitude[] = ComplexF64(0.0)
    Dynamiq.update!(gc, 3)
    @test gc._coeff[] == 0.0
end

# ── Polarizability & off-resonant scattering (Yb-171) ─────────────────────────

using AtomTwin.Units: c, hbar

@testset "atom-facing polarizability methods dispatch to the species models" begin
    for (atom, λ) in ((Ytterbium171Atom(; levels = [Level("1S0")]), 759.0),
                      (Strontium88Atom(; levels = [Level("1S0")]), 813.4))
        for st in ("1S0", "3P0")
            @test isfinite(light_shift_coeff_Hz_per_Wcm2(atom, st, λ))
            @test isfinite(polarizability_au(atom, st, λ))
            @test scattering_rate_per_Wcm2(atom, st, λ) > 0
        end
    end
end
# The models were refitted to these (Kroeze et al., arXiv:2509.04416, Table 7),
# with the clock magic wavelength weighted at its experimental uncertainty and the
# others at a 0.015 nm model floor.
@testset "Yb clock-state models reproduce the measured state-(in)dependent wavelengths" begin
    g, e = AtomTwin.YB171_POLARIZABILITY_1S0, AtomTwin.YB171_POLARIZABILITY_3P0
    function zero_near(f, λ; w = 1.0)
        lo, hi = λ - w, λ + w
        flo = f(lo)
        @assert sign(flo) != sign(f(hi))
        for _ in 1:80
            mid = (lo + hi) / 2
            sign(f(mid)) == sign(flo) ? (lo = mid) : (hi = mid)
        end
        (lo + hi) / 2
    end
    magic(λ) = zero_near(x -> polarizability_au(e, x) - polarizability_au(g, x), λ)
    @test magic(759.3560633) ≈ 759.3560633 atol = 1e-5      # clock, 394.798 267 2(16) THz
    @test magic(459.5960)    ≈ 459.5960    atol = 0.015
    @test magic(552.61074)   ≈ 552.61074   atol = 0.03
    @test zero_near(x -> polarizability_au(g, x), 553.2936) ≈ 553.2936 atol = 0.03   # ¹S₀ tune-out
    @test zero_near(x -> polarizability_au(e, x), 576.613)  ≈ 576.613  atol = 0.015  # ³P₀ tune-out
end

@testset "scattering_rate_per_Wcm2 matches independent Grimm sum" begin
    # Hand-rolled Grimm-et-al. (2000) rate sum over the 3P0 transition table
    # (Phys. Rev. A 108, 053325) — independent of the implementation under test.
    trans = [(215.870446, 0.306), (461.867846, 1.50182782), (675.141040, 4.0326703),
             (729.293151, 0.62437427), (796.547798, 23.2660865)]
    λ  = 759.0
    ωL = 2π * c / (λ * 1e-9)
    ref = 0.0   # (1/s)/(W/m²)
    for (fTHz, gMHz) in trans
        ω0 = 2π * fTHz * 1e12; Γ = 2π * gMHz * 1e6
        ref += (3π * c^2) / (2 * hbar * ω0^3) * (ωL/ω0)^3 *
               (Γ/(ω0 - ωL) + Γ/(ω0 + ωL))^2
    end
    ref *= 1e4   # → per (W/cm²)
    yb  = Ytterbium171Atom(; levels = [Level("1S0")])
    got = scattering_rate_per_Wcm2(yb, "3P0", λ)
    @test isapprox(got, ref; rtol = 1e-10)
end

# ── Alkali D-line polarizability: line strength vs natural linewidth ───────────

using AtomTwin.Units: ε0, a0
using AtomTwin.Units: e as e_charge

# Independent multi-level scalar polarizability (NOT the PolarizabilityModel code
# path) from a list of (dipole_ea0, freq_Hz) reduced dipoles plus a static core:
#   α(ω) = α_core + (1/(2Jg+1)) Σᵢ (2/3)|dᵢ|² ω0ᵢ /(ħ(ω0ᵢ²-ω²)).
function _alpha_dipoles_au(λ_nm, lines; α_core = 0.0, Jg = 0.5)
    au = 4π * ε0 * a0^3
    ωL = 2π * c / (λ_nm * 1e-9)
    s = 0.0
    for (d_ea0, f) in lines
        ω0 = 2π * f; d = d_ea0 * e_charge * a0
        s += (1 / (2Jg + 1)) * (2 / 3) * d^2 * ω0 / (hbar * (ω0^2 - ωL^2))
    end
    return s / au + α_core
end

@testset "dipole-specified transitions reproduce multi-level α (mechanism)" begin
    # A PolarizabilityModel built from reduced dipoles must equal the independent
    # multi-level sum at EVERY wavelength — the dipole→effective-width conversion
    # is exact. Isolate the API mechanism with a self-contained two-line model.
    ref = [(4.227, 377.107463e12), (5.977, 384.230485e12)]
    m = PolarizabilityModel("5S1/2",
        [(freq_THz = 377.107463, dipole_ea0 = 4.227, J_f = 1//2),   # Rb 5P₁/₂
         (freq_THz = 384.230485, dipole_ea0 = 5.977, J_f = 3//2)];
        J = 1//2)   # Rb 5S₁/₂. Declaring J must NOT perturb a dipole-specified
                    # scalar sum — the 1/(2Jg+1) weight is already inside Γ_eff.
    for λ in (1e7, 1200.0, 1000.0, 900.0, 850.0, 800.0)
        @test isapprox(polarizability_au(m, λ), _alpha_dipoles_au(λ, ref); rtol = 1e-9)
    end

    # Returns line strengths in (e·a₀)² units
    function line_strength_ea0sq(freq_THz, gamma_MHz, Je)
        ω0 = 2π * abs(freq_THz) * 1e12
        Γ  = 2π * gamma_MHz * 1e6
        S  = 3π * AtomTwin.Units.ε0 * AtomTwin.Units.hbar * AtomTwin.Units.c^3 * (2Je + 1) * Γ / ω0^3   # (C·m)²
        return S / (AtomTwin.Units.e * AtomTwin.Units.a0)^2                              # (e·a₀)²
    end
    # D2 carries ~2× the line strength of D1:
    # the computed line strengths should match that ratio
    S1, S2 = line_strength_ea0sq(m.transitions[1].freq_THz, m.transitions[1].gamma_MHz, m.transitions[1].J_f), line_strength_ea0sq(m.transitions[2].freq_THz, m.transitions[2].gamma_MHz, m.transitions[2].J_f)
    @test 1.9 < S2 / S1 < 2.2
end

@testset "shipped Rb-87 5S1/2 model is accurate IR→blue" begin
    m  = AtomTwin.RB87_POLARIZABILITY_5S12
    rb = Rubidium87Atom(; levels = [Level("5S1/2")])
    # The shipped model's full line list + ionic-core offset, reconstructed
    # independently, must reproduce polarizability_au everywhere — including below
    # the D lines, where D-only would err by several %.
    lines   = [(t.dipole_ea0, t.freq_THz * 1e12) for t in (
                (dipole_ea0 = 4.227, freq_THz = 377.107),
                (dipole_ea0 = 5.977, freq_THz = 384.231),
                (dipole_ea0 = 0.342, freq_THz = 710.960),
                (dipole_ea0 = 0.553, freq_THz = 713.477),
                (dipole_ea0 = 0.118, freq_THz = 834.474),
                (dipole_ea0 = 0.207, freq_THz = 835.526))]
    α_core  = 9.08
    for λ in (1064.0, 900.0, 850.0, 700.0, 600.0, 532.0, 500.0)
        @test isapprox(polarizability_au(m, λ),
                       _alpha_dipoles_au(λ, lines; α_core = α_core); rtol = 2e-3)
    end
    # Accepted Rb-87 5S1/2 static scalar polarizability ≈ 318.8 a.u.
    @test isapprox(polarizability_au(m, 1e7), 318.8; rtol = 5e-3)
    # Atom-facing convenience methods wire to the same model.
    @test polarizability_au(rb, "5S1/2", 850.0) ≈ polarizability_au(m, 850.0)
    @test scattering_rate_per_Wcm2(rb, "5S1/2", 850.0) > 0
end

@testset "standard α: U = −α I/(2c ε₀), dipole lines at their true strength" begin
    U = AtomTwin.Units
    UoverI(m, λ) = U.h * light_shift_coeff_Hz_per_Wcm2(m, λ) * 1e-4      # J/(W/m²)
    for (m, λ) in ((AtomTwin.RB87_POLARIZABILITY_5S12, 1064.0),
                   (AtomTwin.SR88_POLARIZABILITY_1S0, 813.4),
                   (AtomTwin.YB174_POLARIZABILITY_1S0, 759.35))
        @test polarizability_si(m, λ) ≈ -2U.c * U.ε0 * UoverI(m, λ) rtol = 1e-12
    end
    # Absolute, independent of AtomTwin's line data: Grimm, Weidemüller &
    # Ovchinnikov (2000) eq. 20 for Rb 5S₁/₂ at 1064 nm, linear polarisation, D2 and
    # D1 with weights 2 : 1, the natural width (Steck), counter-rotating terms kept.
    # The remaining ~1 % is the core and the 6P/7P lines.
    Γ, ωL = 2π * 6.0666e6, 2π * U.c / 1064e-9
    grimm = sum(w * -(π * U.c^2 * Γ / (2ω0^3)) * (1 / (ω0 - ωL) + 1 / (ω0 + ωL))
                for (ω0, w) in ((2π * 384.2304844685e12, 2), (2π * 377.107463380e12, 1)))
    @test UoverI(AtomTwin.RB87_POLARIZABILITY_5S12, 1064.0) ≈ grimm rtol = 0.02
    # Yb ground state, from measured widths: ≈ 141 a.u. static.
    @test polarizability_au(AtomTwin.YB174_POLARIZABILITY_1S0, 1e7) ≈ 141.5 rtol = 0.01
end
@testset "Sr static polarizabilities match Safronova 2013 recommended values" begin
    # Line lists from Phys. Rev. A 87, 012509 (2013) reproduce the recommended
    # static scalar polarizabilities: α(1S0)=197.14, α(3P0)=444.51 a.u.
    sr = Strontium88Atom(; levels = [Level("1S0")])
    @test isapprox(polarizability_au(sr, "1S0", 1.0e7), 197.1; rtol = 0.02)
    @test isapprox(polarizability_au(sr, "3P0", 1.0e7), 444.5; rtol = 0.02)
end

@testset "Sr 1S0/3P0 cross at the measured 813.428 nm magic wavelength" begin
    sr = Strontium88Atom(; levels = [Level("1S0")])
    f(λ) = light_shift_coeff_Hz_per_Wcm2(sr, "3P0", λ) -
           light_shift_coeff_Hz_per_Wcm2(sr, "1S0", λ)
    lo, hi = 800.0, 826.0
    for _ in 1:60
        m = (lo + hi) / 2
        f(lo) * f(m) <= 0 ? (hi = m) : (lo = m)
    end
    @test (lo + hi) / 2 ≈ 813.428 atol = 0.1
end
# ======================================================================
# Angular momentum: line strength, tensor polarizability, trap light shift
# ======================================================================

@testset "line-strength factor f(J,J′) and its sign rule" begin
    # A line BELOW the state carries the opposite sign, which is what makes the two
    # states of a two-level atom shift oppositely.
    f = AtomTwin._line_strength_factor
    @test f(0//1, 1//1, +539.0) == 3//1
    @test f(1//1, 0//1, -539.0) == -1//1    # below in energy
    @test f(1//1, 2//1, +202.0) == 5//3

    above = PolarizabilityModel("g", [(freq_THz = +539.3868, gamma_MHz = 0.183,
                                       J_f = 1//1)]; J = 0//1)
    below = PolarizabilityModel("e", [(freq_THz = -539.3868, gamma_MHz = 0.183,
                                       J_f = 0//1)]; J = 1//1)
    la = light_shift_coeff_Hz_per_Wcm2(above, 767.0)
    lb = light_shift_coeff_Hz_per_Wcm2(below, 767.0)
    @test isapprox(lb / la, -1/3; rtol = 1e-12)   # = f(−1)/f(3), sign included

    # Provenance is recorded and validated (a refit must tell fitted from measured).
    @test PolarizabilityModel("x", [(freq_THz = 100.0, gamma_MHz = 1.0, J_f = 1//1,
                                     source = :fitted)]; J=0//1).transitions[1].source == :fitted
    @test_throws ErrorException PolarizabilityModel("x",
        [(freq_THz = 100.0, gamma_MHz = 1.0, source = :guessed, J_f=1//1)]; J = 0//1)
end

@testset "tensor polarizability: Sr-88 3P1 magic wavelengths and vanishing rules" begin
    # Kestler et al., PRA 105, 012821 (2022) measured two 1S0–3P1 magic
    # wavelengths near 473 nm — precision the Yb 3P1 fit (good to ~3 nm) cannot
    # match, so this is the accuracy gate for the whole tensor path.
    m1S0 = AtomTwin.SR88_POLARIZABILITY_1S0
    m3P1 = AtomTwin.SR88_POLARIZABILITY_3P1
    au   = 4π * AtomTwin.Units.ε0 * AtomTwin.Units.a0^3

    # Scalar and tensor both land inside the paper's own uncertainty.
    @test isapprox(AtomTwin._alpha0_si(m1S0, 473.1445) / au, 3637; atol = 17)
    @test isapprox(AtomTwin._alpha0_si(m3P1, 473.1445) / au, 4146; atol = 117)
    @test isapprox(AtomTwin._alpha2_si(m3P1, 473.1445; F = 1//1, I = 0//1) / au,
                   -509; atol = 15)     # Kestler Table II, an independent CI+all-orders value
    @test isapprox(AtomTwin._alpha2_si(m3P1, 473.375; F = 1//1, I = 0//1) / au,
                   -384; atol = 25)

    # The absolute values above anchor the normalisation, which the splitting
    # identity (invariant under any rescaling of α⁽²⁾) cannot. The identity checks
    # the geometry: with factors −2 (m=0) and +1 (|m|=1) the splitting is −3α⁽²⁾.
    s0 = light_shift_coeff_Hz_per_Wcm2(m3P1, 473.1445; F = 1//1, mF = 0//1,
                                       I = 0//1, ε_z = 1.0)
    s1 = light_shift_coeff_Hz_per_Wcm2(m3P1, 473.1445; F = 1//1, mF = 1//1,
                                       I = 0//1, ε_z = 1.0)
    α2 = AtomTwin._alpha2_si(m3P1, 473.1445; F = 1//1, I = 0//1) / au
    split = -(s0 - s1) * AtomTwin.Units.h * 1e-4 *
             2 *AtomTwin.Units.c * AtomTwin.Units.ε0 / au
    @test isapprox(split, -3 * α2; rtol = 1e-10)

    # The magic wavelengths themselves. The crossing runs at ≈5000 a.u./nm, so the
    # paper's ±117 a.u. is already ±23 pm — the tolerance is the reference's own.
    function magic(mF)
        f(λ) = light_shift_coeff_Hz_per_Wcm2(m1S0, λ) -
               light_shift_coeff_Hz_per_Wcm2(m3P1, λ; F = 1//1, mF = mF, I = 0//1,
                                             ε_z = 1.0)
        lo, hi = 473.0, 473.6
        @assert f(lo) * f(hi) < 0
        for _ in 1:200
            m = (lo + hi) / 2
            f(lo) * f(m) <= 0 ? (hi = m) : (lo = m)
        end
        (lo + hi) / 2
    end
    @test isapprox(magic(0//1), 473.375; atol = 0.03)
    @test isapprox(magic(1//1), 473.145; atol = 0.03)

    # Two INDEPENDENT reasons the tensor term is absent — J ≤ 1/2 (the
    # {1 1 2; J J J′} triangle rule) and F ≤ 1/2 (the prefactor).
    @test AtomTwin._alpha2_si(m1S0, 700.0; F = 0//1, I = 0//1) == 0.0
    @test AtomTwin._alpha2_si(m3P1, 473.3; F = 1//2, I = 1//2) == 0.0
    # The sublevel factor sums to zero: the tensor shift splits a manifold without
    # moving its centre of gravity.
    for F in (1//1, 3//2, 5//2)
        @test isapprox(sum(AtomTwin._tensor_geometry(F, mF) for mF in -F:1//1:F),
                       0.0; atol = 1e-12)
    end
    # …and vanishes at the geometric magic angle.
    @test isapprox(AtomTwin._polarization_factor(cosd(54.735610317245346)), 0.0;
                   atol = 1e-12)
end

@testset "trap geometry: beam polarization and the quantization axis" begin
    b = GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 20e-3)
    @test b.pol == ComplexF64[1, 0, 0]                   # x̂ by default
    @test GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 20e-3,
                       pol = [0, 0, 5]).pol == ComplexF64[0, 0, 1]   # normalised
    @test copy(b).pol == b.pol
    @test_throws ErrorException GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 20e-3,
                                             pol = [0, 0, 0])

    s = System(Ytterbium171Atom(; levels = [Level("1S0")]))
    @test getquantizationaxis(s) == [0.0, 0.0, 1.0]      # ẑ by default
    add_quantization_axis!(s, [1, 0, 1])
    @test isapprox(sum(abs2, getquantizationaxis(s)), 1.0; atol = 1e-12)
    @test_throws ErrorException add_quantization_axis!(s, [0, 1, 0])   # only one
end

@testset "trap α is per-sublevel and tracks the magic angle" begin
    function trap_alphas(θ_deg; levels = nothing)
        e  = HyperfineManifold(1//1, 1; label = "³P₁", term = l"3P1")
        sr = Strontium88Atom(; levels = levels === nothing ? [e...] : levels)
        tw = GaussianBeam(λ = 520e-9, w0 = 1e-6, P = 10e-3,
                          pol = [sind(θ_deg), 0, cosd(θ_deg)])
        s  = System(sr, tw)
        add_quantization_axis!(s, [0, 0, 1])
        inner = AtomTwin.Dynamiq.NLevelAtom(length(sr.levels))
        AtomTwin.initialize!(sr, inner; beams = [tw],
                             q_axis = getquantizationaxis(s))
        inner.alpha[520e-9]
    end

    α0, α90 = trap_alphas(0.0), trap_alphas(90.0)
    @test maximum(α0) - minimum(α0) > 0             # ∥ axis: the manifold splits
    @test isapprox(α0[1], α0[3]; rtol = 1e-12)      # mF = ±1 degenerate (∝ mF²)
    @test (α0[2] - α0[1]) * (α90[2] - α90[1]) < 0   # ordering inverts ∥ vs ⊥
    # At the magic angle the (3cos²θ−1)/2 factor vanishes and degeneracy returns.
    αm = trap_alphas(54.735610317245346)
    @test maximum(αm) - minimum(αm) < 1e-48
    # A J=0 manifold has no tensor part, so its sublevels stay identical.
    g = HyperfineManifold(0//1, 0; label = "¹S₀", term = l"1S0")
    @test length(unique(trap_alphas(0.0; levels = [g...]))) == 1

    # α is cached per wavelength but the tensor term is per beam, so beams sharing
    # a wavelength must agree on polarization. There is no right answer otherwise.
    sr = Strontium88Atom(; levels = [HyperfineManifold(1//1, 1; term = l"3P1")...])
    inner = AtomTwin.Dynamiq.NLevelAtom(length(sr.levels))
    b(p) = GaussianBeam(λ = 520e-9, w0 = 1e-6, P = 10e-3, pol = p)
    @test_throws ErrorException AtomTwin.initialize!(sr, inner;
                                    beams = [b([0,0,1]), b([1,0,0])])
end

@testset "the tensor light shift is quantised along the trap polarisation" begin
    # In general the tensor shift mixes Δm = ±1, ±2; the shifts and the force cannot
    # depend on which axis m is counted along.
    using LinearAlgebra
    D = AtomTwin.Dynamiq
    e = HyperfineManifold(1//1, 1; label = "³P₁", term = l"3P1")
    ε = [sind(45), 0, cosd(45)]
    function trap(q)
        sr  = Strontium88Atom(; levels = [e...])
        tw  = GaussianBeam(λ = 520e-9, w0 = 1e-6, P = 10e-3, pol = ε)
        sys = System(sr, tw)
        add_quantization_axis!(sys, q)
        seq = Sequence(1e-9)
        @sequence seq begin
            Wait(1e-9)
        end
        job = compile(sys, seq)
        fs  = [f for f in job.fields if f isa StarkShiftAC || f isa D.StarkShiftACTensor]
        H   = sum(f -> (D.update!(f, 1); real(f._coeff[]) .* Matrix(D.sparse(f.H))), fs)
        E, V = eigen(Hermitian(H))
        a = job.atoms[1]
        a.x .= [0.3e-6, 0.0, 0.0]
        D.updatepop!(a, ComplexF64.(V[:, 1]))          # the most deeply trapped state
        E, D.force(a, [tw])[1], count(f -> f isa D.StarkShiftACTensor, fs)
    end
    E0, F0, n0 = trap(ε)                     # m along ε: diagonal
    @test n0 == 0
    @test E0[2] ≈ E0[3] rtol = 1e-12         # m_ε = ±1 degenerate
    for q in ([0, 0, 1], [0, 1, 0], [1, 1, 1])
        E, F, n = trap(q)
        @test n == 1
        @test E ≈ E0 rtol = 1e-12
        @test F ≈ F0 rtol = 1e-9             # the force counts the coherences
    end
end

@testset "a trapping beam shifts the levels it traps" begin
    # A trap that holds an atom also shifts its levels: one physical effect, so
    # passing the beam to `System` is the whole user action. Ramsey is the
    # end-to-end check — the fringe follows the differential shift.
    g, e = Level("g"; term = l"1S0"), Level("e"; term = l"3P1")
    function ramsey(T; P = 2e-5, explicit = false)
        sr = Strontium88Atom(; levels = [g, e])
        tw = GaussianBeam(λ = 520e-9, w0 = 1e-6, P = P, pol = [0, 0, 1])
        sys = System(sr, tw)
        add_quantization_axis!(sys, [0, 0, 1])
        Ω = 2π * 1e6
        c = add_coupling!(sys, sr, g => e, Ω; active = false)
        explicit && add_light_shift!(sys, sr, [g, e], tw; reference = g)
        add_detector!(sys, PopulationDetectorSpec(sr, e; name = "Pe"))
        seq = Sequence(2e-10)
        @sequence seq begin
            Pulse(c, π / (2Ω)); Wait(T); Pulse(c, π / (2Ω))
        end
        play(sys, seq; initial_state = g).detectors["Pe"][end]
    end

    # Predict the differential shift from the stored α, then check the fringe.
    sr0 = Strontium88Atom(; levels = [g, e])
    tw0 = GaussianBeam(λ = 520e-9, w0 = 1e-6, P = 2e-5, pol = [0, 0, 1])
    inner = AtomTwin.Dynamiq.NLevelAtom(2)
    AtomTwin.initialize!(sr0, inner; beams = [tw0], q_axis = [0.0, 0.0, 1.0])
    αs = inner.alpha[520e-9]
    # U = -α I/(2 c ε₀), so the angular frequency is α I/(2 c ε₀ ħ).
    Tπ = π / abs((αs[2] - αs[1]) * tw0.I0 /
                 (2 * AtomTwin.Units.c * AtomTwin.Units.ε0 * AtomTwin.Units.hbar))

    @test ramsey(Tπ)                   < 0.01    # automatic: the trap alone
    @test ramsey(Tπ; P = 0.0)          > 0.99    # no trap, no shift
    @test ramsey(Tπ; explicit = true)  < 0.01    # explicit gives the same physics

    # add_light_shift! REPLACES the automatic shift rather than adding to it.
    function n_fields(explicit)
        sr = Strontium88Atom(; levels = [g, e])
        tw = GaussianBeam(λ = 520e-9, w0 = 1e-6, P = 1e-3, pol = [0, 0, 1])
        sys = System(sr, tw)
        add_quantization_axis!(sys, [0, 0, 1])
        explicit && add_light_shift!(sys, sr, [g, e], tw; reference = g)
        seq = Sequence(1e-9)
        @sequence seq begin
            Wait(1e-9)
        end
        job = compile(sys, seq)
        fs = [f for f in job.fields if f isa StarkShiftAC]
        # The applied shift is the trap potential −α I/(2c ε₀ ħ) at the atom, negative
        # for a trapped level. The magnitude lives in the operator and `_coeff`
        # carries the intensity envelope, so the shift is their product.
        shift(f) = real(f.H.forward[1][3] * f._coeff[])
        AtomTwin.Dynamiq.update!(fs[end], 1)
        @test isapprox(shift(fs[end]),
                       - fs[end].alpha * AtomTwin.Dynamiq.intensity(tw, sr.inner.x) /
                       (2 * AtomTwin.Units.c * AtomTwin.Units.ε0 * AtomTwin.Units.hbar);
                       rtol = 1e-12)
        # Absolute anchor: the ground-state shift is the trap potential, −U0/ħ.
        if !explicit
            U0 = AtomTwin.polarizability_si(AtomTwin.SR88_POLARIZABILITY_1S0, 520.0) *
                 tw.I0 / (2AtomTwin.Units.c * AtomTwin.Units.ε0)
            AtomTwin.Dynamiq.update!(fs[1], 1)
            @test U0 > 0
            @test isapprox(shift(fs[1]), -U0 / AtomTwin.Units.hbar; rtol = 1e-9)
        end
        explicit && @test fs[1].alpha == 0.0      # reference = g sits at zero
        length(fs)
    end
    @test n_fields(false) == 2
    @test n_fields(true)  == 2                   # replaced, not added
end

@testset "Yb-174 3P1 model and its documented accuracy" begin
    yb   = Ytterbium174Atom(; levels = [Level("1S0")])
    m1S0 = AtomTwin.YB174_POLARIZABILITY_1S0
    m3P1 = AtomTwin.YB174_POLARIZABILITY_3P1

    @test yb.I == 0//1                            # spin-zero: F = J
    @test m1S0 === AtomTwin.YB171_POLARIZABILITY_1S0   # shared electronic structure

    ls = m3P1.transitions
    i1S0 = findfirst(t -> t.freq_THz < 0, ls)
    @test ls[i1S0].J_f == 0//1                    # the line down to ¹S₀ (J′ = 0)
    @test m3P1 === AtomTwin.YB171_POLARIZABILITY_3P1          # one line list, both isotopes
    @test AtomTwin.getpolarizabilitymodels(Ytterbium171Atom())["3P1"] === m3P1
    # The J′ = 0 member of 6p² ³P (409 nm), which sets the tensor part in the blue.
    @test any(t -> t.J_f == 0//1 && t.freq_THz > 0, ls)
    # Provenance survives, so a refit can find the free parameters.
    @test count(t -> t.source == :fitted, ls) == 8

    # The scalar part is solid — the 1S0 control against digitised thesis curves
    # is RMS 0.075 Hz/(W/cm²) — so the tensor angular dependence is what matters:
    # red-shifted along the axis, blue-shifted across it, crossing in between.
    dV(θ) = light_shift_coeff_Hz_per_Wcm2(m1S0, 759.3) -
            light_shift_coeff_Hz_per_Wcm2(m3P1, 759.3; F = 1//1, mF = 0//1,
                                          I = 0//1, ε_z = cosd(θ))
    @test dV(0.0)  < 0
    @test dV(90.0) > 0

    # Test against measured magic angle of 38.5(9)° at 759.3 nm 
    # for m_J = 0 (Höhn et al., PRX QUANTUM 7, 010303 (2026)).
    lo, hi = 0.0, 90.0
    for _ in 1:200
        m = (lo + hi) / 2
        sign(dV(m)) == sign(dV(lo)) ? (lo = m) : (hi = m)
    end
    @test isapprox((lo + hi) / 2, 38.5; atol = 0.9)
end

@testset "Yb 3P1 model against calibration-free light-shift data" begin
    m, g, g1 = AtomTwin.YB174_POLARIZABILITY_3P1, AtomTwin.YB174_POLARIZABILITY_1S0, AtomTwin.YB171_POLARIZABILITY_1S0
    # Differential polarizability of the m_J sublevel, quantization axis along the trap polarization.
    d(λ, mJ) = polarizability_si(m, λ; F = 1//1, mF = mJ, I = 0//1, ε_z = 1.0) - polarizability_si(g, λ)
    function crossing(f, lo, hi)
        for _ in 1:60
            c = (lo + hi) / 2
            sign(f(c)) == sign(f(lo)) ? (lo = c) : (hi = c)
        end
        (lo + hi) / 2
    end
    # ¹⁷⁴Yb π/σ shift ratio ΔU₀/ΔU₁ (Ma et al., PRX 12, 021028 (2022), Fig. 6); ±2σ.
    for (λ, R, tol) in ((473.0, 2.62, 0.6), (488.0, -0.35, 0.17), (494.0, -0.07, 0.12), (532.0, 0.015, 0.1))
        @test isapprox(d(λ, 0//1) / d(λ, 1//1), R; atol = tol)
    end
    # ¹⁷⁴Yb magic wavelengths near the 1032 nm ¹D₂ resonance, ε ∥ B (Zheng et al., PRA 102, 062805 (2020)); ±3σ.
    @test isapprox(crossing(λ -> d(λ, 0//1), 1035.0, 1036.0), 1035.68; atol = 0.12)
    @test isapprox(crossing(λ -> d(λ, 1//1), 1035.8, 1036.6), 1036.12; atol = 0.09)
    # ¹⁷¹Yb F = 3/2, |m_F| = 1/2 magic wavelength 486.78(10) nm (Ma et al. 2022), axis along ε; ±3σ.
    h(λ) = polarizability_si(m, λ; F = 3//2, mF = 1//2, I = 1//2, ε_z = 1.0) - polarizability_si(g1, λ)
    @test isapprox(crossing(h, 484.0, 489.0), 486.78; atol = 0.3)
end

@testset "the trap light shift follows the local intensity" begin
    # The resonance of a trapped atom sits at its LOCAL differential light shift, so
    # it tracks the Gaussian intensity profile as the atom moves off centre; the
    # centre value anchors the absolute scale.
    gm = HyperfineManifold(0//1, 0; label = "¹S₀", term = l"1S0", g_F = 0.0)
    e  = HyperfineManifold(1//1, 1; label = "³P₁", term = l"3P1", g_F = 1.5)
    θ  = 54.7356                      # tensor-free angle: all three mF degenerate…
    B  = 1e-2                         # …once a field quantises them along ẑ. At B = 0
                                      # the tensor shift is quantised along ε at any θ
                                      # (lines at β∥ and β⊥); 100 G (210 MHz) ≫ the
                                      # ~10 MHz mixing, and m = 0 has no linear Zeeman.
    w0 = 1e-6
    tw = GaussianBeam(λ = 767e-9, w0 = w0, P = 50e-3,
                      pol = [sind(θ), 0.0, cosd(θ)])
    I0 = 2 * 50e-3 / (π * w0^2)

    # Analytic differential shift at the trap centre: (U_e − U_g)/h. ³P₁ is the more
    # deeply trapped here, so the line moves red (Δ0 < 0).
    Δ0 = (light_shift_coeff_Hz_per_Wcm2(AtomTwin.YB174_POLARIZABILITY_3P1, 767.0;
                F = 1//1, mF = 0//1, I = 0//1, ε_z = cosd(θ)) -
          light_shift_coeff_Hz_per_Wcm2(AtomTwin.YB174_POLARIZABILITY_1S0, 767.0)) * I0 * 1e-4
    @test Δ0 < 0

    "Detuning of the peak excitation for a static atom held at x₀."
    function resonance(x0)
        yb = Ytterbium174Atom(; levels = [gm..., e...], x_init = [x0, 0.0, 0.0])
        ds = range(-12.0, 2.0; length = 150)
        y = map(ds) do d
            sys = System(yb, tw)
            add_quantization_axis!(sys, [0.0, 0.0, 1.0])
            add_zeeman_detunings!(sys, yb, e; B = B, delta = 2π * d * 1e6)
            cp = add_coupling!(sys, yb, gm => e, 2π * 20e3, 0.0, 0.0; active = false)
            add_decay!(sys, yb, e => gm, 2π * 182e3)
            add_detector!(sys, PopulationDetectorSpec(yb, e.levels[2]; name = "P"))
            seq = Sequence(2e-9)
            @sequence seq begin
                Pulse(cp, 20e-6)
            end
            ## held: motion is the default, and the trap would pull it to the centre
            play(sys, seq; initial_state = gm[0], shots = 1, frozen = true,
                 density_matrix = true).detectors["P"][end]
        end
        ds[argmax(y)]
    end

    # At the centre the resonance is the full differential shift…
    @test isapprox(resonance(0.0), Δ0 / 1e6; rtol = 0.02)
    # …and off centre it follows the Gaussian intensity profile.
    for x0 in (0.4e-6, 0.8e-6)
        @test isapprox(resonance(x0), Δ0 / 1e6 * exp(-2 * x0^2 / w0^2); rtol = 0.02)
    end
end

@testset "PhotoDetector counts every channel of a manifold decay" begin
    # A manifold decay has one jump per photon polarisation; the detector counts
    # them all, whichever sublevel is driven.
    Γ = 2π * 182e3
    g = HyperfineManifold(0//1, 0; label = "1S0", term = l"1S0")
    e = HyperfineManifold(1//1, 1; label = "3P1", term = l"3P1", g_F = 1.493)

    function run(; Ω_π, Ω_p, Ω_m, shots = 16)
        yb  = Ytterbium174Atom(; levels = [g..., e...])
        sys = System(yb)
        add_quantization_axis!(sys, [1.0, 0.0, 0.0])
        add_zeeman_detunings!(sys, yb, e; B = 0.0, delta = 0.0)
        cp = add_coupling!(sys, yb, g => e; Ω_π = Ω_π, Ω_p = Ω_p, Ω_m = Ω_m,
                           active = false)
        pd = PhotoDetectorSpec(name = "clicks")
        add_detector!(sys, pd)
        add_decay!(sys, yb, e => g, Γ; clicks = pd, λ = 556e-9)
        add_detector!(sys, PopulationDetectorSpec(yb, g[0]; name = "Pg"))
        seq = Sequence(2e-7; downsample = 50)
        @sequence seq begin
            Pulse(cp, 4e-4)
        end
        out = play(sys, seq; initial_state = g[0], shots = shots)
        (mean(sum(out.detectors["clicks"], dims = 1)),
         mean(out.detectors["Pg"][end, :]))
    end

    Ω = 2π * 180e3
    n_pi,  Pg_pi  = run(Ω_π = Ω,   Ω_p = 0.0, Ω_m = 0.0)
    n_sig, Pg_sig = run(Ω_π = 0.0, Ω_p = Ω,   Ω_m = Ω)

    # Both drives cycle the atom, so both must report photons.
    @test Pg_pi  < 0.95        # it really is decaying
    @test Pg_sig < 0.95
    @test n_pi  > 10
    @test n_sig > 10

    # Every excited sublevel decays at the same total rate Γ, so driving one
    # sublevel or two gives comparable counts.
    @test 0.3 < n_pi / n_sig < 3.0
end

@testset "a manifold decay transfers excited coherence to the ground state" begin
    # One jump per photon polarisation: the π branch, amplitudes ∓1/√3, carries the
    # coherence of (|e₊⟩ + |e₋⟩)/√2 to the ground state, ρ₊₋ = −1/6 for J = ½ → ½.
    g = FineManifold(1//2; label = "S")
    e = FineManifold(1//2; label = "P")
    function run(; dm, shots = 1)
        atom = Atom(; levels = [g..., e...])
        sys = System(atom)
        add_decay!(sys, atom, e => g, 2π * 1e6)
        add_detector!(sys, CoherenceDetectorSpec(atom, g[1//2] => g[-1//2]; name = "c"))
        add_detector!(sys, PopulationDetectorSpec(atom, g[1//2]; name = "p"))
        seq = Sequence(1e-8)
        @sequence seq begin
            Wait(2e-6)
        end
        out = play(sys, seq; initial_state = [(e[1//2] + e[-1//2]) / √2],
                   density_matrix = dm, shots = shots)
        mean(out.detectors["c"][end, :]), mean(out.detectors["p"][end, :])
    end
    c, p = run(dm = true)
    @test c ≈ -1/6 atol = 1e-5
    @test p ≈ 1/2 atol = 1e-5
    c, p = run(dm = false, shots = 2000)
    @test real(c) ≈ -1/6 atol = 0.03
    @test p ≈ 1/2 atol = 0.05
end

@testset "measured polarizability for a level with no line model" begin
    # Yb ¹P₁ has no shipped model. `polarizabilities` attaches one to an atom, and
    # a tensor offset lets a measured (scalar, tensor) pair at one wavelength stand
    # without a line list. Muzi Falconi et al. (2025), 532 nm, π-polarised trap:
    # m_J = ±1 magic, m_J = 0 at −11.6 MHz/mK relative to ¹S₀ (whose shift is
    # 20.84 MHz/mK), so r0 = α(0)/α(¹S₀) = 1 − 11.6/20.84 = 0.443. In units of α(¹S₀),
    # with the tensor factor (1 for m = ±1, −2 for m = 0) of a π-polarised trap:
    # α_s + α_t = 1 and α_s − 2α_t = r0, so α_s = (2 + r0)/3, α_t = (1 − r0)/3.
    u   = light_shift_coeff_Hz_per_Wcm2(AtomTwin.YB174_POLARIZABILITY_1S0, 532.0) * 1e-4
    r0  = 1 - 11.6 / 20.84
    αs, αt = (2 + r0) / 3, (1 - r0) / 3
    model = PolarizabilityModel("1P1", NamedTuple[]; J = 1,
                                offset_Hz_per_Wm2 = αs * u, tensor_offset_Hz_per_Wm2 = αt * u)
    g  = HyperfineManifold(0//1, 0; label = "¹S₀", term = l"1S0")
    ex = HyperfineManifold(1//1, 1; label = "¹P₁", term = l"1P1")
    for (εz, ratio0, ratio1) in ((1.0, r0, 1.0),                    # π-polarised trap
                                 (cosd(45), (r0 + 1) / 2, αs + αt / 4))  # 45°: P₂ = 1/4
        yb  = Ytterbium174Atom(; levels = [g..., ex...], polarizabilities = Dict(l"1P1" => model))
        sys = System(yb, GaussianBeam(λ = 532e-9, w0 = 1e-6, P = 1e-3,
                                      pol = [sqrt(1 - εz^2), 0.0, εz]))
        add_quantization_axis!(sys, [0.0, 0.0, 1.0])
        q = Sequence(1e-8); push!(q, Wait(1e-8))          # (an empty Sequence won't compile)
        job = compile(sys, q; initial_state = [g[0]])
        α = job.atoms[1].alpha[532e-9]
        idx(l) = yb.level_indices[l]
        @test α[idx(ex[0])] / α[idx(g[0])] ≈ ratio0 rtol = 1e-12
        @test α[idx(ex[1])] / α[idx(g[0])] ≈ ratio1 rtol = 1e-12
        @test α[idx(ex[-1])] ≈ α[idx(ex[1])]
    end
    # The species' own models are untouched for other atoms.
    @test !haskey(AtomTwin.getpolarizabilitymodels(Ytterbium174Atom()), "1P1")
    # Hyperfine recoupling of the tensor offset: a J = 1 offset on the F levels of
    # I = 1/2 and I = 3/2 isotopes equals what a line of the same J-level tensor gives.
    line = PolarizabilityModel("X", [(freq_THz = 300.0, gamma_MHz = 10.0, J_f = 2//1)]; J = 1)
    αJ = AtomTwin._alpha2_si(line, 532.0; F = 1//1, I = 0//1)
    uJ = -αJ / (2 * AtomTwin.Units.c * AtomTwin.Units.ε0 * AtomTwin.Units.h)
    off = PolarizabilityModel("X", NamedTuple[]; J = 1, tensor_offset_Hz_per_Wm2 = uJ)
    for (F, I) in ((3//2, 1//2), (3//2, 3//2), (5//2, 3//2))
        αF = AtomTwin._alpha2_si(line, 532.0; F = F, I = I)
        @test αF != 0
        @test AtomTwin._alpha2_si(off, 532.0; F = F, I = I) ≈ αF rtol = 1e-12
    end
    @test_throws ArgumentError PolarizabilityModel("X", NamedTuple[]; J = 1//2,
                                                   tensor_offset_Hz_per_Wm2 = 1.0)
end

@testset "a detuning pulse must have real amplitudes" begin
    # A complex detuning would be non-Hermitian.
    l0, l1 = Level("0"), Level("1")
    q = Atom(; levels = [l0, l1]); s = System(q)
    δ = add_detuning!(s, q, l1, 1.0; active = false)
    add_detector!(s, PopulationDetectorSpec(q, l1; name = "P1"))
    seq = Sequence(1e-9)
    push!(seq, Pulse(δ, 1e-7; amplitudes = fill(2π * 1e6 * im, 100), interp = :constant))
    @test_throws ArgumentError play(s, seq; initial_state = l0 + l1)
    @test_throws ArgumentError play(s, seq; initial_state = l0 + l1, density_matrix = true)
    seq2 = Sequence(1e-9); push!(seq2, Pulse(δ, 1e-7; ampl = 1.0im))
    @test_throws ArgumentError play(s, seq2; initial_state = l0 + l1)
    seq3 = Sequence(1e-9); push!(seq3, Pulse(δ, 1e-7; amplitudes = fill(2π * 1e6, 100)))
    @test play(s, seq3; initial_state = l0 + l1).detectors["P1"][end] ≈ 0.5 atol = 1e-10
    seq4 = Sequence(1e-9); push!(seq4, Pulse(δ, 1e-7; ampl = -exp(im * π)))  # rounding residue
    @test play(s, seq4; initial_state = l0 + l1).detectors["P1"][end] ≈ 0.5 atol = 1e-10
end

@testset "GeneralGaussianBeam follows the paraxial axial profile" begin
    # Paraxial: on-axis field ∝ w0/w, Gouy phase −ζ, |E|² consistent with
    # `intensity` and the gradients consistent with finite differences.
    λ, w0 = 759.35e-9, 0.6e-6
    b  = GeneralGaussianBeam(λ, w0, w0, 1e-3, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    zR = π * w0^2 / λ
    E(r) = Dynamiq.efield_scalar(b, r)
    @test abs(E([0.0, 0.0, zR])) / abs(E([0.0, 0.0, 0.0])) ≈ 1 / √2 rtol = 1e-12
    # On axis E(z)/E(0) = e^{i(kz − ζ)}: the phase runs as exp(+ik·r) like PlanarBeam's,
    # and the Gouy phase lags (ζ(z_R) = π/4).
    @test angle(E([0.0, 0.0, zR]) * cis(-2π / λ * zR) / E([0.0, 0.0, 0.0])) ≈ -π / 4 rtol = 1e-9
    pb = Dynamiq.PlanarBeam(λ, 1.0, [0.0, 0.0, 1.0], ComplexF64[1, 0, 0])
    @test sign(angle(E([0.0, 0.0, λ / 8]) / E([0.0, 0.0, 0.0]))) ==
          sign(angle(Dynamiq.efield_scalar(pb, [0.0, 0.0, λ / 8]) / Dynamiq.efield_scalar(pb, [0.0, 0.0, 0.0])))
    @test Dynamiq.intensity(b, [0.0, 0.0, zR]) / Dynamiq.intensity(b, [0.0, 0.0, 0.0]) ≈ 0.5 rtol = 1e-12
    cε = AtomTwin.Units.c * AtomTwin.Units.ε0
    h = 1e-12
    for r in ([0.2e-6, -0.1e-6, 0.7e-6], [0.35e-6, 0.3e-6, -1.8e-6], [0.0, 0.0, 0.4e-6])
        @test cε * abs2(E(r)) / 2 ≈ Dynamiq.intensity(b, r) rtol = 1e-12
        gI, gE = Dynamiq.dIdx(b, r), Dynamiq.dEdr(b, r)
        for i in 1:3
            e = zeros(3); e[i] = h
            fdI = (Dynamiq.intensity(b, r .+ e) - Dynamiq.intensity(b, r .- e)) / 2h
            fdE = (E(r .+ e) - E(r .- e)) / 2h
            @test gI[i] ≈ fdI rtol = 1e-6 atol = 1e-6 * Dynamiq.intensity(b, r) / w0
            @test gE[i] ≈ fdE rtol = 1e-6 atol = 1e-6 * abs(E(r)) / w0
        end
    end
end

# ── Coupling API and polarization ─────────────────────────────────────────────

@testset "MixedPolarization contaminates circular light with the opposite helicity" begin
    σp = ComplexF64[1, im, 0] / √2
    σm = ComplexF64[1, -im, 0] / √2
    ε  = 0.1
    mp = MixedPolarization(σp, ε; k = [0.0, 0.0, 1.0])
    pol = AtomTwin._resolve_node_default(mp)
    ## `k × σ⁺` is parallel to σ⁺; the orthogonal polarization is `k × σ⁺*` ∝ σ⁻.
    @test abs(sum(conj.(σp) .* mp.pol_perp)) < 1e-12
    @test abs2(sum(conj.(σm) .* pol)) ≈ ε^2 / (1 + ε^2) rtol = 1e-9
    lin = MixedPolarization([1.0, 0.0, 0.0], ε; k = [0.0, 0.0, 1.0])  # unchanged for linear
    @test lin.pol_perp ≈ ComplexF64[0, 1, 0]
end

@testset "add_coupling! keyword and positional forms build the same Hamiltonian" begin
    Ω  = 2π * 1e6
    σp = ComplexF64[1, im, 0] / √2
    function σplus(keyword)
        g  = HyperfineManifold(1//2, 0; label = "1S0", term = l"1S0")
        e  = HyperfineManifold(3//2, 1; label = "3P1", term = l"3P1")
        yb = Ytterbium171Atom(; levels = [g..., e...])
        s  = System(yb)
        beam = PlanarBeam(556e-9, 1.0, [0.0, 0.0, 1.0], σp)
        keyword ? add_coupling!(s, yb, g => e; beam = beam, Ω_π = 0.0, Ω_p = Ω, Ω_m = 0.0) :
                  add_coupling!(s, yb, g => e, 0.0, Ω, 0.0; beam = beam)
        H = Matrix(AtomTwin.gethamiltonian(s))
        i(l) = findfirst(==(l), [g..., e...])
        [abs(H[i(e[1//2]), i(g[-1//2])]), abs(H[i(e[3//2]), i(g[1//2])])] ./ (Ω / 2)
    end
    ## σ⁺ pairs m = −½ → +½ and +½ → +3/2 carry CG √(1/3) and 1 in both forms.
    @test σplus(true) ≈ [sqrt(1/3), 1.0] rtol = 1e-9
    @test σplus(true) ≈ σplus(false) rtol = 1e-9
end

@testset "add_coupling! keyword form takes a Parameter, and a Gaussian beam in either form" begin
    g  = HyperfineManifold(0//1, 0; label = "1S0", term = l"1S0")
    e  = HyperfineManifold(1//1, 1; label = "3P1", term = l"3P1")
    yb = Ytterbium174Atom(; levels = [g..., e...])
    beam = GeneralGaussianBeam(556e-9, 5e-6, 5e-6, 1e-3, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    Ωp = Parameter(:Ω, 2π * 1e6)
    kw  = add_coupling!(System(yb), yb, g => e; beam = beam, Ω_π = Ωp, Ω_p = 0.0, Ω_m = 0.0)
    @test length(kw) == 1 && kw[1] isa AtomTwin.Dynamiq.GaussianCoupling
    ## The positional form used to fall back to a GlobalCoupling: no profile, no Doppler phase.
    pos = add_coupling!(System(yb), yb, g => e, 2π * 1e6, 0.0, 0.0; beam = beam)
    @test all(c -> c isa AtomTwin.Dynamiq.GaussianCoupling, pos)
    ## An explicit Ω is the Rabi frequency at the focus; an atom off-centre sees the profile.
    lg, le = Level("g"), Level("e")
    a   = Atom(; levels = [lg, le], x_init = [2.5e-6, 0.0, 0.0])
    s   = System(a)
    add_coupling!(s, a, lg => le, 2π * 1e6; beam = beam)
    seq = Sequence(1e-9); push!(seq, Wait(1e-9))
    job = compile(s, seq; initial_state = [lg])
    c   = only(f for f in job.fields if f isa AtomTwin.Dynamiq.GaussianCoupling)
    @test abs(c.Ω0) ≈ 2π * 1e6 * exp(-(2.5e-6 / 5e-6)^2) rtol = 1e-9
end

@testset "a J = 0 → 0 beam-derived coupling is refused, not silently zero" begin
    gnd  = HyperfineManifold(1//2, 0; label = "1S0", term = l"1S0")
    clk  = HyperfineManifold(1//2, 0; label = "3P0", term = l"3P0")
    yb   = Ytterbium171Atom(; levels = [gnd..., clk...])
    beam = PlanarBeam(578.42e-9, 1e7, [0.0, 0.0, 1.0], ComplexF64[1, im, 0] / √2)
    @test_throws ArgumentError add_coupling!(System(yb), yb, gnd => clk; beam = beam, d_red = 5e-34)
    @test_throws ArgumentError add_coupling!(System(yb), yb, gnd => clk, beam; d_red = 5e-34)
    ## The explicit path builds the F = ½ → ½ σ⁺ pair.
    @test length(add_coupling!(System(yb), yb, gnd => clk, 0.0, 2π * 10.0, 0.0; beam = beam)) == 1
end

@testset "an unusable Rabi frequency is an error, not a zero coupling" begin
    g  = HyperfineManifold(1//2, 0; label = "1S0", term = l"1S0")
    e  = HyperfineManifold(3//2, 1; label = "3P1", term = l"3P1")
    yb = Ytterbium171Atom(; levels = [g..., e...])
    Ωt = t -> 2π * 1e6
    @test_throws MethodError add_coupling!(System(yb), yb, g => e, 0.0, Ωt, 0.0)
    @test_throws MethodError AtomTwin.compute_coupling_strength(g[1//2], e[3//2], yb, 0.0, Ωt, 0.0)
end

@testset "rabi_frequencies accepts a GaussianBeam" begin
    beam = GaussianBeam(578.42e-9, 10e-6, 0.1)                 # x-polarized
    Ωπ, Ωp, Ωm = rabi_frequencies(beam; d_red = 5e-34)
    E0 = abs(Dynamiq.Efield(beam, beam.r0))
    @test abs(Ωπ) < 1e-9 * abs(Ωp)
    @test abs(Ωp) ≈ 5e-34 * E0 / √2 / AtomTwin.Units.hbar rtol = 1e-9
    @test abs(Ωm) ≈ abs(Ωp) rtol = 1e-12
end
