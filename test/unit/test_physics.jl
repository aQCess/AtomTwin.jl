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
    P, t = out.detectors["P_e"], out.times
    @test maximum(P) ≈ 0.5 atol = 0.01                             # Ω²/Ω_eff²
    @test P[argmin(abs.(t .- π / (√2 * Ω)))] ≈ 0.5 atol = 0.01      # the first maximum...
    @test P[argmin(abs.(t .- 2π / (√2 * Ω)))] ≈ 0.0 atol = 0.01     # ...and return, at Ω_eff = √2 Ω
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

@testset "Lindblad: pure dephasing in MCWF agrees with the master equation" begin
    # The trajectory realisation of L = √γ|e⟩⟨e| (a no-jump decay of the |e⟩
    # amplitude at γ/2 and projective jumps) must average to the same γ/2 coherence
    # decay with the populations untouched.
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    γ = 2π * 1e6
    add_dephasing!(sys, atom, e, γ)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))
    add_detector!(sys, CoherenceDetectorSpec(atom, g => e; name = "coh"))
    seq = Sequence(1e-9; downsample = 1000)
    @sequence seq begin
        Wait(1e-6)
    end
    N = 4000
    out = play(sys, seq; initial_state = (1/√2)*g + (1/√2)*e, shots = N, rng = MersenneTwister(5))
    σ = 0.5 / sqrt(N)
    @test abs(mean(out.detectors["coh"][end, :])) ≈ 0.5 * exp(-γ * 1e-6 / 2) atol = 4σ
    @test mean(out.detectors["P_e"][end, :]) ≈ 0.5 atol = 4σ
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
    m    = AtomTwin.ATOM_DEFAULTS[:Ytterbium171].mass
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

@testset "polarizability wavelengths are in nanometres; metres are rejected" begin
    yb = Ytterbium171Atom(; levels = [Level("1S0")])
    for f in (light_shift_coeff_Hz_per_Wcm2, polarizability_au, scattering_rate_per_Wcm2)
        @test_throws ArgumentError f(yb, "1S0", 759e-9)
    end
    @test_throws ArgumentError polarizability_si(AtomTwin.YB171_POLARIZABILITY_1S0, 759e-9)
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

@testset "Yb decay rates: lifetimes, branching and their matrix elements" begin
    A = AtomTwin.YB171_DECAY_RATES
    @test AtomTwin.YB174_DECAY_RATES === A
    @test l"3P2".J == 2//1
    # A = ω³|⟨J‖d‖J′⟩|²/(3πε₀ħc³(2J′+1)), ω from the NIST energies (cm⁻¹).
    ea0 = 1.602176634e-19 * 5.29177210903e-11
    Arate(ν, d) = (2π * 2.99792458e10 * ν)^3 * (d * ea0)^2 /
                  (3π * 8.8541878128e-12 * 1.054571817e-34 * 2.99792458e8^3 * 3)
    E3P = (17288.439, 17992.007, 19710.388)
    # (6s6d)³D₁: the CI+all-order elements 1.84 (³P₀) and 1.62 (³P₁), and the LS ³P₂ one.
    for (J, d) in ((0, 1.84), (1, 1.62), (2, 1.84 / sqrt(20)))
        @test isapprox(A["(6s6d)3D1" => "3P$J"], Arate(39808.72 - E3P[J+1], d); rtol = 1e-3)
    end
    @test isapprox(1 / sum(A["(6s6d)3D1" => "3P$J"] for J in 0:2), 22.0e-9; rtol = 0.01)
    # (5d6s)³D₁: Kroeze Table 12 — lifetime 332 ns and β = 0.638 / 0.352 / 0.0099.
    a = [A["(5d6s)3D1" => "3P$J"] for J in 0:2]
    @test isapprox(1 / sum(a), 332e-9; rtol = 0.01)
    @test isapprox(a ./ sum(a), [0.638, 0.352, 0.0099]; atol = 2e-3)
    # (6s7s)³S₁: Kroeze Table 13 — τ = 13.8 ns, β(³P₂) = 0.508, β(³P₀) = 0.130.
    a = [A["(6s7s)3S1" => l] for l in ("3P0", "3P1", "3P2", "1P1")]
    @test isapprox(1 / sum(a), 13.8e-9; rtol = 0.01)
    @test isapprox(a[3] / sum(a), 0.508; atol = 2e-3) && isapprox(a[1] / sum(a), 0.130; atol = 2e-3)
    # ¹P₁: leak out of the cycling transition, 6.4e-8 per photon (Porsev 1999).
    leak = (A["1P1" => "(5d6s)3D1"] + A["1P1" => "(5d6s)3D2"]) / A["1P1" => "1S0"]
    @test isapprox(leak, 6.4e-8; rtol = 0.02)
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

@testset "PhotoDetector aperture: each photon collected by its own direction" begin
    # A σ (or π) photon from J = 1 → 0 about ẑ, through an NA 0.7 objective: the fraction
    # collected is its dipole pattern integrated over the cone, c = cos θ = √(1 − NA²):
    #   along ẑ  f_σ = ⅜[(1 − c) + (1 − c³)/3],   f_π = ½ − ¾c + ¼c³
    #   along x̂  f_σ = ⅜(1 − c) + (3/16)(⅔ − c + c³/3)
    g = HyperfineManifold(0//1, 0; term = l"1S0")
    e = HyperfineManifold(1//1, 1; term = l"3P1")
    Γ, Ω, T = 2π * 182e3, 2π * 1e6, 1e-3
    function counts(pol; aperture = nothing, efficiency = 1.0, λ = 556e-9, shots = 24)
        yb  = Ytterbium174Atom(; levels = [g..., e...])
        sys = System(yb)
        cp  = add_coupling!(sys, yb, g => e; Ω_π = pol == :π ? Ω : 0.0, Ω_p = pol == :σ ? Ω : 0.0,
                            active = false)
        pd  = PhotoDetectorSpec(; name = "n", tspan = [T], aperture, efficiency)
        add_detector!(sys, pd)
        add_decay!(sys, yb, e => g, Γ; λ, clicks = pd)
        seq = Sequence(1e-7)
        @sequence seq begin
            Pulse(cp, T)
        end
        sum(play(sys, seq; initial_state = g[0], shots, frozen = true,
                 rng = MersenneTwister(3)).detectors["n"])
    end
    c = sqrt(1 - 0.7^2)
    within(n, N, f) = abs(n / N - f) < 4 * sqrt(f * (1 - f) / N)
    ## with λ the direction comes from the recoil draw: same trajectories, a subset counted
    for (pol, axis, f) in ((:σ, [0.0, 0, 1], 3 / 8 * ((1 - c) + (1 - c^3) / 3)),
                           (:π, [0.0, 0, 1], 1 / 2 - 3c / 4 + c^3 / 4),
                           (:σ, [1.0, 0, 0], 3 / 8 * (1 - c) + 3 / 16 * (2 / 3 - c + c^3 / 3)))
        N = counts(pol)
        n = counts(pol; aperture = (axis = axis, NA = 0.7))
        @test n <= N && within(n, N, f)
    end
    ## efficiency alone, and an aperture on a decay without a recoil (direction drawn for it)
    N = counts(:σ)
    @test within(counts(:σ; efficiency = 0.5), N, 0.5)
    @test within(counts(:σ; λ = nothing, aperture = (axis = [0, 0, 1], NA = 0.7)), counts(:σ; λ = nothing),
                 3 / 8 * ((1 - c) + (1 - c^3) / 3))
    @test_throws ArgumentError PhotoDetectorSpec(aperture = (axis = [0, 0, 1], NA = 1.2))
    @test_throws ArgumentError PhotoDetectorSpec(efficiency = 1.5)
end

@testset "manifold decay: recoil pattern and clicks by photon polarisation" begin
    # J = 1 → 0 about a tilted axis (x): e₀ emits π photons only, e₊ σ photons only.
    g = HyperfineManifold(0//1, 0; label = "1S0", term = l"1S0")
    e = HyperfineManifold(1//1, 1; label = "3P1", term = l"3P1", g_F = 1.493)
    function run(; Ω_π, Ω_p, shots = 8)
        yb  = Ytterbium174Atom(; levels = [g..., e...])
        sys = System(yb)
        pdπ, pdσ = PhotoDetectorSpec(name = "pi"), PhotoDetectorSpec(name = "sigma")
        add_detector!(sys, pdπ); add_detector!(sys, pdσ)
        add_decay!(sys, yb, e => g, 2π * 182e3; λ = 556e-9, clicks = (π = pdπ, σ = pdσ))
        add_quantization_axis!(sys, [1.0, 0.0, 0.0])          # after the decay: order is free
        add_zeeman_detunings!(sys, yb, e; B = 0.0, delta = 0.0)
        cp = add_coupling!(sys, yb, g => e; Ω_π = Ω_π, Ω_p = Ω_p, Ω_m = 0.0, active = false)
        seq = Sequence(2e-7; downsample = 100)
        @sequence seq begin
            Pulse(cp, 1e-4)
        end
        out = play(sys, seq; initial_state = g[0], shots = shots, frozen = true)
        ## the run's atom (the job's private copy) carries the photon dipoles about the axis
        a = compile(sys, seq; initial_state = [g[0]]).atoms[1]
        dip = Dict(yb.level_indices[e[m]] => a.dipole[yb.level_indices[e[m]] => yb.level_indices[g[0]]]
                   for m in (-1, 0, 1))
        sum(out.detectors["pi"]), sum(out.detectors["sigma"]), dip, yb
    end
    nπ, nσ, dip, yb = run(Ω_π = 2π * 180e3, Ω_p = 0.0)
    @test nπ > 20 && nσ == 0
    nπ, nσ, _, _ = run(Ω_π = 0.0, Ω_p = 2π * 180e3)
    @test nπ == 0 && nσ > 20
    ## π dipole along the axis, σ dipoles circular in the plane ⊥ to it
    dπ = dip[yb.level_indices[e[0]]]
    @test abs(dπ[1]) ≈ 1 && abs(dπ[2]) < 1e-12 && abs(dπ[3]) < 1e-12
    for m in (-1, 1)
        d = dip[yb.level_indices[e[m]]]
        @test abs(d[1]) < 1e-12 && abs(d[2]) ≈ abs(d[3]) ≈ 1 / √2
    end
    @test_throws ErrorException add_decay!(System(Ytterbium174Atom(; levels = [g..., e...])),
        Ytterbium174Atom(; levels = [g..., e...]), e => g, 1.0; clicks = (sigma = nothing,))
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

@testset "g-factors: the Landé formula with species g_J and g_I" begin
    # ¹⁷¹Yb: Kroeze et al. measure ³P₁ F′ = 3/2, ½ at 1392.674 and 2786.10 kHz/G and ¹S₀ at
    # −0.7505 kHz/G; the Landé formula with g_J(³P₁) = 1.49282 and g_I reproduces them.
    μB_kHz_per_G = 1399.6245
    S0  = HyperfineManifold(1//2, 0; term = l"1S0")
    P32 = HyperfineManifold(3//2, 1; term = l"3P1")
    P12 = HyperfineManifold(1//2, 1; term = l"3P1")
    yb  = Ytterbium171Atom(; levels = [S0..., P32..., P12...])
    @test zeeman_g(yb, S0[1//2])  * μB_kHz_per_G ≈ -0.7505  rtol = 1e-9
    @test zeeman_g(yb, P32[3//2]) * μB_kHz_per_G ≈ 1392.674 rtol = 1e-6
    @test zeeman_g(yb, P12[1//2]) * μB_kHz_per_G ≈ 2786.10  rtol = 1e-6
    @test lande_gF(1, 1, 0; g_J = 1.5, g_I = 0.0) == 1.5                   # no nuclear spin: g_F = g_J
    @test lande_gF(0, 0, 0; g_J = 0.0, g_I = 0.0) == 0.0
    ## ¹⁷⁴Yb: I = 0, so ³P₁ has g_F = g_J
    P1 = HyperfineManifold(1, 1; term = l"3P1")
    @test zeeman_g(Ytterbium174Atom(; levels = [P1...]), P1[1]) == 1.49282
    ## a declared g_F wins; a term the species does not list (³P₀) must be declared
    @test zeeman_g(yb, HyperfineManifold(1//2, 0; term = l"3P0", g_F = -6.8e-4)[1//2]) == -6.8e-4
    @test_throws ErrorException zeeman_g(yb, HyperfineManifold(1//2, 0; term = l"3P0")[1//2])
    @test_throws ErrorException zeeman_g(Atom(; levels = [Level("g")]), FineManifold(1//2)[1//2])
    ## add_zeeman_detunings! uses them: H(m) = −m g_F μ_B B
    s = System(yb)
    add_zeeman_detunings!(s, yb, P32; B = 1e-2)
    H = gethamiltonian(s)
    i = yb.level_indices[P32[3//2]]
    @test real(H[i, i]) ≈ 1.5 * 1392.674e3 * 100 * 2π rtol = 1e-6       # 3/2 × 1392.674 kHz/G × 100 G
end

@testset "species line data: d_red for a beam-derived coupling" begin
    # ¹⁷¹Yb 556 nm: a σ⁺ beam at the cycling transition's saturation intensity drives
    # |½, ½⟩ → |3/2, 3/2⟩ at Ω = Γ/√2, and every excited sublevel decays at Γ.
    S0  = HyperfineManifold(1//2, 0; label = "1S0", term = l"1S0")
    P32 = HyperfineManifold(3//2, 1; label = "3P1", term = l"3P1")
    P12 = HyperfineManifold(1//2, 1; label = "3P1'", term = l"3P1")
    yb  = Ytterbium171Atom(; levels = [S0..., P32..., P12...])
    Γ, ω = getdecayrates(yb)["3P1" => "1S0"], 2π * getlinefrequencies(yb)["3P1" => "1S0"]
    d = reduced_dipole(yb, S0[1//2], P32[3//2])
    @test reduced_dipole(yb, P32[3//2], S0[1//2]) == d                 # either order
    Isat = AtomTwin.Units.hbar * ω^3 * Γ / (12π * AtomTwin.Units.c^2)     # πhcΓ/3λ³
    beam = PlanarBeam(2π * AtomTwin.Units.c / ω, Isat, [0.0, 0, 1], [1.0, 1im, 0] / √2)
    s = System(yb)
    add_coupling!(s, yb, S0 => P32, beam)                               # no d_red
    H = gethamiltonian(s)
    i(l) = yb.level_indices[l]
    @test abs(H[i(P32[3//2]), i(S0[1//2])]) ≈ Γ / (2√2) rtol = 1e-12
    for e in (P32[1//2], P12[1//2])
        Σd² = sum(abs2(AtomTwin.dipole_matrix_element(yb, g, e) * d) for g in S0.levels)
        @test ω^3 * Σd² / (3π * AtomTwin.Units.ε0 * AtomTwin.Units.hbar * AtomTwin.Units.c^3) ≈ Γ rtol = 1e-12
    end
    @test_throws ErrorException reduced_dipole(Atom(; levels = [S0..., P32...]), S0[1//2], P32[3//2])
end

@testset "a beam-derived plane-wave coupling pushes the atom like an explicit one" begin
    # Radiation pressure acts through `PlanarCoupling`s; a beam-derived coupling to a
    # `PlanarBeam` must be one, so a moving atom scatters exactly as with explicit Ω.
    S0  = HyperfineManifold(1//2, 0; label = "1S0", term = l"1S0")
    P32 = HyperfineManifold(3//2, 1; label = "3P1", term = l"3P1")
    yb0 = Ytterbium171Atom()
    Γ, ω = getdecayrates(yb0)["3P1" => "1S0"], 2π * getlinefrequencies(yb0)["3P1" => "1S0"]
    Isat = AtomTwin.Units.hbar * ω^3 * Γ / (12π * AtomTwin.Units.c^2)
    function clicks(form)
        yb  = Ytterbium171Atom(; levels = [S0..., P32...])
        sys = System(yb)
        b = PlanarBeam(2π * AtomTwin.Units.c / ω, 15Isat, [0.0, 0, 1], [1.0, 1im, 0] / √2)
        cp = form == :explicit ? add_coupling!(sys, yb, S0 => P32, 0.0, Γ * sqrt(7.5), 0.0; beam = b, active = false) :
             form == :keyword  ? add_coupling!(sys, yb, S0 => P32; beam = b, active = false) :
                                 add_coupling!(sys, yb, S0 => P32, b; active = false)
        pd = PhotoDetectorSpec(name = "n", tspan = [50e-6])
        add_detector!(sys, pd)
        add_decay!(sys, yb, P32 => S0, Γ; λ = 2π * AtomTwin.Units.c / ω, clicks = pd)
        seq = Sequence(25e-9)
        @sequence seq begin
            Pulse(cp, 50e-6)
        end
        play(sys, seq; initial_state = S0[1//2], shots = 4, rng = MersenneTwister(1)).detectors["n"]
    end
    @test clicks(:keyword) ≈ clicks(:explicit)
    @test clicks(:positional) ≈ clicks(:explicit)
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

@testset "manifold decays and Zeeman detunings return what they add" begin
    S0  = HyperfineManifold(1//2, 0; label = "1S0", term = l"1S0")
    P32 = HyperfineManifold(3//2, 1; label = "3P1", term = l"3P1")
    leak = Level("leak")
    yb  = Ytterbium171Atom(; levels = [S0..., P32..., leak])
    sys = System(yb)
    js = add_decay!(sys, yb, P32 => S0, 2π * 1e5)
    @test js isa Vector{Jump} && length(js) == 3            # one per polarisation
    jl = add_decay!(sys, yb, P32 => leak, 2π * 1e4)
    @test jl isa Vector{Jump} && length(jl) == length(P32)  # one per sublevel
    @test add_dephasing!(sys, yb, S0, 2π * 1e3) isa Vector{Jump}
    zd = add_zeeman_detunings!(sys, yb, P32; B = 1e-4)
    @test eltype(zd) <: AtomTwin.Dynamiq.Detuning && length(zd) == length(P32)
end

@testset "a beam drives fine-structure manifolds as the I = 0 hyperfine ones" begin
    # ¹⁷⁴Yb 556 nm with FineManifolds: the σ⁺ stretched line at I_sat has Ω = Γ/√2, and
    # every form gives the Hamiltonian of the equivalent F = J hyperfine manifolds.
    S0f, P1f = FineManifold(0; label = "1S0", term = l"1S0"), FineManifold(1; label = "3P1", term = l"3P1")
    S0h = HyperfineManifold(0//1, 0; label = "1S0", term = l"1S0")
    P1h = HyperfineManifold(1//1, 1; label = "3P1", term = l"3P1")
    yb0 = Ytterbium174Atom()
    Γ, ω = getdecayrates(yb0)["3P1" => "1S0"], 2π * getlinefrequencies(yb0)["3P1" => "1S0"]
    Isat = AtomTwin.Units.hbar * ω^3 * Γ / (12π * AtomTwin.Units.c^2)
    λ = 2π * AtomTwin.Units.c / ω
    pw = PlanarBeam(λ, Isat, [0.0, 0, 1], [1.0, 1im, 0] / √2)
    gb = GaussianBeam(λ, 20e-6, 1e-6; pol = [1.0, 0.3im, 0.2])
    function H(S0, P1, beam, form)
        yb = Ytterbium174Atom(; levels = [S0..., P1...]); s = System(yb)
        form == :positional ? add_coupling!(s, yb, S0 => P1, beam) :
                              add_coupling!(s, yb, S0 => P1; beam)
        gethamiltonian(s), yb
    end
    Hf, yb = H(S0f, P1f, pw, :positional)
    @test abs(Hf[yb.level_indices[P1f[1//1]], yb.level_indices[S0f[0//1]]]) ≈ Γ / (2√2) rtol = 1e-12
    for beam in (pw, gb), form in (:positional, :keyword)
        @test H(S0f, P1f, beam, form)[1] ≈ H(S0h, P1h, beam, form)[1] rtol = 1e-12
    end
end

@testset "a coupling rescaled by a complex rate stays Hermitian" begin
    # A Gaussian-beam coupling is refreshed per shot at the atom's position, with the
    # beam phase exp(ik·z): the rescale of its operator must conjugate the reverse half.
    g, e = Level("g"), Level("e")
    beam = GaussianBeam(556e-9, 5e-6, 1e-3)
    a = Atom(; levels = [g, e], x_init = gaussian(0.0, 0.0, 100e-9))
    s = System(a)
    c = add_coupling!(s, a, g => e, 2π * 1e6; beam, active = false)
    add_detector!(s, PopulationDetectorSpec(a, g; name = "Pg"))
    add_detector!(s, PopulationDetectorSpec(a, e; name = "Pe"))
    q = Sequence(1e-9)
    @sequence q begin
        Pulse(c, 2e-6)
    end
    o = play(s, q; initial_state = g, shots = 4, frozen = true, rng = MersenneTwister(1))
    @test maximum(abs.(o.detectors["Pg"] .+ o.detectors["Pe"] .- 1)) < 1e-9
    H = AtomTwin.Dynamiq.Op(s.basis, a.inner, 1 => 2, 2π * 1e6)
    AtomTwin.Dynamiq.rescale!(H, cis(0.7))
    M = Matrix(AtomTwin.Dynamiq.sparse(H))
    @test M ≈ M'
end

@testset "add_vdwinteraction! reads `transition` as add_interaction! does" begin
    g, r = Level("g"), Level("r")
    a1 = Atom(; levels = [g, r], x_init = [0.0, 0, 0])
    a2 = Atom(; levels = [g, r], x_init = [3e-6, 0, 0])
    for tr in ((r, r) => (r, r), (g, r) => (g, r), (g, r) => (r, g))
        s1 = System([a1, a2]); add_interaction!(s1, (a1, a2), tr, 1.0)
        s2 = System([a1, a2]); f = add_vdwinteraction!(s2, (a1, a2), tr, 1.0; V_cap = Inf)
        @test findall(!iszero, Matrix(gethamiltonian(s1))) ==
              findall(!iszero, Matrix(AtomTwin.Dynamiq.sparse(f.H)))
    end
    ## different level objects on the two atoms
    g1, r1, g2, r2 = Level("g1"), Level("r1"), Level("g2"), Level("r2")
    b1 = Atom(; levels = [g1, r1], x_init = [0.0, 0, 0])
    b2 = Atom(; levels = [g2, r2], x_init = [3e-6, 0, 0])
    s = System([b1, b2])
    @test add_interaction!(s, (b1, b2), (r1, r2) => (r1, r2), 1.0) isa AtomTwin.Dynamiq.AbstractField
    @test add_vdwinteraction!(s, (b1, b2), (r1, r2) => (r1, r2), 1.0) isa AtomTwin.Dynamiq.AbstractField
    ## a parametric C6 with the default cap
    s3 = System([a1, a2])
    @test add_vdwinteraction!(s3, (a1, a2), (r, r) => (r, r), Parameter(:C6, 2π * 1e6 * (4e-6)^6)) isa
          AtomTwin.Dynamiq.AbstractField
end

@testset "a Gaussian-beam coupling is built at the beam's focus and follows the atom per shot" begin
    g, e = Level("g"), Level("e")
    site = [10e-6, 0.0, 0.0]                        # 10 waists from the origin
    beam = GaussianBeam(556e-9, 1e-6, 1e-3; r0 = site)
    ## explicit Ω: an atom `x_init` places at the site sees the full Ω
    a = Atom(; levels = [g, e], x_init = site); s = System(a)
    c = add_coupling!(s, a, g => e, 2π * 1e6; beam, active = false)
    add_detector!(s, PopulationDetectorSpec(a, e; name = "Pe"))
    q = Sequence(1e-9)
    @sequence q begin
        Pulse(c, 0.5e-6)                              # a π pulse at the focus
    end
    @test play(s, q; initial_state = g, frozen = true).detectors["Pe"][end] ≈ 1 atol = 1e-6
    ## beam-derived: every |Δm| ≤ 1 pair is kept (the axis may still change its strength),
    ## although the atom sits at the origin until compile
    S0 = HyperfineManifold(1//2, 0; label = "1S0", term = l"1S0")
    P1 = HyperfineManifold(3//2, 1; label = "3P1", term = l"3P1")
    yb = Ytterbium171Atom(; levels = [S0..., P1...], x_init = site)
    @test length(add_coupling!(System(yb), yb, S0 => P1; beam)) == 6
    ## a shot beyond the beam's cutoff zeroes the operator; the next shot gets it back
    node = only(n for n in s.nodes if n isa AtomTwin.GaussianCouplingNode)
    f = node._field
    AtomTwin._refresh_gaussian!(node, f, [0.0, 0, 0], Dict{Symbol,Any}(), Random.default_rng())
    @test f.Ω0 == 0
    AtomTwin._refresh_gaussian!(node, f, site, Dict{Symbol,Any}(), Random.default_rng())
    @test abs(f.Ω0) ≈ 2π * 1e6 && abs(f.H.forward[1][3]) ≈ π * 1e6
end

@testset "build-time tol keeps a parametric Ω or rate whose default is zero" begin
    S0 = HyperfineManifold(1//2, 0; label = "1S0", term = l"1S0")
    P1 = HyperfineManifold(3//2, 1; label = "3P1", term = l"3P1")
    yb = Ytterbium171Atom(; levels = [S0..., P1...]); s = System(yb)
    cs = add_coupling!(s, yb, S0 => P1; Ω_p = Parameter(:Ω, 0.0), active = false)
    @test length(cs) == 2
    add_detector!(s, PopulationDetectorSpec(yb, P1[3//2]; name = "Pe"))
    q = Sequence(1e-9)
    @sequence q begin
        Pulse(cs, 0.5e-6)
    end
    @test play(s, q; initial_state = S0[1//2], frozen = true, Ω = 2π * 1e6).detectors["Pe"][end] ≈ 1 atol = 1e-6
    @test length(add_decay!(System(yb), yb, P1 => S0, Parameter(:Γ, 0.0))) == 3
end

@testset "a hyperfine → fine coupling adds its polarizations in quadrature" begin
    g = HyperfineLevel(2//1, 1//2, 0//1)                 # |F = 2, m = 0⟩ of a J = 1/2 ground state
    e = FineLevel(1//2, 1//2)
    rb = Rubidium87Atom(; levels = [g, e])              # I = 3/2
    Ωπ = AtomTwin.compute_coupling_strength(g, e, rb, 1.0, 0.0, 0.0)
    Ωσ = AtomTwin.compute_coupling_strength(g, e, rb, 0.0, 1.0, 0.0)
    both = AtomTwin.compute_coupling_strength(g, e, rb, 1.0, 1.0, 0.0)
    @test both ≈ hypot(Ωπ, Ωσ)
    @test both ≈ AtomTwin.compute_coupling_strength(g, e, rb, 1.0, -1.0, 0.0)   # the relative sign is unphysical
    @test AtomTwin.compute_coupling_strength(g, e, rb, Parameter(:Ω, 1.0), 0.0, 0.0) isa ParametricExpression
    @test_throws ArgumentError AtomTwin.compute_coupling_strength(g, e, rb, Parameter(:Ω, 1.0), 1.0, 0.0)
end

@testset "rabi_frequencies(atom, g, e, beam) reproduces the beam-derived Hamiltonian" begin
    S0 = HyperfineManifold(1//2, 0; label = "1S0", term = l"1S0")
    P1 = HyperfineManifold(3//2, 1; label = "3P1", term = l"3P1")
    yb0 = Ytterbium171Atom()
    Γ, ω = getdecayrates(yb0)["3P1" => "1S0"], 2π * getlinefrequencies(yb0)["3P1" => "1S0"]
    Isat = AtomTwin.Units.hbar * ω^3 * Γ / (12π * AtomTwin.Units.c^2)
    beam = PlanarBeam(2π * AtomTwin.Units.c / ω, Isat, [0.0, 0, 1], [1.0, 0.3im, 0.2])
    H(f) = (yb = Ytterbium171Atom(; levels = [S0..., P1...]); s = System(yb); f(s, yb); gethamiltonian(s))
    Hb = H((s, yb) -> add_coupling!(s, yb, S0 => P1, beam))
    Hp = H((s, yb) -> add_coupling!(s, yb, S0 => P1, rabi_frequencies(yb, S0, P1, beam)...))
    @test Hp ≈ Hb rtol = 1e-12
    ## the bare form takes the stretched-transition element, √3 below the reduced one on this line
    d = reduced_dipole(yb0, S0[1//2], P1[3//2])
    Hr = H((s, yb) -> add_coupling!(s, yb, S0 => P1, rabi_frequencies(beam; d_red = d)...))
    @test maximum(abs.(Hr)) ≈ √3 * maximum(abs.(Hb)) rtol = 1e-9
end

@testset "beam-derived couplings keep one phase convention across manifold pairs" begin
    # ⁸⁷Rb D1 (J = J′ = ½, I = 3/2): the far-detuned π–π Raman amplitude |1, 1⟩ → |2, 1⟩
    # summed over F′ = 1, 2 vanishes (no rank-2 electronic operator for J = ½, rank 0 is
    # diagonal in F); it does not for σ⁺–π. A sign fixed pair by pair broke this.
    rb = Rubidium87Atom()
    lv(F, m) = HyperfineLevel(F, 1//2, m, 0.0, "")
    d(F, m, Fp, mp) = AtomTwin.dipole_matrix_element(rb, lv(F, m), lv(Fp, mp))
    raman(m1, q1, m2) = sum(d(2//1, m2, Fp, m1 + q1) * d(1//1, m1, Fp, m1 + q1)
                            for Fp in (1//1, 2//1) if abs(m1 + q1) <= Fp)
    @test abs(raman(1//1, 0, 1//1)) < 1e-12          # π–π
    @test abs(raman(0//1, 0, 0//1)) < 1e-12
    @test abs(raman(0//1, 1, 1//1)) > 0.1             # σ⁺–π
end

@testset "d_red is honoured where there is no line data, and E1 selection rules are checked" begin
    g, e = Level("g"), Level("e")
    S = FineManifold(0; label = "S"); P = FineManifold(1; label = "P"); D = FineManifold(2; label = "D")
    a = Atom(; levels = [S..., P..., D...])
    pw = PlanarBeam(600e-9, 1.0, [0.0, 0, 1], [1.0, 0, 0])
    gb = GaussianBeam(600e-9, 10e-6, 1e-3)
    pb = GaussianBeam(600e-9, 10e-6, Parameter(:P, 1e-3))
    ## every |Δm| ≤ 1 pair, resolved at compile about the system's axis; an explicit
    ## `q_axis` fixes the strengths, and `tol` then drops the π pair of x̂ light
    @test length(add_coupling!(System(a), a, S => P; beam = pw, d_red = 1e-29)) == 3
    @test length(add_coupling!(System(a), a, S => P, gb; d_red = 1e-29)) == 3
    @test length(add_coupling!(System(a), a, S => P, pb; d_red = 1e-29)) == 3
    @test length(add_coupling!(System(a), a, S => P; beam = pw, d_red = 1e-29, q_axis = [0, 0, 1.0])) == 2
    @test_throws ErrorException add_coupling!(System(a), a, S => P; beam = pw)          # no line data, no d_red
    @test_throws ArgumentError add_coupling!(System(a), a, S => D; beam = pw, d_red = 1e-29)   # |ΔJ| = 2
    ## noise is not silently dropped with a beam
    b = Atom(; levels = [g, e])
    @test_throws ArgumentError add_coupling!(System(b), b, g => e, 1.0; beam = pw, noise = LaserPhaseNoiseModel())
    ## a parametric Gaussian beam carries its polarization
    rp = AtomTwin._resolve_beam_default(GaussianBeam(600e-9, Parameter(:w, 1e-6), 1e-3; pol = [0, 1, 0]))
    @test rp.pol ≈ ComplexF64[0, 1, 0]
end

@testset "atom construction rejects duplicate levels and an F outside |J−I| … J+I" begin
    g1, g2 = HyperfineManifold(1//2, 0), HyperfineManifold(1//2, 0)
    @test_throws ArgumentError Ytterbium171Atom(; levels = [g1..., g2...])
    a = Ytterbium171Atom(; levels = [HyperfineManifold(1//2, 0; label = "a")...,
                                     HyperfineManifold(1//2, 0; label = "b")...])
    @test length(a.level_indices) == 4
    P32 = HyperfineManifold(3//2, 1; label = "3P1", term = l"3P1")
    @test_throws ArgumentError Ytterbium174Atom(; levels = [P32...])      # I = 0 has no F = 3/2
    @test Ytterbium171Atom(; levels = [P32...]) isa AbstractAtom
end

@testset "polarizability bookkeeping warns once and checks the level's J against its model" begin
    g, e = Level("g"), Level("e")
    a = Atom(; levels = [g, e])
    beam = GaussianBeam(1064e-9, 1e-6, 10e-3)
    @test_logs (:warn, r"no polarizability models") AtomTwin._init_species_data!(a, a.inner, [beam])
    @test_logs AtomTwin._init_species_data!(a, a.inner, AtomTwin.AbstractBeam[])     # no beam, no warning
    yb = Ytterbium171Atom(; levels = [Level("1S0"), Level("leak")])
    @test_logs (:warn, r"No polarizability model for 1 level") AtomTwin._init_species_data!(yb, yb.inner, [GaussianBeam(759e-9, 1e-6, 1e-3)])
    bad = Ytterbium171Atom(; levels = [HyperfineManifold(1//2, 0; term = "3P1")...])   # a string term skips the J check
    @test_throws ErrorException AtomTwin._init_species_data!(bad, bad.inner, [GaussianBeam(520e-9, 1e-6, 1e-3)])
    ## a fine level is the I = 0 picture on any isotope, as for its couplings
    P1 = FineManifold(1; label = "3P1", term = l"3P1")
    a171, a174 = Ytterbium171Atom(; levels = [P1...]), Ytterbium174Atom(; levels = [P1...])
    b = GaussianBeam(520e-9, 1e-6, 1e-3)
    AtomTwin._init_species_data!(a171, a171.inner, [b]); AtomTwin._init_species_data!(a174, a174.inner, [b])
    λ = AtomTwin.getwavelength(b)
    @test a171.inner.alpha[λ] ≈ a174.inner.alpha[λ]
    @test a171.inner.alpha[λ][1] != a171.inner.alpha[λ][2]                 # the tensor part is there
end

@testset "species masses are isotope masses" begin
    # AME2020 (Wang et al., Chin. Phys. C 45, 030003): ¹⁷⁴Yb 173.938 866 4 u,
    # ⁸⁷Rb 86.909 180 5 u, ¹⁷¹Yb 170.936 330 2 u, ⁸⁸Sr 87.905 612 2 u.
    @test Ytterbium174Atom().m / AtomTwin.Units.amu ≈ 173.9388664 rtol = 1e-7
    @test Rubidium87Atom().m / AtomTwin.Units.amu ≈ 86.9091805 rtol = 1e-7
    @test Ytterbium171Atom().m / AtomTwin.Units.amu ≈ 170.9363302 rtol = 1e-7
    @test Strontium88Atom().m / AtomTwin.Units.amu ≈ 87.9056122 rtol = 1e-7
end

@testset "gethamiltonian refreshes explicit light shifts for the current axis; elliptical trap light warns" begin
    P1 = HyperfineManifold(1//1, 1; label = "3P1", term = l"3P1")
    tw = GaussianBeam(520e-9, 1e-6, 1e-3; pol = [1.0, 0, 0])
    function H(order)
        sr = Strontium88Atom(; levels = [P1...]); s = System(sr)
        order == :axis_first && add_quantization_axis!(s, [1.0, 0, 0])
        add_light_shift!(s, sr, P1, tw)
        order == :axis_last && add_quantization_axis!(s, [1.0, 0, 0])
        real.(diag(gethamiltonian(s)))
    end
    @test H(:axis_last) ≈ H(:axis_first)
    @test H(:axis_last)[1] != H(:axis_last)[2]                   # ε ∥ axis: m = 0 differs from |m| = 1
    sr = Strontium88Atom(; levels = [P1...])
    circ = GaussianBeam(520e-9, 1e-6, 1e-3; pol = [1.0, 1im, 0] / √2)
    @test_logs (:warn, r"not linearly polarised") match_mode = :any AtomTwin._init_species_data!(sr, sr.inner, [circ])
    @test_logs AtomTwin._init_species_data!(sr, sr.inner, [tw])
end

@testset "laser phase noise reproduces its spectrum: white floor and servo bump" begin
    # S_ν in Hz²/Hz, φ = 2π∫δν dt: a white floor h₀ gives Var[φ(T) − φ(0)] = 2π² h₀ T,
    # a servo bump Var[φ] = ∫ S_ν/f² df.
    g, e = Level("g"), Level("e")
    function trace(nm, T, dt)
        a = Atom(; levels = [g, e]); s = System(a)
        add_coupling!(s, a, g => e, 1.0; noise = nm), collect(dt:dt:T)
    end
    h0, T, dt = 2.5e4, 100e-6, 2e-8
    nf, tspan = trace(LaserPhaseNoiseModel(bump_ampl = 0.0, powerlaw_ampl = h0), T, dt)
    Random.seed!(nf.rng, 11)
    v = 0.0
    for _ in 1:300
        z = AtomTwin.generate_noise(tspan, 0.0, nf)
        φ = cumsum(vcat(0.0, angle.(z[2:end] ./ z[1:end-1])))
        v += (φ[end] - φ[1])^2 / 300
    end
    @test 0.75 < v / (2π^2 * h0 * T) < 1.1
    hg, fg, σ = 2e5, 1e6, 2e5
    nf2, tspan2 = trace(LaserPhaseNoiseModel(bump_ampl = hg, bump_center = fg, bump_width = σ, powerlaw_ampl = 0.0), 20e-6, 2e-9)
    Random.seed!(nf2.rng, 12)
    v2 = sum(angle(AtomTwin.generate_noise(tspan2, 0.0, nf2)[end])^2 for _ in 1:300) / 300
    f = exp.(range(log(1e4), log(1e8); length = 20000))
    @test v2 ≈ sum(AtomTwin.laser_phase_psd(f, nf2.noise) .* f .* log(f[2] / f[1])) rtol = 0.15
    @test AtomTwin.laser_phase_psd(fg, nf2.noise) ≈ hg / fg^2        # S_φ = S_ν/f²
end

@testset "laser phase noise damps a Rabi oscillation at π² S_ν(Ω/2π)/2" begin
    # A white frequency-noise floor h₀ (one-sided, Hz²/Hz) on a resonant drive: the
    # noise at the Rabi frequency drives transitions between the dressed states and
    # the oscillation's envelope decays at Γ = π² h₀ / 2 (half the Ramsey dephasing
    # rate π² h₀), so P_e(t) = ½(1 − cos(Ωt) e^{−Γt}). Checked through the whole
    # path: model → NoisyField → per-shot synthesis → solver.
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    Ω, h0 = 2π * 1e6, 4e4
    Γ = π^2 * h0 / 2
    noise = LaserPhaseNoiseModel(bump_ampl = 0.0, powerlaw_ampl = h0, powerlaw_exp = 0.0)
    c = add_coupling!(sys, atom, g => e, Ω; noise, active = false)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))
    T = 9.5e-6                                     # cos(ΩT) = −1
    seq = Sequence(2π / (20Ω); tol = 1e-4)
    @sequence seq begin
        Pulse(c, T)
    end
    N = 600
    out = play(sys, seq; initial_state = g, shots = N, rng = MersenneTwister(11))
    Pe = mean(out.detectors["P_e"][end, :])
    @test Pe ≈ 0.5 * (1 + exp(-Γ * T)) atol = 4 * 0.5 / sqrt(N)   # 0.58; the old synthesis gave 0.9
    @test Pe < 0.75
end

@testset "a symbolic operator materialises as the low-level operator does" begin
    g, e = Level("g"), Level("e")
    a1, a2 = Atom(; levels = [g, e]), Atom(; levels = [g, e])
    s = System([a1, a2])
    P = Matrix(AtomTwin.Dynamiq.sparse(AtomTwin.Op(AtomTwin.projector(e), s, a2)))
    s2 = System([a1, a2]); add_detuning!(s2, a2, e, -1.0)            # H_ee = −δ = |e⟩⟨e| on atom 2
    @test P ≈ gethamiltonian(s2)
    X = Matrix(AtomTwin.Dynamiq.sparse(AtomTwin.Op(AtomTwin.transition(g => e), s, a1)))
    s3 = System([a1, a2]); add_coupling!(s3, a1, g => e, 2.0)
    @test X + X' ≈ gethamiltonian(s3)
end
