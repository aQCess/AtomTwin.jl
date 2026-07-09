# Physics regression tests: Rabi, dissipation, selection rules, GaussianBeam, DAG parameters
#
# These tests fail when physics is wrong, not just when struct layout changes.
# Also includes play() state-management regression tests.

import AtomTwin.Dynamiq

# ── A. Rabi oscillations ──────────────────────────────────────────────────────

@testset "Rabi: resonant π-pulse flips population" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    Ω    = 2π * 1e6
    coupling = add_coupling!(sys, atom, g => e, Ω; active = false)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))
    T_pi = π / Ω   # 500 ns
    seq  = Sequence(1e-9)
    @sequence seq begin
        Pulse(coupling, T_pi)
    end
    out = play(sys, seq; initial_state = g)
    @test out.detectors["P_e"][end] > 0.99
end

@testset "Rabi: resonant 2π-pulse returns to ground state" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    Ω    = 2π * 1e6
    coupling = add_coupling!(sys, atom, g => e, Ω; active = false)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))
    T_2pi = 2π / Ω   # 1000 ns
    seq   = Sequence(1e-9)
    @sequence seq begin
        Pulse(coupling, T_2pi)
    end
    out = play(sys, seq; initial_state = g)
    @test out.detectors["P_e"][end] < 0.01
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
    # Regression: the density-matrix dissipator applied L ρ L† in place and then
    # read the modified ρ back for the -½{L†L,ρ} term, so the gain and loss acted
    # on different inputs and the trace leaked (Tr ρ ≈ 0.96 at t=1µs, γ=2π·1MHz).
    # Pure dephasing L = √γ|e⟩⟨e| must leave populations and trace exactly fixed
    # and damp coherences at γ/2.
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

    # Start in |+⟩ = (|g⟩+|e⟩)/√2: this is the case that exposed the bug — it has
    # both population on the dephased level and a coherence to damp.
    out = play(sys, seq; initial_state = (1/√2)*g + (1/√2)*e, density_matrix = true)

    # Trace preserved at every recorded time (pre-fix leaked to Tr ≈ 0.96)
    trace = out.detectors["P_g"] .+ out.detectors["P_e"]
    @test all(isapprox.(trace, 1.0; atol = 1e-9))
    # Pure dephasing leaves populations fixed (P_g = P_e = 1/2)
    @test all(isapprox.(out.detectors["P_e"], 0.5; atol = 1e-9))
    @test all(isapprox.(out.detectors["P_g"], 0.5; atol = 1e-9))
    # Coherence damps at γ/2 (standard L=√γ|e⟩⟨e| convention), not γ
    @test isapprox(abs(out.detectors["coh"][end]), 0.5 * exp(-γ * 1e-6 / 2); rtol = 1e-2)
end

# ── C. Selection rules ────────────────────────────────────────────────────────

@testset "Selection rules: |ΔmF| > 1 returns zero coupling" begin
    # F=1, mF=-1  →  F=2, mF=+1 : ΔmF = 2, forbidden
    g_level = HyperfineLevel(1//1, 0//1, -1//1, 1.0, "g")
    e_level = HyperfineLevel(2//1, 1//1,  1//1, 1.0, "e")
    @test AtomTwin.compute_coupling_strength(g_level, e_level, nothing, 1.0, 1.0, 1.0) == 0.0
end

@testset "Selection rules: σ⁺ drives Δm=+1, π cannot" begin
    # F=1, mF=0  →  F=2, mF=+1 : ΔmF = +1 (allowed for σ⁺)
    g_level  = HyperfineLevel(1//1, 0//1, 0//1, 1.0, "g")
    e_level  = HyperfineLevel(2//1, 1//1, 1//1, 1.0, "e")
    # σ⁺ polarisation couples Δm=+1
    @test AtomTwin.compute_coupling_strength(g_level, e_level, nothing, 0.0, 1.0, 0.0) != 0.0
    # π polarisation cannot couple Δm=+1
    @test AtomTwin.compute_coupling_strength(g_level, e_level, nothing, 1.0, 0.0, 0.0) == 0.0
end

# ── D. GaussianBeam analytical checks ────────────────────────────────────────

@testset "GaussianBeam: peak intensity equals 2P/(π w0²)" begin
    λ  = 1e-6
    w0 = 50e-6
    P  = 1e-3
    b  = GaussianBeam(λ, w0, P)
    I0_analytic = 2 * P / (π * w0^2)
    @test isapprox(Dynamiq.intensity(b, [0.0, 0.0, 0.0]), I0_analytic, rtol = 1e-10)
end

@testset "GaussianBeam: |E(0,0,0)| matches analytical formula" begin
    c_v  = 2.997_924_58e8
    ε0_v = 8.854_187_812_8e-12
    λ  = 1e-6
    w0 = 50e-6
    P  = 1e-3
    b  = GaussianBeam(λ, w0, P)
    I0 = 2 * P / (π * w0^2)
    E_analytical = sqrt(2 * I0 / (c_v * ε0_v))
    E_computed   = abs(Dynamiq.Efield(b, [0.0, 0.0, 0.0]))
    @test isapprox(E_computed, E_analytical, rtol = 1e-10)
    # Cross-check: |E|² · c·ε0/2 must equal intensity
    @test isapprox(E_computed^2 * c_v * ε0_v / 2, Dynamiq.intensity(b, [0.0, 0.0, 0.0]), rtol = 1e-10)
end

@testset "GaussianBeam: 8-waist cutoff returns zero field" begin
    λ  = 1e-6
    w0 = 50e-6
    P  = 1e-3
    b  = GaussianBeam(λ, w0, P)
    # 10·w0 in x-direction: (10w0)²/w0² = 100 ≫ 16 → zero
    r_far = [10 * w0, 0.0, 0.0]
    @test Dynamiq.Efield(b, r_far) == 0.0 + 0.0im
    @test Dynamiq.intensity(b, r_far) == 0.0
end

# ── E. DAG parameter resolution: end-to-end ───────────────────────────────────
#
# Each test runs a π-pulse where Ω is encoded as a DAG expression and checks
# that the simulation produces the same result as the direct numeric case.

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

@testset "DAG: multiplicative expression 2*p resolves correctly" begin
    # 2 * Parameter(:Omega, 2π*0.5e6) should behave like Ω = 2π*1e6
    Ω_expr = 2.0 * Parameter(:Omega, 2π * 0.5e6)
    out = _run_pi_pulse(Ω_expr)
    @test out.detectors["P_e"][end] > 0.99
end

@testset "DAG: additive expression p+q resolves correctly" begin
    # Parameter(:A, 2π*0.5e6) + Parameter(:B, 2π*0.5e6) = 2π*1e6
    Ω_expr = Parameter(:A, 2π * 0.5e6) + Parameter(:B, 2π * 0.5e6)
    out = _run_pi_pulse(Ω_expr)
    @test out.detectors["P_e"][end] > 0.99
end

@testset "DAG: subtraction expression p-q resolves correctly" begin
    # Parameter(:A, 2π*1.5e6) - Parameter(:B, 2π*0.5e6) = 2π*1e6
    Ω_expr = Parameter(:A, 2π * 1.5e6) - Parameter(:B, 2π * 0.5e6)
    out = _run_pi_pulse(Ω_expr)
    @test out.detectors["P_e"][end] > 0.99
end

@testset "DAG: inverse expression inv(p) resolves correctly" begin
    # inv(Parameter(:inv_Omega, 1/(2π*1e6))) = 2π*1e6
    Ω_expr = inv(Parameter(:inv_Omega, 1.0 / (2π * 1e6)))
    out = _run_pi_pulse(Ω_expr)
    @test out.detectors["P_e"][end] > 0.99
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

# ── Z. play() state-management regression ────────────────────────────────────

@testset "play: ME with initial_state then multi-shot MCWF on same sys does not crash" begin
    # Regression: play(dm_job, sys; initial_state=..., density_matrix=true) sets
    # sys.state[] to a Matrix.  A subsequent multi-shot WF play on the same sys
    # would have crashed in recompile! (Vector .= Matrix DimensionMismatch).
    g, e  = Level("g"), Level("e")
    atom  = Atom(; levels=[g, e])
    sys   = System(atom)
    Ω     = 2π * 1e6
    coup  = add_coupling!(sys, atom, g => e, Ω; active=false)
    add_dephasing!(sys, atom, e, 2π * 0.1e6)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name="P_e"))
    seq   = Sequence(1e-9)
    @sequence seq begin Pulse(coup, 10e-9) end

    # compile expects initial_state as a vector (high-level play calls _tovector)
    job_me   = compile(sys, seq; density_matrix=true,  initial_state=[g])
    job_mcwf = compile(sys, seq; density_matrix=false, initial_state=[g])

    # play(job, sys; initial_state=g, density_matrix=true) sets sys.state[] to a Matrix
    play(job_me, sys; initial_state=g, density_matrix=true)
    @test sys.state[] isa Matrix

    # Multi-shot WF play on the same sys must not crash
    out = play(job_mcwf, sys; shots=5)
    @test length(out.times) == 10
    @test haskey(out.detectors, "P_e")
end

# ── Polarizability & off-resonant scattering (Yb-171) ─────────────────────────

using AtomTwin.Units: c, hbar

@testset "Yb-171 atom-facing polarizability methods resolve" begin
    # Regression: these used a stale `Polarizability.` module qualifier and threw
    # UndefVarError. They must now dispatch and return finite numbers.
    yb = Ytterbium171Atom(; levels = [Level("1S0")])
    for st in ("1S0", "3P0")
        @test isfinite(light_shift_coeff_Hz_per_Wcm2(yb, st, 759.0))
        @test isfinite(polarizability_au(yb, st, 759.0))
        @test scattering_rate_per_Wcm2(yb, st, 759.0) > 0
    end
end

@testset "759 nm is magic: 1S0 and 3P0 trap depths agree" begin
    yb  = Ytterbium171Atom(; levels = [Level("1S0")])
    ls1 = light_shift_coeff_Hz_per_Wcm2(yb, "1S0", 759.0)
    ls3 = light_shift_coeff_Hz_per_Wcm2(yb, "3P0", 759.0)
    @test ls1 < 0 && ls3 < 0                       # both states trapped (red shift)
    @test abs(ls1 - ls3) / abs(ls1) < 0.01         # equal depth to <1% ⇒ magic
end

@testset "scattering_rate_per_Wcm2 matches independent Grimm sum" begin
    # Hand-rolled Grimm-et-al. (2000) rate sum over the 3P0 transition table
    # (Phys. Rev. A 108, 053325) — independent of the implementation under test.
    trans = [(215.870446, 0.308), (461.867846, 1.516), (675.141040, 4.081),
             (729.293151, 0.625), (797.204099, 22.889)]
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

@testset "per-line Grimm identity Γsc = (Γ/Δ)·U_dip/ħ" begin
    # In the RWA far-detuned limit, a single line's scattering rate and light
    # shift obey Γsc = (Γ/Δ)·U_dip/ħ exactly. Check on each 3P0 line.
    trans = [(215.870446, 0.308), (461.867846, 1.516), (675.141040, 4.081),
             (729.293151, 0.625), (797.204099, 22.889)]
    ωL = 2π * c / (759.0 * 1e-9)
    for (fTHz, gMHz) in trans
        ω0 = 2π * fTHz * 1e12; Γ = 2π * gMHz * 1e6; Δ = ωL - ω0
        Udip = (3π * c^2 / (2 * ω0^3)) * (Γ / Δ)          # J/(W/m²), RWA single line
        Γsc  = (3π * c^2 / (2 * hbar * ω0^3)) * (Γ / Δ)^2 # 1/s/(W/m²), RWA single line
        @test isapprox(abs(Γsc), abs((Γ/Δ) * Udip / hbar); rtol = 1e-12)
    end
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
        [(freq_THz = 377.107463, dipole_ea0 = 4.227),
         (freq_THz = 384.230485, dipole_ea0 = 5.977)])
    for λ in (1e7, 1200.0, 1000.0, 900.0, 850.0, 800.0)
        @test isapprox(polarizability_au(m, λ), _alpha_dipoles_au(λ, ref); rtol = 1e-9)
    end
    # D2 carries ~2× the line strength of D1: the produced effective widths encode
    # that ratio (unlike the near-equal natural widths 5.75 / 6.07 MHz).
    g1, g2 = m.transitions[1].gamma_MHz, m.transitions[2].gamma_MHz
    @test 1.9 < g2 / g1 < 2.2
end

@testset "natural-Γ alkali doublet is wrong off-resonance (the trap it avoids)" begin
    # The naive user path — the two D lines with their natural linewidths — agrees
    # with the truth at the static limit but drifts toward the D lines. The
    # dipole-specified model does not. This pins the value of the dipole API.
    ref   = [(4.227, 377.107463e12), (5.977, 384.230485e12)]
    m_nat = PolarizabilityModel("5S1/2",
        [(freq_THz = 377.107463, gamma_MHz = 5.7500),   # D1 natural Γ
         (freq_THz = 384.230485, gamma_MHz = 6.0666)])  # D2 natural Γ
    @test isapprox(polarizability_au(m_nat, 1e7), _alpha_dipoles_au(1e7, ref); rtol = 0.02)
    @test !isapprox(polarizability_au(m_nat, 850.0), _alpha_dipoles_au(850.0, ref); rtol = 0.02)
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

@testset "gamma_MHz path unchanged; Yb 759 nm magic wavelength intact" begin
    # Yb models are gamma_MHz-based; the dipole extension must not perturb them.
    yb  = Ytterbium171Atom(; levels = [Level("1S0")])
    ls1 = light_shift_coeff_Hz_per_Wcm2(yb, "1S0", 759.0)
    ls3 = light_shift_coeff_Hz_per_Wcm2(yb, "3P0", 759.0)
    @test ls1 < 0 && ls3 < 0
    @test abs(ls1 - ls3) / abs(ls1) < 0.01
    # A gamma_MHz line and the dipole line that produces the same effective width
    # (at the same frequency) give identical polarizability — round-trip of the
    # normalisation. Take g_eff from a dipole model at the SAME freq to keep it exact.
    m_d = PolarizabilityModel("x", [(freq_THz = 377.107463, dipole_ea0 = 4.227)])
    m_g = PolarizabilityModel("x",
        [(freq_THz = 377.107463, gamma_MHz = m_d.transitions[1].gamma_MHz)])
    @test polarizability_au(m_g, 850.0) ≈ polarizability_au(m_d, 850.0)
end

