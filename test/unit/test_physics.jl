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
         (freq_THz = 384.230485, dipole_ea0 = 5.977)];
        J = 1//2)   # Rb 5S₁/₂. Declaring J must NOT perturb a dipole-specified
                    # scalar sum — the 1/(2Jg+1) weight is already inside Γ_eff.
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
    # Both models must land on the same `f`: leave J/J_f at their defaults so the
    # dipole line (f = 3 by construction) and the gamma line (f(0,1) = 3) agree.
    m_d = PolarizabilityModel("x", [(freq_THz = 377.107463, dipole_ea0 = 4.227)])
    m_g = PolarizabilityModel("x",
        [(freq_THz = 377.107463, gamma_MHz = m_d.transitions[1].gamma_MHz)])
    @test polarizability_au(m_g, 850.0) ≈ polarizability_au(m_d, 850.0)
end

@testset "Sr atom-facing polarizability methods resolve" begin
    sr = Strontium88Atom(; levels = [Level("1S0")])
    for st in ("1S0", "3P0")
        @test isfinite(light_shift_coeff_Hz_per_Wcm2(sr, st, 813.4))
        @test isfinite(polarizability_au(sr, st, 813.4))
        @test scattering_rate_per_Wcm2(sr, st, 813.4) > 0
    end
end

@testset "Sr static polarizabilities match Safronova 2013 recommended values" begin
    # Line lists from Phys. Rev. A 87, 012509 (2013) reproduce the recommended
    # static scalar polarizabilities: α(1S0)=197.14, α(3P0)=444.51 a.u.
    sr = Strontium88Atom(; levels = [Level("1S0")])
    @test isapprox(polarizability_au(sr, "1S0", 1.0e7), 197.1; rtol = 0.02)
    @test isapprox(polarizability_au(sr, "3P0", 1.0e7), 444.5; rtol = 0.02)
end

@testset "813.4 nm is magic: 1S0 and 3P0 trap depths agree" begin
    # The 3P0 tail offset is anchored to the measured 813.428 nm crossing, so the
    # two trap depths must be equal there (to <1%) and the crossing near 813.4 nm.
    sr  = Strontium88Atom(; levels = [Level("1S0")])
    ls1 = light_shift_coeff_Hz_per_Wcm2(sr, "1S0", 813.428)
    ls3 = light_shift_coeff_Hz_per_Wcm2(sr, "3P0", 813.428)
    @test ls1 < 0 && ls3 < 0                       # both states trapped (red shift)
    @test abs(ls1 - ls3) / abs(ls1) < 0.01         # equal depth ⇒ magic

    # Locate the crossing by bisection; it must land within 0.1 nm of 813.428.
    f(λ) = light_shift_coeff_Hz_per_Wcm2(sr, "3P0", λ) -
           light_shift_coeff_Hz_per_Wcm2(sr, "1S0", λ)
    lo, hi = 800.0, 826.0
    for _ in 1:60
        m = (lo + hi) / 2
        f(lo) * f(m) <= 0 ? (hi = m) : (lo = m)
    end
    @test isapprox((lo + hi) / 2, 813.428; atol = 0.1)
end


# ======================================================================
# Angular momentum: line strength, tensor polarizability, trap light shift
# ======================================================================

@testset "line-strength factor f(J,J′) and its sign rule" begin
    # The kernel used to hard-code f = 3, correct only for J=0 → J′=1. A line
    # BELOW the state carries f = −1 — the opposite sign — which is what makes the
    # two states of a two-level atom shift oppositely.
    f = AtomTwin._line_strength_factor
    @test f(0//1, 1//1, +539.0) == 3//1     # the historical case, unchanged
    @test f(1//1, 0//1, -539.0) == -1//1    # below in energy
    @test f(1//1, 2//1, +202.0) == 5//3

    above = PolarizabilityModel("g", [(freq_THz = +539.3868, gamma_MHz = 0.183,
                                       J_f = 1//1)]; J = 0//1)
    below = PolarizabilityModel("e", [(freq_THz = -539.3868, gamma_MHz = 0.183,
                                       J_f = 0//1)]; J = 1//1)
    la = light_shift_coeff_Hz_per_Wcm2(above, 767.0)
    lb = light_shift_coeff_Hz_per_Wcm2(below, 767.0)
    @test isapprox(lb / la, -1/3; rtol = 1e-12)   # = f(−1)/f(3), sign included

    # A dipole-specified line already has 1/(2Jg+1) inside Γ_eff, so declaring J
    # must NOT perturb its scalar sum — applying f(J,J′) again would double-count.
    d0 = polarizability_au(
        PolarizabilityModel("x", [(freq_THz = 377.1, dipole_ea0 = 4.227)]), 850.0)
    dJ = polarizability_au(
        PolarizabilityModel("x", [(freq_THz = 377.1, dipole_ea0 = 4.227)];
                            J = 1//2), 850.0)
    @test d0 == dJ

    # Provenance is recorded and validated (a refit must tell fitted from measured).
    @test PolarizabilityModel("x", [(freq_THz = 100.0, gamma_MHz = 1.0,
                                     source = :fitted)]).transitions[1].source == :fitted
    @test_throws ErrorException PolarizabilityModel("x",
        [(freq_THz = 100.0, gamma_MHz = 1.0, source = :guessed)])
    @test_throws ErrorException PolarizabilityModel("x",
        [(freq_THz = 100.0, gamma_MHz = 1.0, J = 2//1)]; J = 1//1)
end

@testset "tensor polarizability: Sr-88 3P1 magic wavelengths and vanishing rules" begin
    # Kestler et al., PRA 105, 012821 (2022) measured two 1S0–3P1 magic
    # wavelengths near 473 nm — precision the Yb 3P1 fit (good to ~3 nm) cannot
    # match, so this is the accuracy gate for the whole tensor path.
    m1S0 = AtomTwin.SR88_POLARIZABILITY_1S0
    m3P1 = AtomTwin.SR88_POLARIZABILITY_3P1
    au   = 4π * AtomTwin.Units.ε0 * AtomTwin.Units.a0^3

    # Scalar and tensor both land inside the paper's own uncertainty.
    @test isapprox(polarizability_au(m1S0, 473.1445), 3637; atol = 17)
    @test isapprox(polarizability_au(m3P1, 473.1445), 4146; atol = 117)
    @test isapprox(AtomTwin._alpha2_si(m3P1, 473.1445; F = 1//1, I = 0//1) / au,
                   -509; atol = 15)

    # THE INVARIANT. With geometric factors −2 (m=0) and +1 (|m|=1) the splitting
    # is exactly −3α⁽²⁾. α⁽²⁾ and the U/I conversion must share one convention:
    # pairing the notes' 3π prefactor with AtomTwin's 1/(cε₀) halves this while
    # leaving α⁽²⁾ itself looking right against a published table.
    s0 = light_shift_coeff_Hz_per_Wcm2(m3P1, 473.1445; F = 1//1, mF = 0//1,
                                       I = 0//1, ε_z = 1.0)
    s1 = light_shift_coeff_Hz_per_Wcm2(m3P1, 473.1445; F = 1//1, mF = 1//1,
                                       I = 0//1, ε_z = 1.0)
    α2 = AtomTwin._alpha2_si(m3P1, 473.1445; F = 1//1, I = 0//1) / au
    split = -(s0 - s1) * AtomTwin.Units.h * 1e-4 *
             AtomTwin.Units.c * AtomTwin.Units.ε0 / au
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
    # THE DELIVERABLE: sublevels of one manifold no longer share a polarizability.
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

@testset "a trapping beam shifts the levels it traps" begin
    # A trap that holds an atom also shifts its levels: one physical effect, so
    # passing the beam to `System` is the whole user action. Ramsey is the
    # end-to-end check — the fringe follows the differential shift.
    g, e = Level("g"; term = l"1S0"), Level("e"; term = l"3P1")
    function ramsey(T; P = 1e-3, explicit = false)
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
    tw0 = GaussianBeam(λ = 520e-9, w0 = 1e-6, P = 1e-3, pol = [0, 0, 1])
    inner = AtomTwin.Dynamiq.NLevelAtom(2)
    AtomTwin.initialize!(sr0, inner; beams = [tw0], q_axis = [0.0, 0.0, 1.0])
    αs = inner.alpha[520e-9]
    Tπ = π / abs((αs[2] - αs[1]) * tw0.I0 / AtomTwin.Units.hbar)

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
        # The coefficient is α·I/ħ at the atom — an angular frequency.
        AtomTwin.Dynamiq.update!(fs[end], 1)
        @test isapprox(real(fs[end]._coeff[]),
                       fs[end].alpha * AtomTwin.Dynamiq.intensity(tw, sr.inner.x) /
                       AtomTwin.Units.hbar; rtol = 1e-12)
        explicit && @test fs[1].alpha == 0.0      # reference = g sits at zero
        length(fs)
    end
    @test n_fields(false) == 2
    @test n_fields(true)  == 2                   # NOT 4

    # The shipped motion examples are unaffected because 759 nm is magic for the
    # Yb clock pair: the shift IS applied, the differential is just negligible.
    yb = Ytterbium171Atom(; levels = [Level("1S0"), Level("3P0")])
    twyb = GaussianBeam(λ = 759e-9, w0 = 1.0e-6, P = 50e-3)
    iyb = AtomTwin.Dynamiq.NLevelAtom(2)
    AtomTwin.initialize!(yb, iyb; beams = [twyb])
    a = iyb.alpha[759e-9]
    @test abs((a[2] - a[1]) / a[1]) < 0.01
end
