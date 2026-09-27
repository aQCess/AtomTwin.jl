# Classical motion regression test: trap frequency of atom in a Gaussian tweezer.
#
# Validates that a Ytterbium-171 atom displaced from the beam centre oscillates
# at the analytically expected trap frequency.  Fails if the force calculation,
# polarizability model, or classical integrator is broken.
#
# Pattern follows atom_sorting.jl: classical-only dynamics via play(sys, seq)
# without an initial quantum state.

@testset "Classical motion: trap oscillation at correct frequency" begin
    c_v  = 2.997_924_58e8        # m/s
    ε0_v = 8.854_187_812_8e-12   # F/m
    amu  = 1.660_539_066_60e-27  # kg
    m    = 171 * amu             # Yb-171 mass

    # Tweezer parameters
    λ_nm = 759.0
    λ    = λ_nm * 1e-9
    w0   = 2e-6    # 2 µm waist
    P    = 0.1     # 100 mW

    # Analytical trap frequency
    α_SI   = polarizability_si(AtomTwin.YB171_POLARIZABILITY_1S0, λ_nm)
    I0     = 2 * P / (π * w0^2)
    ω_trap = sqrt(4 * α_SI * I0 / (m * c_v * ε0_v * w0^2))
    T_trap = 2π / ω_trap

    # Initial displacement: 10% of waist in x
    x0 = 0.10 * w0

    # Single-level atom (no quantum dynamics) displaced from the beam centre
    atom = Ytterbium171Atom(;
        levels = [Level("1S0")],
        x_init = [x0, 0.0, 0.0],
        v_init = [0.0, 0.0, 0.0],
    )
    tweezer = GaussianBeam(λ, w0, P)
    sys = System(atom, tweezer)
    add_detector!(sys, MotionDetectorSpec(atom; dims = [1], name = "x"))

    # Simulate for 3 trap periods using Wait (no quantum instruction needed)
    T_sim = 3 * T_trap
    dt    = min(T_trap / 200, 100e-9)
    seq   = Sequence(dt)
    @sequence seq begin
        Wait(T_sim)
    end

    # No initial_state → classical dynamics only, matching atom_sorting pattern.
    # Explicitly expect (and capture) the resulting warning so it doesn't appear as noise.
    out = @test_logs (:warn, r"Initial state not specified") play(sys, seq)
    x_traj = out.detectors["x"][:, 1]
    t_traj = out.times

    # 1. Atom must stay trapped (not escape)
    @test maximum(abs.(x_traj)) < 2 * x0

    # 2. First zero crossing must exist: x(t)=x0·cos(ω_trap·t) → zero at T_trap/4
    sign0 = sign(x_traj[1])
    first_zero_idx = findfirst(i -> sign(x_traj[i]) != sign0, 2:length(x_traj))
    @test !isnothing(first_zero_idx)

    if !isnothing(first_zero_idx)
        t_first_zero = t_traj[first_zero_idx]
        ω_measured   = π / (2 * t_first_zero)   # quarter-period → full frequency

        # 3. Measured frequency within 10% of analytical value
        @test isapprox(ω_measured, ω_trap, rtol = 0.10)
    end
end

# Regression: a MoveCol must not accumulate the displacement across Monte Carlo
# shots or across repeated `play` calls. Move modifiers mutate beam.r0 in place;
# the compiled job must own private beam copies and restore them per shot so that
# (a) every shot's tweezer ends at the same position and (b) the source System /
# TweezerArray is never mutated.
@testset "MoveCol resets beam position across shots and plays" begin
    f0  = 8.3e6
    dx  = 3e-6 / 1e6          # 3 µm/MHz
    x0  = dx * f0
    d   = 1e-6                # 1 µm move
    Δf  = d / dx

    tw = TweezerArray(λ = 759e-9, w0 = 0.7e-6, P_total = 1e-3,
                      row_freqs = [0.0], col_freqs = [f0], dx = dx, dy = dx)
    atom = Ytterbium171Atom(; levels = [Level("1S0")], x_init = [x0, 0.0, 0.0])
    sys  = System([atom], [tw])
    add_detector!(sys, MotionDetectorSpec(tw[1]; dims = [1], name = "twz"))

    seq = Sequence(0.5e-6)
    @sequence seq begin
        MoveCol(tw, 1, Δf, 100e-6; sweep = :min_jerk)
    end

    x_target = x0 + d

    # Multi-shot: every shot must end at the same (single-move) target, not x0 + s·d.
    # 1D detector with shots>1 → Matrix indexed [time, shot].
    out = play(sys, seq; initial_state = Level("1S0"), shots = 4)
    twz_ends = [out.detectors["twz"][end, s] for s in 1:4]
    @test all(isapprox.(twz_ends, x_target; atol = 1e-9))

    # Source array must be pristine after the run (not left at the moved position)
    @test isapprox(tw[1].r0[1], x0; atol = 1e-12)

    # A second play on the same system must reproduce the first, not drift further
    out2 = play(sys, seq; initial_state = Level("1S0"), shots = 1)
    @test isapprox(out2.detectors["twz"][end], x_target; atol = 1e-9)
    @test isapprox(tw[1].r0[1], x0; atol = 1e-12)
end

@testset "spontaneous emission carries a recoil kick" begin
    # `recoil!` shipped in v0.1.0 but was never called, and `atom.lambda` — which
    # it reads — was never populated. So decay was radiatively correct but
    # momentum-free: an atom could scatter thousands of photons without heating.
    # `add_decay!(...; λ)` now records the photon wavelength and enables the kick.
    g = Level("g"; term = l"1S0")
    e = Level("e"; term = l"3P1")
    m    = 174 * AtomTwin.Units.amu
    vrec = AtomTwin.Units.hbar * (2π / 556e-9) / m

    function run(; λ = nothing, shots = 60, seed = 20260917)
        yb = Ytterbium174Atom(; levels = [g, e], x_init = [0.0, 0.0, 0.0],
                              v_init = [0.0, 0.0, 0.0])
        # Wide and weak, so the dipole force is negligible and the velocity is
        # set by the kicks alone.
        tw  = GaussianBeam(λ = 767e-9, w0 = 50e-6, P = 1e-3, pol = [0, 0, 1])
        sys = System(yb, tw)
        add_quantization_axis!(sys, [0.0, 0.0, 1.0])
        cp = add_coupling!(sys, yb, g => e, 2π * 200e3; active = false)
        pd = PhotoDetectorSpec(name = "clicks")
        add_detector!(sys, pd)
        if λ === nothing
            add_decay!(sys, yb, e => g, 2π * 182e3; clicks = pd)
        else
            add_decay!(sys, yb, e => g, 2π * 182e3; clicks = pd, λ = λ)
        end
        add_detector!(sys, MotionDetectorSpec(yb; dims = [1, 2, 3], name = "x"))
        seq = Sequence(20e-9; downsample = 1000)
        @sequence seq begin
            Pulse(cp, 400e-6)
        end
        o = play(sys, seq; initial_state = g, shots = shots,
                 rng = MersenneTwister(seed))
        x  = o.detectors["x"]
        dt = o.times[end] - o.times[end-1]
        v  = [sqrt(sum(((x[end, :, s] .- x[end-1, :, s]) ./ dt).^2))
              for s in 1:size(x, 3)]
        ## RMS, not mean: <|v|^2> = N*v_rec^2 exactly for an isotropic walk, while
        ## <|v|> carries a Maxwell-distribution factor sqrt(8/3pi) = 0.921.
        (mean(sum(o.detectors["clicks"], dims = 1)), sqrt(mean(v .^ 2)))
    end

    # Without λ the atom scatters but never moves.
    N0, v0 = run()
    @test N0 > 50                  # it really is scattering
    @test v0 < 1e-4                # …and going nowhere

    # With λ the speed follows the isotropic random walk, |v| = √N·v_rec.
    N, v = run(λ = 556e-9)
    @test N > 50
    ## Seeded, because the ratio's shot-to-shot spread is 5.4% (1 sigma, measured
    ## over 12 seeds at 60 shots): an unseeded 5% tolerance fails about half the
    ## time. 20% covers the full observed 0.885-1.039 range with margin, and is
    ## still far tighter than the 0/1 distinction this test exists to make.
    @test isapprox(v, sqrt(N) * vrec; rtol = 0.20)
end

@testset "shots start from a fresh atom, not where the last one stopped" begin
    # `initialize!` reset the atom's position only when `x_init` was given (and its
    # velocity only when `v_init` was), but `inner` is reused across shots. An atom
    # with a thermal velocity and no explicit position therefore began shot n where
    # shot n-1 ended: an ensemble was silently one continuous trajectory, and a
    # bound atom appeared to escape once a few shots of drift had accumulated.
    #
    # The same function already carried a `reset_force!` for exactly this hazard,
    # so the omission was an oversight.
    g  = HyperfineManifold(0//1, 0; label = "1S0", term = l"1S0")
    yb = Ytterbium174Atom(; levels = [g...], v_init = maxwellboltzmann(T = 5e-6))
    tw  = GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 1e-3, pol = [1.0, 0.0, 0.0])
    sys = System(yb, tw)
    add_quantization_axis!(sys, [1.0, 0.0, 0.0])
    add_detector!(sys, MotionDetectorSpec(yb; dims = [1, 2, 3], name = "x"))

    seq = Sequence(5e-8; downsample = 20)
    @sequence seq begin
        Wait(2e-4)
    end
    x = play(sys, seq; initial_state = g[0], shots = 6).detectors["x"]

    # Every shot starts at the origin; 1 µs of thermal drift is ~0.02 µm, so 0.1 µm
    # is loose enough not to be a noise test and tight enough to catch the leak
    # (which put later shots at 0.2–0.8 µm).
    r0 = [sqrt(x[1, 1, s]^2 + x[1, 2, s]^2) for s in 1:size(x, 3)]
    @test maximum(r0) < 0.1e-6

    # …and the atom really is bound, so the test is not passing on a frozen atom.
    @test maximum(abs, x[:, 1, :]) > 1e-8
end

@testset "a replayed job starts where it was compiled" begin
    # Later shots re-`initialize!` their atoms, but shot 1 of a replay took them as
    # the previous run left them: a free atom at 1 m/s ended 1 µm out, then 2 µm on
    # the next `play` of the same job.
    g   = Level("g")
    yb  = Ytterbium174Atom(; levels = [g], x_init = zeros(3), v_init = [1.0, 0.0, 0.0])
    sys = System(yb)
    add_detector!(sys, MotionDetectorSpec(yb; dims = [1], name = "x"))
    seq = Sequence(1e-8); push!(seq, Wait(1e-6; downsample = 100))
    job = compile(sys, seq; initial_state = [g])
    x1  = play(job, sys).detectors["x"][end]
    @test x1 ≈ 1e-6 rtol = 1e-9
    @test play(job, sys).detectors["x"][end] == x1
    @test play(job, sys; shots = 2).detectors["x"][end, :] ≈ [x1, x1] rtol = 1e-12
end

# `recompile!` called `initialize!` without the quantization axis, so from the
# second shot on the tensor polarizability was computed against ẑ: identical,
# deterministic shots then ended in different places.
@testset "quantization axis survives recompile" begin
    gm = HyperfineManifold(0//1, 0; label = "¹S₀", term = l"1S0", g_F = 0.0)
    em = HyperfineManifold(1//1, 1; label = "³P₁", term = l"3P1", g_F = 1.5)
    λ  = 767e-9
    yb = Ytterbium174Atom(; levels = [gm..., em...],
                          x_init = [0.3e-6, 0.0, 0.0], v_init = [0.05, 0.0, 0.0])
    tw = GaussianBeam(λ = λ, w0 = 1e-6, P = 50e-3, pol = [1.0, 0.0, 0.0])
    sys = System(yb, tw)
    add_quantization_axis!(sys, [1.0, 0.0, 0.0])
    add_detector!(sys, MotionDetectorSpec(yb; dims = [1], name = "x"))
    seq = Sequence(0.05e-6)
    @sequence seq begin
        Wait(5e-6)
    end

    job = compile(sys, seq; initial_state = [em[1]])
    α1  = copy(job.atoms[1].alpha[λ])
    AtomTwin.recompile!(job, sys)
    @test job.atoms[1].alpha[λ] == α1

    X = play(sys, seq; initial_state = [em[1]], shots = 3).detectors["x"]
    @test X[:, 2] == X[:, 1] && X[:, 3] == X[:, 1]
end

# `qme_semiclassical` never refreshed the level populations from ρ, so the
# dipole force weighted α by whatever a PopulationDetector had last written --
# nothing at all without one, and the atom moved ballistically.
@testset "density-matrix motion feels the state-dependent force" begin
    gm = HyperfineManifold(0//1, 0; label = "¹S₀", term = l"1S0", g_F = 0.0)
    em = HyperfineManifold(1//1, 1; label = "³P₁", term = l"3P1", g_F = 1.5)
    g, e = gm[0], em[0]
    function run(; dm, popdet)
        yb  = Ytterbium174Atom(; levels = [g, e], x_init = [0.4e-6, 0, 0],
                               v_init = [1e-3, 0, 0])
        sys = System(yb, GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 50e-3, pol = [0, 0, 1.0]))
        popdet && add_detector!(sys, PopulationDetectorSpec(yb, e; name = "Pe"))
        add_detector!(sys, MotionDetectorSpec(yb; dims = [1], name = "x"))
        seq = Sequence(0.05e-6)
        @sequence seq begin
            Wait(10e-6)
        end
        play(sys, seq; initial_state = [e], density_matrix = dm).detectors["x"][end]
    end
    x_sv = run(dm = false, popdet = false)      # deterministic: no jumps
    x_dm = run(dm = true,  popdet = false)
    @test abs(x_sv - (0.4e-6 + 1e-3 * 10e-6)) > 1e-8   # the force did something
    @test isapprox(x_dm, x_sv; atol = 1e-12)
    @test run(dm = true, popdet = true) == x_dm
end

# A ramp read its starting amplitude from the beam at COMPILE time, before any
# earlier `AmplRow` had run, so `AmplRow(0.1)` then a ramp to 1 ramped 1 → 1.
@testset "a ramp starts from the amplitude the beam has at run time" begin
    beam = GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 1e-3)
    ramps, n = AtomTwin.ramp([beam], [1.0], 1e-6, 1e-7)
    @test n == 10
    beam._coeff[] = 0.1                     # what an earlier AmplRow leaves
    m = ramps[1]
    AtomTwin.Dynamiq.begin_instruction!(m)
    AtomTwin.Dynamiq.update!(m, 0.5e-6)
    @test beam._coeff[] ≈ 0.55
    AtomTwin.Dynamiq.update!(m, 1e-6)
    @test beam._coeff[] == 1.0
end

@testset "plane-wave drives: quantised absorption, stimulated emission pushes back" begin
    # Radiation pressure used to be a mean force ħ R_tot Σ ŵ_b k_b sampled once per
    # step, and only in the MCWF solver: a π pulse moved ~0 instead of ħk,
    # stimulated emission into an opposing beam nothing, the TDSE and density-matrix
    # solvers no radiation pressure at all, and the spread of the absorbed momentum
    # was set by `dt` (0.35 E_r per photon at Γ·dt = 0.18, 1.33 at 4.6; quantised
    # 1.26 at s = 40). Now: the Ehrenfest force of the plane-wave phase, integrated
    # over sub-steps, with one photon's momentum per MCWF emission cycle.
    g = Level("g"; term = l"1S0")
    e = Level("e"; term = l"3P1")
    m = 174 * AtomTwin.Units.amu
    vrec(λ) = AtomTwin.Units.hbar * (2π / λ) / m

    function drive(; Ω, Γ, λ, pulses, shots, dm = false, decay = true, seed = 20260927)
        yb = Ytterbium174Atom(; levels = [g, e], x_init = [0.0, 0.0, 0.0],
                              v_init = [0.0, 0.0, 0.0])
        sys = System(yb)            # free space: motion is on by default now
        bp = PlanarBeam(λ, 1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        bm = PlanarBeam(λ, 1.0, [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        cp = add_coupling!(sys, yb, g => e, Ω; beam = bp, active = false)
        cm = add_coupling!(sys, yb, g => e, Ω; beam = bm, active = false)
        pd = dm ? nothing : PhotoDetectorSpec(name = "clicks")
        dm || add_detector!(sys, pd)
        decay && add_decay!(sys, yb, e => g, Γ; clicks = pd, λ = λ)
        add_detector!(sys, MotionDetectorSpec(yb; dims = [1, 2, 3], name = "x"))
        seq = Sequence(1e-9)
        for (sgn, T) in pulses
            push!(seq, Pulse(sgn > 0 ? cp : cm, T; dt = T / 100, downsample = 100))
            push!(seq, Wait(2e-9; dt = 1e-9, downsample = 1))
        end
        o = play(sys, seq; initial_state = g, shots = shots, density_matrix = dm,
                 rng = MersenneTwister(seed))
        x, t = o.detectors["x"], o.times
        vx = shots == 1 ?
            [(x[3k, 1] - x[3k-1, 1]) / (t[3k] - t[3k-1]) for k in 1:length(pulses)] :
            [(x[3k, 1, s] - x[3k-1, 1, s]) / (t[3k] - t[3k-1]) for k in 1:length(pulses), s in 1:shots]
        N = dm ? nothing : sum(o.detectors["clicks"], dims = 1)
        (vx ./ vrec(λ), N)
    end

    # Coherent, TDSE (no jumps at all): π pulses +k, −k, +k give 1, 2, 3 ħk —
    # absorb, stimulated emission into the −k beam, absorb.
    Ω = 2π * 10e6
    vx, _ = drive(; Ω = Ω, Γ = 0.0, λ = 556e-9, decay = false, shots = 1,
                  pulses = [(+1, π / Ω), (-1, π / Ω), (+1, π / Ω)])
    @test vx ≈ [1.0, 2.0, 3.0] atol = 3e-3

    # One solver step per π pulse: the TDSE sub-divides the step to integrate the
    # impulse (to `RAD_THETA` of rotation) and still moves one ħk, to the
    # trapezoid's ≲ 0.3 %.
    let yb = Ytterbium174Atom(; levels = [g, e], x_init = zeros(3), v_init = zeros(3))
        sys = System(yb)
        c = add_coupling!(sys, yb, g => e, Ω; active = false,
                          beam = PlanarBeam(556e-9, 1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
        add_detector!(sys, MotionDetectorSpec(yb; dims = [1], name = "x"))
        seq = Sequence(1e-9)
        push!(seq, Pulse(c, π / Ω; dt = π / Ω, downsample = 1))
        push!(seq, Wait(2e-9; dt = 1e-9, downsample = 1))
        o = play(sys, seq; initial_state = g)
        x, t = o.detectors["x"], o.times
        @test (x[3] - x[2]) / (t[3] - t[2]) / vrec(556e-9) ≈ 1.0 atol = 5e-3
    end

    # Density matrix, CW on one saturated beam: the ensemble-mean force ħk R with
    # R = (Γ/2) s/(1+s), after the ~1/Γ transient.
    Γ = 2π * 29.1e6; s = 40.0; T = 400e-9
    vx, _ = drive(; Ω = Γ * sqrt(s / 2), Γ = Γ, λ = 398.9e-9, pulses = [(+1, T)],
                  shots = 1, dm = true)
    @test vx[1] / (Γ / 2 * s / (1 + s) * T) ≈ 1.0 rtol = 0.02

    # MCWF, same drive: mean Nħk, and every emission cycle absorbs exactly ħk, so
    # var(p) = var(N) + N/3 ≈ (1 + Q + 1/3) N, Mandel Q = −3s/(1+s)² = −0.07.
    vx, N = drive(; Ω = Γ * sqrt(s / 2), Γ = Γ, λ = 398.9e-9, pulses = [(+1, T)],
                  shots = 1500)
    p = vec(vx); Nm = mean(N)
    @test mean(p) / Nm ≈ 1.0 rtol = 0.03
    ## 1σ of the variance ratio at 1500 shots ≈ 4 %; the old force gave 0.35
    @test 1.10 < var(p) / Nm < 1.42

    # MCWF conditioning: a π/2 pulse leaves the atom half excited. A trajectory that
    # then emits nothing was never excited and must end at rest, not with the ħk/2
    # the mean force gave it; one that emits once absorbed exactly one photon, so
    # |p − ħk x̂| = ħk, the emission recoil, to rounding.
    Ω = 2π * 300e6; Tp = π / (2Ω); λ = 398.9e-9; shots = 400
    yb  = Ytterbium174Atom(; levels = [g, e], x_init = zeros(3), v_init = zeros(3))
    sys = System(yb)
    c   = add_coupling!(sys, yb, g => e, Ω; active = false,
                        beam = PlanarBeam(λ, 1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
    pd  = PhotoDetectorSpec(name = "clicks"); add_detector!(sys, pd)
    add_decay!(sys, yb, e => g, Γ; clicks = pd, λ = λ)
    add_detector!(sys, MotionDetectorSpec(yb; dims = [1, 2, 3], name = "x"))
    seq = Sequence(1e-9)
    push!(seq, Pulse(c, Tp; dt = Tp / 50, downsample = 50))
    push!(seq, Wait(300e-9; downsample = 300)); push!(seq, Wait(2e-9; downsample = 1))
    o = play(sys, seq; initial_state = g, shots = shots, rng = MersenneTwister(7))
    x, t = o.detectors["x"], o.times
    p = [(x[end, d, s] - x[end-1, d, s]) / (t[end] - t[end-1]) / vrec(λ) for d in 1:3, s in 1:shots]
    N = vec(sum(o.detectors["clicks"], dims = 1))
    n0, n1 = findall(==(0), N), findall(==(1), N)
    @test length(n0) > 50 && length(n1) > 50
    @test maximum(abs, p[:, n0]) < 1e-6                 # was 0.5
    @test maximum(abs(hypot(p[1, s] - 1, p[2, s], p[3, s]) - 1) for s in n1) < 1e-6
end

@testset "motion is on by default; frozen = true freezes it" begin
    # `play` used to freeze an atom unless a beam in the SYSTEM could exert a dipole
    # force on it: a free atom scattering plane-wave light, with recoil, never moved.
    g = Level("g"; term = l"1S0"); e = Level("e"; term = l"3P1")
    function run(; frozen = false, trap = false)
        yb  = Ytterbium174Atom(; levels = [g, e], x_init = [0.0, 0.0, 0.0],
                               v_init = [0.0, 0.0, 0.0])
        sys = trap ? System(yb, GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 1e-3, pol = [0, 0, 1])) :
                     System(yb)
        c = add_coupling!(sys, yb, g => e, 2π * 1e6; active = false,
                          beam = PlanarBeam(556e-9, 1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
        add_decay!(sys, yb, e => g, 2π * 182e3; λ = 556e-9)
        add_detector!(sys, MotionDetectorSpec(yb; dims = [1, 2, 3], name = "x"))
        seq = Sequence(20e-9); push!(seq, Pulse(c, 20e-6; downsample = 1000))
        o = play(sys, seq; initial_state = g, shots = 4, frozen = frozen,
                 rng = MersenneTwister(1))
        maximum(abs, o.detectors["x"])
    end
    @test run() > 1e-8                       # pushed along the beam
    @test run(frozen = true) == 0.0
    @test run(trap = true, frozen = true) == 0.0
end

@testset "a run in which nothing can move takes the frozen solvers" begin
    # Motion is the default, but an atom at rest, without recoil or plane-wave
    # drives, at the exact centre of its trap cannot move: `_is_static` sends such
    # a run to the frozen solvers. The answer must be identical to `frozen = true`.
    g = Level("g"; term = l"1S0"); e = Level("e"; term = l"3P1")
    function build(x0; neighbour = false)
        yb  = Ytterbium174Atom(; levels = [g, e], x_init = x0, v_init = [0.0, 0.0, 0.0])
        tw  = GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 1e-3, pol = [0, 0, 1])
        sys = neighbour ? System([yb], [tw, GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 1e-3,
                                                          r0 = [4e-6, 0, 0], pol = [0, 0, 1])]) :
                          System(yb, tw)
        c = add_coupling!(sys, yb, g => e, 2π * 1e6; active = false)
        add_decay!(sys, yb, e => g, 2π * 0.2e6)
        add_detector!(sys, PopulationDetectorSpec(yb, e; name = "Pe"))
        seq = Sequence(10e-9); push!(seq, Pulse(c, 2e-6; downsample = 20))
        sys, seq
    end
    sys, seq = build([0.0, 0.0, 0.0])
    a = play(sys, seq; initial_state = g, shots = 8, rng = MersenneTwister(3))
    b = play(sys, seq; initial_state = g, shots = 8, rng = MersenneTwister(3), frozen = true)
    @test a.detectors["Pe"] == b.detectors["Pe"]
    job = compile(sys, seq; initial_state = [g])
    @test AtomTwin._is_static(job)
    # Off-centre, the trap pushes it: not static.
    sys2, seq2 = build([0.3e-6, 0.0, 0.0])
    @test !AtomTwin._is_static(compile(sys2, seq2; initial_state = [g]))
    # A neighbouring tweezer's tail is never exactly zero, but far too weak to move
    # the atom over the run: still static.
    sys3, seq3 = build([0.0, 0.0, 0.0]; neighbour = true)
    @test AtomTwin._is_static(compile(sys3, seq3; initial_state = [g]))
end

@testset "MCWF radiation pressure through a ladder of plane-wave drives" begin
    # g → e (beam 1) → r (beam 2): the second drive moves population WITHIN the
    # excited branch. Booked as absorption into it, its flow was drained back out
    # as if unobserved and the momentum came out 0.62× the Ehrenfest value, with no
    # jump at all. Without a jump a trajectory carries ħk₁(P_e + P_r) + ħk₂P_r.
    g, e, r = Level("g"), Level("e"), Level("r")
    λ1, λ2 = 780e-9, 480e-9
    at  = Atom(; levels = [g, e, r], x_init = zeros(3), v_init = zeros(3))
    sys = System(at)
    c1 = add_coupling!(sys, at, g => e, 2π * 5e6; active = false,
                       beam = PlanarBeam(λ1, 1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
    c2 = add_coupling!(sys, at, e => r, 2π * 5e6; active = false,
                       beam = PlanarBeam(λ2, 1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
    add_decay!(sys, at, r => g, 2π * 1e3)     # MCWF; a jump in 100 ns is ~1e-4 likely
    add_detector!(sys, MotionDetectorSpec(at; dims = [1], name = "x"))
    add_detector!(sys, PopulationDetectorSpec(at, e; name = "Pe"))
    add_detector!(sys, PopulationDetectorSpec(at, r; name = "Pr"))
    seq = Sequence(1e-9)
    push!(seq, Pulse([c1, c2], 100e-9; dt = 0.5e-9, downsample = 200))
    push!(seq, Wait(2e-9; dt = 1e-9, downsample = 1))
    o = play(sys, seq; initial_state = g, shots = 2, rng = MersenneTwister(5))
    x, t = o.detectors["x"], o.times
    m = compile(sys, seq; initial_state = [g]).atoms[1].m
    for s in 1:2
        p  = (x[3, s] - x[2, s]) / (t[3] - t[2]) * m / AtomTwin.Units.hbar
        Pe = real(o.detectors["Pe"][1, s]); Pr = real(o.detectors["Pr"][1, s])
        @test p ≈ 2π / λ1 * (Pe + Pr) + 2π / λ2 * Pr rtol = 2e-3
    end
end
