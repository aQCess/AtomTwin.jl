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
