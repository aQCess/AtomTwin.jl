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
