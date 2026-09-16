# Convergence-order regression tests.
#
# These pin the OBSERVED order of the solver, which no other test covers. An
# order regression is silent — results stay plausible, just needlessly
# inaccurate — so it is exactly the failure mode worth asserting on.
using Test, AtomTwin, Printf, Random, LinearAlgebra, SparseArrays

"Final P_e after a pulse on a driven two-level atom, at solver step `dt`."
function _final_Pe(dt; shaped::Bool, interp = :cubic, npts = 401, T = 2e-6)
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    c = add_coupling!(sys, atom, g => e, 2π * 1e6; active = false)
    add_detuning!(sys, atom, e, 2π * 0.3e6)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "Pe"))
    seq = Sequence(dt; downsample = 1)
    if shaped
        env = [exp(-(((k - 1) / (npts - 1) - 0.5) / 0.18)^2) for k in 1:npts]
        @sequence seq begin
            Pulse(c, T; amplitudes = env, interp = interp)
        end
    else
        @sequence seq begin
            Pulse(c, T)
        end
    end
    play(sys, seq; initial_state = g, shots = 1).detectors["Pe"][end]
end

"Least-squares convergence order of |f(dt) - ref| over a dt sweep."
function _observed_order(f, dts, ref)
    errs = [abs(f(dt) - ref) for dt in dts]
    # slope of log(err) vs log(dt)
    x = log.(dts); y = log.(errs)
    n = length(x)
    (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x .^ 2) - sum(x)^2)
end

# A probe that counts how often the controller calls it. Defined at top level:
# `probe!` is a method on `AtomTwin.Dynamiq.probe!`, which cannot be added from
# inside a testset's local scope.
mutable struct CountProbe <: AtomTwin.Dynamiq.SolverProbe
    n::Int
    kmax::Int
end
AtomTwin.Dynamiq.probe!(p::CountProbe, info) =
    (p.n += 1; p.kmax = max(p.kmax, info.k); nothing)

@testset "Solver convergence order" begin
    dts = [2e-9, 1e-9, 5e-10, 2.5e-10]

    @testset "constant pulse is exact (no measurable order)" begin
        # With the Chebyshev propagator a constant-H step is exact to machine
        # precision, so the error does NOT fall with dt -- it sits at the
        # reference's own floor (~1e-12). Asserting an order here would be
        # asserting the propagator is inaccurate.
        # The propagator is exact for constant H, so the reference only has to sit
        # below the sweep -- it does not have to be converged. Measured flat at
        # 1e-12 all the way from dt=2e-9 to 1e-10, so 1e-10 is ample, at ~200x
        # less work than the 2e-11 this used to use.
        ref = _final_Pe(1e-10; shaped = false)
        errs = [abs(_final_Pe(dt; shaped = false) - ref) for dt in dts]
        @test all(<(1e-10), errs)              # exact at every step size
        # Bound the error rather than the spread between errors: at 1e-26 to
        # 1e-30 against the analytic solution, a ratio is round-off noise and
        # swings with any change to the Chebyshev degree.
        @test errs[end] < 1e-10 && errs[1] < 1e-10
    end

    @testset "shaped pulse is 2nd order (midpoint sampling)" begin
        # Left-endpoint sampling would give order 1 here. See compile.jl.
        # 1e-10 differs from a 2e-11 reference by 2.2e-9, orders below the errors
        # asserted here, at ~200x less work.
        ref = _final_Pe(1e-10; shaped = true)
        p = _observed_order(dt -> _final_Pe(dt; shaped = true), dts, ref)
        @test 1.7 < p < 2.4
    end

    @testset "piecewise-constant envelope stays 1st order by design" begin
        # The staircase is physical (an AWG holds each bin), so this must NOT be
        # "fixed" to midpoint sampling.
        # A coarse staircase (51 bins) converges cleanly, so the fitted order is
        # stable in the reference step: 1.08 at refdt=5e-11 vs 1.04 at 2e-11. The
        # 401-bin version needed a 2e-11 reference and still swung 0.91 -> 3.33
        # when that was loosened, i.e. it was fitting the reference's own error.
        ref = _final_Pe(5e-11; shaped = true, interp = :piecewise_constant, npts = 51)
        p = _observed_order(dt -> _final_Pe(dt; shaped = true,
                                            interp = :piecewise_constant, npts = 51),
                            dts, ref)
        @test 0.7 < p < 1.5
    end
end

@testset "order keyword rejects unstable truncations" begin
    # Truncated Taylor is norm-contractive on the imaginary axis only for
    # p ≡ 0, 3 (mod 4). Orders 1, 2, 5, 6 amplify at EVERY step size, so they
    # must not be silently accepted.
    using LinearAlgebra, SparseArrays, Random
    D = AtomTwin.Dynamiq
    Random.seed!(51); d = 8
    A = randn(ComplexF64, d, d); H = Matrix(Hermitian(A + A'))
    psi = normalize(randn(ComplexF64, d)); q1 = similar(psi); q2 = similar(psi)
    Hl = [(Ref(ComplexF64(1.0)), D.Op(sparse(H)))]

    for p in (1, 2, 5, 6, 9)
        @test_throws ArgumentError D.fquantum!(1e-3, copy(psi), Hl, q1, q2; order = p)
    end
    for p in (3, 4, 7, 8)
        @test (D.fquantum!(1e-3, copy(psi), Hl, q1, q2; order = p); true)
    end
end

@testset "classical integrator is 2nd order (velocity Verlet)" begin
    D = AtomTwin.Dynamiq

    # Atom falling through a Gaussian dipole trap; compare final x against a
    # highly-resolved reference.
    function _finalx(nsteps; T = 2e-3)
        atom = D.NLevelAtom(2; x = [2e-6, 0.0, 0.0], v = [0.0, 0.0, 0.0],
                            m = 87 * 1.66053906660e-27,
                            alphas = Dict(1064e-9 => [1e-39, 1e-39]))
        beam = D.GaussianBeam(1064e-9, 20e-6, 0.5; r0 = [0.0, 0.0, 0.0])
        fill!(atom._P, 0.0); atom._P[1] = 1.0
        D.newton([atom], collect(range(T / nsteps, T, nsteps)); beams = [beam])
        atom.x[1]
    end

    ref = _finalx(400_000)
    e1 = abs(_finalx(2_000)  - ref)
    e2 = abs(_finalx(8_000)  - ref)
    @test 1.7 < log2(e1 / e2) / 2 < 2.3       # two doublings between 2k and 8k

    @testset "force cache is invalidated, not silently reused" begin
        # A stale F_old corresponds to a different position and corrupts the
        # trajectory; reset_force! must restore agreement with a fresh atom.
        function _traj(; poison::Bool, reset::Bool)
            atom = D.NLevelAtom(2; x = [3e-6, 0.0, 0.0], v = [0.0, 0.0, 0.0],
                                m = 87 * 1.66053906660e-27,
                                alphas = Dict(1064e-9 => [1e-39, 1e-39]))
            beam = D.GaussianBeam(1064e-9, 20e-6, 0.5; r0 = [0.0, 0.0, 0.0])
            if poison
                atom._F[1] = 1e-20; atom._Fvalid = true
            end
            reset && D.reset_force!(atom)
            fill!(atom._P, 0.0); atom._P[1] = 1.0
            D.newton([atom], collect(range(1e-6, 1e-3, 1000)); beams = [beam])
            atom.x[1]
        end
        fresh = _traj(poison = false, reset = false)
        @test _traj(poison = true,  reset = false) != fresh   # stale cache DOES matter
        @test _traj(poison = true,  reset = true)  == fresh   # and reset fixes it
    end
end

@testset "step-size guard warns before divergence" begin
    D = AtomTwin.Dynamiq
    using LinearAlgebra, SparseArrays

    @testset "gershgorin_bound is an upper bound on ‖H‖" begin
        Random.seed!(91)
        for d in (4, 16, 64)
            A = sprand(ComplexF64, d, d, min(1.0, 8 / d)); H = A + A'
            terms = [(Ref(ComplexF64(1.0)), D.Op(H))]
            @test D.gershgorin_bound(terms) >= opnorm(Matrix(H)) - 1e-9
        end
    end

    @testset "fires on a step that diverges, silent on one that does not" begin
        # Blockade-stiff two-atom system: dt=4e-9 genuinely produces NaN.
        function _run(dt)
            g, r = Level("g"), Level("r")
            a1, a2 = Atom(; levels = [g, r]), Atom(; levels = [g, r])
            sys = System([a1, a2])
            c1 = add_coupling!(sys, a1, g => r, 2π * 1e6; active = false)
            c2 = add_coupling!(sys, a2, g => r, 2π * 1e6; active = false)
            add_interaction!(sys, (a1, a2), (r, r) => (r, r), 2π * 200e6)
            add_detector!(sys, PopulationDetectorSpec(a1, r; name = "Pr"))
            seq = Sequence(dt; downsample = 1)
            @sequence seq begin
                Pulse([c1, c2], 1e-6)
            end
            play(sys, seq; initial_state = [g, g], shots = 1).detectors["Pr"][end]
        end
        # The Chebyshev propagator has NO stability limit, so the step that
        # used to diverge now simply works. The warning machinery still guards
        # `fquantum!` (used by the MCWF path), but `play` on a statevector no
        # longer diverges at any step size -- which is the point of Phase 3.
        # H is constant here, so the propagator is exact and a 1e-10 baseline is
        # indistinguishable from 1e-11 at the 1e-3 tolerance asserted -- at 10x
        # less work.
        @test !isnan(_run(1e-10))
        @test !isnan(_run(4e-9))                # was NaN with Taylor-4
        @test isapprox(_run(4e-9), _run(1e-10); atol = 1e-3)
    end
end

@testset "automatic time step (Sequence without dt)" begin
    function _rabi(seqf)
        g, e = Level("g"), Level("e")
        atom = Atom(; levels = [g, e]); sys = System(atom)
        c = add_coupling!(sys, atom, g => e, 2π * 1e6; active = false)
        add_detuning!(sys, atom, e, 2π * 0.3e6)
        add_detector!(sys, PopulationDetectorSpec(atom, e; name = "Pe"))
        seq = seqf()
        @sequence seq begin
            Pulse(c, 2e-6)
        end
        play(sys, seq; initial_state = g, shots = 1).detectors["Pe"][end]
    end

    ref = _rabi(() -> Sequence(2e-9; downsample = 1))

    @testset "derived step reaches the requested tolerance" begin
        @test abs(_rabi(() -> Sequence(; tol = 1e-6)) - ref) < 1e-6
        @test abs(_rabi(() -> Sequence(; tol = 1e-9)) - ref) < 1e-8
    end

    @testset "accuracy improves monotonically with tol" begin
        # Regression: before dt was snapped to divide the pulse duration, a
        # tighter tol could give a WORSE answer, because a non-dividing step made
        # `tsteps = round(duration/dt)` run the pulse for the wrong total time
        # (~0.05% off => ~1e-3 population error, swamping integration error).
        # With the Chebyshev propagator both tolerances land on the reference's
        # own floor (~1e-12), so their ORDER is floating-point noise -- the
        # original 1e-3 discrepancy this guarded against would still be caught,
        # but a strict inequality between two 1e-12 values is not meaningful.
        e_loose = abs(_rabi(() -> Sequence(; tol = 1e-4)) - ref)
        e_tight = abs(_rabi(() -> Sequence(; tol = 1e-9)) - ref)
        @test e_tight <= max(e_loose, 1e-10)
    end

    @testset "derived step is the output resolution, not the step" begin
        # `_derive_dt` returns how often results are RECORDED. With no `dt` or
        # `steps` on the sequence the default is one sample per instruction: a
        # user who wrote only `Sequence(; tol)` wants an accurate final state,
        # and one who wants a trace asks for it. The integration step is chosen
        # separately, by the solver's own error estimators.
        #
        # With no Hamiltonian there is no accuracy bound either, so the whole
        # instruction is one sample.
        for T in (2e-6, 1e-6, 3.7e-6), tol in (1e-5, 1e-8)
            seq = Sequence(; tol = tol)
            push!(seq, Wait(T))
            dt = AtomTwin._derive_dt(seq, AtomTwin.Dynamiq.AbstractField[])
            @test dt ≈ T
            @test dt > 0
        end
    end

    @testset "handles a stiff system that diverges at a naive step" begin
        g, r = Level("g"), Level("r")
        a1, a2 = Atom(; levels = [g, r]), Atom(; levels = [g, r])
        sys = System([a1, a2])
        c1 = add_coupling!(sys, a1, g => r, 2π * 1e6; active = false)
        c2 = add_coupling!(sys, a2, g => r, 2π * 1e6; active = false)
        add_interaction!(sys, (a1, a2), (r, r) => (r, r), 2π * 200e6)
        add_detector!(sys, PopulationDetectorSpec(a1, r; name = "Pr"))
        seq = Sequence(; tol = 1e-6)
        @sequence seq begin
            Pulse([c1, c2], 1e-6)
        end
        p = play(sys, seq; initial_state = [g, g], shots = 1).detectors["Pr"][end]
        @test !isnan(p)
        @test isapprox(p, 0.4645530578, atol = 1e-6)
    end
end

@testset "duration is exact; dt gives way" begin
    function _rabi(dt)
        g, e = Level("g"), Level("e")
        atom = Atom(; levels = [g, e]); sys = System(atom)
        c = add_coupling!(sys, atom, g => e, 2π * 1e6; active = false)
        add_detuning!(sys, atom, e, 2π * 0.3e6)
        add_detector!(sys, PopulationDetectorSpec(atom, e; name = "Pe"))
        seq = Sequence(dt; downsample = 1)
        @sequence seq begin
            Pulse(c, 2e-6)
        end
        play(sys, seq; initial_state = g, shots = 1).detectors["Pe"][end]
    end

    @testset "accuracy is monotone even for non-dividing dt" begin
        # Regression: `tsteps = round(duration/dt)` let a non-dividing step run
        # the pulse for the wrong total time. dt=8.4081e-9 measured 1.794e-3
        # error — 30000x WORSE than the larger dt=1e-8 — until `stepgrid` began
        # adjusting dt instead of the duration.
        ref = _rabi(2e-9)
        errs = [abs(_rabi(dt) - ref) for dt in (1.0e-8, 8.4081e-9, 3.1530e-9, 1.0e-9)]
        # The original defect made a NON-DIVIDING dt far worse than a larger
        # dividing one (8.4081e-9 measured 30000x worse than 1e-8). With an
        # exact propagator the errors are now all at the reference floor, so
        # they no longer decrease with dt -- assert the property that matters:
        # no step size is anomalously bad.
        @test all(<(1e-6), errs)
        # As above: bound the error, not the spread. The property under test is
        # that a non-dividing `dt` does not blow up.
        @test maximum(errs) < 1e-9
    end

    @testset "stepgrid realises the duration exactly" begin
        for (T, dt) in ((2e-6, 8.4081e-9), (1e-6, 7e-10), (3.7e-6, 3.3e-10))
            n, dte = AtomTwin.stepgrid(T, dt)
            @test abs(n * dte - T) < 1e-20        # duration preserved
            @test abs(dte - dt) / dt < 1 / n      # dt perturbed by < one part in n
        end
    end

    @testset "Sequence(duration, tsteps) matches the equivalent dt" begin
        # Stating "2 µs in 2000 steps" must be identical to stating dt = 1 ns.
        function _via_tsteps()
            g, e = Level("g"), Level("e")
            atom = Atom(; levels = [g, e]); sys = System(atom)
            c = add_coupling!(sys, atom, g => e, 2π * 1e6; active = false)
            add_detuning!(sys, atom, e, 2π * 0.3e6)
            add_detector!(sys, PopulationDetectorSpec(atom, e; name = "Pe"))
            seq = Sequence(2e-6, 2000)
            @sequence seq begin
                Pulse(c, 2e-6)
            end
            play(sys, seq; initial_state = g, shots = 1).detectors["Pe"][end]
        end
        @test _rabi(1e-9) == _via_tsteps()
        @test_throws ArgumentError Sequence(2e-6, 0)
        @test_throws ArgumentError Sequence(-1.0, 100)
    end

    @testset "Kraus dissipators: order and complete positivity" begin
        # A two-level system with decay, as raw Dynamiq structures, so the
        # dissipator is exercised against exp(dt·𝓛_D) without a surrounding
        # Hamiltonian step.
        d, γ = 2, 1.0
        σ   = ComplexF64[0 1; 0 0]
        Jop = AtomTwin.Dynamiq.Op(Tuple{Int,Int,ComplexF64}[(1, 2, 1.0 + 0im)],
                                  Tuple{Int,Int,ComplexF64}[], d)
        J = Tuple{Base.RefValue{ComplexF64},AtomTwin.Dynamiq.Op,Vector{Float64}}[
                (Ref(ComplexF64(sqrt(γ))), Jop, [0.0, 1.0])]
        Id = Matrix(I, d, d)
        𝓛 = γ * (kron(conj(σ), σ) -
                 0.5 * (kron(Id, σ'σ) + kron(transpose(σ'σ), Id)))
        ρ0 = ComplexF64[0.6 0.3+0.1im; 0.3-0.1im 0.4]

        # The Kraus set is EXACT for a single channel -- the whole point of
        # the per-source-level Kraus set -- so assert exactness, not an order.
        # (An order fitted to machine-noise errors is meaningless.)
        for dt in (0.4, 0.2, 0.1, 0.05)
            ref = reshape(exp(𝓛 * dt) * vec(ρ0), d, d)
            ρ = copy(ρ0); q1 = copy(ρ0); q2 = copy(ρ0)
            AtomTwin.Dynamiq.fdissipator2!(dt, ρ, J, q1, q2)
            @test norm(ρ - ref) < 1e-13        # exact at ANY step size
        end

        # Complete positivity is unconditional in dt for both: every term is a
        # congruence with a non-negative coefficient. Checked far past any step
        # size a run would use.
        for f in (AtomTwin.Dynamiq.fdissipator2!,)
            for dt in (1.0, 10.0, 100.0)
                ρ = copy(ρ0); q1 = copy(ρ0); q2 = copy(ρ0)
                f(dt, ρ, J, q1, q2)
                @test minimum(eigvals(Hermitian((ρ + ρ') / 2))) > -1e-12
                @test maximum(abs, ρ - ρ') < 1e-12
            end
        end

        # fdissipator2! is trace-preserving by construction (final rescaling).
        for dt in (0.1, 1.0, 10.0)
            ρ = copy(ρ0); q1 = copy(ρ0); q2 = copy(ρ0)
            AtomTwin.Dynamiq.fdissipator2!(dt, ρ, J, q1, q2)
            @test abs(real(tr(ρ)) - 1) < 1e-12
        end

        # Allocation-free hot loop.
        ρ = copy(ρ0); q1 = copy(ρ0); q2 = copy(ρ0)
        AtomTwin.Dynamiq.fdissipator2!(0.1, ρ, J, q1, q2)
        copyto!(ρ, ρ0)
        @test (@allocated AtomTwin.Dynamiq.fdissipator2!(0.1, ρ, J, q1, q2)) == 0
    end


    @testset "dissipator is exact or CPTP across jump topologies" begin
        # The per-source-level Kraus set is EXACT where the channels leaving a
        # level are the only ones acting on it, and second order otherwise. In
        # every topology it must stay trace-preserving and positive at any step
        # size -- an unfavourable topology may lose exactness, never physicality.
        #
        # This covers the case an earlier construction got wrong: pure dephasing
        # must leave populations exactly fixed.
        mkJ(specs, d) = begin
            J = Tuple{Base.RefValue{ComplexF64},AtomTwin.Dynamiq.Op,Vector{Float64}}[]
            for (p, q, g) in specs
                dg = zeros(Float64, d); dg[q] = 1.0
                op = AtomTwin.Dynamiq.Op(Tuple{Int,Int,ComplexF64}[(p, q, 1.0 + 0im)],
                                         Tuple{Int,Int,ComplexF64}[], d)
                push!(J, (Ref(ComplexF64(sqrt(g))), op, dg))
            end
            J
        end
        denseL(specs, d) = begin
            Id = Matrix(I, d, d); L = zeros(ComplexF64, d*d, d*d)
            for (p, q, g) in specs
                M = zeros(ComplexF64, d, d); M[p, q] = 1
                L += g * (kron(conj(M), M) -
                          0.5 * (kron(Id, M'M) + kron(transpose(M'M), Id)))
            end
            L
        end

        d = 4
        ρ0 = zeros(ComplexF64, d, d)
        ρ0[4,4] = 0.5; ρ0[3,3] = 0.3; ρ0[1,1] = 0.2; ρ0[3,4] = 0.2; ρ0[4,3] = 0.2

        exact_cases = (
            ("single decay",        [(1, 3, 2.0)]),
            ("pure dephasing",      [(3, 3, 2.0)]),
            ("branching, 2 sources", [(1,3,2.0), (2,3,1.0), (1,4,1.5), (2,4,0.5)]),
            # Dephasing shares a source level with the decay but transfers no
            # population, so it takes no share of the branching: the two damp
            # different parts of ρ and the Kraus set stays exact.
            ("decay + dephasing on one level", [(1,3,2.0), (3,3,1.5)]),
        )
        for (name, specs) in exact_cases
            J = mkJ(specs, d); 𝓛 = denseL(specs, d)
            for dt in (0.4, 0.1, 0.025)
                ref = reshape(exp(𝓛 * dt) * vec(ρ0), d, d)
                ρ = copy(ρ0); q1 = copy(ρ0); q2 = copy(ρ0)
                AtomTwin.Dynamiq.fdissipator2!(dt, ρ, J, q1, q2)
                @test norm(ρ - ref) < 1e-12          # exact, any dt ($name)
            end
        end

        # Non-commuting topologies: second order, still CPTP. A cascade is the
        # real case -- population leaving 4 lands on 3, which is itself decaying
        # within the same step, and no single-step Kraus set captures that.
        approx_cases = (
            ("cascade 4->3->{1,2}",            [(3,4,2.0), (1,3,1.5), (2,3,0.5)]),
        )
        for (name, specs) in approx_cases
            J = mkJ(specs, d); 𝓛 = denseL(specs, d)
            errs = map((0.2, 0.1, 0.05)) do dt
                ref = reshape(exp(𝓛 * dt) * vec(ρ0), d, d)
                ρ = copy(ρ0); q1 = copy(ρ0); q2 = copy(ρ0)
                AtomTwin.Dynamiq.fdissipator2!(dt, ρ, J, q1, q2)
                norm(ρ - ref)
            end
            @test 1.6 < log(errs[1] / errs[3]) / log(4) < 2.3   # ~2nd order ($name)
        end

        # Physicality holds in EVERY topology, at brutal step sizes.
        for (_, specs) in (exact_cases..., approx_cases...)
            J = mkJ(specs, d)
            for dt in (1.0, 10.0, 100.0)
                ρ = copy(ρ0); q1 = copy(ρ0); q2 = copy(ρ0)
                AtomTwin.Dynamiq.fdissipator2!(dt, ρ, J, q1, q2)
                @test abs(real(tr(ρ)) - 1) < 1e-12
                @test minimum(eigvals(Hermitian((ρ + ρ') / 2))) > -1e-12
            end
        end
    end



    @testset "automatic step honours tol on a dissipative system" begin
        # With dissipation the Strang splitting error binds, not the propagator,
        # so the solver sub-divides each step until it meets `tol`. Pins the
        # property that matters: achieved error tracks the request, and a looser
        # tol really does buy a bigger step.
        Ω, γ, Δ, T = 2π*1e6, 2π*0.5e6, 2π*2e6, 4e-6
        function run(seqf)
            g, e = Level("g"), Level("e")
            atom = Atom(; levels = [g, e]); sys = System(atom)
            add_coupling!(sys, atom, g => e, Ω; active = true)
            add_detuning!(sys, atom, e, Δ)
            add_decay!(sys, atom, e => g, γ)
            add_detector!(sys, PopulationDetectorSpec(atom, e; name = "Pe"))
            seq = seqf()
            @sequence seq begin
                Wait(T)
            end
            out = play(sys, seq; initial_state = g, density_matrix = true)
            out.detectors["Pe"][end], length(out.times)
        end

        # `tol` bounds ONE integration step. What this measures is the error
        # accumulated over the whole instruction, so it is larger, and it falls
        # more slowly than `tol` itself: the step error is O(h³) and the number
        # of steps grows as tol^(-1/3), so the accumulated error goes as
        # tol^(2/3) -- roughly 2.15x per decade, not 10x. Measured 8.7, 18.5,
        # 40.6, 87.3 (err/tol) for tol = 1e-6 … 1e-9.
        ref, _ = run(() -> Sequence(5e-12))
        errs = Float64[]
        for tol in (1e-4, 1e-5, 1e-6)
            v, _ = run(() -> Sequence(; tol = tol))
            push!(errs, abs(v - ref) / abs(ref))
        end

        # The controller responds to `tol`: tighter always means more accurate.
        @test issorted(errs; rev = true)

        # And it responds at the rate the scheme dictates. A decade of `tol`
        # buys 10^(2/3) = 4.64x; allow a factor of two either side, which still
        # fails loudly if the controller stops tracking `tol` (ratio -> 1) or if
        # something makes it track linearly (ratio -> 10).
        for r in (errs[1] / errs[2], errs[2] / errs[3])
            @test 2.3 < r < 9.3
        end
    end


    @testset "downsample keeps the output grid uniform and endpoint exact" begin
        # `dt` is refined so `downsample` divides the step count. Without that the
        # final group is partial: its sample sits closer to its predecessor than
        # the rest, leaving `out.times` non-uniform in exactly one interval -- a
        # grid that looks uniform and breaks `diff(t)` at the boundary. Worse,
        # sizing the buffer with `÷` dropped that sample entirely, losing the
        # endpoint whenever the two did not divide evenly.
        g = 2π * 1e6
        T = 2π / g
        function run(ds)
            l0, l1 = Level("0"), Level("1")
            q1 = Atom(; levels = [l0, l1]); q2 = Atom(; levels = [l0, l1])
            sys = System([q1, q2])
            add_interaction!(sys, (q1, q2), (l0, l1) => (l1, l0), g / 2)
            add_detector!(sys, PopulationDetectorSpec(q1, l1; name = "P1"))
            seq = Sequence(; tol = 1e-6, downsample = ds)
            push!(seq, Wait(T))
            play(sys, seq; initial_state = [l0, l1], density_matrix = true)
        end

        for ds in (1, 7, 20, 33)      # 7 and 33 do not divide the natural count
            out = run(ds)
            t = out.times
            @test abs(t[end] / T - 1) < 1e-12          # endpoint exact
            if length(t) > 2
                d = diff(t)
                @test (maximum(d) - minimum(d)) < 1e-15 * T   # uniform spacing
            end
        end

        # A `downsample` larger than the step count must still yield one sample,
        # not inflate the instruction to a full group.
        for ds in (150, 5000)
            out = run(ds)
            @test length(out.times) == 1
            @test abs(out.times[end] / T - 1) < 1e-12
        end
    end


    @testset "Chebyshev hot loop is allocation-free" begin
        # The Bessel coefficients depend only on (ΔE·dt, tol), both constant
        # within an instruction, so they are built once by `plan_step` and the
        # per-step call does no coefficient work and touches no global state.
        # Computing them per step instead cost 29 us against 100 ns here.
        d = 2
        Ω = 2π * 5e5
        op = AtomTwin.Dynamiq.Op(
            Tuple{Int,Int,ComplexF64}[(1, 2, ComplexF64(Ω))],
            Tuple{Int,Int,ComplexF64}[(2, 1, ComplexF64(Ω))], d)
        H = Tuple{Base.RefValue{ComplexF64},AtomTwin.Dynamiq.Op}[
                (Ref(ComplexF64(1.0)), op)]
        spec = AtomTwin.Dynamiq.spectral_spec(H)
        Hm = ComplexF64[0 Ω; Ω 0]
        ψ0 = ComplexF64[1, 0]

        for dt in (1e-9, 1e-7)
            plan = AtomTwin.Dynamiq.plan_step(AtomTwin.Dynamiq.Chebyshev(), ψ0, dt,
                                              spec; tol = 1e-12)
            ψ = copy(ψ0)
            AtomTwin.Dynamiq.chebyshev!(ψ, H, plan)          # warm up
            @test norm(ψ - exp(-1im * Hm * dt) * ψ0) < 1e-12  # still exact

            copyto!(ψ, ψ0)
            @test (@allocated AtomTwin.Dynamiq.chebyshev!(ψ, H, plan)) == 0
        end
    end


    @testset "Taylor(order) is honoured" begin
        # `propagate!(::Taylor, ...)` hardcoded `order = 4`, so `Taylor(8)`
        # silently ran at order 4. `TaylorPlan` now carries the order.
        d = 2
        Ω = 2π * 5e5
        op = AtomTwin.Dynamiq.Op(
            Tuple{Int,Int,ComplexF64}[(1, 2, ComplexF64(Ω))],
            Tuple{Int,Int,ComplexF64}[(2, 1, ComplexF64(Ω))], d)
        H = Tuple{Base.RefValue{ComplexF64},AtomTwin.Dynamiq.Op}[
                (Ref(ComplexF64(1.0)), op)]
        spec = AtomTwin.Dynamiq.spectral_spec(H)
        Hm = ComplexF64[0 Ω; Ω 0]
        ψ0 = ComplexF64[1, 0]
        dt = 3e-7                       # coarse enough that the order matters

        errs = map((3, 4, 7, 8)) do ord
            plan = AtomTwin.Dynamiq.plan_step(AtomTwin.Dynamiq.Taylor(ord), ψ0, dt, spec)
            ψ = copy(ψ0)
            AtomTwin.Dynamiq.propagate!(AtomTwin.Dynamiq.Taylor(ord), ψ, H, plan)
            norm(ψ - exp(-1im * Hm * dt) * ψ0)
        end
        # Strictly decreasing: each stable order is more accurate than the last.
        @test issorted(errs; rev = true)
        @test errs[4] < errs[2] / 100   # order 8 well clear of order 4
    end


    @testset "stability warning is integrator-aware" begin
        # The warning describes Taylor's failure mode, and `evolve!` used to ask
        # for it with `get(kwargs, :order, 4)` -- but `order` moved onto
        # `Taylor(order)`, so that always returned 4 and every Chebyshev run was
        # warned against a limit it does not have. Chebyshev at theta = 63 is
        # accurate to 3.5e-13 where Taylor-4 is wrong by 6.5e+05.
        Ω = 2π * 5e6
        op = AtomTwin.Dynamiq.Op(
            Tuple{Int,Int,ComplexF64}[(1, 2, ComplexF64(Ω))],
            Tuple{Int,Int,ComplexF64}[(2, 1, ComplexF64(Ω))], 2)
        H = Tuple{Base.RefValue{ComplexF64},AtomTwin.Dynamiq.Op}[
                (Ref(ComplexF64(1.0)), op)]
        dt = 1e-6                                   # theta ~ 31, far past 2.828

        # Chebyshev: silent, and says so by returning 0.
        @test_logs AtomTwin.Dynamiq.warn_if_step_too_large(
            H, dt, AtomTwin.Dynamiq.Chebyshev())

        # Taylor: warns, and reports the limit of the order actually requested.
        @test_logs (:warn,) AtomTwin.Dynamiq.warn_if_step_too_large(
            H, dt, AtomTwin.Dynamiq.Taylor(4))
        @test_logs (:warn,) AtomTwin.Dynamiq.warn_if_step_too_large(
            H, dt, AtomTwin.Dynamiq.Taylor(8))

        # Chebyshev really is accurate where the old warning claimed divergence.
        spec = AtomTwin.Dynamiq.spectral_spec(H)
        Hm = ComplexF64[0 Ω; Ω 0]
        ψ0 = ComplexF64[1, 0]
        plan = AtomTwin.Dynamiq.plan_step(AtomTwin.Dynamiq.Chebyshev(), ψ0, dt,
                                          spec; tol = 1e-12)
        ψ = copy(ψ0)
        AtomTwin.Dynamiq.propagate!(AtomTwin.Dynamiq.Chebyshev(), ψ, H, plan)
        @test norm(ψ - exp(-1im * Hm * dt) * ψ0) < 1e-10
    end


    @testset "solver probe" begin
        # The controller's instrumentation hook. `NoProbe` must cost nothing --
        # the field was previously `::Any`, and the point of typing it is that
        # an uninstrumented run is unchanged -- and a custom probe must actually
        # observe the controller, which is otherwise unreachable without
        # patching the solver.
        g, e = Level("g"), Level("e")
        atom = Atom(; levels = [g, e])
        sys  = System([atom])
        c    = add_coupling!(sys, atom, g => e, 2π * 1e6; active = false)
        add_decay!(sys, atom, e => g, 2π * 1e5)
        add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P"))
        seq = Sequence(2π / (20 * 2π * 1e6); tol = 1e-8)
        @sequence seq begin
            Pulse([c], 4e-6)
        end
        runit() = play(sys, seq; initial_state = [g], density_matrix = true)

        runit()                                    # warm up
        GC.gc(); a1 = @allocated runit()
        GC.gc(); a2 = @allocated runit()
        @test a1 == a2                             # stable, not drifting

        ctl = AtomTwin.Dynamiq.strang_control(2)
        @test ctl.probe isa AtomTwin.Dynamiq.NoProbe

        ctl.probe = CountProbe(0, 0)
        out = runit()
        fired, kmax = ctl.probe.n, ctl.probe.kmax
        ctl.probe = AtomTwin.Dynamiq.NoProbe()     # leave the cache clean

        @test fired > 0                            # the hook is reachable
        @test kmax >= 1                            # and carries the controller state
        # Instrumenting must not change the answer: this is the converged value
        # the same system gives uninstrumented (see bench/bench1_small.jl).
        @test abs(out.detectors["P"][end] - 0.4220190) < 1e-6
    end

end
