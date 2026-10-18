# Solver accuracy, convergence order and step control. An order or accuracy
# regression is silent -- results stay plausible -- so these pin the observed
# behaviour.
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

@testset "Propagators" begin
    dts = [2e-9, 1e-9, 5e-10, 2.5e-10]

    @testset "constant pulse is exact" begin
        # Chebyshev is exact for constant H: the error sits at the reference's floor
        # at every step size, so there is no order to fit.
        ref = _final_Pe(1e-10; shaped = false)
        errs = [abs(_final_Pe(dt; shaped = false) - ref) for dt in dts]
        @test all(<(1e-10), errs)
    end

    @testset "shaped pulse is 2nd order (midpoint sampling)" begin
        # Left-endpoint sampling would give order 1 here.
        ref = _final_Pe(1e-10; shaped = true)
        p = _observed_order(dt -> _final_Pe(dt; shaped = true), dts, ref)
        @test 1.7 < p < 2.4
    end

    @testset "piecewise-constant envelope is 1st order by design" begin
        # The staircase is physical (an AWG holds each bin), so it stays first order.
        ref = _final_Pe(5e-11; shaped = true, interp = :piecewise_constant, npts = 51)
        p = _observed_order(dt -> _final_Pe(dt; shaped = true,
                                            interp = :piecewise_constant, npts = 51),
                            dts, ref)
        @test 0.7 < p < 1.5
    end

    @testset "Chebyshev is exact and allocation-free" begin
        # The Bessel coefficients are built once per instruction by `plan_step`.
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

    @testset "Chebyshev has no stability limit" begin
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
        # Blockade-stiff (‖H‖·dt ≈ 5 at 4 ns, past the Taylor-4 limit): Chebyshev has
        # no stability limit.
        @test !isnan(_run(4e-9))
        @test isapprox(_run(4e-9), _run(1e-10); atol = 1e-3)
    end

    @testset "Taylor: unstable orders rejected, stable orders honoured" begin
        # Truncated Taylor is norm-contractive on the imaginary axis only for p ≡ 0, 3
        # (mod 4); other orders amplify at every step size and are rejected.
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

        # Each stable order is more accurate than the last.
        Ω = 2π * 5e5
        op = AtomTwin.Dynamiq.Op(
            Tuple{Int,Int,ComplexF64}[(1, 2, ComplexF64(Ω))],
            Tuple{Int,Int,ComplexF64}[(2, 1, ComplexF64(Ω))], 2)
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
        @test issorted(errs; rev = true)
        @test errs[4] < errs[2] / 100   # order 8 well clear of order 4
    end

    @testset "stability warning: Gershgorin bound, integrator-aware" begin
        D = AtomTwin.Dynamiq
        Random.seed!(91)
        for d in (4, 16, 64)
            A = sprand(ComplexF64, d, d, min(1.0, 8 / d)); H = A + A'
            terms = [(Ref(ComplexF64(1.0)), D.Op(H))]
            @test D.gershgorin_bound(terms) >= opnorm(Matrix(H)) - 1e-9
        end

        # Taylor's stability limit is warned about; Chebyshev has none.
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

        # …and is accurate there.
        spec = AtomTwin.Dynamiq.spectral_spec(H)
        Hm = ComplexF64[0 Ω; Ω 0]
        ψ0 = ComplexF64[1, 0]
        plan = AtomTwin.Dynamiq.plan_step(AtomTwin.Dynamiq.Chebyshev(), ψ0, dt,
                                          spec; tol = 1e-12)
        ψ = copy(ψ0)
        AtomTwin.Dynamiq.propagate!(AtomTwin.Dynamiq.Chebyshev(), ψ, H, plan)
        @test norm(ψ - exp(-1im * Hm * dt) * ψ0) < 1e-10
    end
end

# The Chebyshev plan is built once per instruction. A trap ramped or moved under
# a frozen atom, or a van der Waals term, moves the spectrum within it; the plan
# must follow, or the expansion leaves its interval and stops being unitary.
@testset "Chebyshev plan follows a spectrum that moves within an instruction" begin
    gm = HyperfineManifold(0//1, 0; label = "¹S₀", term = l"1S0", g_F = 0.0)
    em = HyperfineManifold(1//1, 1; label = "³P₁", term = l"3P1", g_F = 1.5)
    g, e = gm[0], em[0]

    function trap_run(dt, mode; decay = false, shots = 1)
        tw  = TweezerArray(λ = 767e-9, w0 = 1e-6, P_total = 50e-3,
                           row_freqs = [0.0], col_freqs = [0.0])
        yb  = Ytterbium174Atom(; levels = [g, e])
        sys = System([yb], [tw])
        c   = add_coupling!(sys, yb, g => e, 2π * 1e6; active = false)
        decay && add_decay!(sys, yb, e => g, 2π * 1e3)
        add_detector!(sys, PopulationDetectorSpec(yb, e; name = "Pe"))
        seq = dt === nothing ? Sequence(; tol = 1e-6) : Sequence(dt)
        part = mode === :ramp ? RampRow(tw, 1, 0.1, 2e-6) : MoveCol(tw, 1, 1e6, 2e-6)
        @sequence seq begin
            Parallel([Pulse([c], 2e-6), part])
        end
        # frozen: this tests the frozen solvers' tracking (a moving trap makes the
        # run non-static, which would otherwise send it to the semiclassical ones).
        Pe = play(sys, seq; initial_state = [g], shots = shots,
                  frozen = true).detectors["Pe"]
        return shots == 1 ? Pe[end] : sum(Pe[end, :]) / shots
    end

    for mode in (:ramp, :move)
        ref = trap_run(1e-9, mode)
        @test 0 < ref < 1
        # Coarse steps carry a discretisation error, but must stay physical.
        @test isapprox(trap_run(50e-9, mode), ref; atol = 0.02)
        @test 0 <= trap_run(nothing, mode) <= 1
        @test 0 <= trap_run(50e-9, mode; decay = true, shots = 2) <= 1   # wfmc
    end

    # Two static atoms: vdW must match the same constant interaction exactly.
    gg, rr = Level("g"), Level("r")
    V, d = 2π * 50e6, 4e-6
    function pair_run(kind, dt)
        a1 = Atom(; levels = [gg, rr], x_init = [0.0, 0, 0])
        a2 = Atom(; levels = [gg, rr], x_init = [d, 0, 0])
        sys = System([a1, a2])
        cs = [add_coupling!(sys, a, gg => rr, 2π * 1e6; active = false) for a in (a1, a2)]
        kind === :vdw ? add_vdwinteraction!(sys, (a1, a2), (rr, rr) => (rr, rr), V * d^6) :
                        add_interaction!(sys, (a1, a2), (rr, rr) => (rr, rr), V)
        add_detector!(sys, PopulationDetectorSpec(a1, rr; name = "P1"))
        seq = Sequence(dt)
        @sequence seq begin
            Pulse(cs, 0.35e-6)
        end
        play(sys, seq; initial_state = [gg, gg]).detectors["P1"][end]
    end
    @test isapprox(pair_run(:vdw, 20e-9), pair_run(:const, 20e-9); atol = 1e-9)
end

@testset "Time grid" begin
    @testset "automatic step (Sequence without dt)" begin
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

        @testset "derived step is the output resolution" begin
            # `_derive_dt` returns how often results are recorded: with no `dt`, one sample
            # per instruction (with no Hamiltonian there is no accuracy bound either).
            for T in (2e-6, 1e-6, 3.7e-6), tol in (1e-5, 1e-8)
                seq = Sequence(; tol = tol)
                push!(seq, Wait(T))
                dt = AtomTwin._derive_dt(seq, AtomTwin.Dynamiq.AbstractField[])
                @test dt ≈ T
                @test dt > 0
            end
        end

        @testset "a stiff system at the derived step" begin
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

        @testset "non-dividing dt" begin
            # A step that does not divide the duration is adjusted, not the duration.
            ref = _rabi(2e-9)
            errs = [abs(_rabi(dt) - ref) for dt in (1.0e-8, 8.4081e-9, 3.1530e-9, 1.0e-9)]
            @test maximum(errs) < 1e-9
        end

        @testset "stepgrid realises the duration exactly" begin
            for (T, dt) in ((2e-6, 8.4081e-9), (1e-6, 7e-10), (3.7e-6, 3.3e-10))
                n, dte = AtomTwin.stepgrid(T, dt)
                @test abs(n * dte - T) < 1e-20        # duration preserved
                @test abs(dte - dt) / dt < 1 / n      # dt perturbed by < one part in n
            end
        end

        @testset "stepgrid never inflates an instruction shorter than downsample" begin
            @test AtomTwin.stepgrid(250e-9, 10e-9, 10_000)[1] == 25
            @test AtomTwin.stepgrid(250e-9, 10e-9, 25)[1] == 25
            @test AtomTwin.stepgrid(250e-9, 10e-9, 10)[1] == 30   # whole groups
            g, e = Level("g"), Level("e")
            atom = Atom(; levels = [g, e]); sys = System(atom)
            c = add_coupling!(sys, atom, g => e, 2π * 1e6; active = false)
            add_detector!(sys, PopulationDetectorSpec(atom, e; name = "Pe"))
            seq = Sequence(10e-9; downsample = 10_000)
            push!(seq, Pulse(c, 250e-9))
            out = play(sys, seq; initial_state = g)
            @test length(out.times) == 1 && out.times[end] ≈ 250e-9
            @test out.detectors["Pe"][end] ≈ sin(2π * 1e6 * 250e-9 / 2)^2 atol = 1e-6
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
    end

    @testset "downsample keeps the output grid uniform and endpoint exact" begin
        # `dt` is refined so `downsample` divides the step count.
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
end

@testset "Density matrix" begin
    @testset "dissipator is exact or CPTP across jump topologies" begin
        # The per-source-level Kraus set is exact where the channels leaving a level are
        # the only ones acting on it, second order otherwise, and trace-preserving and
        # positive at any step size in every topology.
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

        # Allocation-free hot loop.
        J = mkJ([(1, 3, 2.0)], d)
        ρ = copy(ρ0); q1 = copy(ρ0); q2 = copy(ρ0)
        AtomTwin.Dynamiq.fdissipator2!(0.1, ρ, J, q1, q2)
        copyto!(ρ, ρ0)
        @test (@allocated AtomTwin.Dynamiq.fdissipator2!(0.1, ρ, J, q1, q2)) == 0
    end

    @testset "automatic step honours tol on a dissipative system" begin
        # With dissipation the splitting error binds, and the solver sub-divides to meet `tol`.
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

        # `tol` bounds one step; the error accumulated over the instruction goes as
        # tol^(2/3): the step error is O(h³) and the step count grows as tol^(-1/3).
        ref, _ = run(() -> Sequence(5e-12))
        errs = Float64[]
        for tol in (1e-4, 1e-5, 1e-6)
            v, _ = run(() -> Sequence(; tol = tol))
            push!(errs, abs(v - ref) / abs(ref))
        end

        # The controller responds to `tol`: tighter always means more accurate.
        @test issorted(errs; rev = true)

        # A decade of `tol` buys 10^(2/3) = 4.64×; a factor of two either side.
        for r in (errs[1] / errs[2], errs[2] / errs[3])
            @test 2.3 < r < 9.3
        end
    end

    @testset "error control without dissipation" begin
        g, e = Level("g"), Level("e")
        Ω = 2π * 10e6
        for dt in (10e-9, 100e-9)
            a = Atom(; levels = [g, e])
            sys = System([a])
            c = add_coupling!(sys, a, g => e, Ω; active = false)
            add_detector!(sys, PopulationDetectorSpec(a, e; name = "Pe"))
            seq = Sequence(dt)
            @sequence seq begin
                Pulse([c], 0.95e-6)
            end
            Pe = play(sys, seq; initial_state = [g], density_matrix = true).detectors["Pe"][end]
            @test isapprox(Pe, sin(Ω * 0.95e-6 / 2)^2; atol = 1e-3)
        end
    end

    @testset "solver probe" begin
        # The controller's instrumentation hook: `NoProbe` costs nothing, and a custom
        # probe observes the controller without changing the answer.
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

@testset "Classical motion" begin
    @testset "velocity Verlet is 2nd order" begin
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
            # A stale cached force corrupts the trajectory; reset_force! clears it.
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

    @testset "moves converge at second order (semiclassical solvers)" begin
        gm = HyperfineManifold(0//1, 0; label = "¹S₀", term = l"1S0", g_F = 0.0)
        em = HyperfineManifold(1//1, 1; label = "³P₁", term = l"3P1", g_F = 1.5)
        g, e = gm[0], em[0]
        function run(dt; density_matrix = false, Γ = 0.0)
            tw  = TweezerArray(λ = 767e-9, w0 = 1e-6, P_total = 50e-3,
                               row_freqs = [0.0], col_freqs = [0.0])
            yb  = Ytterbium174Atom(; levels = [g, e])
            sys = System([yb], [tw])
            c   = add_coupling!(sys, yb, g => e, 2π * 1e6; active = false)
            Γ > 0 && add_decay!(sys, yb, e => g, Γ)
            add_detector!(sys, PopulationDetectorSpec(yb, e; name = "Pe"))
            # A tight `tol` keeps the density-matrix solver's adaptive sub-steps from
            # limiting the error: with the default, the controller's tolerance is
            # comparable to the dt² error at coarse dt and hides its scaling.
            seq = Sequence(dt; tol = 1e-10)
            @sequence seq begin
                Parallel([Pulse([c], 1e-6), MoveCol(tw, 1, 0.5e6, 1e-6; sweep = :min_jerk)])
            end
            real(play(sys, seq; initial_state = [g], density_matrix = density_matrix,
                      rng = MersenneTwister(1)).detectors["Pe"][end])
        end
        # Every ratio across a sweep from coarse to fine steps must be ≈ 4: at coarse
        # dt a large dt² term can mask first-order ones. Each semiclassical solver has
        # its own step loop, so each is checked:
        # statevector (`tdse_semiclassical`), density matrix with decay
        # (`qme_semiclassical`), and MCWF (`wfmc_semiclassical`). The MCWF decay is
        # weak enough that no jump fires, which keeps the run deterministic: shot
        # noise would swamp the ~1e-7 differences measured here.
        dts = [8e-9 / 2^k for k in 0:8]          # 8 ns down to 0.03125 ns
        for kw in ((;), (density_matrix = true, Γ = 2π * 10e3), (Γ = 2π * 10.0,))
            P = [run(dt; kw...) for dt in dts]
            for k in 1:length(P) - 2
                ratio = (P[k] - P[k+1]) / (P[k+1] - P[k+2])
                @test 3.8 < ratio < 4.2         # 2 for first order, 4 for second
            end
        end
    end

    # Velocity Verlet's end-of-step force needs the beams at the step's end time;
    # any other time is first order once the trap moves.
    @testset "moves converge at second order (classical solver)" begin
        g = Level("1S0")
        function run(dt; moving)
            tw  = TweezerArray(λ = 767e-9, w0 = 1e-6, P_total = 50e-3,
                               row_freqs = [0.0], col_freqs = [0.0])
            yb  = Ytterbium174Atom(; levels = [g], x_init = [moving ? 0.0 : 0.1e-6, 0.0, 0.0],
                                   v_init = [0.0, 0.0, 0.0])
            sys = System([yb], [tw])
            add_detector!(sys, MotionDetectorSpec(yb; dims = [1], name = "x"))
            seq = Sequence(dt)
            if moving
                # The trap moves 1.5 µm in 1 µs, faster than the atom can follow, so
                # the force on the atom changes in time as the trap sweeps past.
                @sequence seq begin
                    MoveCol(tw, 1, 0.5e6, 1e-6; sweep = :min_jerk)
                end
            else
                # Control: a static trap, with the atom displaced from its centre.
                @sequence seq begin
                    Wait(1e-6)
                end
            end
            o = @test_logs (:warn, r"Initial state not specified") play(sys, seq)
            o.detectors["x"][end, 1]
        end
        dts = [8e-9 / 2^k for k in 0:8]          # 8 ns down to 0.03125 ns
        for moving in (false, true)
            X = [run(dt; moving) for dt in dts]
            for k in 1:length(X) - 2
                ratio = (X[k] - X[k+1]) / (X[k+1] - X[k+2])
                @test 3.8 < ratio < 4.2         # 2 for first order, 4 for second
            end
        end
    end
end

@testset "MCWF" begin
    @testset "photon counts match the master equation" begin
        # Saturated line: photon counts equal Γ∫P_e dt from the master equation. The
        # manifold decay (three channels, two dark) checks that each jump is drawn from
        # the state at its time, which holds the decaying amplitude.
        Γ = 2π * 29.1e6; s = 40.0; Ω = Γ * sqrt(s / 2); T = 400e-9; dt = 1e-9
        function build(clicks, manifold)
            if manifold
                gm, em = HyperfineManifold(0//1, 0; label = "g"), HyperfineManifold(1//1, 1; label = "e")
                a = Atom(; levels = [gm..., em...]); g, e, decay = gm[0], em[0], em => gm
            else
                g, e = Level("g"), Level("e")
                a = Atom(; levels = [g, e]); decay = e => g
            end
            sys = System(a)
            c = add_coupling!(sys, a, g => e, Ω; active = false)
            if clicks
                pd = PhotoDetectorSpec(name = "c"); add_detector!(sys, pd)
                add_decay!(sys, a, decay, Γ; clicks = pd)
            else
                add_decay!(sys, a, decay, Γ)
                add_detector!(sys, PopulationDetectorSpec(a, e; name = "Pe"))
            end
            seq = Sequence(dt)
            push!(seq, Pulse(c, T; downsample = 1)); push!(seq, Wait(60e-9; downsample = 1))
            sys, seq, g
        end
        for manifold in (false, true)
            sys, seq, g = build(false, manifold)
            o = play(sys, seq; initial_state = g, density_matrix = true)
            t = [0.0; o.times]; P = [0.0; real.(o.detectors["Pe"])]
            N_exact = Γ * sum(0.5 * (P[i] + P[i-1]) * (t[i] - t[i-1]) for i in 2:length(t))
            sys, seq, g = build(true, manifold)
            N = vec(sum(play(sys, seq; initial_state = g, shots = 6000,
                             rng = MersenneTwister(11)).detectors["c"], dims = 1))
            se = std(N) / sqrt(length(N))
            @test abs(mean(N) - N_exact) < 4se  
            @test abs(mean(N) / N_exact - 1) < 0.006
        end
    end

    # Each jump is placed where the norm crosses its threshold, on the Chebyshev
    # step's dense output: a step that is a whole number of Rabi periods, or that
    # holds several jumps, gives the master-equation result.
    @testset "jump times are exact, whatever the sampling step" begin
        function twolevel(Ω, Γ, T, nsamp; dm = false, shots = 1000, Δ = 0.0, clicks = false)
            g, e = Level("g"), Level("e"); a = Atom(; levels = [g, e]); s = System(a)
            c = add_coupling!(s, a, g => e, Ω; active = false)
            Δ != 0 && add_detuning!(s, a, e, -Δ)
            if clicks
                pd = PhotoDetectorSpec(name = "clicks"); add_detector!(s, pd)
                add_decay!(s, a, e => g, Γ; clicks = pd)
            else
                add_decay!(s, a, e => g, Γ)
            end
            add_detector!(s, PopulationDetectorSpec(a, e; name = "P"))
            q = Sequence(T, nsamp); @sequence q begin Pulse(c, T) end
            play(s, q; initial_state = g, density_matrix = dm, shots = dm ? 1 : shots,
                 rng = MersenneTwister(11))
        end
        # Sampling step = 10 Rabi periods (the preprint's Rabi benchmark).
        Ω, Γ, T = 2π * 1e6, 2π * 0.5e3, 1e-3
        ref = twolevel(Ω, Γ, T, 100; dm = true).detectors["P"]
        P   = twolevel(Ω, Γ, T, 100; shots = 2000).detectors["P"]
        @test mean(P[end, :]) ≈ ref[end] atol = 4 * 0.5 / sqrt(2000)
        # Saturated line sampled at Γ·dt = 10: several jumps per step.
        Ω, Γ = 2π * 2e6, 2π * 1e6
        ref = twolevel(Ω, Γ, 20e-6, 20; dm = true).detectors["P"]
        P   = twolevel(Ω, Γ, 20e-6, 20; shots = 2000).detectors["P"]
        @test mean(P[end, :]) ≈ ref[end] atol = 4 * 0.5 / sqrt(2000)
        # Photon counts on a far-detuned line equal Γ∫P_e dt at any sampling step.
        Ω, Δ, Γ, T = 2π * 10e6, 2π * 50e6, 2π * 1e6, 5e-6
        fine = twolevel(Ω, Γ, T, 5000; dm = true, Δ = Δ).detectors["P"]
        expected = Γ * sum(fine) * (T / 5000)
        for nsamp in (500, 12)                             # dt = 10 ns and 417 ns
            n = vec(sum(twolevel(Ω, Γ, T, nsamp; shots = 20000, Δ = Δ, clicks = true).detectors["clicks"], dims = 1))
            @test mean(n) ≈ expected atol = 4 * std(n) / sqrt(length(n))
        end
    end
end

@testset "a shaped envelope on a position-dependent coupling is resampled within the step" begin
    # The coefficient of a position-dependent field (amplitude × geometric factor)
    # follows the envelope at every sub-step. At the origin a frozen PlanarCoupling
    # is a GlobalCoupling, so the two must agree.
    g, e = Level("g"), Level("e")
    function run(planar; dm)
        a = Atom(; levels = [g, e], x_init = zeros(3), v_init = zeros(3))
        sys = System(a)
        c = planar ? add_coupling!(sys, a, g => e, 2π * 5e6; active = false,
                                   beam = PlanarBeam(780e-9, 1.0, [1.0, 0, 0], [0, 0, 1.0])) :
                     add_coupling!(sys, a, g => e, 2π * 5e6; active = false)
        add_detuning!(sys, a, e, 2π * 3e6)
        add_decay!(sys, a, e => g, 2π * 1e6)
        add_detector!(sys, PopulationDetectorSpec(a, e; name = "Pe"))
        seq = Sequence(50e-9)                        # 4 samples across the pulse
        push!(seq, Pulse(c, 200e-9; amplitudes = sin.(range(0, π, length = 101)) .^ 2))
        o = play(sys, seq; initial_state = g, frozen = true, density_matrix = dm,
                 shots = dm ? 1 : 200, rng = MersenneTwister(2))
        o.detectors["Pe"]
    end
    @test run(true; dm = true) ≈ run(false; dm = true) rtol = 1e-10
    @test run(true; dm = false) == run(false; dm = false)
end
