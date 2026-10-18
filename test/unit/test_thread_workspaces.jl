# Thread safety of the solver scratch caches. Per-thread keys do not make a
# shared `Dict` safe -- concurrent inserts corrupt it during `rehash!` -- so each
# `ThreadCache` holds one store per thread. The statevector propagators keep their
# scratch in the per-instruction plan; the density-matrix scratch and spectral
# bounds use `ThreadCache`s (`Dynamiq._ENGINE_CACHES`), which these tests cover.
# They need more than one thread to show anything.
using Test, AtomTwin, Random, Statistics

const _D = AtomTwin.Dynamiq

@testset "ThreadCache concurrent inserts" begin
    # Many distinct dims per thread, so inserts keep crossing rehash! points.
    function hammer(ndims::Int)
        c   = _D.ThreadCache{Vector{Float64}}(d -> zeros(d))
        bad = Threads.Atomic{Int}(0)
        Threads.@threads :static for _ in 1:max(Threads.nthreads(), 2)
            for d in 1:ndims
                length(_D.get_ws!(c, d)) == d || Threads.atomic_add!(bad, 1)
            end
        end
        bad[]
    end

    # Fresh cache each trial: the cold-start path is the one that raced.
    @test all(hammer(60) == 0 for _ in 1:200)

    # A given thread must get the SAME object back — the cache exists so the
    # propagator allocates nothing in the stepping loop.
    c = _D.ThreadCache{Vector{Float64}}(d -> zeros(d))
    @test _D.get_ws!(c, 16) === _D.get_ws!(c, 16)
    @test _D.get_ws!(c, 16) !== _D.get_ws!(c, 32)

    # Distinct dims are distinct entries, each the right size.
    @test length(_D.get_ws!(c, 7)) == 7
    @test length(_D.get_ws!(c, 33)) == 33
end

@testset "threaded shots match serial" begin
    # End-to-end: if workspaces were shared across concurrent shots the states
    # would cross-talk. Same seed, so threaded and serial must agree exactly.
    function run(; parallel::Bool)
        g, e = Level("g"), Level("e")
        atom = Atom(; levels = [g, e])
        sys  = System(atom)
        c = add_coupling!(sys, atom, g => e, 2π * 1e6; active = false)
        add_decay!(sys, atom, e => g, 2π * 0.05e6; active = true)
        add_detector!(sys, PopulationDetectorSpec(atom, e; name = "Pe"))
        seq = Sequence(2e-9)
        @sequence seq begin
            Pulse(c, 2e-6)
        end
        job = compile(sys, seq; density_matrix = false, initial_state = [g])
        out = play(job, sys; shots = 32, rng = MersenneTwister(20260913),
                   parallel_thresh = parallel ? 4 : typemax(Int))
        out.detectors["Pe"]
    end

    @test run(parallel = true) ≈ run(parallel = false)
end

@testset "per-shot beam and position sampling is independent of threading" begin
    # `recompile!` resolves a sampled beam into the system's nodes and reads atom
    # positions through the system, so each thread needs its own copy of both.
    function run(random_power; parallel)
        gm = HyperfineManifold(0//1, 0; label = "1S0", term = l"1S0", g_F = 0.0)
        em = HyperfineManifold(1//1, 1; label = "3P1", term = l"3P1", g_F = 1.5)
        yb = Ytterbium174Atom(; levels = [gm..., em...],
                              x_init = random_power ? zeros(3) : gaussian(0.4e-6, 0.4e-6, 0.0))
        P = random_power ? Parameter(:P, 1e-9; std = 0.3e-9) : 1e-9
        beam = GeneralGaussianBeam(556e-9, 1e-6, 1e-6, P, [1.0, 0, 0], [0.0, 0, 1.0])
        sys = System(yb)
        cs = add_coupling!(sys, yb, gm => em, beam; d_red = 1e-31)
        add_detector!(sys, PopulationDetectorSpec(yb, em[0]; name = "Pe"))
        seq = Sequence(1e-8)
        @sequence seq begin
            Parallel([Pulse(c, 1e-6) for c in cs])
        end
        out = play(sys, seq; initial_state = gm[0], shots = 64, frozen = true,
                   rng = MersenneTwister(7),
                   parallel_thresh = parallel ? 1 : typemax(Int))
        out.detectors["Pe"][end, :]
    end
    for random_power in (true, false)
        s = run(random_power; parallel = false)
        @test std(s) > 1e-3                      # the sampling matters
        @test run(random_power; parallel = true) ≈ s
    end
end
