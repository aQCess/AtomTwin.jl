# Thread safety of the solver scratch caches. Per-thread keys do not make a
# shared `Dict` safe -- concurrent inserts corrupt it during `rehash!` -- so each
# `ThreadCache` holds one store per thread. The statevector propagators keep their
# scratch in the per-instruction plan; the density-matrix scratch and spectral
# bounds use `ThreadCache`s (`Dynamiq._ENGINE_CACHES`), which these tests cover.
# They need more than one thread to show anything.
using Test, AtomTwin

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
