@testset "Sequence rejects non-positive dt and downsample" begin
    @test_throws ArgumentError Sequence(0.0)
    @test_throws ArgumentError Sequence(-1e-9)
    @test_throws ArgumentError Sequence(1e-9; downsample=0)
    @test_throws ArgumentError Sequence(1e-9; downsample=-1)
end

@testset "downsample thins detector output and times" begin
    g, e   = Level("g"), Level("e")
    atom   = Atom(; levels=[g, e])
    sys    = System(atom)
    coup   = add_coupling!(sys, atom, g => e, 2π * 1e6; active=false)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name="P_e"))

    dt = 1e-9
    ds = 10
    T  = 100e-9   # exactly 100 steps

    seq_full = Sequence(dt)
    seq_ds   = Sequence(dt; downsample=ds)
    @sequence seq_full begin Pulse(coup, T) end
    @sequence seq_ds   begin Pulse(coup, T) end

    out_full = play(sys, seq_full; initial_state=g)
    out_ds   = play(sys, seq_ds;   initial_state=g)

    n_full = round(Int, T / dt)
    n_ds   = n_full ÷ ds

    @test length(out_full.times) == n_full
    @test length(out_ds.times)   == n_ds
    @test length(out_ds.detectors["P_e"]) == n_ds

    # downsampled times must equal every ds-th full time
    @test out_ds.times ≈ out_full.times[ds:ds:end]

    # downsampled values must equal every ds-th full value
    @test out_ds.detectors["P_e"] ≈ out_full.detectors["P_e"][ds:ds:end]
end

@testset "instruction constructors agree on durations, vectors and Parameters" begin
    g, e = Level("g"), Level("e")
    a = Atom(; levels = [g, e]); s = System(a)
    c = add_coupling!(s, a, g => e, 2π * 1e6)
    @test Pulse(c, 1).duration === 1.0 && Pulse([c], 2).duration === 2.0 && Wait(3).duration === 3.0
    @test On([c]; dt = 1e-9).dt == 1e-9 && Off([c]; downsample = 2).downsample == 2
    tw = TweezerArray(λ = 1e-6, w0 = 1e-6, P_total = 1e-3, row_freqs = [0.0, 1e6],
                      col_freqs = [0.0, 1e6], dx = 1e-12, dy = 1e-12)
    d = Parameter(:d, 1e6)
    for (row, col) in ((MoveRow(tw, 1, d, 1e-6), MoveCol(tw, 1, d, 1e-6)),
                       (MoveRow(tw, [1, 2], 1e6, 1), MoveCol(tw, [1, 2], 1e6, 1)))
        @test row.rows == col.cols && row.delta == col.delta && row.duration === col.duration
    end
    @test RampRow(tw, [1, 2], 0.5, 1).rows == RampCol(tw, [1, 2], 0.5, 1).cols == [1, 2]
    @test RampRow(tw, 1, 0.5, 1).ramp_time === 1.0
end

@testset "instantaneous instructions take no time; a Parallel part ends on time" begin
    g, e = Level("g"), Level("e")
    Ω = 2π * 1e6; T = 0.25e-6; dt = 1e-9
    function run(build)
        a = Atom(; levels = [g, e]); s = System(a)
        c = add_coupling!(s, a, g => e, Ω; active = false)
        add_detector!(s, PopulationDetectorSpec(a, e; name = "P"))
        q = Sequence(dt)
        build(q, c)
        o = play(s, q; initial_state = g, frozen = true)
        o.detectors["P"][end], o.times
    end
    ref, t = run((q, c) -> @sequence q begin Pulse(c, T); Wait(T) end)
    @test ref ≈ sin(Ω * T / 2)^2 atol = 1e-9
    ## On/Off are instantaneous: no steps, no time, no extra drive
    p, t2 = run((q, c) -> @sequence q begin On(c); Wait(T); Off(c); Wait(T) end)
    @test p ≈ ref atol = 1e-12
    @test t2 == t
    ## a constant pulse shorter than the Parallel ends at its own duration
    p, t3 = run((q, c) -> @sequence q begin Parallel([Pulse(c, T), Wait(2T)]) end)
    @test p ≈ ref atol = 1e-12
    @test t3 ≈ t
    ## so does a shaped pulse whose envelope does not end at zero
    env = fill(1.0, 11)
    refs, _ = run((q, c) -> @sequence q begin Pulse(c, T; amplitudes = env, interp = :linear); Wait(T) end)
    ps, _ = run((q, c) -> @sequence q begin Parallel([Pulse(c, T; amplitudes = env, interp = :linear), Wait(2T)]) end)
    @test ps ≈ refs atol = 1e-12
    ## an Off inside a Parallel switches off at its start
    po, _ = run((q, c) -> @sequence q begin On(c); Parallel([Off(c), Wait(T)]) end)
    @test po < 1e-12
    ## a zero-length instruction leaves no sample in the grid
    _, t4 = run((q, c) -> @sequence q begin Wait(T); Wait(0.0); Wait(T) end)
    @test t4 == t
    @test allunique(t4)
end

@testset "FreqCol/AmplCol retune in no time and keep the array's offset" begin
    g = Level("g")
    a = Atom(; levels = [g])
    tw = TweezerArray(λ = 1e-6, w0 = 1e-6, P_total = 1e-3, r0 = (1e-6, 2e-6),
                      row_freqs = [0.0], col_freqs = [0.0, 1e6], dx = 1e-12, dy = 1e-12)
    s = System(a, tw)
    add_detector!(s, MotionDetectorSpec(tw[1, 2]; dims = [1, 2], name = "b"))
    q = Sequence(1e-8)
    @sequence q begin
        Wait(1e-7)
        FreqCol(tw, 2, 3e6)
        AmplCol(tw, 2, 0.5)
        Wait(1e-7)
    end
    o = play(s, q; frozen = true)
    @test length(o.times) == 20 && o.times[end] ≈ 2e-7
    @test o.detectors["b"][10, :] ≈ [1e-6 + 1e-12 * 1e6, 2e-6]       # before the retune
    @test o.detectors["b"][11, :] ≈ [1e-6 + 1e-12 * 3e6, 2e-6]       # r0 offset kept
end
