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
