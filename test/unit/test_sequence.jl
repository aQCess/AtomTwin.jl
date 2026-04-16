@testset "Sequence rejects non-positive dt" begin
    @test_throws ArgumentError Sequence(0.0)
    @test_throws ArgumentError Sequence(-1e-9)
end

@testset "Sequence rejects non-positive downsample" begin
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
