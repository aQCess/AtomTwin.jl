@testset "getqstate raises before simulation" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys = System(atom)
    @test_throws ErrorException getqstate(sys)
end

@testset "play returns valid detector output" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys = System(atom)

    coupling = add_coupling!(sys, atom, g => e, 2π * 1e6; active = false)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))

    seq = Sequence(1e-9)
    @sequence seq begin
        Pulse(coupling, 100e-9)
    end

    out = play(sys, seq; initial_state = g)
    @test haskey(out.detectors, "P_e")
    @test length(out.times) > 0
    @test all(x -> 0.0 - 1e-6 ≤ x ≤ 1.0 + 1e-6, out.detectors["P_e"])
end

# With no detectors the per-instruction detector list came out as `Vector{Any}`,
# which the solvers' signatures reject: `play` threw a TypeError.
@testset "play runs with no detectors" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys = System(atom)
    c = add_coupling!(sys, atom, g => e, 2π * 1e6; active = false)
    seq = Sequence(1e-8)
    @sequence seq begin
        Pulse(c, 0.25e-6)
    end
    out = play(sys, seq; initial_state = g, savefinalstate = true)
    @test isempty(out.detectors)
    @test sum(abs2, out.final_states[1]) ≈ 1
end

# `@threads :static` cannot nest. A parameter sweep that itself runs under
# `Threads.@threads` must still be able to call a multi-shot `play`.
@testset "multi-shot play inside a threaded loop" begin
    g, e = Level("g"), Level("e")
    function sweep(Ω)
        a = Atom(; levels = [g, e]); s = System([a])
        c = add_coupling!(s, a, g => e, Ω; active = false)
        add_decay!(s, a, e => g, 2π * 1e5)
        add_detector!(s, PopulationDetectorSpec(a, e; name = "Pe"))
        q = Sequence(1e-8)
        @sequence q begin
            Pulse([c], 1e-7)
        end
        sum(play(s, q; initial_state = [g], shots = 8).detectors["Pe"][end, :]) / 8
    end
    r = zeros(4)
    Threads.@threads for k in 1:4
        r[k] = sweep(2π * 1e6 * k)
    end
    @test all(x -> 0 < x < 1, r)
end

@testset "atom equality" begin
    a = Atom(; levels = [Level("g")])
    b = Atom(; levels = [Level("g")])
    @test isequal(a, a)                      # was a MethodError (ambiguous)
    @test !isequal(a, b)
end

# `min(V, V_cap)` with an attractive C6 (so a negative default cap) pinned the
# interaction at the cap for every separation.
@testset "attractive vdW interaction is capped in magnitude only" begin
    r = Level("r")
    a1, a2 = Atom(; levels = [r]), Atom(; levels = [r])
    sys = System([a1, a2])
    C6 = -2π * 1e9 * (1e-6)^6
    f = add_vdwinteraction!(sys, (a1, a2), (r, r) => (r, r), C6)
    a2.inner.x[1] = 2e-6                    # positions are set by `compile`; set directly
    AtomTwin.Dynamiq.update!(f, 0)
    @test real(f._coeff[]) ≈ C6 / (2e-6)^6          # not the 1 µm cap
    a2.inner.x[1] = 0.5e-6
    AtomTwin.Dynamiq.update!(f, 0)
    @test real(f._coeff[]) ≈ C6 / (1e-6)^6          # capped inside 1 µm
end

# Fields whose `update!` recomputes `_coeff` from geometry used to overwrite the
# commanded amplitude every step: a planar-beam coupling, a vdW interaction and a
# light shift were permanently on, whatever `active`, `Pulse` or `ampl` said.
@testset "position-dependent fields obey active, Pulse and ampl" begin
    g, e = Level("g"), Level("e")
    function planar(prog)
        a = Atom(; levels = [g, e]); s = System([a])
        pb = PlanarBeam(780e-9, 1e3, [1.0, 0, 0], [0, 0, 1.0])
        c = add_coupling!(s, a, g => e, 2π * 1e6; beam = pb, active = false)
        add_detector!(s, PopulationDetectorSpec(a, e; name = "Pe"))
        q = Sequence(1e-9)
        prog === :wait  && @sequence q begin Wait(0.5e-6) end
        prog === :pulse && @sequence q begin Pulse(c, 0.5e-6) end
        prog === :half  && @sequence q begin Pulse(c, 0.5e-6; ampl = 0.5) end
        ## frozen: this checks the amplitude logic. With motion (the default) the
        ## absorbed ħk Doppler-shifts the atom and P_e = 0.999986, as it should.
        play(s, q; initial_state = g, frozen = true).detectors["Pe"][end]
    end
    @test planar(:wait) < 1e-12
    @test planar(:pulse) ≈ 1 atol = 1e-6            # π pulse
    @test planar(:half) ≈ 0.5 atol = 1e-6           # sin²(π/4)

    gg, rr = Level("g"), Level("r")
    function blockade(active)
        a1 = Atom(; levels = [gg, rr], x_init = [0.0, 0, 0])
        a2 = Atom(; levels = [gg, rr], x_init = [4e-6, 0, 0])
        s = System([a1, a2])
        cs = [add_coupling!(s, a, gg => rr, 2π * 1e6; active = false) for a in (a1, a2)]
        add_vdwinteraction!(s, (a1, a2), (rr, rr) => (rr, rr), 2π * 50e6 * (4e-6)^6;
                            active = active)
        add_detector!(s, PopulationDetectorSpec(a1, rr; name = "P1"))
        q = Sequence(1e-9)
        @sequence q begin
            Pulse(cs, 0.5e-6)
        end
        play(s, q; initial_state = [gg, gg]).detectors["P1"][end]
    end
    @test blockade(false) ≈ 1 atol = 1e-6            # independent π pulses
    @test blockade(true) < 0.5                       # blockaded

    gm = HyperfineManifold(0//1, 0; label = "¹S₀", term = l"1S0", g_F = 0.0)
    em = HyperfineManifold(1//1, 1; label = "³P₁", term = l"3P1", g_F = 1.5)
    function shifted(active)
        yb = Ytterbium174Atom(; levels = [gm[0], em[0]])
        tw = GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 50e-3, pol = [0, 0, 1.0])
        s = System(yb, tw)
        add_light_shift!(s, yb, [gm[0], em[0]], tw; active = active)
        c = add_coupling!(s, yb, gm[0] => em[0], 2π * 1e6; active = false)
        add_detector!(s, PopulationDetectorSpec(yb, em[0]; name = "Pe"))
        q = Sequence(1e-9)
        @sequence q begin
            Pulse([c], 0.5e-6)
        end
        play(s, q; initial_state = [gm[0]]).detectors["Pe"][end]
    end
    @test shifted(false) ≈ 1 atol = 1e-6             # on resonance: unshifted
    @test shifted(true) < 1e-2                       # shifted off resonance
end

# A coupling left switched on at the end of a shot (`On` with no `Off`) stayed
# on for the next shot, and for the next `play`.
@testset "every shot starts from the declared switch state" begin
    g, e = Level("g"), Level("e")
    a = Atom(; levels = [g, e]); s = System([a])
    c = add_coupling!(s, a, g => e, 2π * 1e6; active = false)
    add_detector!(s, PopulationDetectorSpec(a, e; name = "Pe"))
    q = Sequence(1e-8)
    @sequence q begin
        Wait(0.25e-6)
        On(c)
        Wait(1e-8)
    end
    o = play(s, q; initial_state = g, shots = 3)
    @test all(o.detectors["Pe"][25, :] .< 1e-12)
    @test play(s, q; initial_state = g).detectors["Pe"][25] < 1e-12
end

# A per-shot rate was rescaled from a value stored on the shared node, which was
# never updated: successive shots compounded (V₂·V₃/V₀ instead of V₃). Each
# field now carries its own rate, as `GlobalCoupling` always did.
@testset "per-shot rates do not compound" begin
    g, r = Level("g"), Level("r")
    a1, a2 = Atom(; levels = [g, r]), Atom(; levels = [g, r])
    s = System([a1, a2])
    Ω = Parameter(:Ω, 2π * 1e6)
    V = Parameter(:V, 2π * 10e6)
    add_coupling!(s, a1, g => r, Ω)
    add_coupling!(s, a2, g => r, Ω; beam = PlanarBeam(780e-9, 1e3, [1.0, 0, 0], [0, 0, 1.0]))
    add_interaction!(s, (a1, a2), (r, r) => (r, r), V)
    q = Sequence(1e-8)
    @sequence q begin
        Wait(1e-8)
    end
    job = compile(s, q; initial_state = [g, g])
    for (Ωk, Vk) in ((2π * 2e6, 2π * 20e6), (2π * 3e6, 2π * 30e6))
        AtomTwin.recompile!(job, s; Ω = Ωk, V = Vk)
        for f in job.fields
            want = f isa AtomTwin.Dynamiq.Interaction ? Vk : Ωk / 2
            @test f.rate ≈ (f isa AtomTwin.Dynamiq.Interaction ? Vk : Ωk)
            @test maximum(x -> abs(x[3]), f.H.forward) ≈ abs(want)
        end
    end
end

# The ramps ignored the row × column factorisation that AmplRow/AmplCol apply,
# and did not update the bookkeeping those read.
@testset "tweezer ramps respect the row–column factorisation" begin
    tw = TweezerArray(λ = 767e-9, w0 = 1e-6, P_total = 1e-3,
                      row_freqs = [0.0], col_freqs = [0.0, 1e6])
    atom = Atom(; levels = [Level("g")])
    ta = AtomTwin.resolve(tw, Dict{Symbol,Any}())
    ta.col_amplitudes[2] = 0.5
    mods, _, _ = AtomTwin.compile([atom], RampRow(ta, 1, 0.8, 1e-6), 1e-7)
    targets = sort([m.target for m in mods]; by = real)
    @test real.(targets) ≈ [0.4, 0.8]
    @test ta.row_amplitudes[1] == 0.8
end

# `play(job, sys; initial_state)` wrote the requested state only to
# `sys.state[]`, which the solvers never read, so a reused job started from the
# previous run's FINAL state and process tomography ran every input from the
# first (an X gate with 20% Rabi noise scored fidelity 0.25 instead of ~0.9).
@testset "every run starts from its initial state" begin
    g, e = Level("g"), Level("e")
    a = Atom(; levels = [g, e]); s = System(a)
    c = add_coupling!(s, a, g => e, 2π * 1e6; active = false)
    pd = PhotoDetectorSpec(name = "clicks")
    add_decay!(s, a, e => g, 2π * 1e6; clicks = pd)
    add_detector!(s, pd)
    add_detector!(s, PopulationDetectorSpec(a, e; name = "Pe"))
    q = Sequence(1e-9)
    @sequence q begin
        Pulse(c, 0.25e-6)
    end
    job = compile(s, q; initial_state = [g])
    o1 = play(job, s; rng = Random.Xoshiro(1))
    o2 = play(job, s; rng = Random.Xoshiro(1))
    @test o2.detectors["Pe"] == o1.detectors["Pe"]        # not from the final state
    @test o2.detectors["clicks"] == o1.detectors["clicks"]  # clicks do not accumulate
    oe = play(job, s; initial_state = [e], rng = Random.Xoshiro(1))
    @test oe.detectors["Pe"][1] > 0.9                      # starts in e
    @test play(job, s; rng = Random.Xoshiro(1)).detectors["Pe"] == o1.detectors["Pe"]

    # Identity process: each input must come out as itself.
    idle = Sequence(1e-9)
    @sequence idle begin
        Wait(1e-8)
    end
    s2 = System(a)
    add_decay!(s2, a, e => g, 1.0)
    for dm in (true, false)
        outs = AtomTwin.simulate_process(s2, idle, [g, e]; shots = 4, density_matrix = dm)
        @test real(outs[1][1, 1]) ≈ 1 atol = 1e-6
        @test real(outs[2][2, 2]) ≈ 1 atol = 1e-6
    end
end

# With no Hamiltonian terms the Chebyshev plan saw zero spectral width and took
# the degenerate shortcut -- a global phase -- dropping the anti-Hermitian decay
# of the MCWF effective Hamiltonian: an excited atom never decayed.
@testset "MCWF decays with no Hamiltonian" begin
    g, e = Level("g"), Level("e")
    a = Atom(; levels = [g, e]); s = System(a)
    add_decay!(s, a, e => g, 2π * 1e6)
    add_detector!(s, PopulationDetectorSpec(a, e; name = "Pe"))
    q = Sequence(1e-8)
    @sequence q begin
        Wait(0.5e-6)
    end
    N = 4000
    o = play(s, q; initial_state = e, shots = N, rng = Random.Xoshiro(3))
    p = exp(-2π * 1e6 * 0.5e-6)
    @test abs(sum(o.detectors["Pe"][end, :]) / N - p) < 5 * sqrt(p * (1 - p) / N)
end

# `force` indexed the polarizability table unguarded, so a classical run with an
# atom that has no model at the beam's wavelength threw a KeyError.
@testset "no polarizability at a beam's wavelength means no force" begin
    a = Atom(; levels = [Level("g")], x_init = [0.0, 0, 0], v_init = [1.0, 0, 0])
    s = System([a], [GaussianBeam(λ = 800e-9, w0 = 1e-6, P = 1e-3)])
    add_detector!(s, MotionDetectorSpec(a; dims = [1], name = "x"))
    q = Sequence(1e-8)
    @sequence q begin
        Wait(1e-7)
    end
    @test play(s, q).detectors["x"][end] ≈ 1e-7          # ballistic
end
