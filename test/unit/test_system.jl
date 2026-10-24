# System construction, play's state handling, and field switching.

@testset "play: detector output, no detectors, getqstate before a run" begin
    let
        g, e = Level("g"), Level("e")
        atom = Atom(; levels = [g, e])
        sys = System(atom)
        @test_throws ErrorException getqstate(sys)
    end

    let
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

    let
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
end

# `@threads :static` cannot nest; a sweep under `Threads.@threads` runs its shots serially.
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
    @test isequal(a, a)                  
    @test !isequal(a, b)
end

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

# A field whose coefficient is built from geometry (planar coupling, vdW, light
# shift) keeps the commanded amplitude separate from the geometric factor.
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
        ## frozen: this checks the amplitude logic, not the Doppler shift.
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

@testset "every shot and run starts from its declared state" begin
    ## Switch states: a coupling left on (`On` without `Off`) is off again next shot.
    let
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

    ## Quantum state and photon clicks; process tomography runs each input from itself.
    let
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
end

# Each job field carries its own rate, so a parameter resampled per shot is set,
# not rescaled from a shared value.
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

# With no Hamiltonian the effective Hamiltonian is purely anti-Hermitian; the
# degenerate (zero-width) spectral case must still decay.
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

@testset "detectors report on their own tspan" begin
    g, e = Level("g"), Level("e")
    function run(tspans; shots)
        a = Atom(; levels = [g, e], x_init = [0.0, 0, 0], v_init = [1.0, 2.0, 0])
        s = System(a)
        c = add_coupling!(s, a, g => e, 2π * 5e6)
        pd = PhotoDetectorSpec(name = "n", tspan = tspans)
        add_detector!(s, pd)
        add_decay!(s, a, e => g, 2π * 2e6; clicks = pd)
        add_detector!(s, PopulationDetectorSpec(a, e; name = "Pe", tspan = tspans))
        add_detector!(s, MotionDetectorSpec(a; dims = 1:6, name = "xv", tspan = tspans))
        q = Sequence(1e-8)
        @sequence q begin
            Wait(1e-6)
        end
        play(s, q; initial_state = g, shots = shots, rng = Random.Xoshiro(7))
    end
    full = run(nothing; shots = 3)
    t = full.times
    te = [0.3e-6, 0.75e-6]
    sub = run(te; shots = 3)
    k = [argmin(abs.(t .- x)) for x in te]
    in1, in2 = t .<= te[1] + 1e-15, te[1] + 1e-15 .< t .<= te[2] + 1e-15
    @test size(sub.detectors["n"]) == (2, 3)
    @test sub.detectors["n"][1, :] == vec(sum(full.detectors["n"][in1, :]; dims = 1))
    @test sub.detectors["n"][2, :] == vec(sum(full.detectors["n"][in2, :]; dims = 1))
    @test sub.detectors["Pe"] == full.detectors["Pe"][k, :]
    @test sub.detectors["xv"] == full.detectors["xv"][k, :, :]
    one = run(te; shots = 1)
    @test size(one.detectors["n"]) == (2,) && size(one.detectors["xv"]) == (2, 6)
    ## free flight: dims 4–6 are the velocity
    @test full.detectors["xv"][end, 4:6, 1] == [1.0, 2.0, 0.0]
    @test full.detectors["xv"][end, 1:2, 1] ≈ [1.0, 2.0] .* t[end]
    ## tspan must be sorted and within the run; dims within 1:6 (1:3 for a beam)
    @test_throws ArgumentError run([0.75e-6, 0.3e-6]; shots = 1)
    @test_throws ArgumentError run([0.3e-6, 2e-6]; shots = 1)
    @test_throws ArgumentError MotionDetectorSpec(Atom(; levels = [g, e]); dims = [7])
    @test_throws ArgumentError MotionDetectorSpec(PlanarBeam(1e-6, 1.0, [1.0, 0, 0], [0.0, 1, 0]); dims = [4])
end

@testset "gethamiltonian includes the trapping beams' light shifts" begin
    g = HyperfineManifold(1//2, 0; label = "1S0", term = l"1S0")
    e = HyperfineManifold(3//2, 1; label = "3P1", term = l"3P1")
    yb = Ytterbium171Atom(; levels = [g.levels; e.levels])
    s = System(yb, GaussianBeam(λ = 759e-9, w0 = 1e-6, P = 1e-3, pol = [1.0, 0, 0]))
    add_quantization_axis!(s, [0.0, 0, 1])          # ⊥ trap polarisation: tensor mixing
    add_zeeman_detunings!(s, yb, e; B = 1e-3)
    q = Sequence(1e-9)
    @sequence q begin
        Wait(1e-9)
    end
    job = compile(s, q)
    H = zeros(ComplexF64, 6, 6)
    for f in job.fields
        AtomTwin.Dynamiq.update!(f, 0)
        H .+= f._coeff[] .* Matrix(f.H)
    end
    Hs = gethamiltonian(s)
    @test Hs ≈ H rtol = 1e-12
    @test abs(Hs[1, 1]) > 2π * 1e5                  # the ¹S₀ trap shift (≈ 560 kHz) is there
    @test any(abs(Hs[i, j]) > 0 for i in 3:6, j in 3:6 if i != j)    # tensor mixing in ³P₁
end

@testset "an unknown species needs its mass and nuclear spin" begin
    @test_throws ArgumentError AtomTwin.AtomWrapper{:Cesium133}()
    @test_throws ArgumentError AtomTwin.AtomWrapper{:Cesium133}(; mass = 133 * AtomTwin.Units.amu)
    cs = AtomTwin.AtomWrapper{:Cesium133}(; mass = 133 * AtomTwin.Units.amu, I = 7//2)
    @test cs.m ≈ 133 * AtomTwin.Units.amu
    @test cs.I == 7//2
end

@testset "a coupling to a level the atom lacks warns and adds nothing" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e]); sys = System(atom)
    c = @test_logs (:warn, r"not found") add_coupling!(sys, atom, g => Level("x"), 2π * 1e6)
    @test isempty(c)
    @test isempty(sys.nodes)
end

@testset "beam constructors take any real numbers, positionally or by keyword" begin
    b1 = GaussianBeam(532e-9, 1e-6, 1)
    b2 = GaussianBeam(; λ = 532e-9, w0 = 1e-6, P = 1, r0 = [0, 0, 0])
    @test b1 isa GaussianBeam && b2 isa GaussianBeam && b1.I0 == b2.I0
    g1 = GeneralGaussianBeam(532e-9, 1e-6, 2e-6, 1, [0, 0, 1], [1, 0, 0])
    g2 = GeneralGaussianBeam(; λ = 532e-9, w0x = 1e-6, w0y = 2e-6, P = 1)
    @test g1 isa GeneralGaussianBeam && g2 isa GeneralGaussianBeam && g1.I0 == g2.I0
    @test GeneralGaussianBeam(532e-9, Parameter(:w, 1e-6), 2e-6, 1.0, [0, 0, 1], [1, 0, 0]) isa
          AtomTwin.ParametricBeam
    p1 = PlanarBeam(556e-9, 1.0, [0, 1, 0], [0, 0, 1])
    p2 = PlanarBeam(; λ = 556e-9, I = 1, direction = [0, 1, 0], polarization = [0, 0, 1])
    @test p1.k == p2.k && p1.pol == p2.pol && p1.I == p2.I
end

@testset "detector specs take the same keywords" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e]); sys = System(atom)
    name = SubString("P_e", 1)
    @test PopulationDetectorSpec(atom, e; name).params.name == "P_e"
    @test CoherenceDetectorSpec(atom, g => e; name).params.name == "P_e"
    d = ExpectationDetectorSpec(sys, atom, AtomTwin.projector(e); name, tspan = [1e-6])
    @test d.tspan == [1e-6] && d.params.name == "P_e"
end

@testset "gethamiltonian rejects keywords it does not take" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e]); sys = System(atom)
    add_coupling!(sys, atom, g => e, 2π * 1e6)
    @test size(gethamiltonian(sys)) == (2, 2)
    @test_throws MethodError gethamiltonian(sys; bogus = 1)
end

@testset "Dynamiq basis and operator utilities" begin
    D = AtomTwin.Dynamiq
    a = D.NLevelAtom(2); b = D.Basis([a], [(1,), (2,)])
    @test a._pidx == [[1], [2]]                       # explicit elements set the index lists
    @test D.productstate(b, (2,)) == ComplexF64[0, 1]
    @test_throws ArgumentError D.productstate(b, (3,))
    # A scaled operator keeps its forward/reverse split, so a complex time
    # coefficient still gives a Hermitian term.
    H = D.Op(b, a, 1 => 2, 1.0)
    function applied(op, c)
        M = zeros(ComplexF64, 2, 2)
        for j in 1:2
            v = zeros(ComplexF64, 2); v[j] = 1; w = zeros(ComplexF64, 2)
            D.apply!(w, [(Ref(ComplexF64(c)), op)], v, ComplexF64(1)); M[:, j] = w
        end
        M
    end
    M2 = applied(2 * H, cis(0.3)); @test M2 ≈ M2'
    @test applied(2 * H, cis(0.3)) ≈ 2 * applied(H, cis(0.3))
end

@testset "an instant retune moves the atom in a quantum run, at second order" begin
    g = Level("1S0")
    function run(dt; quantum)
        tw = TweezerArray(λ = 759e-9, w0 = 1e-6, P_total = 20e-3, row_freqs = [0.0], col_freqs = [0.0])
        yb = Ytterbium174Atom(; levels = [g], x_init = [0.0, 0, 0], v_init = [0.0, 0, 0])
        s = System([yb], [tw])
        add_detector!(s, MotionDetectorSpec(yb; dims = [1, 4], name = "x"))
        q = Sequence(dt)
        @sequence q begin
            Wait(2e-6)
            FreqCol(tw, 1, 0.3e6)
            Wait(2e-6)
        end
        o = quantum ? play(s, q; initial_state = g) : play(s, q)
        o.detectors["x"][end, :]
    end
    xq, xc = run(4e-9; quantum = true), run(4e-9; quantum = false)
    @test abs(xq[1]) > 1e-8                       # the atom follows the displaced trap
    @test xq ≈ xc rtol = 1e-9                     # as the classical solver has it
    ## the first drift after the retune uses the force at the new trap position
    x = [run(dt; quantum = false)[1] for dt in (8e-9, 4e-9, 2e-9)]
    @test 3.3 < (x[1] - x[2]) / (x[2] - x[3]) < 4.7
end

@testset "a parametric beam on a single level pair is refused, not silently dropped" begin
    g, e = Level("g"), Level("e")
    a = Atom(; levels = [g, e])
    pb = GaussianBeam(556e-9, Parameter(:w0, 1e-6), 1e-3)
    @test_throws ArgumentError add_coupling!(System(a), a, g => e, 2π * 1e6; beam = pb)
end

@testset "GeneralGaussianBeam waists lie along the lab axes; the bare field vector" begin
    D = AtomTwin.Dynamiq
    b = GeneralGaussianBeam(532e-9, 1e-6, 2e-6, 1e-3, [0, 0, 1], [1, 0, 0])
    I0 = D.intensity(b, [0.0, 0, 0])
    @test D.intensity(b, [1e-6, 0, 0]) / I0 ≈ exp(-2)          # w0x = 1 µm along lab x
    @test D.intensity(b, [0, 1e-6, 0]) / I0 ≈ exp(-0.5)        # w0y = 2 µm along lab y
    @test b.u ≈ [1, 0, 0] && b.v ≈ [0, 1, 0]
    bx = GeneralGaussianBeam(532e-9, 1e-6, 2e-6, 1e-3, [1, 0, 0], [0, 1, 0])
    @test bx.u ≈ [0, 1, 0] && bx.v ≈ [0, 0, 1]
    r = [0.3e-6, 0.2e-6, 1e-6]
    b._coeff[] = 0.5
    @test D.efield_vector(b, r) ≈ D.Efield(b, r) ./ 0.5          # efield_vector excludes _coeff
    @test D.efield_scalar(b, r) ≈ 0.5 * D._efield_bare(b, r)
    b._coeff[] = 1.0
    ## the force evaluation allocates nothing
    a = Ytterbium171Atom(; levels = [Level("1S0")], x = [0.2e-6, 0.1e-6, 0.3e-6])
    bs = [b]
    AtomTwin._init_species_data!(a, a.inner, bs)
    a.inner._P[1] = 1.0
    measure(at, beams) = (AtomTwin.Dynamiq.force(at, beams); @allocated AtomTwin.Dynamiq.force(at, beams))
    @test measure(a.inner, bs) == 0
end

@testset "force over a Vector{AbstractBeam} allocates nothing per beam" begin
    # The job's beam list is abstractly typed; the polarization lookup must stay
    # concrete there or every accumulation boxes (a 60× slower sorting example).
    # Julia 1.11 keeps a fixed 112 B per call (1.12: 0 B); what must not happen is
    # an allocation that grows with the number of beams.
    a = Ytterbium171Atom(; levels = [Level("1S0")], x = [0.2e-6, 0.1e-6, 0.3e-6])
    beams(n) = (bs = AbstractBeam[GaussianBeam(759e-9, 1e-6, 1e-3; r0 = [1e-6 * k, 0, 0]) for k in 0:n-2];
                push!(bs, GeneralGaussianBeam(759e-9, 1e-6, 2e-6, 1e-3, [0, 0, 1], [0, 1, 0])); bs)
    f(atom, bs) = AtomTwin.Dynamiq.force(atom, bs)   # qualified: a captured local would box
    bs5, bs20 = beams(5), beams(20)
    AtomTwin._init_species_data!(a, a.inner, bs20)
    a.inner._P[1] = 1.0
    f(a.inner, bs5); f(a.inner, bs20)
    a5, a20 = (@allocated f(a.inner, bs5)), (@allocated f(a.inner, bs20))
    @test a20 == a5
    @test a5 <= 128
    @test sum(abs, f(a.inner, bs5)) > 0
end

@testset "same-wavelength beams interfere according to their polarizations" begin
    D = AtomTwin.Dynamiq
    λ = 759e-9
    b1 = GeneralGaussianBeam(λ, 5e-6, 5e-6, 10e-3, [0, 0, 1], [0, 1, 0])      # k ∥ z, pol ŷ
    b2 = GeneralGaussianBeam(λ, 5e-6, 5e-6, 10e-3, [1, 0, 0], [0, 1, 0])      # k ∥ x, pol ŷ: interferes
    b3 = GeneralGaussianBeam(λ, 5e-6, 5e-6, 10e-3, [1, 0, 0], [0, 0, 1])      # k ∥ x, pol ẑ: does not
    r = [λ / 8, 0.0, λ / 8]
    force(beams) = (a = Ytterbium171Atom(; levels = [Level("1S0")], x = r);
                    AtomTwin._init_species_data!(a, a.inner, beams; q_axis = [1.0, 0, 0]);
                    a.inner._P[1] = 1.0;                               # the level is occupied
                    collect(D.force(a.inner, beams)))
    single = force([b1]) .+ force([b3])
    @test force([b1, b3]) ≈ single rtol = 1e-9
    @test !isapprox(force([b1, b2]), force([b1]) .+ force([b2]); rtol = 0.1)
end

@testset "detector names are unique; a plane wave is not a trap" begin
    g, e = Level("g"), Level("e")
    a = Atom(; levels = [g, e]); s = System(a)
    add_detector!(s, PopulationDetectorSpec(a, g))
    @test_throws ArgumentError add_detector!(s, PopulationDetectorSpec(a, e))
    add_detector!(s, PopulationDetectorSpec(a, e; name = "Pe"))
    @test length(s.detector_specs) == 2
    pw = PlanarBeam(759e-9, 1e9, [0.0, 0, 1], [1.0, 0, 0])
    @test_throws ArgumentError System(a, pw)
    @test_throws ArgumentError MotionDetectorSpec(pw; dims = [1])
end
