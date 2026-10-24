# DAG: node dependencies, topological order, parameter rebinding

# Test-local node type for cycle detection test (must be at file scope in Julia)
mutable struct _CyclicTestNode <: AtomTwin.AbstractNode
    peer::Union{Nothing, _CyclicTestNode}
end
_CyclicTestNode() = _CyclicTestNode(nothing)
AtomTwin.node_dependencies(n::_CyclicTestNode) =
    n.peer === nothing ? AtomTwin.AbstractNode[] : AtomTwin.AbstractNode[n.peer]

@testset "node_dependencies" begin
    let
        g, e = Level("g"), Level("e")
        atom = Atom(; levels = [g, e])
        beam_node = AtomTwin.BeamNode(PlanarBeam(578e-9, 1.0, [1.0, 0.0, 0.0], [0, 1, 0]))
        decay_node = AtomTwin.DecayNode(1e6, atom, g => e)
        det_node   = AtomTwin.DetuningNode(1e6, atom, e)
        @test isempty(AtomTwin.node_dependencies(beam_node))
        @test isempty(AtomTwin.node_dependencies(decay_node))
        @test isempty(AtomTwin.node_dependencies(det_node))
    end

    ## A coupling depends on a beam node only through a BeamRabiFrequency.
    let
        g, e = Level("g"), Level("e")
        atom = Atom(; levels = [g, e])
        node = AtomTwin.CouplingNode(2π * 1e6, atom, g => e)
        @test isempty(AtomTwin.node_dependencies(node))
    end

    let
        g, e = Level("g"), Level("e")
        atom = Atom(; levels = [g, e])
        beam_node = AtomTwin.BeamNode(PlanarBeam(578e-9, 1.0, [1.0, 0.0, 0.0], [0, 1, 0]))
        brf  = AtomTwin.BeamRabiFrequency(beam_node, atom, g, e, [0.0, 0.0, 1.0], 1e-29)
        node = AtomTwin.CouplingNode(brf, atom, g => e)
        deps = AtomTwin.node_dependencies(node)
        @test length(deps) == 1
        @test deps[1] === beam_node
    end
end

@testset "_topological_sort: insertion order, dependencies first, cycles rejected" begin
    let
        g, e = Level("g"), Level("e")
        atom = Atom(; levels = [g, e])
        n1 = AtomTwin.CouplingNode(1.0, atom, g => e)
        n2 = AtomTwin.DetuningNode(1.0, atom, e)
        n3 = AtomTwin.DecayNode(1.0, atom, g => e)
        nodes  = AtomTwin.AbstractNode[n1, n2, n3]
        sorted = AtomTwin._topological_sort(nodes)
        @test sorted == nodes
    end

    let
        g, e = Level("g"), Level("e")
        atom = Atom(; levels = [g, e])
        beam_node = AtomTwin.BeamNode(PlanarBeam(578e-9, 1.0, [1.0, 0.0, 0.0], [0, 1, 0]))
        brf  = AtomTwin.BeamRabiFrequency(beam_node, atom, g, e, [0.0, 0.0, 1.0], 1e-29)
        coupling_node = AtomTwin.CouplingNode(brf, atom, g => e)
        # Reversed: CouplingNode first, BeamNode second
        nodes  = AtomTwin.AbstractNode[coupling_node, beam_node]
        sorted = AtomTwin._topological_sort(nodes)
        @test sorted[1] === beam_node
        @test sorted[2] === coupling_node
    end

    let
        a = _CyclicTestNode()
        b = _CyclicTestNode()
        a.peer = b
        b.peer = a
        @test_throws ErrorException AtomTwin._topological_sort(AtomTwin.AbstractNode[a, b])
    end
end

@testset "recompile! rebinds Parameters when auto light shifts are present" begin
    # `job.fields` holds the automatic light shifts first, then one field per node;
    # `recompile!` must rebind each node's Parameter to its own field. Checked through
    # behaviour: detuning off resonance switches the scattering off.
    g = HyperfineManifold(0//1, 0; label = "1S0", term = l"1S0")
    e = HyperfineManifold(1//1, 1; label = "3P1", term = l"3P1", g_F = 1.493)
    δ = Parameter(:δ_img, 2.595)

    yb  = Ytterbium174Atom(; levels = [g..., e...],
                           v_init = maxwellboltzmann(T = 5e-6))
    tw  = GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 2e-3, pol = [1.0, 0.0, 0.0])
    sys = System(yb, tw)
    add_quantization_axis!(sys, [1.0, 0.0, 0.0])
    add_zeeman_detunings!(sys, yb, e; B = 7.18e-5, delta = -2π * 1e6 * δ)
    cp = add_coupling!(sys, yb, g => e; Ω_π = 0.0, Ω_p = 2π*90e3, Ω_m = 2π*90e3,
                       active = false)
    pd = PhotoDetectorSpec(name = "clicks")
    add_detector!(sys, pd)
    add_decay!(sys, yb, e => g, 2π * 182e3; clicks = pd, λ = 556e-9)

    seq = Sequence(2e-7; downsample = 200)
    @sequence seq begin
        Pulse(cp, 5e-4)
    end
    st  = AtomTwin._tovector(g[0])
    job = compile(sys, seq; initial_state = st, shots = 32, δ_img = 2.595)

    # The trap contributes light-shift fields that have no node behind them.
    @test job.n_auto_fields > 0

    photons(o) = mean(sum(o.detectors["clicks"], dims = 1))

    # On resonance with the trap-shifted σ⁺ line (m_F = +1), from the light-shift model.
    I0 = 2 * 2e-3 / (π * (1e-6)^2) * 1e-4                                  # W/cm²
    LS = (light_shift_coeff_Hz_per_Wcm2(AtomTwin.YB174_POLARIZABILITY_3P1, 767.0;
              F = 1//1, mF = 1//1, I = 0//1, ε_z = 1.0) -
          light_shift_coeff_Hz_per_Wcm2(AtomTwin.YB174_POLARIZABILITY_1S0, 767.0)) * I0
    Z  = 1.493 * AtomTwin.BOHR_MAGNETON_RAD_S_TESLA * 7.18e-5 / 2π
    δ_on = -(Z + LS) / 1e6  

    on  = photons(play(job, sys; shots = 32, initial_state = st, δ_img = δ_on))
    off = photons(play(job, sys; shots = 32, initial_state = st, δ_img = δ_on - 2.0))

    @test on > 20                # on resonance it really is scattering
    @test off < 0.2 * on         # …and the Parameter change actually took effect
end

# A compiled job is a snapshot with its own copy of the system: compiling or
# playing one never changes the user's System or another job.
@testset "a job owns a private copy of the system" begin
    g, e = Level("g"), Level("e")
    a = Atom(; levels = [g, e])
    s = System(a)
    Ω = Parameter(:Ω, 2π * 1e6)
    c = add_coupling!(s, a, g => e, Ω; active = false)
    add_detector!(s, PopulationDetectorSpec(a, e; name = "P"))
    q = Sequence(1e-9)
    @sequence q begin
        Pulse(c, 0.25e-6)                       # π/2 at 1 MHz, π at 2 MHz
    end
    H0 = gethamiltonian(s)
    job1 = compile(s, q; initial_state = [g], Ω = 2π * 1e6)
    job2 = compile(s, q; initial_state = [g], Ω = 2π * 2e6)
    @test job1.fields[1] !== job2.fields[1] && job1.fields[1] !== s.nodes[1]._field
    @test job1.atoms[1] !== a.inner
    @test play(job1, s).detectors["P"][end] ≈ 0.5 atol = 1e-3   # not job2's 2 MHz
    @test play(job2, s).detectors["P"][end] ≈ 1.0 atol = 1e-3
    # a sampled run leaves nothing behind in the user's system
    play(s, q; initial_state = [g], shots = 3, Ω = Parameter(:Ω, 2π * 1e6; std = 2π * 0.3e6),
         rng = MersenneTwister(1))
    @test s.nodes[1]._field.rate == 2π * 1e6
    @test gethamiltonian(s) == H0
    # `copy(sys)` is an independent system
    s2 = copy(s)
    @test s2.nodes[1]._field !== s.nodes[1]._field && gethamiltonian(s2) == H0
end

@testset "an instruction drives the job's copy of every field it names" begin
    # `Detuning` is an immutable struct: a handle map built from `deepcopy` alone missed
    # it, so a `Pulse` on an inactive `add_detuning!` term switched the user's object and
    # the job's stayed off. Ramsey: Rx(π/2)·Z(π)·Rx(π/2) returns |g⟩; without Z, |e⟩.
    Ω = Δ = 2π * 1e6
    function ramsey(kind)
        g, e = Level("g"), Level("e")
        atom = Atom(; levels = [g, e])
        sys  = System(atom)
        sq   = add_coupling!(sys, atom, g => e, Ω; active = false)
        z    = kind == :none ? nothing : add_detuning!(sys, atom, e, Δ; active = false)
        add_detector!(sys, PopulationDetectorSpec(atom, e; name = "Pe"))
        seq = Sequence(1e-9)
        push!(seq, Pulse(sq, π / 2 / Ω))
        if kind == :pulse
            push!(seq, Pulse(z, π / Δ))
        elseif kind == :shaped
            push!(seq, Pulse(z, π / Δ; amplitudes = ones(20)))
        elseif kind == :onoff
            push!(seq, On(z)); push!(seq, Wait(π / Δ)); push!(seq, Off(z))
        else
            push!(seq, Wait(π / Δ))
        end
        push!(seq, Pulse(sq, π / 2 / Ω))
        play(sys, seq; initial_state = g).detectors["Pe"][end]
    end
    @test ramsey(:none) ≈ 1 atol = 1e-6
    for kind in (:pulse, :shaped, :onoff)
        @test ramsey(kind) ≈ 0 atol = 1e-6
    end
end

@testset "a replayed job applies its keywords and draws afresh from shot 1" begin
    g, e = Level("g"), Level("e")
    a = Atom(; levels = [g, e])
    s = System(a)
    c = add_coupling!(s, a, g => e, Parameter(:Ω, 2π * 1e6); active = false)
    add_detector!(s, PopulationDetectorSpec(a, e; name = "P"))
    q = Sequence(1e-9)
    @sequence q begin
        Pulse(c, 0.25e-6)
    end
    job = compile(s, q; initial_state = [g])
    one  = play(job, s; Ω = 2π * 2e6)
    @test one.detectors["P"][end] ≈ 1.0 atol = 1e-3           # shots = 1 saw the override
    two  = play(job, s; Ω = 2π * 2e6, shots = 2)
    @test all(two.detectors["P"][end, :] .≈ 1.0)
    kept = play(job, s)                                        # the job remembers the last value
    @test kept.detectors["P"][end] ≈ 1.0 atol = 1e-3
    AtomTwin.recompile!(job, s; Ω = 2π * 1e6)                  # `recompile!` sets it too
    back = play(job, s)
    @test back.detectors["P"][end] ≈ 0.5 atol = 1e-3
    # a sampled Parameter is redrawn on every replay, shot 1 included
    Ωs = Parameter(:Ω, 2π * 1e6; std = 2π * 0.2e6)
    p = [play(job, s; Ω = Ωs, rng = MersenneTwister(k)).detectors["P"][end] for k in 1:4]
    @test length(unique(p)) == 4
    # single-shot outputs are the caller's, not the job's buffers
    o1 = play(job, s)
    v1 = copy(o1.detectors["P"])
    play(job, s; Ω = 2π * 2e6)
    @test o1.detectors["P"] == v1
    # the solver is the one the job was compiled with
    @test_throws ArgumentError play(job, s; density_matrix = true)
end

@testset "a keyword that names no Parameter is an error" begin
    g, e = Level("g"), Level("e")
    a = Atom(; levels = [g, e])
    s = System(a)
    c = add_coupling!(s, a, g => e, Parameter(:Omega, 2π * 1e6); active = false)
    q = Sequence(1e-9)
    @sequence q begin
        Pulse(c, 0.25e-6)
    end
    @test_throws ArgumentError play(s, q; initial_state = [g], Omeg = 2π * 2e6)
    @test_throws ArgumentError compile(s, q; initial_state = [g], delta = 1.0)
    job = compile(s, q; initial_state = [g])
    @test_throws ArgumentError play(job, s; seed = 1)
    # the keywords of `play` and a Parameter of the sequence pass
    tw = TweezerArray(λ = 759e-9, w0 = 1e-6, P_total = 1e-3, row_freqs = [0.0], col_freqs = [0.0])
    s2 = System(a, tw)
    q2 = Sequence(1e-8)
    @sequence q2 begin
        MoveCol(tw, [1], Parameter(:dx, 1e-6), 2e-7)
    end
    @test length(play(s2, q2; dx = 2e-6, shots = 2, savefinalstate = false,
                      parallel_thresh = 8).times) == 20
    @test_throws ArgumentError compile(s, Sequence(1e-9); initial_state = [g])  # no instructions
end

@testset "a quantization axis may be sampled or parametric per component" begin
    g = HyperfineManifold(0//1, 0; label = "1S0", term = l"1S0")
    e = HyperfineManifold(1//1, 1; label = "3P1", term = l"3P1", g_F = 1.5)
    yb = Ytterbium174Atom(; levels = [g..., e...])
    tw = GaussianBeam(λ = 767e-9, w0 = 1e-6, P = 2e-3, pol = [0.0, 0.0, 1.0])
    seq = Sequence(1e-8)
    @sequence seq begin
        Wait(1e-7)
    end
    s = System(yb, tw)
    add_quantization_axis!(s, Parameter(:B, [0.0, 0.0, 1.0]; std = 0.05))
    add_zeeman_detunings!(s, yb, e; B = 1e-4)
    out = play(s, seq; initial_state = [g[0]], shots = 3, rng = MersenneTwister(2))
    @test length(out.times) == 10                               # three shots ran
    s2 = System(yb, tw)
    θ = Parameter(:θ, 0.0)
    add_quantization_axis!(s2, [θ, 0.0, 1.0])
    add_zeeman_detunings!(s2, yb, e; B = 1e-4)
    job = compile(s2, seq; initial_state = [g[0]], θ = 1.0)
    @test getquantizationaxis(job.system) ≈ [1, 0, 1] / √2
    @test getquantizationaxis(s2) == [0.0, 0.0, 1.0]           # the system keeps its default
end

@testset "beam-derived couplings follow the system's quantization axis" begin
    S0 = FineManifold(0; label = "1S0", term = l"1S0")
    P1 = FineManifold(1; label = "3P1", term = l"3P1")
    yb = Ytterbium174Atom(; levels = [S0..., P1...], x_init = zeros(3))
    beam = GeneralGaussianBeam(556e-9, 20e-6, 20e-6, 1e-6, [0, 0, 1], [1, 0, 0])   # k ∥ z, pol x̂
    plane = PlanarBeam(556e-9, 1.0, [0, 0, 1.0], [1.0, 0, 0])
    seq = Sequence(1e-8)
    @sequence seq begin
        Wait(1e-8)
    end
    idx(a, l) = a.level_indices[l]
    function drives(sys)                           # |Ω| on each m′ from the compiled fields
        job = compile(sys, seq; initial_state = [S0[0]])
        Dict(f.transition[2] - idx(yb, S0[0]) => abs(hasproperty(f, :rate) ? f.rate : f.Ω0) for f in job.fields)
    end
    for b in (beam, plane)
        # no axis: ẑ, an x̂-polarized beam is σ⁺ + σ⁻
        s = System(yb)
        add_coupling!(s, yb, S0 => P1; beam = b)
        d = drives(s)
        @test d[1] ≈ d[3] && d[1] > 0
        @test abs(d[2]) < 1e-6 * d[1]
        # the axis set after the coupling, along the polarization: pure π
        s = System(yb)
        add_coupling!(s, yb, S0 => P1; beam = b)
        add_quantization_axis!(s, [1.0, 0.0, 0.0])
        dπ = drives(s)
        @test dπ[2] ≈ d[1] * √2
        @test abs(dπ[1]) < 1e-6 * dπ[2]
        @test abs(gethamiltonian(s)[idx(yb, S0[0]), idx(yb, P1[0])]) ≈ dπ[2] / 2
        # an explicit `q_axis` overrides the system's
        s = System(yb)
        add_quantization_axis!(s, [1.0, 0.0, 0.0])
        add_coupling!(s, yb, S0 => P1; beam = b, q_axis = [0, 0, 1.0])
        dz = drives(s)                                  # the π pair is dropped by `tol`, or zero
        @test !haskey(dz, 2) || abs(dz[2]) < 1e-6 * d[1]
        @test dz[1] ≈ d[1]
    end
end

@testset "product states build in linear time; initial_state forms" begin
    g, e = Level("g"), Level("e")
    N = 12
    atoms = [Atom(; levels = [g, e]) for _ in 1:N]
    s = System(atoms)
    st = fill(g + e, N)
    getqstate(s, st)
    t = @elapsed ψ = getqstate(s, st)
    @test length(ψ) == 2^N && all(abs2.(ψ) .≈ 1 / 2^N) && t < 0.5
    @test (@allocated getqstate(s, st)) < 3 * 2^N * 16             # the state and little else
    @test getqstate(s, [g]) == getqstate(s, fill(g, N))          # one level for every atom
    @test_throws ArgumentError getqstate(s, [g, e])
    s2 = System([Atom(; levels = [g, e]), Atom(; levels = [g, e])])
    ψ = [0, 1, 0, 0] / 1.0
    @test getqstate(s2, ψ) == ComplexF64.(ψ)
    @test getqstate(s2, ψ * ψ'; density_matrix = true) == ComplexF64.(ψ * ψ')
    @test_throws ArgumentError getqstate(s2, ψ * ψ')
    @test_throws ArgumentError getqstate(s2, [1.0, 0.0])
end

@testset "sampled rates leave serial and threaded runs bit-identical" begin
    g, e = Level("g"), Level("e")
    a = Atom(; levels = [g, e])
    s = System(a)
    c = add_coupling!(s, a, g => e, Parameter(:Ω, 2π * 1e6; std = 2π * 0.2e6); active = false)
    add_decay!(s, a, e => g, 2π * 1e5)
    add_detector!(s, PopulationDetectorSpec(a, e; name = "P"))
    q = Sequence(1e-9)
    @sequence q begin
        Pulse(c, 0.5e-6)
    end
    serial   = play(s, q; initial_state = [g], shots = 16, rng = MersenneTwister(3), parallel_thresh = 100)
    threaded = play(s, q; initial_state = [g], shots = 16, rng = MersenneTwister(3), parallel_thresh = 2)
    @test serial.detectors["P"] == threaded.detectors["P"]
    @test length(unique(serial.detectors["P"][end, :])) == 16
end
