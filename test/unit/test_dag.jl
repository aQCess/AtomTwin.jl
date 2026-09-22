# DAG topological sort tests

# Test-local node type for cycle detection test (must be at file scope in Julia)
mutable struct _CyclicTestNode <: AtomTwin.AbstractNode
    peer::Union{Nothing, _CyclicTestNode}
end
_CyclicTestNode() = _CyclicTestNode(nothing)
AtomTwin.node_dependencies(n::_CyclicTestNode) =
    n.peer === nothing ? AtomTwin.AbstractNode[] : AtomTwin.AbstractNode[n.peer]

@testset "node_dependencies: default returns empty" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    beam_node = AtomTwin.BeamNode(PlanarBeam(578e-9, 1.0, [1.0, 0.0, 0.0], [0, 1, 0]))
    decay_node = AtomTwin.DecayNode(1e6, atom, g => e)
    det_node   = AtomTwin.DetuningNode(1e6, atom, e)
    @test isempty(AtomTwin.node_dependencies(beam_node))
    @test isempty(AtomTwin.node_dependencies(decay_node))
    @test isempty(AtomTwin.node_dependencies(det_node))
end

@testset "node_dependencies: CouplingNode without BeamRabiFrequency returns empty" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    node = AtomTwin.CouplingNode(2π * 1e6, atom, g => e)
    @test isempty(AtomTwin.node_dependencies(node))
end

@testset "node_dependencies: CouplingNode with BeamRabiFrequency returns beam_node" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    beam_node = AtomTwin.BeamNode(PlanarBeam(578e-9, 1.0, [1.0, 0.0, 0.0], [0, 1, 0]))
    brf  = AtomTwin.BeamRabiFrequency(beam_node, atom, g, e, [0.0, 0.0, 1.0], 1e-29)
    node = AtomTwin.CouplingNode(brf, atom, g => e)
    deps = AtomTwin.node_dependencies(node)
    @test length(deps) == 1
    @test deps[1] === beam_node
end

@testset "_topological_sort: preserves insertion order with no dependencies" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    n1 = AtomTwin.CouplingNode(1.0, atom, g => e)
    n2 = AtomTwin.DetuningNode(1.0, atom, e)
    n3 = AtomTwin.DecayNode(1.0, atom, g => e)
    nodes  = AtomTwin.AbstractNode[n1, n2, n3]
    sorted = AtomTwin._topological_sort(nodes)
    @test sorted == nodes
end

@testset "_topological_sort: BeamNode pushed after dependent CouplingNode is reordered" begin
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

@testset "_topological_sort: error on cycle" begin
    a = _CyclicTestNode()
    b = _CyclicTestNode()
    a.peer = b
    b.peer = a
    @test_throws ErrorException AtomTwin._topological_sort(AtomTwin.AbstractNode[a, b])
end

@testset "recompile! rebinds Parameters when auto light shifts are present" begin
    # `compile` builds job.fields as [auto light shifts..., then one per DAG node],
    # but `recompile!` walked nodes while indexing job.fields from 1 — so every node
    # updated the field `n_auto_fields` earlier than its own (a DetuningNode writing
    # into a StarkShiftAC). Nothing errored; a Parameter simply stopped responding,
    # on any system with a trapping beam and polarizability data.
    #
    # Tested through behaviour rather than field internals: detuning far off
    # resonance must switch the scattering off, whether the job is reused or rebuilt.
    g = HyperfineManifold(0//1, 0; label = "1S0", term = l"1S0")
    e = HyperfineManifold(1//1, 1; label = "3P1", term = l"3P1", g_F = 1.493)
    δ = Parameter(:δ_img, -2.595)

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
    job = compile(sys, seq; initial_state = st, shots = 32, δ_img = -2.595)

    # The trap contributes light-shift fields that have no node behind them; that
    # offset is precisely what used to be missing.
    @test job.n_auto_fields > 0

    photons(o) = mean(sum(o.detectors["clicks"], dims = 1))
    on  = photons(play(job, sys; shots = 32, initial_state = st, δ_img = -2.595))
    off = photons(play(job, sys; shots = 32, initial_state = st, δ_img = -3.5))

    @test on > 20                # on resonance it really is scattering
    @test off < 0.2 * on         # …and the Parameter change actually took effect
end
