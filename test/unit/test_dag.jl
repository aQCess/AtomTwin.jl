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
