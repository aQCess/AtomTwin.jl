@testset "Parameter resolution via DAG" begin
    using Random
    rng = Xoshiro(42)

    p = Parameter(:Ω, 5.0)
    @test AtomTwin._resolve_node_default(p) ≈ 5.0
    @test AtomTwin._resolve_node_value(p, Dict{Symbol,Any}(), rng) ≈ 5.0

    # Override via param_values dict
    @test AtomTwin._resolve_node_value(p, Dict(:Ω => 7.0), rng) ≈ 7.0

    # Plain numbers pass through unchanged
    @test AtomTwin._resolve_node_default(3.14) ≈ 3.14
    @test AtomTwin._resolve_node_value(2.71, Dict{Symbol,Any}(), rng) ≈ 2.71

    # Multiplication: 2*p = 10
    expr = 2.0 * p
    @test AtomTwin._resolve_node_default(expr) ≈ 10.0
    @test AtomTwin._resolve_node_value(expr, Dict{Symbol,Any}(), rng) ≈ 10.0

    # Addition
    q = Parameter(:δ, 3.0)
    @test AtomTwin._resolve_node_default(p + q) ≈ 8.0

    # Subtraction
    @test AtomTwin._resolve_node_default(p - q) ≈ 2.0

    # Inverse
    @test AtomTwin._resolve_node_default(inv(p)) ≈ 0.2

    # Nested: 2*(p + q) = 16
    @test AtomTwin._resolve_node_default(2.0 * (p + q)) ≈ 16.0
end

@testset "Parameter with noise" begin
    using Random
    rng = Xoshiro(0)
    p_noisy = Parameter(:Ω, 0.0; std = 1.0)
    samples = [AtomTwin._resolve_node_value(p_noisy, Dict{Symbol,Any}(), rng) for _ in 1:1000]
    @test abs(mean(samples)) < 0.1    # mean ≈ 0
    @test 0.8 < std(samples) < 1.2   # std ≈ 1
end

@testset "every resolver evaluates the same expressions" begin
    using Random
    p, q = Parameter(:p, 2.0), Parameter(:q, 4.0)
    none = Dict{Symbol,Any}()
    for (ex, val) in ((-p, -2.0), (p - 1.0, 1.0), (1.0 - p, -1.0), (1.0 / p, 0.5),
                      (p / q, 0.5), (p / (2q), 0.25), (-(p + q), -6.0), (abs(p - q), 2.0))
        @test AtomTwin._resolve_node_default(ex) ≈ val
        @test AtomTwin._resolve_node_value(ex, none, Xoshiro(1)) ≈ val
        @test AtomTwin.resolve(ex, none) ≈ val
    end
end

@testset "resolve draws a noisy Parameter from the rng it is given" begin
    using Random
    p = Parameter(:p, 1.0; std = 0.1)
    d = Dict{Symbol,Any}()
    @test AtomTwin.resolve(p, d; rng = Xoshiro(7)) == AtomTwin.resolve(p, d; rng = Xoshiro(7))
    @test AtomTwin.resolve(p, d; rng = Xoshiro(7)) != AtomTwin.resolve(p, d; rng = Xoshiro(8))
end
