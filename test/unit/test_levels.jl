# Superposition algebra: scalar multiply/divide/negate must compose to a FLAT,
# correctly-weighted coefficient dict for arbitrary state expressions — never nest
# a Superposition as a dict key. Regression for `a*(ℓ1+ℓ2)` silently malforming
# the state (Superposition <: AbstractLevel, so the scalar-times-level fallback
# used to swallow the whole superposition).

@testset "Superposition algebra" begin
    g0, g1, g2 = Level("0"), Level("1"), Level("2")

    # keys are always bare levels, never a nested Superposition
    flat(s) = all(k -> !(k isa AtomTwin.Superposition), keys(s.coeffs))

    # scalar × (sum): the case that used to nest
    s = (1/√2) * (g0 + g1)
    @test flat(s)
    @test Set(keys(s.coeffs)) == Set([g0, g1])
    @test s.coeffs[g0] ≈ 1/√2
    @test s.coeffs[g1] ≈ 1/√2

    # division and the a*ℓ1 + b*ℓ2 form agree
    s2 = (g0 + g1) / √2
    @test flat(s2)
    @test s2.coeffs[g0] ≈ 1/√2
    @test s2.coeffs[g1] ≈ 1/√2

    s3 = (1/√2)*g0 + (1/√2)*g1
    @test s3.coeffs[g0] ≈ 1/√2
    @test s3.coeffs[g1] ≈ 1/√2

    # right-multiply commutes; unary minus; nested scaling
    @test (g0 * 2).coeffs[g0] ≈ 2
    @test (-g0).coeffs[g0] ≈ -1
    @test (2 * (g0 + g1)).coeffs[g0] ≈ 2
    @test ((g0 + g1) * 3).coeffs[g1] ≈ 3

    # a richer expression with cancellation collapses correctly
    s4 = 2*g0 - 0.5*g1 + 0.5*g1            # the g1 terms cancel
    @test flat(s4)
    @test s4.coeffs[g0] ≈ 2
    @test get(s4.coeffs, g1, 0.0) ≈ 0.0 atol=1e-12

    # three-level normalised state
    s5 = (g0 + g1 + g2) / √3
    @test flat(s5)
    @test all(abs(s5.coeffs[l]) ≈ 1/√3 for l in (g0, g1, g2))

    # getqstate must place the coefficients on the right basis indices and
    # normalise — i.e. the malformed-key path is gone end to end.
    atom = Atom(; levels = [g0, g1, g2])
    sys  = System(atom)
    ψ = getqstate(sys, (1/√2)*g0 + (1/√2)*g1)
    @test length(ψ) == 3
    @test abs(ψ[atom.level_indices[g0]]) ≈ 1/√2
    @test abs(ψ[atom.level_indices[g1]]) ≈ 1/√2
    @test abs(ψ[atom.level_indices[g2]]) ≈ 0 atol=1e-12
    @test ψ'ψ ≈ 1
end

# ======================================================================
# Term symbols
# ======================================================================

@testset "term symbols bind levels to species data" begin
    # `label` is display text — the shipped examples write "³P₀" — while model
    # dictionaries are keyed ASCII "3P0". Matching one against the other silently
    # missed and the level took α = 0. Terms are resolved at parse time instead.
    @test (l"3P1").J == 1//1
    @test (l"1S0").J == 0//1
    @test AtomTwin.Ytterbium171._3P1 === l"3P1"   # species-qualified: same object
    @test AtomTwin.Strontium88._1S0  === l"1S0"   # a term is the state, not the atom
    @test_throws Exception eval(:(@l_str "3Q9"))  # unknown term fails where written

    # A term propagates to every sublevel, and `label` stays cosmetic.
    e = HyperfineManifold(1//1, 1; label = "³P₁", term = l"3P1")
    @test e.term == "3P1"
    @test all(l -> l.term == "3P1" && l.label == "³P₁", e.levels)
    @test FineManifold(1//1; term = l"3P1").term == "3P1"

    # A term knows its own J, so HyperfineManifold(F, J) with the arguments
    # swapped is an error at the point of declaration.
    @test_throws ErrorException HyperfineManifold(1//1, 0; term = l"3P1")

    # Nothing pre-term changes: a bare Level's label doubles as its term, the
    # positional constructors still work, and a String is still accepted.
    @test AtomTwin._level_term(Level("1S0")) == "1S0"
    @test AtomTwin._level_term(Level("ground"; term = l"1S0")) == "1S0"
    @test copy(Level("g"; term = l"1S0")).term == "1S0"
    @test HyperfineLevel(1//1, 1//1, 0//1, 1.0, "x").term == ""
    @test FineLevel(1//1, 0//1, 1.0, "x").term == ""
    @test HyperfineManifold(1//1, 1; term = "3P1").term == "3P1"
end
