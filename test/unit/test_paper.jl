# Paper listing reproducibility tests
# ─────────────────────────────────────────────────────────────────────────────
# Each @testset corresponds to one code listing in the paper.  Every API call
# mirrors the listing exactly; simulation parameters are scaled down for speed
# (fewer shots, shorter durations) where the listing itself does not constrain
# the value.

using AtomTwin.Dynamiq.Units    # µm, nm, MHz, µK, mW, G, µB, hbar, e, a0 …

# ═════════════════════════════════════════════════════════════════════════════
# Section 3 (Implementation) listings
# ═════════════════════════════════════════════════════════════════════════════

# ── Listing 1: minimal system definition ─────────────────────────────────────
@testset "impl-listing1: HyperfineLevel, Ytterbium171Atom, GaussianBeam keyword ctor" begin
    g = HyperfineLevel(1//2, 0, -1//2; label = "1S0, m_F=-1/2")
    e = HyperfineLevel(1//2, 0, +1//2; label = "3P0, m_F=1/2")

    atom    = Ytterbium171Atom(; levels = [g, e])
    tweezer = GaussianBeam(λ = 759e-9, w0 = 0.8e-6, P = 0.5e-3)
    system  = System(atom, tweezer)

    coupling = add_coupling!(system, atom, g => e, 2π * 2e3)
    decay    = add_decay!(system,   atom, e => g, 2π * 0.0076)

    # GaussianBeam passed at System construction goes into system.beams, not system.nodes
    @test !isempty(system.beams)
    node_types = map(typeof, system.nodes)
    @test any(t -> t <: AtomTwin.CouplingNode,  node_types)
    @test any(t -> t <: AtomTwin.DecayNode,     node_types)
end

# ── Listing 2: Parameter with shot-to-shot disorder ───────────────────────────
@testset "impl-listing2: Parameter with std, multi-shot play, keyword override" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    system = System(atom)

    Ω = Parameter(:Ω, 2π * 5e3; std = 2π * 0.1e3)

    coupling = add_coupling!(system, atom, g => e, Ω)
    add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))

    seq = Sequence(1e-8)
    @sequence seq begin
        Wait(2e-8)   # 2 steps — minimal
    end

    # each shot draws Ω from N(5 kHz, 0.1 kHz); use 2 shots for speed
    out = play(system, seq; initial_state = g, shots = 2)
    @test haskey(out.detectors, "P_e")

    # fix Ω at a specific sweep point via keyword override
    out2 = play(system, seq; initial_state = g, Ω = 2π * 6e3)
    @test haskey(out2.detectors, "P_e")
end

# ── Listing 3: push!-based Sequence construction ──────────────────────────────
@testset "impl-listing3: push!-based Sequence loop" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    c    = add_coupling!(sys, atom, g => e, 2π * 1e6; active = false)

    dt        = 1e-9
    couplings = [c, c]
    durations = [50e-9, 50e-9]
    gap       = 10e-9

    seq = Sequence(dt; downsample = 10)
    for (coupling, duration) in zip(couplings, durations)
        push!(seq, Pulse(coupling, duration))
        push!(seq, Wait(gap))
    end

    # each loop iteration adds one Pulse and one Wait
    @test length(seq.instructions) == 2 * length(couplings)
end

# ── Listing 4: @sequence macro with dynamical-decoupling loop ─────────────────
@testset "impl-listing4: @sequence with DD loop" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    Ω_val    = 2π * 1e6
    coupling = add_coupling!(sys, atom, g => e, Ω_val; active = false)

    N_pi = 2     # reduced from 8; same pattern
    tau  = 50e-9
    Ω    = Ω_val
    dt   = 1e-9

    seq = Sequence(dt)
    @sequence seq begin
        Pulse(coupling, π/2 / Ω)
        for k in 1:N_pi
            Wait(tau)
            Pulse(coupling, π / Ω)
        end
        Wait(tau)
        Pulse(coupling, π/2 / Ω)
    end

    # 1 (π/2) + N_pi*(Wait+Pulse) + 1 (Wait) + 1 (π/2) = 2*N_pi + 3
    @test length(seq.instructions) == 2 * N_pi + 3
end

# ── Listing 5: compile + recompile! + parametric sweep ───────────────────────
@testset "impl-listing5: compile / recompile! / play(job, sys) sweep" begin
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    system = System(atom)
    coupling = add_coupling!(system, atom, g => e, 2π * 1e6; active = false)
    add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))

    Δ = Parameter(:Δ, 0.0)
    add_detuning!(system, atom, e, Δ)

    dt   = 1e-9
    T_pi = π / (2π * 1e6)
    seq  = Sequence(dt)
    @sequence seq begin
        Pulse(coupling, T_pi)
    end

    job = compile(system, seq; initial_state = [g])

    Δ_range = [0.0, 2π * 0.5e6]    # resonant and off-resonant
    # Extract the scalar immediately: play(job,sys) returns job.detector_outputs by
    # reference, so storing the NamedTuple and inspecting it after a second play call
    # would see the overwritten (second-run) values in both entries.
    pe_vals = map(Δ_range) do Δ_val
        recompile!(job, system; Δ = Δ_val)
        out = play(job, system; shots = 1)
        @test haskey(out.detectors, "P_e")
        out.detectors["P_e"][end]   # extract value before next recompile! overwrites buffer
    end

    @test length(pe_vals) == 2
    # resonant π-pulse flips the qubit; off-resonant does not
    @test pe_vals[1] > pe_vals[2]
end

# ═════════════════════════════════════════════════════════════════════════════
# Appendix A (Getting started) — first simulation listing
# ═════════════════════════════════════════════════════════════════════════════

@testset "gs-listing: first simulation from Appendix A" begin
    # Exact code from the getting-started listing
    g, e   = Level(; label = "g"), Level(; label = "e")
    atom   = Atom(; levels = [g, e])
    system = System(atom)

    add_coupling!(system, atom, g => e, 2pi * 1e6)   # Ω/2π = 1 MHz
    add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))

    seq = Sequence(1e-9)   # fixed time step: 1 ns
    @sequence seq begin
        Wait(5e-6)         # evolve for 5 µs → 5 000 steps
    end

    out = play(system, seq; initial_state = g)

    @test haskey(out.detectors, "P_e")
    @test length(out.times) ≥ 5000
    @test all(x -> 0.0 - 1e-6 ≤ x ≤ 1.0 + 1e-6, out.detectors["P_e"])
end

# ═════════════════════════════════════════════════════════════════════════════
# Section 4 (Application example) listings
# ═════════════════════════════════════════════════════════════════════════════

# ── lst:model — HyperfineManifold, atoms, beam, rabi_frequencies ──────────────
@testset "app-listing-model: HyperfineManifold, maxwellboltzmann, GeneralGaussianBeam, rabi_frequencies" begin
    g_eff = 2.357
    d_eff = 0.001071 * e * a0

    met     = HyperfineManifold(1//2, 0; label = "3P0",     g_F = -0.00067875)
    rydberg = HyperfineManifold(1//2, 0; label = "54.28S1", g_F = g_eff)
    r_lvl, leak = rydberg[-1//2], rydberg[+1//2]

    atoms = [Ytterbium171Atom(; levels = [met..., rydberg...],
                                v_init = maxwellboltzmann(T = 3µK))
             for _ in 1:4]

    w_ryd   = 12µm
    P_ryd   = 20mW
    k_ryd   = [1.0, 0.0, 0.0]
    pol_ryd = [0.0, 1.0, 0.0]
    B_vec   = [4.88G, 0.0, 0.0]
    beam    = GeneralGaussianBeam(302nm, w_ryd, w_ryd, P_ryd, k_ryd, pol_ryd)

    Ω_π, Ω_p, Ω_m = rabi_frequencies(beam; q_axis = B_vec, d_red = d_eff)

    @test length(atoms) == 4
    @test r_lvl isa HyperfineLevel
    @test leak  isa HyperfineLevel
    @test Ω_π   isa Number
    @test Ω_p   isa Number
    @test Ω_m   isa Number
end

# ── lst:system — multi-atom system assembly ───────────────────────────────────
@testset "app-listing-system: zeeman detunings, vdW interaction, manifold coupling, decay" begin
    g_eff = 2.357
    d_eff = 0.001071 * e * a0
    C6    = 2π * 34 * (GHz * µm^6)
    B_mag = 4.88G

    met     = HyperfineManifold(1//2, 0; label = "3P0",     g_F = -0.00067875)
    rydberg = HyperfineManifold(1//2, 0; label = "54.28S1", g_F = g_eff)
    r_lvl   = rydberg[-1//2]
    Δ_ryd   = -0.5 * g_eff * µB * B_mag / hbar
    Γ_ryd   = 1 / (2π * 56µs)

    # two atoms: minimal blockade pair (paper uses four)
    atoms   = [Ytterbium171Atom(; levels = [met..., rydberg...],
                                  v_init = maxwellboltzmann(T = 0µK))
               for _ in 1:2]
    tweezer = GaussianBeam(λ = 759nm, w0 = 1µm, P = 10mW)

    w_ryd = 12µm; P_ryd = 20mW; k_ryd = [1.0,0.0,0.0]; pol_ryd = [0.0,1.0,0.0]
    B_vec = [B_mag, 0.0, 0.0]
    beam  = GeneralGaussianBeam(302nm, w_ryd, w_ryd, P_ryd, k_ryd, pol_ryd)
    Ω_π, Ω_p, Ω_m = rabi_frequencies(beam; q_axis = B_vec, d_red = d_eff)

    sys = System(atoms, [tweezer])
    for atom in atoms
        add_zeeman_detunings!(sys, atom, met,     B = B_mag)
        add_zeeman_detunings!(sys, atom, rydberg, B = B_mag, delta = Δ_ryd)
    end
    add_vdwinteraction!(sys, (atoms[1], atoms[2]),
                        (r_lvl, r_lvl) => (r_lvl, r_lvl), C6)
    ryd = vcat([add_coupling!(sys, atom, met => rydberg;
                              beam = beam, Ω_π = Ω_π, Ω_p = Ω_p, Ω_m = Ω_m,
                              active = false)
                for atom in atoms]...)
    for atom in atoms
        add_decay!(sys, atom, rydberg => met, Γ_ryd)
    end

    @test !isempty(sys.nodes)
    @test !isempty(ryd)
end

# ── lst:cz_gate — Pulse with shaped amplitude envelope ───────────────────────
@testset "app-listing-cz: Pulse with amplitudes and :piecewise_constant interpolation" begin
    # Key API: Pulse(r, T_gate; amplitudes=ryd_amplitudes, interp=:piecewise_constant)
    # Exercised on a minimal 2-level system to avoid long compile times.
    g, e = Level("g"), Level("e")
    atom = Atom(; levels = [g, e])
    sys  = System(atom)
    r_c  = add_coupling!(sys, atom, g => e, 2π * 2.5e6; active = false)
    add_detector!(sys, PopulationDetectorSpec(atom, e; name = "P_e"))

    n_seg    = 10
    T_pulse  = Float64(n_seg) * 1e-9       # 10 ns total
    ryd_amps = ComplexF64[cis(-k * 0.1) for k in 0:n_seg-1]

    seq = Sequence(1e-9; downsample = 10)
    @sequence seq begin
        Pulse(r_c, T_pulse; amplitudes = ryd_amps, interp = :piecewise_constant)
    end

    out = play(sys, seq; initial_state = g, shots = 1)
    @test haskey(out.detectors, "P_e")
end

# ── lst:helpers — gate helper function definitions ────────────────────────────
@testset "app-listing-helpers: RZ / H / X / CZ return AbstractInstruction vectors" begin
    g, e  = Level("g"), Level("e")
    atom  = Atom(; levels = [g, e])
    sys   = System(atom)
    sq_c  = add_coupling!(sys, atom, g => e, 2π * 1e6; active = false)
    det_c = add_detuning!( sys, atom, e,     2π * 1e6; active = false)

    Ω_sq   = 2π * 1e6
    T_π    = π / Ω_sq
    Δ_det  = 2π * 1e6
    θ_cz   = 2.1663
    T_gate = 7.612 / (2π * 2.501e6)
    n_seg  = 10
    amps   = ComplexF64[cis(-k * 0.1) for k in 0:n_seg-1]

    # Scalar-coupling stand-ins for the paper's indexed sq[t] / det[t]
    RZ(t, ph)  = [Pulse(det_c, mod(ph, 2π) / Δ_det)]
    H_fn(t)    = [RZ(t, π/2)..., Pulse(sq_c, T_π/2), RZ(t, π/2)...]
    X_fn(t)    = [Pulse(sq_c, T_π)]
    CZ_fn()    = [Pulse(sq_c, T_gate; amplitudes = amps, interp = :piecewise_constant),
                  RZ(nothing, θ_cz)...]

    @test all(i -> i isa AbstractInstruction, RZ(nothing, π/2))
    @test all(i -> i isa AbstractInstruction, H_fn(nothing))
    @test all(i -> i isa AbstractInstruction, X_fn(nothing))
    @test all(i -> i isa AbstractInstruction, CZ_fn())
end

# ── lst:seq — @sequence with MoveRow and gate-helper instructions ─────────────
@testset "app-listing-seq: @sequence with MoveRow builds successfully" begin
    g_eff = 2.357
    d_eff = 0.001071 * e * a0
    C6    = 2π * 34 * (GHz * µm^6)
    B_mag = 4.88G

    met     = HyperfineManifold(1//2, 0; label = "3P0",     g_F = -0.00067875)
    rydberg = HyperfineManifold(1//2, 0; label = "54.28S1", g_F = g_eff)
    r_lvl   = rydberg[-1//2]
    Δ_ryd   = -0.5 * g_eff * µB * B_mag / hbar
    Γ_ryd   = 1 / (2π * 56µs)

    dy      = 10µm
    dy_gate = 2µm
    xsep    = 10µm
    Ω_sq    = 2π * 1e6
    T_π     = π / Ω_sq
    Δ_det   = 2π * 1e6
    θ_cz    = 2.1663
    T_gate  = 7.612 / (2π * 2.501e6)
    T_move  = 1µs       # shortened from ~81 µs; duration only affects instruction metadata
    dt_base = 1e-9

    tweezer = TweezerArray(
        λ         = 759nm,
        w0        = 1µm,
        P_total   = 100mW,
        row_freqs = dy / µm * [-0.5, +0.5] * MHz,
        col_freqs = xsep / µm * [-0.5, +0.5] * MHz,
        dx        = 1µm / MHz,
        dy        = 1µm / MHz,
    )
    atoms = [Ytterbium171Atom(; levels = [met..., rydberg...],
                                x_init = copy(getposition(t)),
                                v_init = maxwellboltzmann(T = 0µK))
             for t in tweezer]

    w_ryd = 12µm; P_ryd = 20mW; k_ryd = [1.0,0.0,0.0]; pol_ryd = [0.0,1.0,0.0]
    B_vec = [B_mag, 0.0, 0.0]
    beam  = GeneralGaussianBeam(302nm, w_ryd, w_ryd, P_ryd, k_ryd, pol_ryd)
    Ω_π_v, Ω_p_v, Ω_m_v = rabi_frequencies(beam; q_axis = B_vec, d_red = d_eff)

    sys = System(atoms, [tweezer])
    for atom in atoms
        add_zeeman_detunings!(sys, atom, met,     B = B_mag)
        add_zeeman_detunings!(sys, atom, rydberg, B = B_mag, delta = Δ_ryd)
    end
    add_vdwinteraction!(sys, (atoms[1], atoms[3]),
                        (r_lvl, r_lvl) => (r_lvl, r_lvl), C6)
    add_vdwinteraction!(sys, (atoms[2], atoms[4]),
                        (r_lvl, r_lvl) => (r_lvl, r_lvl), C6)
    ryd = vcat([add_coupling!(sys, atom, met => rydberg;
                              beam = beam, Ω_π = Ω_π_v, Ω_p = Ω_p_v, Ω_m = Ω_m_v,
                              active = false)
                for atom in atoms]...)
    sq  = [add_coupling!(sys, atom, met[-1//2] => met[+1//2], Ω_sq; active = false)
           for atom in atoms]
    det = [add_detuning!( sys, atom, met[+1//2], Δ_det; active = false)
           for atom in atoms]
    for atom in atoms
        add_decay!(sys, atom, rydberg => met, Γ_ryd)
    end

    n_seg    = 10
    ryd_amps = ComplexF64[cis(-k * 0.1) for k in 0:n_seg-1]

    RZ(targets, ph) = [Pulse(det[targets], mod(ph, 2π) / Δ_det)]
    H_fn(targets)   = [RZ(targets, π/2)..., Pulse(sq[targets], T_π/2), RZ(targets, π/2)...]
    X_fn(targets)   = [Pulse(sq[targets], T_π)]
    CZ_fn()         = [Pulse(ryd, T_gate; amplitudes = ryd_amps, interp = :piecewise_constant),
                       RZ(1:4, θ_cz)...]

    seq = Sequence(dt_base; downsample = 1000)
    @sequence seq begin
        Wait(0.1µs)
        H_fn(1:4)
        MoveRow(tweezer, 1,  (dy - dy_gate) / µm * MHz, T_move; dt = 5dt_base)
        CZ_fn()
        X_fn(1:4)
        MoveRow(tweezer, 1, -(dy - dy_gate) / µm * MHz, T_move; dt = 5dt_base)
        X_fn(1:4)
        H_fn(3:4)
    end

    @test !isempty(seq.instructions)
    @test any(i -> i isa MoveRow, seq.instructions)
end
