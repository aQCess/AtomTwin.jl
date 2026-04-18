# Benchmark: N-atom Rydberg blockade — AtomTwin vs QuantumOptics.jl
#
# N two-level atoms (|g⟩,|r⟩), all-to-all blockade V≫Ω, dephasing γ on |r⟩.
# Reference: QuantumOptics adaptive DP5 (default tolerances).
# Accuracy: max|AT − QO| over the trajectory (AT uses first-order dissipation).
#
# Run: julia --project=benchmark --threads=auto benchmark/rydberg_benchmark.jl

using AtomTwin, QuantumOptics, BenchmarkTools, StatsBase, Printf

const Ω       = 2π * 1.0e6          # Rabi frequency (rad/s)
const V       = 2π * 100.0e6        # blockade interaction (rad/s); V/Ω = 100
const γ       = 2π * 250.0e3        # dephasing rate on |r⟩ (rad/s)
const T       = 1 / (Ω / 2π)        # 1 bare Rabi period (s)
const SHOTS   = 100
const SAMPLES = 3
dt_n(n) = 1 / (25 * sqrt(n) * V / 2π)

# ── Toggles ───────────────────────────────────────────────────────────────────

const BENCH_FULL      = false    # full Hilbert space (2^N), AT vs QO, N=2:8
const BENCH_BLOCKADED = true    # blockaded subspace (N+1), AT only,  N=2:24
const BENCH_MCWF      = false   # MCWF trajectories (slow)

# ── Builders ──────────────────────────────────────────────────────────────────

function build_atomtwin(n; blockaded=false, dephasing=false)
    g, r  = Level("g"), Level("r")
    atoms = [Atom(; levels=[g, r]) for _ in 1:n]
    sys   = blockaded ? System(atoms; maxoccupations=[(r, 1)]) : System(atoms)
    coups = [add_coupling!(sys, a, g => r, Ω; active=false) for a in atoms]
    for i in 1:n, j in (i+1):n
        add_interaction!(sys, (atoms[i], atoms[j]), (r,r) => (r,r), V)
    end
    if dephasing
        for a in atoms; add_dephasing!(sys, a, r, γ); end
    end
    for (i, a) in enumerate(atoms)
        add_detector!(sys, PopulationDetectorSpec(a, r; name="P_r$i"))
    end
    seq = Sequence(dt_n(n))
    @sequence seq begin Pulse(coups, T) end
    return sys, seq, fill(g, n)
end

function build_qo(n)
    b1  = NLevelBasis(2)
    σ₊  = transition(b1, 2, 1)
    σ₋  = dagger(σ₊)
    n_r = transition(b1, 2, 2)
    I1  = identityoperator(b1)
    embed(op, i) = tensor([j == i ? op : I1 for j in 1:n]...)
    H       = (Ω/2) * sum(embed(σ₊, i) + embed(σ₋, i) for i in 1:n)
    H      += V * sum(embed(n_r, i) * embed(n_r, j) for i in 1:n for j in (i+1):n)
    J       = [sqrt(γ) * embed(n_r, i) for i in 1:n]
    n_r_tot = sum(embed(n_r, i) for i in 1:n)
    ψ0      = tensor([nlevelstate(b1, 1) for _ in 1:n]...)
    return H, J, QuantumOptics.dm(ψ0), ψ0, n_r_tot
end

P_r_sum(out, n)   = sum(out.detectors["P_r$i"] for i in 1:n)
qo_ex(op, states) = real.([expect(op, s) for s in states[2:end]])
maxerr(a, b)      = maximum(abs.(a .- b))

# ── Header ────────────────────────────────────────────────────────────────────

println("=" ^ 93)
@printf("Rydberg blockade: Ω/2π=%.0f MHz  V/Ω=%d  γ/2π=%.0f kHz  T=%.2f μs  SHOTS=%d  threads=%d\n",
        Ω/2π/1e6, round(Int, V/Ω), γ/2π/1e3, T*1e6, SHOTS, Threads.nthreads())
println("=" ^ 93)

# ── Full Hilbert space: AT vs QO ──────────────────────────────────────────────

if BENCH_FULL
    println("\n── Full Hilbert space (d=2^N): AtomTwin vs QuantumOptics, N=2:7 ─────────────")
    println("Reference: QuantumOptics adaptive DP5 (default tolerances; same run used for timing)")
    println("Accuracy:  max|AT − QO|")
    @printf("\n  %2s %5s │ %9s %8s %9s │ %9s %8s %9s │ %9s %9s\n",
            "N", "d",
            "AT SE(ms)", "max|err|", "QO SE(ms)",
            "AT ME(ms)", "max|err|", "QO ME(ms)",
            "AT MC(ms)", "QO MC(ms)")
    println("  " * "-"^93)

    for n in 2:7
        sys_u, seq_u, init = build_atomtwin(n)
        sys_d, seq_d, _    = build_atomtwin(n; dephasing=true)
        job_se = compile(sys_u, seq_u; initial_state=init)
        job_me = compile(sys_d, seq_d; density_matrix=true,  initial_state=init)
        job_mc = compile(sys_d, seq_d; density_matrix=false, initial_state=init)

        H, J, ρ0, ψ0, n_r_tot = build_qo(n)

        out_se = play(job_se, sys_u)
        tspan  = vcat(0.0, out_se.times)
        out_me = play(job_me, sys_d)

        _, ψts = timeevolution.schroedinger(tspan, ψ0, H)
        _, ρts = timeevolution.master_h(tspan, ρ0, H, J)
        ref_se = qo_ex(n_r_tot, ψts)
        ref_me = qo_ex(n_r_tot, ρts)

        err_se = maxerr(P_r_sum(out_se, n), ref_se)
        err_me = maxerr(P_r_sum(out_me, n), ref_me)

        recompile!(job_se, sys_u); recompile!(job_me, sys_d); recompile!(job_mc, sys_d)

        t_at_se = minimum(@benchmark(begin recompile!($job_se, $sys_u); play($job_se, $sys_u) end,
                           samples=SAMPLES, evals=1).times) / 1e6
        t_qo_se = minimum(@benchmark(timeevolution.schroedinger($tspan, $ψ0, $H),
                           samples=SAMPLES, evals=1).times) / 1e6
        @printf("  %2d %5d │ %9.1f %8.1e %9.1f │", n, 2^n, t_at_se, err_se, t_qo_se); flush(stdout)

        t_at_me = minimum(@benchmark(begin recompile!($job_me, $sys_d); play($job_me, $sys_d) end,
                           samples=SAMPLES, evals=1).times) / 1e6
        t_qo_me = minimum(@benchmark(timeevolution.master_h($tspan, $ρ0, $H, $J),
                           samples=SAMPLES, evals=1).times) / 1e6
        @printf(" %9.1f %8.1e %9.1f │", t_at_me, err_me, t_qo_me); flush(stdout)

        t_at_mc = minimum(@benchmark(begin recompile!($job_mc, $sys_d); play($job_mc, $sys_d; shots=SHOTS) end,
                           samples=SAMPLES, evals=1).times) / 1e6
        t_qo_mc = minimum(@benchmark(begin
                               _acc = [zeros(length($tspan)) for _ in 1:Threads.nthreads()]
                               Threads.@threads :dynamic for _ in 1:$SHOTS
                                   _, ψts = timeevolution.mcwf($tspan, $ψ0, $H, $J;
                                       display_beforeevent=false, display_afterevent=false,
                                       seed=rand(UInt))
                                   _acc[Threads.threadid()] .+= real.(expect($n_r_tot, ψts))
                               end
                           end, samples=SAMPLES, evals=1).times) / 1e6
        @printf(" %9.1f %9.1f\n", t_at_mc, t_qo_mc)
    end
end

# ── Blockaded subspace: AT only ───────────────────────────────────────────────

if BENCH_BLOCKADED
    println("\n── Blockaded subspace (d=N+1): AtomTwin only, N=2:24 ────────────────────────")
    @printf("\n  %2s %5s │ %9s │ %9s │ %9s\n",
            "N", "d", "AT SE(ms)", "AT ME(ms)", "AT MC(ms)")
    println("  " * "-"^52)

    for n in 2:24
        sys_u, seq_u, init = build_atomtwin(n; blockaded=true)
        sys_d, seq_d, _    = build_atomtwin(n; blockaded=true, dephasing=true)
        job_se = compile(sys_u, seq_u; initial_state=init)
        job_me = compile(sys_d, seq_d; density_matrix=true,  initial_state=init)
        job_mc = compile(sys_d, seq_d; density_matrix=false, initial_state=init)

        t_at_se = minimum(@benchmark(begin recompile!($job_se, $sys_u); play($job_se, $sys_u) end,
                           samples=SAMPLES, evals=1).times) / 1e6
        @printf("  %2d %5d │ %9.1f │", n, n+1, t_at_se); flush(stdout)

        t_at_me = minimum(@benchmark(begin recompile!($job_me, $sys_d); play($job_me, $sys_d) end,
                           samples=SAMPLES, evals=1).times) / 1e6
        @printf(" %9.1f │", t_at_me); flush(stdout)

        t_at_mc = minimum(@benchmark(begin recompile!($job_mc, $sys_d); play($job_mc, $sys_d; shots=SHOTS) end,
                           samples=SAMPLES, evals=1).times) / 1e6
        @printf(" %9.1f\n", t_at_mc)
    end
end

println("\n" * "=" ^ 93)
