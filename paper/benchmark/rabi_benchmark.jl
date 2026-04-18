# Benchmark: Rabi oscillations with decay
#
# Compare AtomTwin (Schrödinger, master equation, MCWF) against QuantumOptics.jl
# on an identical two-level system.
#
# System: |g⟩ ↔ |e⟩, resonant drive Ω, spontaneous decay Γ = Ω/200.
# Duration: T = 1000 / (Ω/2π), covering 1000 Rabi periods.
# Time step: dt = T / 100_000 (100 fixed steps per Rabi period).
#   QuantumOptics uses an adaptive integrator internally and outputs on the
#   same fixed grid — the accuracy difference is measured below.
#
# For each method: accuracy is measured first (single run, outside @benchmark),
# then timing.  Both use the same compiled job / operators so parameters are
# identical.  play() resets the quantum state at entry, so @benchmark samples
# always start from |g⟩.
#
# Run with:
#   julia --project=benchmark --threads=auto benchmark/rabi_benchmark.jl

using AtomTwin
using QuantumOptics
using BenchmarkTools
using StatsBase
using Printf

# ── Parameters ────────────────────────────────────────────────────────────────

Ω     = 2π * 1.0e6       # Rabi frequency (rad/s)
Γ     = 2π * 0.5e3       # Spontaneous decay rate |e⟩ → |g⟩ (rad/s)
T     = 1000 / (Ω/2π)    # Total time: 1000 Rabi periods (s)
dt    = T / 100_000      # 50 fixed steps per Rabi period
SHOTS = 100              # Monte Carlo trajectories
SAMPLES = 10

# ── Analytical reference ──────────────────────────────────────────────────────
#
# Master equation (resonant Lindblad with L = √Γ |g⟩⟨e|):
#   Exact closed-form via Bloch equations.
#   Bloch matrix A₂ = [[-Γ/2, Ω], [-Ω, -Γ]] has eigenvalues
#   λ = -3Γ/4 ± iΩ_R,  Ω_R = √(Ω² - Γ²/16).
#   Steady state: z_ss = -Γ²/(Γ²+2Ω²).
#   Transient: δz(t) = exp(A₂t) · δz₀  (closed-form via 2×2 matrix exp).
#   Setting γ=0 recovers the SE result P_e(t) = sin²(Ωt/2).

function analytical_me(t; γ=Γ)
    denom = γ^2 + 2Ω^2
    z_ss  = -γ^2 / denom
    δy₀   =  2Ω*γ / denom      # -y_ss
    δz₀   = -2Ω^2 / denom      # -1 - z_ss
    Ω_R   = sqrt(Ω^2 - γ^2/16)
    e  = exp(-3γ * t / 4)
    cs = cos(Ω_R * t)
    sn = sin(Ω_R * t) / Ω_R
    δz = e * (-Ω * sn * δy₀ + (cs - (γ/4) * sn) * δz₀)
    (1 + z_ss + δz) / 2
end

# Reference arrays for the last Rabi period.
# AtomTwin produces T/dt points on (dt…T]; QO produces T/dt+1 on [0…T].
# Both end at T, so end-indexed slices of length n_last cover the same times.
const n_last  = round(Int, (1/(Ω/2π)) / dt)   # = 100 steps per Rabi period
const t_last  = T .- (n_last-1:-1:0) .* dt    # last n_last AtomTwin time points
const P_se_ref = analytical_me.(t_last; γ=0.0)
const P_me_ref = analytical_me.(t_last)

maxerr(sim, ref) = maximum(abs.(sim[end-n_last+1:end] .- ref))

# ── Helpers ───────────────────────────────────────────────────────────────────

function report_time(label, trial)
    t_ms  = minimum(trial).time / 1e6
    alloc = minimum(trial).allocs
    mem   = minimum(trial).memory / 1024^2
    @printf("  %-46s  %7.1f ms   %6d alloc   %5.1f MiB\n", label, t_ms, alloc, mem)
end

report_acc(label, val) = @printf("  %-46s  max|err| = %.2e\n", label, val)

println("=" ^ 72)
println("Rabi benchmark — $(round(T*1e6, digits=1)) μs, $(round(dt*1e9,digits=2)) ns step, $SHOTS shots")
println("Ω/2π = $(Ω/2π/1e6) MHz,  Γ/2π = $(Γ/2π/1e3) kHz,  $(round(Int, T*Ω/2π)) Rabi periods")
println("=" ^ 72)

# ── AtomTwin ──────────────────────────────────────────────────────────────────

function build_atomtwin(; decay=false)
    g, e   = Level("g"), Level("e")
    atom   = Atom(; levels = [g, e])
    system = System(atom)
    drive  = add_coupling!(system, atom, g => e, Ω; active = false)
    decay && add_decay!(system, atom, e => g, Γ; active = true)
    add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))
    seq = Sequence(dt)
    @sequence seq begin Pulse(drive, T) end
    return system, seq, g
end

println("\n── AtomTwin ──────────────────────────────────────────────────────────")

let
    sys_u, seq_u, g = build_atomtwin(decay = false)
    sys,   seq,   _ = build_atomtwin(decay = true)

    job_se   = compile(sys_u, seq_u; initial_state = [g])
    job_me   = compile(sys,   seq;   density_matrix = true,  initial_state = [g])
    job_mcwf = compile(sys,   seq;   density_matrix = false, initial_state = [g])
    
    println("  Accuracy (max|err| last Rabi period, analytical reference):")
    report_acc("Schrödinger (unitary)",
        maxerr(play(job_se, sys_u).detectors["P_e"], P_se_ref))
    report_acc("master equation",
        maxerr(play(job_me, sys).detectors["P_e"], P_me_ref))

    P_mcwf = mean(play(job_mcwf, sys; shots = SHOTS).detectors["P_e"], dims = 2)
    report_acc("MCWF ($SHOTS shots avg)", maxerr(P_mcwf, P_me_ref))
    println()

    println("  Timing (minimum over samples):")
    report_time("Schrödinger (unitary)", @benchmark play($job_se,   $sys_u)            samples=SAMPLES evals=1)
    report_time("master equation",       @benchmark play($job_me,   $sys)              samples=SAMPLES evals=1)
    report_time("MCWF ($SHOTS shots, sequential)", @benchmark play($job_mcwf, $sys; shots=SHOTS, parallel_thresh=typemax(Int)) samples=SAMPLES evals=1)
    report_time("MCWF ($SHOTS shots, threaded)",   @benchmark play($job_mcwf, $sys; shots=SHOTS)                               samples=SAMPLES evals=1)
end

# ── QuantumOptics.jl ──────────────────────────────────────────────────────────

println("\n── QuantumOptics.jl ──────────────────────────────────────────────────")

let
    b     = NLevelBasis(2)
    ψ0    = nlevelstate(b, 1)
    ρ0    = dm(ψ0)
    H     = (Ω/2) * (transition(b, 2, 1) + transition(b, 1, 2))
    J     = [sqrt(Γ) * transition(b, 1, 2)]
    tspan = collect(range(0.0, T; step = dt))
    Pe(states) = real.([expect(transition(b, 2, 2), s) for s in states])

    println("  Accuracy (max|err| last Rabi period, analytical reference):")
    _, ψ_se = timeevolution.schroedinger(tspan, ψ0, H)
    report_acc("Schrödinger (unitary)", maxerr(Pe(ψ_se), P_se_ref))

    _, ρ_me = timeevolution.master_h(tspan, ρ0, H, J)
    report_acc("master equation", maxerr(Pe(ρ_me), P_me_ref))

    # Per-thread accumulators avoid data races. :static scheduler required so
    # threadid() is stable — :dynamic (Julia 1.11+ default) allows task migration.
    thread_acc = [zeros(length(tspan)) for _ in 1:Threads.nthreads()]
    Threads.@threads :static for _ in 1:SHOTS
        _, ψ_traj = timeevolution.mcwf(tspan, ψ0, H, J;
            display_beforeevent=false, display_afterevent=false, seed=rand(UInt))
        thread_acc[Threads.threadid()] .+= Pe(ψ_traj)
    end
    P_mcwf = reduce(.+, thread_acc)
    report_acc("MCWF ($SHOTS shots avg)", maxerr(P_mcwf ./ SHOTS, P_me_ref))
    println()

    println("  Timing (minimum over samples):")
    report_time("Schrödinger (unitary)", @benchmark timeevolution.schroedinger($tspan, $ψ0, $H) samples=SAMPLES evals=1)
    report_time("master equation",       @benchmark timeevolution.master_h($tspan, $ρ0, $H, $J) samples=SAMPLES evals=1)
    report_time("MCWF ($SHOTS shots, sequential)", @benchmark begin
        for _ in 1:SHOTS
            timeevolution.mcwf($tspan, $ψ0, $H, $J;
                display_beforeevent=false, display_afterevent=false, seed=rand(UInt))
        end
    end samples=SAMPLES evals=1)
    report_time("MCWF ($SHOTS shots, threaded, $(Threads.nthreads()) threads)", @benchmark begin
        _acc = [zeros(length($tspan)) for _ in 1:Threads.nthreads()]
        Threads.@threads :static for _ in 1:$SHOTS
            _, ψ_traj = timeevolution.mcwf($tspan, $ψ0, $H, $J;
                display_beforeevent=false, display_afterevent=false, seed=rand(UInt))
            _acc[Threads.threadid()] .+= $Pe(ψ_traj)
        end
        # Note: Pe allocates O(N_times) per trajectory (QO API returns full trajectories);
        # AtomTwin writes detector scalars incrementally, so its allocation count is lower.
    end samples=SAMPLES evals=1)
end

println("\nNote: QuTiP results are in rabi_benchmark_qutip.py")
println("=" ^ 72)
