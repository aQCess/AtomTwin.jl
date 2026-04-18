# Collective Rabi oscillations — AtomTwin publication figure
#
# (a) P_r vs time (μs): full Hilbert space (solid) and blockaded subspace (dashed)
# (b) Fitted Ω_eff/(2π) vs N: demonstrates √N collective Rabi scaling
#
# Run with:
#   julia --project=benchmark benchmark/rydberg_rabi_figure.jl

using AtomTwin
using Plots
using LaTeXStrings
using Serialization

# ── Parameters ────────────────────────────────────────────────────────────────

const Ω          = 2π * 1.0e6    # single-atom Rabi frequency (rad/s)
const V          = 2π * 100.0e6  # blockade interaction (V/Ω = 100)
const γ          = 2π * 250.0e3  # dephasing rate on |r⟩ (rad/s); set 0.0 to disable
const T          = 1 / (Ω / 2π)  # 1 bare Rabi period (s)
const N_FULL     = 1:8           # full Hilbert space (d = 2^N)
const N_TRUNC    = 1:24          # blockaded subspace (d = N+1)
dt_n(n) = 1 / (25 * sqrt(n) * V / 2π)

# ── Builder ───────────────────────────────────────────────────────────────────

function build_collective(n; blockaded=false)
    g, r  = Level("g"), Level("r")
    atoms = [Atom(; levels=[g, r]) for _ in 1:n]
    sys   = blockaded ? System(atoms; maxoccupations=[(r, 1)]) : System(atoms)
    coups = [add_coupling!(sys, a, g => r, Ω; active=false) for a in atoms]
    for i in 1:n, j in (i+1):n
        add_interaction!(sys, (atoms[i], atoms[j]), (r, r) => (r, r), V)
    end
    if γ > 0
        for a in atoms; add_dephasing!(sys, a, r, γ); end
    end
    for (i, a) in enumerate(atoms)
        add_detector!(sys, PopulationDetectorSpec(a, r; name="P_r$i"))
    end
    dt = blockaded ? 2π / (sqrt(n) * Ω) / 1000 : dt_n(n)
    seq = Sequence(dt)
    @sequence seq begin Pulse(coups, T) end
    return sys, seq, fill(g, n)
end

P_r_sum(out, n) = sum(out.detectors["P_r$i"] for i in 1:n)

# ── Frequency estimate: first peak of sin²(Ω_eff·t/2) at t_peak = π/Ω_eff ──

estimate_omega(t, y) = π / t[argmax(y)]

# ── Simulations (cached) ──────────────────────────────────────────────────────

const DATAFILE = joinpath(@__DIR__, "rydberg_rabi_data.jls")

if isfile(DATAFILE)
    println("Loading cached data from $DATAFILE")
    results_full, results_trunc = deserialize(DATAFILE)
else
    println("Running simulations...")
    dm = γ > 0
    results_full  = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    results_trunc = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    for n in N_FULL
        sys, seq, init = build_collective(n; blockaded=false)
        out = play(compile(sys, seq; initial_state=init, density_matrix=dm), sys)
        Pr  = P_r_sum(out, n)
        results_full[n] = (out.times .* (Ω / 2π), Pr)
        println("  full  N=$n  (d=$(2^n))  steps=$(length(Pr))  peak≈$(round(maximum(Pr), digits=3))")
    end
    for n in N_TRUNC
        sys, seq, init = build_collective(n; blockaded=true)
        out = play(compile(sys, seq; initial_state=init, density_matrix=dm), sys)
        Pr  = P_r_sum(out, n)
        results_trunc[n] = (out.times .* (Ω / 2π), Pr)
        println("  trunc N=$n  (d=$(n+1))  steps=$(length(Pr))  peak≈$(round(maximum(Pr), digits=3))")
    end
    serialize(DATAFILE, (results_full, results_trunc))
    println("Saved data to $DATAFILE")
end

# ── Frequency estimates ───────────────────────────────────────────────────────

to_s(τ) = τ ./ (Ω / 2π)   # τ = t·Ω/(2π) → physical seconds

ωfit_full  = Dict(n => estimate_omega(to_s(results_full[n][1]),  results_full[n][2])  for n in N_FULL)
ωfit_trunc = Dict(n => estimate_omega(to_s(results_trunc[n][1]), results_trunc[n][2]) for n in N_TRUNC)

# ── Colours ───────────────────────────────────────────────────────────────────

_cgrad = cgrad(:turbo, maximum(N_TRUNC); categorical=true)
c(n)   = _cgrad[n]

# ── Figure ────────────────────────────────────────────────────────────────────

fig = plot(layout = (1, 2),
           size   = (1200, 520), dpi = 200,
           guidefontsize = 13, tickfontsize = 11,
           left_margin = 8Plots.mm, bottom_margin = 6Plots.mm,
           right_margin = 4Plots.mm, top_margin = 4Plots.mm)

# Panel (a) ───────────────────────────────────────────────────────────────────

to_us(τ) = to_s(τ) .* 1e6

for n in N_TRUNC                              # blockaded — dashed, semi-transparent
    τ, Pr = results_trunc[n]
    plot!(fig[1], to_us(τ), Pr;
          color = c(n), lw = 1.4, ls = :dash, alpha = 0.6, label = false)
end
for n in reverse(N_FULL)                      # full space — solid, on top
    τ, Pr = results_full[n]
    plot!(fig[1], to_us(τ), Pr; color = c(n), lw = 1.8, ls = :solid, label = false)
end
# Legend entries in order N=1,2,... (full where available, else blockaded)
for n in N_TRUNC
    plot!(fig[1], [NaN], [NaN]; color = c(n), lw = 1.8,
          ls = n in N_FULL ? :solid : :dash, label = "N=$n")
end

plot!(fig[1];
      xlabel = "Time (μs)", ylabel = L"P_r",
      title  = "Collective Rabi oscillations  (V/Ω = $(round(Int, V/Ω)))",
      titlefontsize = 12,
      xlims = (0, T * 1e6), ylims = (-0.04, 1.12),
      legend = :bottomright, legendcolumns = 4, legendfontsize = 8,
      framestyle = :box, grid = true)
annotate!(fig[1], -0.18 * T * 1e6, 1.22, text("(a)", :left, :top, 13, :black, :bold))

# Panel (b) ───────────────────────────────────────────────────────────────────

Ns_full  = collect(N_FULL)
Ns_trunc = collect(N_TRUNC)
N_ref    = range(first(N_TRUNC), last(N_TRUNC), length=200)

plot!(fig[2], N_ref, sqrt.(N_ref) .* Ω / 2π / 1e6;
      color = :black, lw = 2.0, ls = :solid, label = L"\sqrt{N}\,\Omega/2\pi")
scatter!(fig[2], Ns_trunc, [ωfit_trunc[n] / 2π / 1e6 for n in Ns_trunc];
         color = :white, markerstrokecolor = [c(n) for n in Ns_trunc],
         markershape = :square, markersize = 6, markerstrokewidth = 1.5,
         label = "Blockaded (est.)")
scatter!(fig[2], Ns_full, [ωfit_full[n] / 2π / 1e6 for n in Ns_full];
         color = [c(n) for n in Ns_full],
         markershape = :circle, markersize = 6, markerstrokewidth = 0.5,
         label = "Full (est.)")

plot!(fig[2];
      xlabel = "N", ylabel = L"\Omega_\mathrm{eff}/2\pi\mathrm{\ (MHz)}",
      title  = "Collective Rabi scaling  (√N·Ω)", titlefontsize = 12,
      xticks = (Ns_trunc, [iseven(n) ? string(n) : "" for n in Ns_trunc]),
      legend = :bottomright, legendfontsize = 9,
      framestyle = :box, grid = true)
annotate!(fig[2], -3, 5.36, text("(b)", :left, :top, 13, :black, :bold))

# ── Save ──────────────────────────────────────────────────────────────────────

savefig(fig, joinpath(@__DIR__, "rydberg_rabi_figure.pdf"))
println("Saved benchmark/rydberg_rabi_figure.pdf")
display(fig)
