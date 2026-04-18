# Rabi figure — AtomTwin only
#
# Layout:
#   ┌─────────────────────────┬──────────────┐
#   │  P_e(t)  over 1000      │  Last Rabi   │
#   │  Rabi cycles            │  cycle zoom  │
#   ├─────────────────────────┤──────────────┤
#   │  Residual vs analytic   │              │
#   └─────────────────────────┴──────────────┘
#
# Run with:
#   julia --project=benchmark benchmark/rabi_figure.jl

using AtomTwin
using StatsBase
using Plots
using LaTeXStrings
using Random
#gr()

# ── Parameters (match rabi_benchmark.jl) ──────────────────────────────────────

Ω     = 2π * 1.0e6
Γ     = 2π * 0.5e3
T     = 1000 / (Ω/2π)
dt    = T / 100_000
SHOTS = 100
SEED = 133

# ── Analytical solution (Bloch equation, exact) ───────────────────────────────

function analytical(t; γ = Γ)
    denom = γ^2 + 2Ω^2
    z_ss  = -γ^2 / denom
    δy₀   =  2Ω*γ / denom
    δz₀   = -2Ω^2 / denom
    Ω_R   = sqrt(Ω^2 - γ^2/16)
    e     = exp(-3γ * t / 4)
    cs    = cos(Ω_R * t)
    sn    = sin(Ω_R * t) / Ω_R
    δz    = e * (-Ω * sn * δy₀ + (cs - (γ/4) * sn) * δz₀)
    (1 + z_ss + δz) / 2
end

# Analytic envelope: P_ss ± A(t)/2 where A(t) = exp(-3γt/4)·√(C²+S²),
# C = δz₀, S = -Ω/Ω_R·δy₀ - γ/(4Ω_R)·δz₀  (envelope of the Bloch oscillation)
function analytical_envelope(t; γ = Γ)
    denom = γ^2 + 2Ω^2
    z_ss  = -γ^2 / denom
    δy₀   =  2Ω*γ / denom
    δz₀   = -2Ω^2 / denom
    Ω_R   = sqrt(Ω^2 - γ^2/16)
    C     = δz₀
    S     = -Ω/Ω_R * δy₀ - γ/(4Ω_R) * δz₀
    amp   = exp(-3γ * t / 4) * sqrt(C^2 + S^2) / 2
    mid   = (1 + z_ss) / 2
    mid + amp, mid - amp   # upper, lower
end

# ── Time axes ─────────────────────────────────────────────────────────────────

const N      = round(Int, T / dt)
const t_full = range(dt, T; length = N)
const n_last = round(Int, (1 / (Ω/2π)) / dt)   # 100 steps = 1 Rabi period

const P_an_se = analytical.(t_full; γ = 0.0)
const P_an_me = analytical.(t_full)

# Envelopes on thinned grid (smooth curves, no need for full resolution)
const env_me_upper, env_me_lower = ntuple(i -> getindex.(analytical_envelope.(t_full), i), 2)
# SE envelope: constant [0, 1] — just used for residual reference, not plotted

# Thin for full-range panels (100 k → 10 k points, still smooth)
const stride = 10
const idx    = 1:stride:N

# ── Build and run AtomTwin ─────────────────────────────────────────────────────

function build(; decay = false)
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

sys_u, seq_u, g_u = build(decay = false)
sys,   seq,   g   = build(decay = true)

job_se   = compile(sys_u, seq_u; initial_state = [g_u])
job_me   = compile(sys,   seq;   density_matrix = true,  initial_state = [g])
job_mcwf = compile(sys,   seq;   density_matrix = false, initial_state = [g])

println("Running SE and ME...")
P_se = play(job_se, sys_u).detectors["P_e"]          # (N,)
P_me = play(job_me, sys).detectors["P_e"]            # (N,)

println("Running $SHOTS MCWF trajectories...")
mc_out    = play(job_mcwf, sys; shots = SHOTS, initial_state=[g], rng = Random.MersenneTwister(SEED)).detectors["P_e"]   # (N, SHOTS)
P_mc_mean = vec(mean(mc_out, dims = 2))
println("Done.")

# ── Colours ───────────────────────────────────────────────────────────────────

c_se = "#0072B2"    # Wong blue    — dark   (L ≈ 0.15)
c_me = "#CC79A7"    # Wong pink    — medium (L ≈ 0.28)
c_mc = "#E69F00"    # Wong orange  — light  (L ≈ 0.42)

# ── Figure ────────────────────────────────────────────────────────────────────

t_µs      = collect(t_full) .* 1e6
t_last_µs = t_µs[end-n_last+1:end]

res_se   = P_se      .- P_an_se
res_me   = P_me      .- P_an_me
res_mcmn = P_mc_mean .- P_an_me

# Residual y-limits: sized to SE/ME (MCWF will clip — intentional)
res_ylim = maximum(max.(abs.(res_se), abs.(res_me))) * 1.5
res_ylim = max(res_ylim, 1e-12)   # guard against exact zeros

l = grid(2, 2; widths = [0.60, 0.40], heights = [0.75, 0.25])

fig = plot(layout = l,
           size   = (1200, 580),
           dpi    = 200,
           guidefontsize  = 13,
           tickfontsize   = 11,
           left_margin   = 8Plots.mm,
           bottom_margin = 5Plots.mm,
           right_margin  = 3Plots.mm)

# ── Panel 1 (top-left): full oscillation ──────────────────────────────────────

for s in 1:SHOTS
    plot!(fig[1], t_µs[idx], mc_out[idx, s];
          color = c_mc, alpha = 0.06, lw = 0.3, label = false)
end
plot!(fig[1], t_µs[idx], P_se[idx];  color = c_se, lw = 1.2, label = "AtomTwin SE")
plot!(fig[1], t_µs[idx], P_mc_mean[idx]; color = c_mc, lw = 1.2, label = "MCWF mean ($SHOTS shots)")
plot!(fig[1], t_µs[idx], P_me[idx];  color = c_me, lw = 1.2, label = "AtomTwin ME")

# Analytic envelope (ME only — SE envelope is constant [0,1])
plot!(fig[1], t_µs[idx], env_me_upper[idx]; color = :black, lw = 1.5, ls = :dash, label = "Analytic envelope")
plot!(fig[1], t_µs[idx], env_me_lower[idx]; color = :black, lw = 1.5, ls = :dash, label = false)

plot!(fig[1];
      ylabel     = L"P_e(t)",
      title      = "Rabi oscillations  (1000 cycles,  Γ/Ω = $(round(Γ/Ω, sigdigits=3)))",
      xlims      = extrema(t_µs),
      ylims      = (-0.02, 1.02),
      legend     = :topright,
      titlefontsize = 12,
      legendfontsize = 9,
      xformatter = _ -> "",
      framestyle = :box,
      grid       = true)
annotate!(fig[1], t_µs[1] - 0.2*(t_µs[end]-t_µs[1]), 1.1,
          text("(a)", :left, :top, 13, :black))

# grid(2,2) order: [1]=top-left, [2]=top-right, [3]=bottom-left, [4]=bottom-right

# ── Panel 2 (top-right): last Rabi cycle zoom ────────────────────────────────

sl = (N-n_last+1):N

for s in 1:SHOTS
    plot!(fig[2], t_last_µs, mc_out[sl, s];
          color = c_mc, alpha = 0.12, lw = 0.5, label = false)
end
plot!(fig[2], t_last_µs, P_se[sl];      color = c_se, lw = 1.8, label = "AtomTwin SE")
plot!(fig[2], t_last_µs, P_me[sl];      color = c_me, lw = 1.8, label = "AtomTwin ME")
plot!(fig[2], t_last_µs, P_mc_mean[sl]; color = c_mc, lw = 1.8, label = "MCWF mean")
# Analytic on top
plot!(fig[2], t_last_µs, P_an_se[sl]; color = :black, lw = 2.0, ls = :dash, label = "Analytic SE")
plot!(fig[2], t_last_µs, P_an_me[sl]; color = :black, lw = 2.0, ls = :dot,  label = "Analytic ME")

plot!(fig[2];
      ylabel     = L"P_e(t)",
      title      = "Last Rabi cycle",
      ylims      = (-0.02, 1.02),
      xformatter = _ -> "",
      legend     = :topright,
      titlefontsize = 12,
      legendfontsize = 9,
      framestyle = :box,
      grid       = true)
annotate!(fig[2], t_last_µs[1] - 0.35*(t_last_µs[end]-t_last_µs[1]), 1.1,
          text("(b)", :left, :top, 13, :black, :bold))

# ── Panel 3 (bottom-left): residuals (full range) ────────────────────────────

hline!(fig[3], [0.0]; color = :black, lw = 0.7, ls = :dot, label = false)
plot!(fig[3], t_µs[idx], res_mcmn[idx]; color = c_mc, lw = 0.7, label = false)
plot!(fig[3], t_µs[idx], res_me[idx];   color = c_me, lw = 0.9, label = false)
plot!(fig[3], t_µs[idx], res_se[idx];   color = c_se, lw = 0.9, label = false)

plot!(fig[3];
      ylabel     = "Residual",
      xlabel     = "Time (µs)",
      xlims      = extrema(t_µs),
      ylims      = (-res_ylim, res_ylim),
      legend     = false,
      framestyle = :box,
      grid       = true)

# ── Panel 4 (bottom-right): residuals zoom, same y-scale ─────────────────────

hline!(fig[4], [0.0]; color = :black, lw = 0.7, ls = :dot, label = false)
plot!(fig[4], t_last_µs, res_mcmn[sl]; fillrange = 0, color = c_mc, lw = 0.9, label = false)
plot!(fig[4], t_last_µs, res_me[sl];   fillrange = 0, color = c_me, lw = 1.2, label = false)
plot!(fig[4], t_last_µs, res_se[sl];   fillrange = 0, color = c_se, lw = 1.2, label = false)

plot!(fig[4];
      ylabel     = "Residual",
      xlabel     = "Time (µs)",
      xlims      = extrema(t_last_µs),
      ylims      = (-res_ylim, res_ylim),
      legend     = false,
      framestyle = :box,
      grid       = true)

# ── Save ──────────────────────────────────────────────────────────────────────

savefig(fig, joinpath(@__DIR__, "rabi_figure.pdf"))
println("Saved benchmark/rabi_figure.pdf")
