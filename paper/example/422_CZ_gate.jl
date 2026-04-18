# # [[4,2,2]] CZ Gate — Error Budget
#
# Benchmarks each error source independently and reports a table:
#   - mean population in computational states (ideal: 1.0)
#   - mean single-qubit phase shift (ideal: θ_cz = 2.1663 rad)
#   - single-qubit phase coherence error  1 − |⟨e^{iφ}⟩|
#   - mean CZ phase shift (ideal: −π)
#   - CZ phase coherence error  1 − |⟨e^{iφ_CZ}⟩|
#
# Error sources benchmarked independently:
#   doppler_dephasing, beam_inhomogeneity, spontaneous_decay,
#   finite_blockade, interpair_crosstalk, zeeman_dephasing, sigma_p_coupling
#
# Raw shot data is saved to a JSON file so the table can be reproduced
# without re-running the simulations.

include("422_setup.jl")

const n_shots      = 10_000
const results_file = joinpath(@__DIR__, "422_CZ_gate_results.json")

# (label, stochastic) — stochastic sources need n_shots > 1 to average
const error_sources = [
    (:doppler_dephasing,   true),
    (:beam_inhomogeneity,  false),
    (:spontaneous_decay,   true),
    (:finite_blockade,     false),
    (:interpair_crosstalk, false),
    (:zeeman_dephasing,    false),
    (:sigma_p_coupling,    false),
]

const all_off = NamedTuple(k => false for (k, _) in error_sources)

# Input states for the four computational basis cases
const gate_cases = [("00", qubit0, qubit0),
                    ("01", qubit0, qubit1),
                    ("10", qubit1, qubit0),
                    ("11", qubit1, qubit1)]

# ── Helper: run all four gate inputs and collect per-shot amplitudes ──────────
# proj = initial state = ideal output; amp = ⟨proj|ψ_final⟩ per shot

function _run_inputs(sys, seq, shots; callback = nothing)
    raw   = Dict{String, Any}()
    projs = Dict(lbl => getqstate(sys, [q1, qubit1, q3, qubit1])
                 for (lbl, q1, q3) in gate_cases)
    for (lbl, q1, q3) in gate_cases
        out  = play(sys, seq; initial_state = [q1, qubit1, q3, qubit1],
                    savefinalstate = true, shots = shots, shot_callback = callback)
        a    = [projs[lbl]' * fs for fs in out.final_states]
        raw[lbl] = Dict(
            "diag_pops"  => abs2.(a),
            "amp_re"     => real.(a),
            "amp_im"     => imag.(a),
            # mean pop in each comp. basis state — enables off-diagonal breakdown
            "cross_pops" => Dict(lbl2 =>
                mean(abs2.([projs[lbl2]' * fs for fs in out.final_states]))
                for (lbl2, _, _) in gate_cases),
        )
    end
    return raw
end

# ── Metrics: phase errors and entanglement fidelity from raw amplitudes ───────
#
# Gauge-invariant phases:
#   z_sq = ⟨e^(iδφ_sq)⟩  where δφ_sq = arg(a_01·a_00*) − θ_cz
#   z_CZ = ⟨e^(iδφ_CZ)⟩  where δφ_CZ = arg(a_11·a_01*·a_10*·a_00) − π
#
# sq_bias (correctable by R_Z) and cz_bias (non-local, not correctable) are
# the coherent offsets; ε_sq, ε_CZ are the stochastic dephasing errors.
# F_e is the Nielsen entanglement fidelity for the diagonal CZ channel.

function compute_metrics(raw; θ_cz = 0.0)
    amps = Dict(lbl => complex.(raw[lbl]["amp_re"], raw[lbl]["amp_im"]) for lbl in keys(raw))
    shots = length(amps["00"])

    # Population in computational subspace
    mean_pop = mean(vcat([raw[lbl]["diag_pops"] for lbl in ["00","01","10","11"]]...))

    # Single-qubit and CZ gauge-invariant phase phasors
    sq_phases = vcat(angle.(amps["01"] .* conj.(amps["00"])),
                     angle.(amps["10"] .* conj.(amps["00"])))
    sq_z = mean(cis.(rem2pi.(sq_phases, RoundNearest)))

    cz_phases = [angle(amps["11"][s] * conj(amps["01"][s]) *
                       conj(amps["10"][s]) * amps["00"][s]) for s in 1:shots]
    cz_z = mean(cis.(rem2pi.(cz_phases, RoundNearest)))

    sq_bias    = rem2pi(angle(sq_z) + θ_cz, RoundNearest)
    cz_bias    = rem2pi(angle(cz_z) - π,    RoundNearest)
    ε_sq       = 1.0 - abs(sq_z)
    ε_CZ       = 1.0 - abs(cz_z)

    # Entanglement fidelity (Nielsen formula, d=4 diagonal channel)
    # sq_bias is assumed to be removable through calibration;
    # cz_bias is retained as a non-local phase
    T  = 1.0 + 2*abs(sq_z) + (-cz_z)   # z00=1, z01=z10=|sq_z|, z11=−cz_z
    F_e = mean_pop * abs(T)^2 / 16

    return (P_comp = mean_pop, sq_bias = sq_bias, ε_sq = ε_sq,
            cz_bias = cz_bias, ε_CZ = ε_CZ, F_e = F_e)
end

# ── Error budget ──────────────────────────────────────────────────────────────
# Loads cached results from JSON if available; otherwise runs all simulations.

# Recursively convert Symbol-keyed dicts to String-keyed (JSON round-trip fix)
_str_keys(x) = x
_str_keys(d::AbstractDict) = Dict(string(k) => _str_keys(v) for (k, v) in d)

rows = if isfile(results_file)
    println("Loading results from $results_file")
    _str_keys(JSON3.read(read(results_file, String), Dict{String, Any})["rows"])
else
    println("Running error budget (n_shots = $n_shots per stochastic source)...")
    _rows = Dict{String, Any}()

    # ── Ideal baseline (all error sources off) ────────────────────────────
    (; sys, ryd) = build_system(; all_off...)
    seq = Sequence(dt; downsample = 10)
    @sequence seq begin
        Pulse(ryd, T_gate; amplitudes = ryd_amplitudes, interp = :piecewise_constant)
    end
    t0 = time(); print("  [ideal] "); _rows["ideal"] = _run_inputs(sys, seq, 1)
    @printf " (%.1fs)\n" (time()-t0)

    # ── One error source at a time ────────────────────────────────────────
    for (src, stochastic) in error_sources
        local sys, ryd, seq
        (; sys, ryd) = build_system(; merge(all_off, NamedTuple{(src,)}((true,)))...)
        seq = Sequence(dt; downsample = 10)
        @sequence seq begin
            Pulse(ryd, T_gate; amplitudes = ryd_amplitudes, interp = :piecewise_constant)
        end
        shots_i = stochastic ? n_shots : 1
        cb = stochastic ? ((s, _) -> s % 100 == 0 && print(".")) : nothing
        local t0 = time(); @printf "  [%s] " string(src)
        _rows[string(src)] = _run_inputs(sys, seq, shots_i; callback = cb)
        @printf " (%.1fs)\n" (time()-t0)
    end

    # ── All sources enabled ───────────────────────────────────────────────
    (; sys, ryd) = build_system()
    seq = Sequence(dt; downsample = 10)
    @sequence seq begin
        Pulse(ryd, T_gate; amplitudes = ryd_amplitudes, interp = :piecewise_constant)
    end
    t0 = time(); print("  [total] ")
    _rows["total"] = _run_inputs(sys, seq, n_shots;
        callback = (s, _) -> s % 100 == 0 && print("."))
    @printf " (%.1fs)\n" (time()-t0)

    open(results_file, "w") do io
        JSON3.write(io, Dict("n_shots" => n_shots, "theta_cz" => θ_cz, "rows" => _rows))
    end
    println("Saved results to $results_file")
    _rows
end

# ── Print table ───────────────────────────────────────────────────────────────

function print_row(label, res, stochastic)
    coh_sq = stochastic ? @sprintf("%10.2e", res.ε_sq)  : "         -"
    coh_cz = stochastic ? @sprintf("%10.2e", res.ε_CZ)  : "         -"
    @printf "  %-22s  %10.7f  %10.4f  %s  %10.4f  %s  %10.7f\n" label res.P_comp res.sq_bias coh_sq res.cz_bias coh_cz res.F_e
end

println("\n" * "="^100)
println("CZ gate error budget   (stochastic rows: n_shots = $n_shots)")
println("="^100)
@printf "  %-22s  %10s  %10s  %10s  %10s  %10s  %10s\n" "error source" "<P_comp>" "sq bias/rad" "1-|e^iφ_1|" "CZ bias/rad" "1-|e^iφ_2|" "F_e"
println("-"^100)

print_row("ideal (all off)", compute_metrics(rows["ideal"]; θ_cz), false)
println("-"^100)
for (src, stochastic) in error_sources
    print_row(string(src), compute_metrics(rows[string(src)]; θ_cz), stochastic)
end
println("-"^100)
print_row("total (all on)", compute_metrics(rows["total"]; θ_cz), true)
println("="^100)
@printf "\n  Biases relative to: sq phase = θ_cz = %.4f rad,  CZ phase = -pi\n" θ_cz

# ── Off-diagonal error breakdown ──────────────────────────────────────────────
# For each input state:
#   comp = sum of populations in the 3 off-diagonal computational basis states
#   leak = population outside the 4-state computational subspace (Rydberg leakage)
# Both are shot-averaged means.

function _offdiag_vals(src_raw)
    lbls = ["00","01","10","11"]
    [(begin
        cross = src_raw[lbl]["cross_pops"]
        p_sum = sum(Float64(cross[lbl2]) for lbl2 in lbls)
        (comp = p_sum - Float64(cross[lbl]), leak = 1.0 - p_sum)
    end) for lbl in lbls]
end

function print_offdiag_row(label, src_raw)
    v = _offdiag_vals(src_raw)
    @printf "  %-22s  %8.2e %8.2e  %8.2e %8.2e  %8.2e %8.2e  %8.2e %8.2e\n" label v[1].comp v[1].leak v[2].comp v[2].leak v[3].comp v[3].leak v[4].comp v[4].leak
end

println("\n" * "="^102)
println("Off-diagonal error breakdown")
println("="^102)
@printf "  %-22s  %8s %8s  %8s %8s  %8s %8s  %8s %8s\n"    "error source" "co(00)" "lk(00)" "co(01)" "lk(01)" "co(10)" "lk(10)" "co(11)" "lk(11)"
@printf "  %-22s  %17s  %17s  %17s  %17s\n" "" "(input |00>)" "(input |01>)" "(input |10>)" "(input |11>)"
println("-"^102)

print_offdiag_row("ideal (all off)", rows["ideal"])
println("-"^102)
for (src, _) in error_sources
    print_offdiag_row(string(src), rows[string(src)])
end
println("-"^102)
print_offdiag_row("total (all on)", rows["total"])
println("="^102)
println("  co = off-diagonal comp. pop  (wrong comp. state),  lk = leakage outside comp. subspace")

# ── Figure: pulse phase + Rydberg populations (full) ─────────────────────────

(; sys, ryd, atoms) = build_system()

# Add Rydberg population detectors (resonant and leakage sublevels)
for i in [1, 3]
    add_detector!(sys, PopulationDetectorSpec(atoms[i], r_level;  name = "Pr4_$i"))
    add_detector!(sys, PopulationDetectorSpec(atoms[i], leakage;  name = "Pr4p_$i"))
end

seq_fig = Sequence(dt; downsample = 10)
@sequence seq_fig begin
    Pulse(ryd, T_gate; amplitudes = ryd_amplitudes, interp = :piecewise_constant)
end

gate_data = map(gate_cases) do (lbl, q1, q3)
    col = ["#0072B2","#E69F00","#CC79A7","#009E73"][findfirst(x->x[1]==lbl, gate_cases)]
    out = play(sys, seq_fig; initial_state = [q1, qubit1, q3, qubit1], shots = 1)
    (lbl = "|$lbl>", t = out.times .* 1e6,
     Pr_res = out.detectors["Pr4_1"]  .+ out.detectors["Pr4_3"],
     Pr_off = out.detectors["Pr4p_1"] .+ out.detectors["Pr4p_3"],
     col = col)
end

phi_pulse = angle.(ryd_amplitudes)

gr()
default(; fontfamily = "Helvetica", guidefontsize = 9, tickfontsize = 8,
          legendfontsize = 8, titlefontsize = 9, framestyle = :box,
          grid = true, gridalpha = 0.25, tick_direction = :in)

let ticks = range(floor(minimum(phi_pulse)*4)/4, ceil(maximum(phi_pulse)*4)/4; step = 0.25)
    t_pulse = range(0.0, T_gate * 1e6; length = length(ryd_amplitudes))
    global p_phase = Plots.plot(t_pulse, phi_pulse;
        color = :black, lw = 1.5, legend = false,
        xlabel = "t (µs)", ylabel = "Pulse phase (rad)",
        xlim = [0, T_gate * 1e6], ylim = extrema(ticks),
        yticks = (collect(ticks), string.(round.(ticks; digits = 2))))
end

p_pop = Plots.plot(;
    xlabel = "t (µs)", ylabel = "Rydberg pop. (atoms 1+3)",
    ylim = (0, 1.0), xlim = (0, T_gate * 1e6),
    legendfontsize = 7, legend = :topright,
    legend_title = "Initial state", legend_title_font_pointsize = 7)

for d in gate_data
    Plots.plot!(p_pop, d.t, d.Pr_res; label = d.lbl, color = d.col, lw = 2.0)
end
Plots.plot!(p_pop, gate_data[1].t, gate_data[1].Pr_off .* 10;
    label = "|00> leak ×10", color = :gray, lw = 1.5, ls = :dot)

Plots.annotate!(p_phase, -0.23 * T_gate * 1e6, 1.3*maximum(phi_pulse), Plots.text("(a)", :left, :top, 10))
Plots.annotate!(p_pop,   -0.20 * T_gate * 1e6, 1.05,               Plots.text("(b)", :left, :top, 10))

p_cz = Plots.plot(p_phase, p_pop;
    layout = (1, 2), size = (760, 320),
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, top_margin = 5Plots.mm)

Plots.savefig(p_cz, joinpath(@__DIR__, "fig_cz_gate.pdf"))
display(p_cz)
