include("tensor_polarizabilities.jl")

using CSV
using DataFrames
using PyFormattedStrings



yb = Ytterbium171Atom(;)
models_0 = AtomTwin.getpolarizabilitymodels(yb)

curve = PolarizabilityCurve([models_0["1S0"], models_0["3P0"]])
plot(curve)


λ = 420.0:0.01:800.0
lshift_1S0_st = [scalar_light_shift_coeff_Hz_per_Wcm2(models_st["1S0"], l) for l in λ]
lshift_1S0 = [light_shift_coeff_Hz_per_Wcm2(models_0["1S0"], l) for l in λ]

lshift_3P0_st = [scalar_light_shift_coeff_Hz_per_Wcm2(models_st["3P0"], l) for l in λ]
lshift_3P0 = [light_shift_coeff_Hz_per_Wcm2(models_0["3P0"], l) for l in λ]

diff_1S0 = abs.((lshift_1S0 - lshift_1S0_st)./lshift_1S0)
max_1S0 = maximum(diff_1S0)
mean_1S0 = mean(diff_1S0)
std_1S0 = std(diff_1S0)

diff_3P0 = abs.((lshift_3P0 - lshift_3P0_st)./lshift_3P0)
max_3P0 = maximum(diff_3P0)
mean_3P0 = mean(diff_3P0)
std_3P0 = std(diff_3P0)

println("1S0 relative differences: max = $max_1S0, mean = $mean_1S0, std = $std_1S0")
println("3P0 relative differences: max = $max_3P0, mean = $mean_3P0, std = $std_3P0")

p = plot(λ, lshift_1S0_st; size=(900, 500), ylim=(-30, 10), label="1S0")
plot!(p, λ, lshift_3P0_st; label="3P0")

plot!(p, λ, lshift_1S0; label="1S0 old", ls=:dash)
plot!(p, λ, lshift_3P0; label="3P0 old", ls=:dash)

#savefig(p, "comparison1.png")
#plot(λ, lshift_1S0_st)


#curve_st = PolarizabilityCurve_st([YB171_POLARIZABILITY_1S0_st, YB171_POLARIZABILITY_3P0_st])
#plot(curve_st)

# Yb174
λ = 400.0:0.01:800.0

I = 0//2; F = 1//1; mF = 0//1
angle = 0.0
e_z = cos(angle * π/180)
#lshift_1S0_st = [scalar_light_shift_coeff_Hz_per_Wcm2(models_st["1S0"], l) for l in λ]
#lshift_3P1_scalar = [scalar_light_shift_coeff_Hz_per_Wcm2(models_st["3P1"], l) for l in λ]
#lshift_3P1_tensor = [tensor_light_shift_coeff_Hz_per_Wcm2(models_st["3P1"], l; F=F, I=I, mF=mF, e_z=e_z) for l in λ]
lshift_1S0_st = scalar_light_shift_coeff_Hz_per_Wcm2.(Ref(models_st["1S0"]), λ)
lshift_3P1_scalar = scalar_light_shift_coeff_Hz_per_Wcm2.(Ref(models_st["3P1"]), λ)
lshift_3P1_tensor = tensor_light_shift_coeff_Hz_per_Wcm2.(Ref(models_st["3P1"]), λ; F=F, I=I, mF=mF, e_z=e_z)

"""
data_1S0 = CSV.read("/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb174/1S0.csv", DataFrame; delim=',', header=["Wavelength", "Lightshift"])
pt = plot(λ, lshift_1S0_st; size=(900, 500), ylim=(-30, 10), label="1S0", xlabel="Wavelength [nm]", ylabel="Δν / I  [Hz/(W/cm²)]")
scatter!(pt, data_1S0[!, "Wavelength"], data_1S0[!, "Lightshift"])
"""

## Always check its right file
filepath_3P1 = "No file"
if I==0//1 && F == 1//1
    if mF == 0//1
        if angle == 0.0
            filepath_3P1 = "/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb174/3P1 mF0.csv"
        elseif angle == 37.0
            filepath_3P1 = "/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb174/3P1 mF0_37deg.csv"
        end
    elseif mF==1//1
        if angle == 0.0
            filepath_3P1 = "/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb174/3P1 mF1.csv"
        end
    end
elseif I==1//2 && F == 3//2
    if mF == 1//2
        if angle == 0.0
            filepath_3P1 = "/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb171/3P1 mF1_2.csv"
        elseif angle == 17.0
            filepath_3P1 = "/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb171/3P1 mF1_2_17deg.csv"
        end
    elseif mF == 3//2
        if angle == 90.0
            filepath_3P1 = "/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb171/3P1 mF3_2_90deg.csv"
        end
    end
end


if filepath_3P1 == "No file"
    println("Data file not found!")
end

data_3P1 = CSV.read(filepath_3P1, DataFrame; delim=',', header=["Wavelength", "Lightshift"])
data_1S0[!, "Wavelength"]

p = plot(λ, lshift_1S0_st; size=(900, 500), ylim=(-30, 10), label="1S0", xlabel="Wavelength [nm]", ylabel="Δν / I  [Hz/(W/cm²)]", alpha=1.0)
plot!(p, λ, lshift_3P1_scalar.+lshift_3P1_tensor, label=f"3P1, I={I}, F={F}, mF={mF}, {angle} degrees")
plot!(p, λ, lshift_3P1_scalar, label=f"3P1 scalar")
plot!(p, λ, lshift_3P1_tensor, label=f"3P1 tensor")
scatter!(p, data_3P1[!, "Wavelength"], data_3P1[!, "Lightshift"], label="Paper results", alpha=0.5, markersize=3)
scatter!(p, data_1S0[!, "Wavelength"], data_1S0[!, "Lightshift"], label="1S0 paper", alpha=0.5, markersize=3)

#savefig(p, f"tensor_debugging_data_I={I:.2f}_F={F:.2f}_mF={mF:.2f}_{angle}deg.pdf")



# Plot individual contributions
tensor_contributions = []
Ji = models_st["3P1"].J_i
tlabels = []
for (i, trans) in enumerate(models_st["3P1"].transitions)
    println("Index $i: $(trans.state_f)")
    contrib = tensor_light_shift_coeff_Hz_per_Wcm2_single_transition.(Ref(trans), Ji, λ; F=F, I=I, mF=mF, e_z=e_z)

    s = (i==1) ?    1 : 1
    push!(tensor_contributions, contrib.*s)

    push!(tlabels, trans.state_f)
end

print(tlabels)

idx_555 = 1
idx_680 = 4

#p_contrib = plot(λ, tensor_contributions[idx_555]+tensor_contributions[idx_680]; size=(900, 500), ylim=(-30, 10), xlabel="Wavelength [nm]", ylabel="Δν / I  [Hz/(W/cm²)]", label=tlabels[idx_555]*tlabels[idx_680])
#tensor_contributions
total_tensor = sum(tensor_contributions)

p_contrib = plot(λ, lshift_3P1_scalar; size=(900, 500), ylim=(-30, 10), xlabel="Wavelength [nm]", ylabel="Δν / I  [Hz/(W/cm²)]", label="Scalar")
plot!(p_contrib, λ, total_tensor; label=f"Tensor* I={I}, F={F}, mF={mF}, {angle} degrees")
plot!(p_contrib, λ, total_tensor+lshift_3P1_scalar; label="Total")
scatter!(p_contrib, data_3P1[!, "Wavelength"], data_3P1[!, "Lightshift"], label="Paper results", alpha=0.5, markersize=3)


p1 = plot(λ, lshift_3P1_scalar, ; size=(900, 500), ylim=(-30, 10), xlabel="Wavelength [nm]", ylabel="Δν / I  [Hz/(W/cm²)]")
plot!(p1, λ, tensor_contributions[1]+tensor_contributions[4]+tensor_contributions[5]+tensor_contributions[6])

#savefig(p_contrib, f"tensor_mod1S0-0_5_I={I:.2f}_F={F:.2f}_mF={mF:.2f}_{angle}deg.pdf") 



print(wigner6j(1, 1, 2, 1, 1, 0))

idx = [1]
#p2 = plot(λ, lshift_3P1_scalar; size=(900, 500), ylim=(-30, 10), xlabel="Wavelength [nm]", ylabel="Δν / I  [Hz/(W/cm²)]", alpha=0.0)

p2 = plot(λ, -0.5*tensor_contributions[1]; size=(900, 500), ylim=(-30, 10), xlabel="Wavelength [nm]", ylabel="Δν / I  [Hz/(W/cm²)]", label=f"{tlabels[1]} * -0.5", alpha=0.8)
plot!(p2, λ, tensor_contributions[4]; label=tlabels[4], alpha=0.8)
plot!(p2, λ, tensor_contributions[5]+tensor_contributions[6]; label=f"{tlabels[5]} + {tlabels[6]}")
plot!(p2, λ, tensor_contributions[4]-0.5tensor_contributions[1]+tensor_contributions[5]+tensor_contributions[6]; label="Sum of both", alpha=0.8)

tensor_contributions[1]

Ji = 1//1
# sqrt term
K1 = sqrt((40F*(2F+1)*(2F-1))/(3*(F+1)*(2*F+3)))*(2Ji + 1)
# polarization term
K2 = (3e_z^2 - 1)/2 * (3mF^2 - F*(F+1)/(F*(2F-1)))

A, B, C = -1, -3, 1//9

function quotient(λ_nm, Γ_MHz, f0_THz)
    Γ = Γ_MHz * 2π * 1e6
    ωL = 2π * c / (λ_nm * 1e-9)
    ω0 = f0_THz * 2π * 1e12

    return Γ / (ω0^2 * (ω0^2 - ωL^2))
end

term_1S0 = K1*K2*A*B*C .* quotient.(λ, 0.183, -539.386800) * 3π*ε0*c^3 *1e4 / h / (-2 * c * ε0)

plot!(p2, λ, term_1S0; label="Hand computed", linestyle = :dash)
savefig("Handcomputed.pdf")