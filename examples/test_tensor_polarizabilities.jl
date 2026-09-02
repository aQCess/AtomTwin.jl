# ## Imports
include("tensor_polarizabilities.jl")
using CSV
using DataFrames
using PyFormattedStrings



#savefig(p, "comparison1.png")
#plot(λ, lshift_1S0_st)


#curve_st = PolarizabilityCurve_st([YB171_POLARIZABILITY_1S0_st, YB171_POLARIZABILITY_3P0_st])
#plot(curve_st)

# ## Plot computed model

λ = 400.0:0.01:800.0

I = 0//2; F = 2//2; mF = 0//2
angle = 90.0
e_z = cos(angle * π/180)
lshift_1S0_st = [scalar_light_shift_coeff_Hz_per_Wcm2(models_st["1S0"], l) for l in λ]
lshift_3P1_scalar = [scalar_light_shift_coeff_Hz_per_Wcm2(models_st["3P1"], l) for l in λ]
lshift_3P1_tensor = [tensor_light_shift_coeff_Hz_per_Wcm2(models_st["3P1"], l; F=F, I=I, mF=mF, e_z=e_z) for l in λ]


data_1S0 = CSV.read("/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb171/1S0.csv", DataFrame; delim=',', header=["Wavelength", "Lightshift"])
"""
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
        elseif angle == 90.0
            # This one is extracted from the sweep plot in PRX QUANTUM 7, 010303 (2026)
            filepath_3P1 = "/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb174/3P1 mF0_90deg.csv"
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
            filepath_3P1 = "/Uesers/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb171/3P1 mF3_2_90deg.csv"
        end
    end
end


if filepath_3P1 == "No file"
    println("Data file not found!")
else
    data_3P1 = CSV.read(filepath_3P1, DataFrame; delim=',', header=["Wavelength", "Lightshift"])
end



p = plot(λ, lshift_1S0_st; size=(900, 500), ylim=(-40, 30), label="1S0", xlabel="Wavelength [nm]", ylabel="Δν / I  [Hz/(W/cm²)]", alpha=1.0, linewidth = 2)
plot!(p, λ, lshift_3P1_scalar.+lshift_3P1_tensor, label=f"3P1, I={I}, F={F}, mF={mF}, {angle} degrees", linewidth = 2)
plot!(p, λ, lshift_3P1_scalar, label=f"3P1 scalar", linestyle=:dash)
plot!(p, λ, lshift_3P1_tensor, label=f"3P1 tensor", linestyle=:dash)
if filepath_3P1 != "No file"
    scatter!(p, data_3P1[!, "Wavelength"], data_3P1[!, "Lightshift"], label="Paper results", alpha=0.5, markersize=3)
end
scatter!(p, data_1S0[!, "Wavelength"], data_1S0[!, "Lightshift"], label="1S0 paper", alpha=0.5, markersize=3)
vline!(p, [759], c="grey", linestyle=:dash)

display(p)
#savefig(p, f"25-08_deg_fix_I={I:.2f}_F={F:.2f}_mF={mF:.2f}_{angle}deg.pdf")




## # RMSE test for 3P1 ligthshift

λ_ref, lightshift_3P1_ref = data_3P1[!, "Wavelength"], data_3P1[!, "Lightshift"]
lightshift_3P1_computed = scalar_light_shift_coeff_Hz_per_Wcm2.(Ref(models_st["3P1"]), λ_ref) .+ tensor_light_shift_coeff_Hz_per_Wcm2.(Ref(models_st["3P1"]), λ_ref; F=F, I=I, mF=mF, e_z=e_z)


a = scatter(λ_ref, lightshift_3P1_ref; size=(900, 500), label="Paper", xlabel="Wavelength [nm]", ylabel="Δν / I  [Hz/(W/cm²)]")
scatter!(a, λ_ref, lightshift_3P1_computed; label="Self", alpha=0.5)
#plot!(a, λ, lshift_3P1_scalar.+lshift_3P1_tensor, label=f"3P1, I={I}, F={F}, mF={mF}, {angle} degrees")

diff = (lightshift_3P1_computed .- lightshift_3P1_ref)./lightshift_3P1_ref
mean(abs.(diff))
rmse = sqrt(mean(diff.^2))
ame = mean(abs.(diff))


