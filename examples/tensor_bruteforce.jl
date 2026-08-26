include("tensor_polarizabilities.jl")
using CSV
using DataFrames
using PyFormattedStrings



λ = 400.0:0.01:800.0

I = 0//2; F = 1//1; mF = 0//2
angle = 37.0
e_z = cos(angle * π/180)

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
            filepath_3P1 = "/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb171/3P1 mF3_2_90deg.csv"
        end
    end
end

data_3P1 = CSV.read(filepath_3P1, DataFrame; delim=',', header=["Wavelength", "Lightshift"])

λ_ref, lightshift_3P1_ref = data_3P1[!, "Wavelength"], data_3P1[!, "Lightshift"]



# Generates configurations lazily without allocating memory for all 19,683 elements at once
N = 9
domain = 0:2

potential_configs = []
ame_vals = []
epsilon = 0.2

counter = 1
for J_values in Iterators.product(ntuple(_ -> domain, N)...)
    # 'config' is a 9-element tuple, e.g., (0, 0, 0, 0, 0, 0, 0, 0, 0)
    model = PolarizabilityModel_st(
    "3P1",
    [
        (freq_THz=-539.386800, gamma_MHz=0.183, state_f="(6s2) 1S0", J_f=J_values[1]),    # (6s2) 1S0     #
        (freq_THz=194.778008, gamma_MHz=0.170, state_f="(6s5d) 3D1", J_f=J_f=J_values[2]),     # (6s5d) 3D1
        (freq_THz=202.657933, gamma_MHz=0.280, state_f="(6s5d) 3D2", J_f=J_f=J_values[3]),     # (6s5d) 3D2
        (freq_THz=440.775408, gamma_MHz=3.954, state_f="(6s7s) 3S1", J_f=J_f=J_values[4]),     # (6s7s) 3S1   #
        (freq_THz=654.048602, gamma_MHz=2.783, state_f="(6s6d) 3D1", J_f=J_f=J_values[5]),     # (6s6d) 3D1   #
        (freq_THz=654.927593, gamma_MHz=5.215, state_f="(6s6d) 3D2", J_f=J_f=J_values[6]),     # (6s6d) 3D2   #
        (freq_THz=708.200713, gamma_MHz=1.718, state_f="(6s8s) 3S1", J_f=J_f=J_values[7]),     # (6s8s) 3S1
        (freq_THz=778.975785, gamma_MHz=22.313, state_f="Empirical J=1", J_f=J_f=J_values[8]),    # Empirical J=1
        (freq_THz=778.975785, gamma_MHz=36.687, state_f="Empirical J=2", J_f=J_f=J_values[9]),    # Empirical J=2
    ];
    J_i=1//1,
    offset_Hz_per_Wm2=0.0,
    reference="T. O. Höhn, PhD thesis (2024)",
)
    lightshift_3P1_scalar = scalar_light_shift_coeff_Hz_per_Wcm2.(Ref(model), λ_ref)
    lightshift_3P1_tensor = tensor_light_shift_coeff_Hz_per_Wcm2.(Ref(model), λ_ref; F=F, I=I, mF=mF, e_z=e_z)

    diff = lightshift_3P1_ref .- (lightshift_3P1_scalar.+lightshift_3P1_tensor)
    ame = mean(abs.(diff))

    if ame <= epsilon# || counter%1000==1
        a = scatter(λ_ref, lightshift_3P1_ref; size=(900, 500), ylim=(-30, 10), label="Paper", xlabel="Wavelength [nm]", ylabel="Δν / I  [Hz/(W/cm²)]")
        scatter!(a, λ_ref, lightshift_3P1_scalar .+ lightshift_3P1_tensor; label=f"Self, AME = {ame:.3f}, config = {J_values}", alpha=0.5)
        #plot!(a, λ, lshift_3P1_scalar.+lshift_3P1_tensor, label=f"3P1, I={I}, F={F}, mF={mF}, {angle} degrees")
        display(a)
        #println(J_values)
        println(model.transitions[1].J_f)

        push!(potential_configs, J_values)
        push!(ame_vals, ame)
    end
    counter+= 1
end



print(potential_configs)

length(potential_configs)

range(1, length(potential_configs); step=1)

for i in 1:1:length(potential_configs)
    println(potential_configs[i], ame_vals[i])
end






I = 1//2; F = 3//2; mF = 1//2
angle = 17.0
e_z = cos(angle * π/180)

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
            filepath_3P1 = "/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb171/3P1 mF3_2_90deg.csv"
        end
    end
end

data_3P1 = CSV.read(filepath_3P1, DataFrame; delim=',', header=["Wavelength", "Lightshift"])

λ_ref, lightshift_3P1_ref = data_3P1[!, "Wavelength"], data_3P1[!, "Lightshift"]

potential_configs_2 = []
ame_vals_2 = []


for J_values in potential_configs
    # 'config' is a 9-element tuple, e.g., (0, 0, 0, 0, 0, 0, 0, 0, 0)
    model = PolarizabilityModel_st(
    "3P1",
    [
        (freq_THz=-539.386800, gamma_MHz=0.183, state_f="(6s2) 1S0", J_f=J_values[1]),    # (6s2) 1S0     #
        (freq_THz=194.778008, gamma_MHz=0.170, state_f="(6s5d) 3D1", J_f=J_f=J_values[2]),     # (6s5d) 3D1
        (freq_THz=202.657933, gamma_MHz=0.280, state_f="(6s5d) 3D2", J_f=J_f=J_values[3]),     # (6s5d) 3D2
        (freq_THz=440.775408, gamma_MHz=3.954, state_f="(6s7s) 3S1", J_f=J_f=J_values[4]),     # (6s7s) 3S1   #
        (freq_THz=654.048602, gamma_MHz=2.783, state_f="(6s6d) 3D1", J_f=J_f=J_values[5]),     # (6s6d) 3D1   #
        (freq_THz=654.927593, gamma_MHz=5.215, state_f="(6s6d) 3D2", J_f=J_f=J_values[6]),     # (6s6d) 3D2   #
        (freq_THz=708.200713, gamma_MHz=1.718, state_f="(6s8s) 3S1", J_f=J_f=J_values[7]),     # (6s8s) 3S1
        (freq_THz=778.975785, gamma_MHz=22.313, state_f="Empirical J=1", J_f=J_f=J_values[8]),    # Empirical J=1
        (freq_THz=778.975785, gamma_MHz=36.687, state_f="Empirical J=2", J_f=J_f=J_values[9]),    # Empirical J=2
    ];
    J_i=1//1,
    offset_Hz_per_Wm2=0.0,
    reference="T. O. Höhn, PhD thesis (2024)",
)
    lightshift_3P1_scalar = scalar_light_shift_coeff_Hz_per_Wcm2.(Ref(model), λ_ref)
    lightshift_3P1_tensor = tensor_light_shift_coeff_Hz_per_Wcm2.(Ref(model), λ_ref; F=F, I=I, mF=mF, e_z=e_z)

    diff = lightshift_3P1_ref .- (lightshift_3P1_scalar.+lightshift_3P1_tensor)
    ame = mean(abs.(diff))

    a = scatter(λ_ref, lightshift_3P1_ref; size=(900, 500), ylim=(-30, 10), label="Paper", xlabel="Wavelength [nm]", ylabel="Δν / I  [Hz/(W/cm²)]")
    scatter!(a, λ_ref, lightshift_3P1_scalar .+ lightshift_3P1_tensor; label=f"Self, AME = {ame:.3f}, config = {J_values}", alpha=0.5)
    #plot!(a, λ, lshift_3P1_scalar.+lshift_3P1_tensor, label=f"3P1, I={I}, F={F}, mF={mF}, {angle} degrees")
    display(a)

    if ame <= epsilon# || counter%1000==1
        #println(J_values)
        push!(potential_configs_2, J_values)
        push!(ame_vals_2, ame)
    end
    counter+= 1
end


potential_configs_2