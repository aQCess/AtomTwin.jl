# Unit tests for tensor polarisability changes
using Pkg
Pkg.activate("/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl")
using AtomTwin
using Plots
using CSV, DataFrames
#include("../src/atoms/polarizability.jl")
#include("../ext/AtomTwinPlotsExt.jl")

# ## 1. 1S0 and 3P0 curves
data_1S0 = CSV.read("/Users/hervesv/Documents/Stuff/Projects/AtomTwin.jl/examples/data/yb174/1S0.csv", DataFrame; delim=',', header=["Wavelength", "Lightshift"])

yb = Ytterbium171Atom(;)
models_0 = AtomTwin.getpolarizabilitymodels(yb)

print(models_0["1S0"])
typeof(models_0["1S0"])
curve = PolarizabilityCurve([models_0["1S0"], models_0["3P0"]])
fig1 = plot(curve)
scatter!(fig1, data_1S0[!, "Wavelength"], data_1S0[!, "Lightshift"], label="1S0 paper", alpha=0.5, markersize=3)