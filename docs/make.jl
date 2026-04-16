using Documenter
using Literate

using AtomTwin
using AtomTwin.Visualization
using AtomTwin.Units
using Dates

RUN_EXAMPLES = get(ENV, "DOCUMENTER_RUN_EXAMPLES", "false") == "true"

# Single source of truth: name => title
example_titles = (
    "rabi_with_dissipation"            => "Rabi with dissipation",
    "rabi_with_noise"                  => "Rabi with noise",
    "rabi_with_motion"                 => "Rabi with motion",
    "rabi_with_static_intensity_noise" => "Rabi with static noise",
    "rydberg_blockade"                 => "Rydberg blockade",
    "time-optimal_rydberg_gate"        => "Time-optimal Rydberg gate",
    "gateX_tomography"                 => "GateX tomography",
    "eit_with_dissipation"             => "EIT with dissipation",
    "yb171_raman_gate"                 => "Yb-171 Raman gate",
    "k39_state_prep"                   => "K-39 state preparation",
    "atom_sorting"                     => "Atom sorting using AODs",
)

root    = normpath(joinpath(@__DIR__, ".."))
ex_src  = joinpath(root, "test", "examples_src")
md_out  = joinpath(@__DIR__, "src")
jl_out  = joinpath(root, "examples")

for (name, title) in example_titles
    src = joinpath(ex_src, "$name.jl")
    @info "Processing example $src"

    Literate.markdown(src, md_out;
        execute = RUN_EXAMPLES,
        doctest = false,
        flavor = Literate.CommonMarkFlavor()
    )

    src_lines = readlines(src)
    clean_jl  = joinpath(jl_out, "$name.jl")
    open(clean_jl, "w") do io
        for line in src_lines
            occursin("#src", line) && continue
            println(io, line)
        end
    end
end

format = Documenter.HTML(
    collapselevel = 2,
    assets = ["assets/custom.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
    footer = "Copyright 2025-$(year(now())) University of Strasbourg. All rights reserved.",
)  
example_pages = [
    title => "$name.md"
    for (name, title) in example_titles
]

cp(
    joinpath(@__DIR__, "..", "CONTRIBUTING.md"),
    joinpath(@__DIR__, "src", "CONTRIBUTING.md");
    force = true
)

makedocs(
    sitename = "AtomTwin.jl",
    format = format,
    modules  = [AtomTwin],
    pages = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "Examples" => example_pages,
        "API reference"  => [
            "System building"     => "api_system.md",
            "Physical processes"  => "api_physics.md",
            "DAG system"          => "api_dag.md",
            "Running simulations" => "api_sequence.md",
            "Parameters and noise"=> "api_noise.md",
            "Visualization"       => "api_visualization.md",
            "Internals"           => "internals.md",
        ],
        "Contributing" => "CONTRIBUTING.md"
    ],
    remotes = nothing,
    checkdocs = :none,
)

deploydocs(
    repo = "github.com/aQCess/AtomTwin.jl.git",
    devbranch = "main",
    push_preview = false,
    forcepush = true,
)
