using Test
using AtomTwin
using Printf
using Random
using Statistics
using Statistics

unit_dir = joinpath(@__DIR__, "unit")

@testset "Parameters" begin
    include(joinpath(unit_dir, "test_parameters.jl"))
end

@testset "Sequence" begin
    include(joinpath(unit_dir, "test_sequence.jl"))
end

@testset "Levels" begin
    include(joinpath(unit_dir, "test_levels.jl"))
end

@testset "System" begin
    include(joinpath(unit_dir, "test_system.jl"))
end

@testset "DAG" begin
    include(joinpath(unit_dir, "test_dag.jl"))
end

@testset "Physics" begin
    include(joinpath(unit_dir, "test_physics.jl"))
end

@testset "Motion" begin
    include(joinpath(unit_dir, "test_motion.jl"))
end

@testset "Thread workspaces" begin
    include(joinpath(unit_dir, "test_thread_workspaces.jl"))
end

@testset "Solver order" begin
    include(joinpath(unit_dir, "test_solver_order.jl"))
end

@testset "Paper listings" begin
    include(joinpath(unit_dir, "test_paper.jl"))
end

checksum(vec::AbstractVector{<:Number}) = begin
    h = UInt(0)
    @inbounds for (i, x) in pairs(vec)
        c = reinterpret(UInt, Float64(real(x))) ⊻
            (reinterpret(UInt, Float64(imag(x))) * 0x9e3779b97f4a7c15)
        h ⊻= c * (UInt(i) + 0x85ebca6b)
    end
    h
end

# One-shot runner that assumes globals are set by include(path)
function run_one_shot()
    cs = checksum(checksum_data)
    return runtime, cs, descriptor
end

"""
    run_example(min_shots=3, max_shots=10, time_limit=1.0)

Assumes the example script has already been included and defines
`runtime`, `checksum_data`, `descriptor`. Re-includes it per shot.
"""
function run_example(path; min_shots::Int = 10, max_shots::Int = 100, time_limit::Real = 10.0)
    best_runtime    = Inf
    first_elapsed   = Inf
    total_time      = 0.0
    total_runtime   = 0.0  # New: sum of all runtimes
    first_cs        = UInt(0)
    first_desc      = ""


    n_shots = 0
    while n_shots < max_shots
        Random.seed!(1234)
        t = @elapsed include(path)  # defines runtime, checksum_data, descriptor
        n_shots   += 1
        total_time += t


        runtime, cs, desc = run_one_shot()
        total_runtime += runtime  # New: accumulate runtime
        
        if n_shots == 1
            first_elapsed = t
            first_cs      = cs
            first_desc    = desc
        end


        if runtime < best_runtime
            best_runtime = runtime
        end


        if n_shots ≥ min_shots && total_time ≥ time_limit
            break
        end
    end


    # Everything the script does outside its own `runtime` block on the first
    # include: setup, compilation latency, and any further physics after the
    # timed block (process tomography, plotting). Not build time.
    other_time = first_elapsed - best_runtime
    avg_runtime = total_runtime / n_shots  # New: compute average
    return other_time, best_runtime, avg_runtime, first_cs, first_desc
end


ex_path  = joinpath(@__DIR__, "examples_src")
examples = [
    joinpath(ex_path, "rabi_with_noise.jl"),
    joinpath(ex_path, "rabi_with_dissipation.jl"),
    joinpath(ex_path, "rabi_with_motion.jl"),
    joinpath(ex_path, "rabi_with_static_intensity_noise.jl"),
    joinpath(ex_path, "rydberg_blockade.jl"),
    joinpath(ex_path, "two_qubit_exchange.jl"),
    joinpath(ex_path, "time-optimal_rydberg_gate.jl"),
    joinpath(ex_path, "yb171_raman_gate.jl"),
    joinpath(ex_path, "k39_state_prep.jl"),
    joinpath(ex_path, "gateX_tomography.jl"),
    joinpath(ex_path, "eit_with_dissipation.jl"),
    joinpath(ex_path, "atom_sorting.jl"),
    joinpath(ex_path, "yb174_lightshift_spectroscopy.jl"),
]

RUN_EXAMPLES = get(ENV, "ATOMTWIN_RUN_EXAMPLES", "false") == "true"

# `Pkg.test` forces `--check-bounds=yes`, which overrides every `@inbounds` in
# the package -- including the solver hot loops -- so the runtimes below come out
# roughly 2.5x their real value. Harmless for the checksums, misleading for the
# timings, and silent unless we say so.
if RUN_EXAMPLES && Base.JLOptions().check_bounds == 1
    @info """
    Bounds checking is forced on; the example runtimes below are ~2.5x inflated.
    For real timings:

        Pkg.test("AtomTwin"; julia_args = ["--check-bounds=auto"])
    """
end

if RUN_EXAMPLES; @testset "Example scripts" begin
    rows = String[]
    for path in examples
        include(path)
        buildtime, best_runtime, avg_runtime, cs, desc = run_example(path)

        scriptname = basename(path)  
        push!(rows, @sprintf(
            "%-40s  %8.3f  %8.3f  %8.3f  0x%016x",
            scriptname, best_runtime, avg_runtime, buildtime, cs,
        ))
        println(desc)
    end

    println()
    println(rpad("Example", 40), "    best[s]    avg[s]   other[s]   checksum")
    println(repeat("-", 40 + 2 + 10 + 2 + 10 + 2 + 18))
    for row in rows
        println(row)
    end
    println()
end; end  # if RUN_EXAMPLES

;
