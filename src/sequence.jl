"""
    Sequence

Container for a time-ordered list of low-level control instructions
with sequence-level defaults for time step `dt` and downsampling.

A `Sequence` behaves like a vector of `AbstractInstruction` objects:
it supports indexing, iteration, `length`, and `push!`, and is the
primary container passed to `play` when simulating instruction-level dynamics.

# Fields

- `instructions::Vector{AbstractInstruction}` – ordered list of instructions
- `dt::Float64` – default time step for instructions (seconds)
- `downsample::Int` – default output downsampling factor (default 1 = every step)

# Per-Instruction Customization

Individual instructions can override `dt` and `downsample` by specifying them
in their constructors. If not specified, the sequence-level defaults are used:

```julia
seq = Sequence(1e-8; downsample=1)
push!(seq, Pulse(c, 500e-9))                          # uses seq.dt and seq.downsample
push!(seq, Pulse(c, 100e-9; dt=1e-9))                 # fine dt for accuracy
push!(seq, Wait(1e-6; downsample=10))                 # coarse output
push!(seq, MoveRow(tw, r, d, t; dt=5e-9, downsample=3)) # both customized
```

This enables optimizing simulation cost by using fine `dt` during fast dynamics
and coarse `dt` during slow phases.

Sequences are usually constructed via the convenience constructor:

`Sequence(dt::Float64)`

which creates an empty sequence with the given default time step.
"""
struct Sequence
    instructions::Vector{AbstractInstruction}
    dt::Union{Float64,Nothing}     # `nothing` ⇒ derive from ‖H‖ and `tol` at build time
    downsample::Int
    tol::Float64                   # target local error per step (NOT a global bound)
    jtol::Union{Float64,Nothing}   # target MCWF jump-omission probability (see `_derive_dt`);
                                   # `nothing` ⇒ derive from the shot count at play time
end

# Backwards compatibility: `tol` was added after the struct was public, so the
# original three-field positional form must keep working unchanged.
Sequence(instructions::Vector{<:AbstractInstruction}, dt, downsample::Int) =
    Sequence(instructions, dt, downsample, 1e-4, nothing)
Sequence(instructions::Vector{<:AbstractInstruction}, dt, downsample::Int, tol) =
    Sequence(instructions, dt, downsample, tol, nothing)
Sequence(instructions::Vector{<:AbstractInstruction}, dt) =
    Sequence(instructions, dt, 1, 1e-4, 1e-4)

"""
    Sequence(dt::Float64; downsample::Int = 1)

Create an empty `Sequence` with default time step `dt` and downsampling.

These parameters serve as defaults for all instructions. Individual instructions
can override them by specifying `dt` and `downsample` in their constructors:

```julia
seq = Sequence(1e-8; downsample=1)
push!(seq, Pulse(c, t; dt=1e-9))           # overrides dt for this instruction
push!(seq, Wait(1e-6; downsample=10))      # overrides downsample
```

# Parameters

- `dt::Float64`: Default time step for all instructions (seconds). Must be positive.
  Each instruction uses this unless it specifies its own.
- `downsample::Int`: Default output downsampling factor (default 1 = record every step).
  Controls how often detector output is recorded: every `downsample`-th solver step
  is written to output. The quantum integrator still runs at full `dt`; only output
  is thinned. Applies uniformly to all detector types.

# Use Cases

Use global downsampling when a small `dt` is required for accuracy but
full-resolution output is not needed:

```julia
seq = Sequence(1e-9; downsample=100)  # fine integration, sparse output
```

Use per-instruction overrides to optimize across protocol phases:

```julia
seq = Sequence(1e-8)
push!(seq, Pulse(c, t; dt=1e-9))      # fine dt for fast Rabi dynamics
push!(seq, Wait(1e-6))                # coarse dt (default) for slow evolution
push!(seq, Ramp(tw, r, a, t; downsample=10))  # sparse output for slow ramp
```
"""
function Sequence(dt::Float64; downsample::Int = 1, tol::Float64 = 1e-4,
                  jtol::Union{Float64,Nothing} = nothing)
    dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    downsample > 0 || throw(ArgumentError("downsample must be positive, got $downsample"))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))
    return Sequence(AbstractInstruction[], dt, downsample, tol, jtol)
end

"""
    Sequence(; downsample = 1, tol = 1e-6)

Create a `Sequence` with **no explicit time step**. The solver derives one per
instruction from the Hamiltonian actually present, targeting a local relative
error of `tol` per step.

The step is chosen from `θ = ‖H‖·dt`, where `‖H‖` is a Gershgorin upper bound
computed in O(nnz) at build time. The Taylor propagator's error per step is
O(θ^p), so `θ` is set to `tol^(1/p)` and capped well below the stability limit
(`2√2` at the default order 4). Because the bound is an upper bound, the derived
step is conservative — never optimistic.

```julia
seq = Sequence()                  # solver picks dt for 1e-6 local error
seq = Sequence(; tol = 1e-9)      # tighter: smaller dt
seq = Sequence(1e-9)              # explicit dt, unchanged behaviour
```

**When to set `dt` yourself.** `dt` is also the *control* grid: shaped pulse
envelopes and moving beams are resampled onto it, so it fixes how finely a
protocol is resolved, which is a physical choice rather than a numerical one. If
a pulse has structure the derived step would smooth over, pass `dt` explicitly.
A derived step targets accuracy of the *propagator*, not fidelity to your
envelope.
"""
function Sequence(; downsample::Int = 1, tol::Float64 = 1e-4,
                  jtol::Union{Float64,Nothing} = nothing)
    downsample > 0 || throw(ArgumentError("downsample must be positive, got $downsample"))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))
    return Sequence(AbstractInstruction[], nothing, downsample, tol, jtol)
end

"""
    Sequence(instruction_duration, tsteps::Integer; downsample = 1)

Create a `Sequence` whose step is `instruction_duration / tsteps`, i.e. specify
how many steps an instruction of that length should take rather than how long
each step is.

This is often the clearer way to say it. The duration of an instruction is
physical; the step is a discretisation choice that only has to divide it. Giving
a step count makes that relationship explicit and cannot produce a step which
fails to divide the duration:

```julia
seq = Sequence(2e-6, 2000)     # a 2 µs pulse in 2000 steps  (dt = 1 ns)
seq = Sequence(1e-9)           # equivalent, stated as a step
```

Note `dt` still applies to *every* instruction, so an instruction of a different
length simply takes proportionally more or fewer steps — `tsteps` is not a
per-instruction step count. Use the `dt` keyword on an individual instruction to
override it, or `Sequence(; tol = ...)` to let the solver choose.
"""
function Sequence(instruction_duration::Real, tsteps::Integer; downsample::Int = 1,
                  tol::Float64 = 1e-4, jtol::Union{Float64,Nothing} = nothing)
    instruction_duration > 0 ||
        throw(ArgumentError("instruction_duration must be positive, got $instruction_duration"))
    tsteps > 0 || throw(ArgumentError("tsteps must be positive, got $tsteps"))
    return Sequence(Float64(instruction_duration) / tsteps; downsample = downsample,
                    tol = tol, jtol = jtol)
end

"""
    Base.getindex(seq::Sequence, i...)

Index into the underlying instruction list of `seq`.

This allows `Sequence` to be used like a vector of `AbstractInstruction`,
e.g. `seq[1]` returns the first instruction in the sequence.
"""
Base.getindex(seq::Sequence, i...) = seq.instructions[i...]

"""
    Base.iterate(seq::Sequence, state...)

Iterate over the instructions in `seq`.

Enables use of `for inst in seq` and other iterator-based patterns,
treating `Sequence` as a collection of `AbstractInstruction` objects.
"""
Base.iterate(seq::Sequence, state...) = iterate(seq.instructions, state...)

"""
    Base.length(seq::Sequence)

Return the number of instructions stored in `seq`.
"""
Base.length(seq::Sequence) = length(seq.instructions)

"""
    Base.eltype(::Type{Sequence})

Return the element type of a `Sequence`, which is `AbstractInstruction`.
"""
Base.eltype(::Type{Sequence}) = AbstractInstruction

"""
    Base.push!(seq::Sequence, inst::AbstractInstruction)

Append an instruction `inst` to the end of `seq`.

Instructions can specify their own `dt` and `downsample` via their constructors.
If not specified, the sequence-level defaults are used during compilation.

Returns the modified `Sequence`, allowing idioms such as

```julia
push!(seq, Wait(10e-9))
push!(seq, Pulse(coupling, t; dt=1e-9))         # fine dt for this instruction
push!(seq, Wait(1e-6; downsample=10))           # coarse output for this instruction
```
"""
function Base.push!(seq::Sequence, inst::AbstractInstruction)
    push!(seq.instructions, inst)
    return seq
end

"""
    Base.push!(seq::Sequence, instrs::AbstractVector{<:AbstractInstruction})

Append all instructions in `instrs` to `seq` in order.

Enables user-defined composite gate functions that return a vector of instructions
to be used transparently inside [`@sequence`](@ref) blocks:

```julia
RZ(dets, φ; Δ=1.0) = [Pulse(dets, mod(φ, 2π) / Δ)]
RX(sq, θ, T_pi) = [Pulse(sq, θ / π * T_pi / 2)]
Hadamard(sq, det) = [RZ(det, π/2)..., RX(sq, π/2, T_pi)..., RZ(det, π/2)...]

@sequence seq begin
    Hadamard(sq, det)  # appends all sub-instructions
end
```
"""
function Base.push!(seq::Sequence, instrs::AbstractVector{<:AbstractInstruction})
    for inst in instrs
        push!(seq, inst)
    end
    return seq
end

"""
    @sequence seq begin 
        ...
    end

Macro for building an instruction `Sequence` from imperative-style
code while preserving native Julia control flow.

`@sequence` walks the body of the `begin ... end` block and rewrites
each function call into a `push!(seq, ...)`. Loops, conditionals, and
other control-flow constructs are left intact and execute as normal,
so you can generate instruction patterns programmatically:

```
@sequence seq begin
    Wait(1µs)
    for i in 1:3
        Pulse(coupling, t)
        Wait(i * 500ns)
    end
    if true
        Wait(10ns)
    end
end
```
"""
macro sequence(seq, block)
    function lower(expr)
        # Remove line numbers (more robustly)
        if expr isa LineNumberNode || (expr isa Expr && expr.head == :line)
            return nothing
        end

        # For `begin` blocks and code blocks
        if expr isa Expr && expr.head == :block
            args = filter(!isnothing, map(ex -> begin
                if ex isa Expr && ex.head == :call
                    return Expr(:call, :push!, esc(seq), esc(ex))
                else
                    return lower(ex)
                end
            end, expr.args))
            return Expr(:block, args...)
        # For function calls: push! to collection
        elseif expr isa Expr && expr.head == :call
            return Expr(:call, :push!, esc(seq), esc(expr))
        # For control flow, recurse on their arguments
        elseif expr isa Expr && expr.head in (:for, :while, :if, :let, :begin, :try)
            new_args = map(lower, expr.args)
            return Expr(expr.head, new_args...)
        # For everything else (assignments, etc), just splice as is
        else
            return esc(expr)
        end
    end

    result = lower(block)
    return result
end


