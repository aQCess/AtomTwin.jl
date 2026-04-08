"""
    Sequence

Container for a time-ordered list of low-level control
instructions with a fixed time step `dt`.

A `Sequence` behaves like a vector of `AbstractInstruction` objects:
it supports indexing, iteration, `length`, and `push!`, and is the
primary container passed to `play` when simulating instruction-level
dynamics.

# Fields

- `instructions::Vector{AbstractInstruction}` – ordered list of
  instructions in the sequence
- `dt` – base time step used to interpret instruction durations
  (typically a `Float64` in seconds or an application-specific time unit)
- `downsample::Int` – record detector output every nth solver step (default 1)

Sequences are usually constructed via the convenience constructor

`Sequence(dt::Float64)`

which creates an empty sequence with the given time step.
"""
struct Sequence
    instructions::Vector{AbstractInstruction}
    dt::Float64
    downsample::Int  # record every nth step (1 = every step)
end

"""
    Sequence(dt::Float64; downsample::Int = 1)

Create an empty `Sequence` with time step `dt` and no instructions.

`downsample` controls how often detector output is recorded: every `downsample`-th
solver step is written, so all detector arrays and `job.times` have length
`t_steps ÷ downsample` instead of `t_steps`. The quantum integrator still runs
at full `dt`; only the output is thinned. Applies uniformly to all detector types
(`PopulationDetector`, `CoherenceDetector`, `MotionDetector`, `FieldDetector`).

Use when a small `dt` is required for accuracy but full-resolution output is not
needed, e.g. when the interaction frequency greatly exceeds the observable bandwidth:

```julia
seq = Sequence(dt; downsample=100)  # record every 100th step
```
"""
function Sequence(dt::Float64; downsample::Int = 1)
    dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    downsample > 0 || throw(ArgumentError("downsample must be positive, got $downsample"))
    return Sequence(AbstractInstruction[], dt, downsample)
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

Returns the modified `Sequence`, allowing idioms such as

```
push!(seq, Wait(10e-9))
push!(seq, Pulse(global_coupling, t))
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
    Hadamard(sq, det)   # push!(seq, Hadamard(...)) appends 3 Pulses
end
```
"""
function Base.push!(seq::Sequence, instrs::AbstractVector{<:AbstractInstruction})
    for inst in instrs
        push!(seq.instructions, inst)
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


