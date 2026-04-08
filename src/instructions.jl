"""
Describe low-level hardware actions as instruction types pulses, waits, on/off switching, and movements.
"""

"""
    Switchable

Union of field-like or coupling objects that can be switched, pulsed,
or otherwise controlled by the instruction layer.
"""
const Switchable = Union{Dynamiq.AbstractField, GlobalCoupling, PlanarCoupling, Detuning, NoisyField}

"""
    Pulse(couplings::Vector{<:Switchable}, duration, ampl, amplitudes)

Apply laser coupling for a fixed duration or with a shaped amplitude profile.

# Fields
- `couplings::Vector{Switchable}` – list of `Switchable` objects (e.g. `GlobalCoupling{NLevelAtom}`) to be modulated.
- `duration::Float64` – pulse duration in seconds. The compiled pulse will span exactly this duration.
- `ampl::ComplexF64` – overall coupling strength (relative to the default). Can be complex. It scales the entire amplitude shape.
- `amplitudes::Vector{ComplexF64}` – vector of complex amplitudes defining the pulse shape. If empty, the pulse has constant amplitude `ampl` over `duration`. If non‑empty, the shape defined by `amplitudes` is resampled and interpolated over the full duration `duration` (using third‑order Lagrange interpolation).
"""
struct Pulse <: AbstractInstruction
    couplings::Vector{Switchable}
    duration::Float64
    ampl::ComplexF64
    amplitudes::Vector{ComplexF64}
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    Pulse(couplings::Vector{<:Switchable}, duration; ampl=1.0, amplitudes=ComplexF64[])

Construct a `Pulse` acting on multiple couplings simultaneously, all with the same duration and relative amplitude `ampl`.

# Arguments
- `couplings`: list of `Switchable` objects to be modulated.
- `duration`: pulse duration (seconds). The compiled pulse will span exactly this duration.
- `ampl`: overall coupling strength (relative to default). Scales the entire amplitude shape.
- `amplitudes`: optional vector of amplitudes defining the pulse shape. If empty, the pulse has constant amplitude `ampl` over `duration`. If non‑empty, the shape is resampled and interpolated over the full duration `duration` (using third‑order Lagrange interpolation).

See `Pulse(couplings, duration, ampl, amplitudes)` for details on how the shape is interpolated.
"""
function Pulse(couplings::Vector, duration::Float64; ampl=1.0, amplitudes=ComplexF64[], dt=nothing, downsample=nothing)
    Pulse(couplings, duration, ampl, amplitudes, dt, downsample)
end

"""
    Pulse(coupling::Switchable, duration; ampl=1.0, amplitudes=ComplexF64[])

Convenience constructor for a single‑coupling pulse. Equivalent to `Pulse([coupling], duration; ampl, amplitudes)`.

See `Pulse(couplings, duration, ampl, amplitudes)` for details on how the shape is interpolated.
"""
Pulse(coupling::Switchable, duration::Float64; ampl=1.0, amplitudes=ComplexF64[], dt=nothing, downsample=nothing) = Pulse([coupling], duration, ampl, amplitudes, dt, downsample)

"""
    On(couplings)

Turn on one or more couplings.

- `couplings` is a vector of `Switchable` objects to be switched on.
"""
struct On <: AbstractInstruction
    couplings::Vector{Switchable}
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    On(coupling::Switchable)

Convenience constructor for a single coupling. Equivalent to `On([coupling])`.
"""
On(coupling::Switchable; dt=nothing, downsample=nothing) = On([coupling], dt, downsample)

"""
    Off(couplings)

Turn off one or more couplings.

- `couplings` is a vector of `Switchable` objects to be switched off.
"""
struct Off <: AbstractInstruction
    couplings::Vector{Switchable}
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    Off(coupling::Switchable)

Convenience constructor for a single coupling. Equivalent to `Off([coupling])`.
"""
Off(coupling::Switchable; dt=nothing, downsample=nothing) = Off([coupling], dt, downsample)

"""
    Wait(duration)

Idle period with no explicit control actions.

- `duration` is the wait time (seconds).
"""
struct Wait <: AbstractInstruction
    duration::Float64
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    Wait(duration; dt=nothing, downsample=nothing)

Convenience constructor for Wait with optional per-instruction overrides.
"""
Wait(duration::Float64; dt=nothing, downsample=nothing) = Wait(duration, dt, downsample)

"""
    MoveCol(tweezers, cols, delta, duration; sweep = :min_jerk)

Move one or more columns of tweezers simultaneously.

- `cols` can be a single integer or a vector of integers specifying the columns.
- `delta` is the frequency shift (Hz).
- `duration` is the total time for the move (seconds).
- `sweep` is the sweep profile: built-in symbol (`:linear`, `:cosine`, `:min_jerk`)
  or custom function `s -> f(s)` mapping [0,1] → [0,1].
"""
mutable struct MoveCol <: AbstractInstruction
    tweezers::TweezerArray
    cols::AbstractVector{Int}
    delta::Union{Float64, Parameter, ParametricExpression}
    duration::Float64
    sweep::Union{Symbol, Function}
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    MoveCol(tweezers, col, delta, duration; sweep = :min_jerk)

Convenience constructor for moving a single column. `col` can be an `Int`
or a parametric expression; it is stored in the general `cols` field.
"""
MoveCol(ta::TweezerArray, col, delta, duration; sweep=:min_jerk, dt=nothing, downsample=nothing) = MoveCol(
    ta, col isa Int ? [col] : col, delta, duration, sweep, dt, downsample)

"""
    MoveRow(tweezers, rows, delta, duration; sweep = :min_jerk)

Move one or more rows of tweezers simultaneously.

- `rows` can be a single integer or a vector of integers specifying the rows.
- `delta` is the frequency shift (Hz).
- `duration` is the total time for the move (seconds).
- `sweep` is the sweep profile: built-in symbol (`:linear`, `:cosine`, `:min_jerk`)
  or custom function `s -> f(s)` mapping [0,1] → [0,1].
"""
struct MoveRow <: AbstractInstruction
    tweezers::TweezerArray
    rows::AbstractVector{Int}
    delta::Union{Float64, Parameter, ParametricExpression}
    duration::Float64
    sweep::Union{Symbol, Function}
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    MoveRow(tweezers, row, delta, duration; sweep = :min_jerk)

Convenience constructor for moving a single row. Wraps `row` into a vector.
"""
MoveRow(ta::TweezerArray, row::Int, delta::Float64, duration::Float64; sweep=:min_jerk, dt=nothing, downsample=nothing) = MoveRow(ta, [row], delta, duration, sweep, dt, downsample)

"""
    RampRow(tweezers, rows, final_amplitude, ramp_time)

Ramp the amplitude of one or more rows of tweezers simultaneously.

- `rows` can be a single integer or a vector of integers specifying the rows.
- `final_amplitude` is the value to ramp each row to (scalar or vector).
- `ramp_time` is the total duration of the ramp (seconds).
"""
struct RampRow <: AbstractInstruction
    tweezers::TweezerArray
    rows::AbstractVector{Int}
    final_amplitude::Union{Float64,AbstractVector{Float64}}
    ramp_time::Float64
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    RampRow(tweezers, row, final_amplitude, ramp_time)

Convenience constructor for ramping a single row. Wraps `row` into a vector.
"""
RampRow(ta::TweezerArray, row::Int, final_amplitude::Float64, ramp_time::Float64; dt=nothing, downsample=nothing) =
    RampRow(ta, [row], final_amplitude, ramp_time, dt, downsample)

"""
    RampCol(tweezers, cols, final_amplitude, ramp_time)

Ramp the amplitude of one or more columns of tweezers simultaneously.

- `cols` can be a single integer or a vector of integers specifying the columns.
- `final_amplitude` is the value to ramp each column to (scalar or vector).
- `ramp_time` is the total duration of the ramp (seconds).
"""
struct RampCol <: AbstractInstruction
    tweezers::TweezerArray
    cols::AbstractVector{Int}
    final_amplitude::Union{Float64,AbstractVector{Float64}}
    ramp_time::Float64
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    RampCol(tweezers, col, final_amplitude, ramp_time)

Convenience constructor for ramping a single column. Wraps `col` into a vector.
"""
RampCol(ta::TweezerArray, col::Int, final_amplitude::Float64, ramp_time::Float64; dt=nothing, downsample=nothing) =
    RampCol(ta, [col], final_amplitude, ramp_time, dt, downsample)

"""
    AmplCol(tweezers, col, ampl)

Set the relative amplitude of a single tweezer column.

- `col` is the column index.
- `ampl` is the new relative amplitude.
"""
struct AmplCol <: AbstractInstruction
    tweezers::TweezerArray
    col::Int
    ampl::Float64
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    AmplCol(tweezers, col, ampl; dt=nothing, downsample=nothing)

Convenience constructor for AmplCol with optional per-instruction overrides.
"""
AmplCol(tweezers::TweezerArray, col::Int, ampl::Real; dt=nothing, downsample=nothing) =
    AmplCol(tweezers, col, Float64(ampl), dt, downsample)

"""
    AmplRow(tweezers, row, ampl)

Set the relative amplitude of a single tweezer row.

- `row` is the row index.
- `ampl` is the new relative amplitude.
"""
struct AmplRow <: AbstractInstruction
    tweezers::TweezerArray
    row::Int
    ampl::Float64
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    AmplRow(tweezers, row, ampl; dt=nothing, downsample=nothing)

Convenience constructor for AmplRow with optional per-instruction overrides.
"""
AmplRow(tweezers::TweezerArray, row::Int, ampl::Real; dt=nothing, downsample=nothing) =
    AmplRow(tweezers, row, Float64(ampl), dt, downsample)

"""
    FreqCol(tweezers, col, freq)

Set the frequency of a single tweezer column.

- `col` is the column index.
- `freq` is the absolute frequency (Hz).
"""
struct FreqCol <: AbstractInstruction
    tweezers::TweezerArray
    col::Int
    freq::Float64
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    FreqCol(tweezers, col, freq; dt=nothing, downsample=nothing)

Convenience constructor for FreqCol with optional per-instruction overrides.
"""
FreqCol(tweezers::TweezerArray, col::Int, freq::Real; dt=nothing, downsample=nothing) =
    FreqCol(tweezers, col, Float64(freq), dt, downsample)

"""
    FreqRow(tweezers, row, freq)

Set the frequency of a single tweezer row.

- `row` is the row index.
- `freq` is the absolute frequency (Hz).
"""
struct FreqRow <: AbstractInstruction
    tweezers::TweezerArray
    row::Int
    freq::Float64
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    FreqRow(tweezers, row, freq; dt=nothing, downsample=nothing)

Convenience constructor for FreqRow with optional per-instruction overrides.
"""
FreqRow(tweezers::TweezerArray, row::Int, freq::Real; dt=nothing, downsample=nothing) =
    FreqRow(tweezers, row, Float64(freq), dt, downsample)

"""
    Parallel(parts::Vector{<:AbstractInstruction})

Group several instructions into a single instruction that is executed
in parallel.

Each sub-instruction is compiled with the same time interval. The
resulting `Parallel` segment:

- Uses a common time span from the earliest start to the latest
  end time among all sub-instructions.
- Concatenates all emitted modifiers from the sub-instructions.
- Relies on each modifier's `update!` method being a no-op once the
  modifier has exhausted its internal data, so shorter instructions
  naturally hold their final value while longer ones continue.

This effectively "pads" shorter instructions in time up to the longest
instruction without modifying the modifiers themselves.
"""
struct Parallel <: AbstractInstruction
    parts::Vector{AbstractInstruction}
    dt::Union{Float64, Nothing}
    downsample::Union{Int, Nothing}
end

"""
    Parallel(parts; dt=nothing, downsample=nothing)

Convenience constructor for Parallel with optional per-instruction overrides.
"""
Parallel(parts::AbstractVector{<:AbstractInstruction}; dt=nothing, downsample=nothing) =
    Parallel(AbstractInstruction[parts...], dt, downsample)
