"""
Compile high-level instructions into low-level Dynamiq modifiers for simulation.
"""

"""
    BUILTIN_SWEEPS

Dictionary of built-in sweep profiles for motion instructions.

Supported keys:

- `:linear`    – linear sweep `s ↦ s`
- `:min_jerk`  – minimum-jerk profile `10s^3 - 15s^4 + 6s^5`
- `:cosine`    – smooth cosine-based profile
"""
const BUILTIN_SWEEPS = Dict{Symbol, Function}(
    :linear    => s -> s,
    :min_jerk  => s -> 10*s^3 - 15*s^4 + 6*s^5,
    :cosine    => s -> 1 - cos(π/2*s)^2,
    # Add more here if desired
)

#-----------------------------------------------------------------------------
# Generic motion and ramp helpers
#-----------------------------------------------------------------------------

"""
    move(atoms, beams, displacement, duration, sweep, dt) -> (moves, nsteps)

Helper to create a scheduled displacement for `beams`.

Arguments:

- `atoms`: atom collection (used by modifiers if needed)
- `beams`: collection of beam objects to move
- `displacement`: 3-vector total displacement
- `duration`: total time for the move (seconds)
- `sweep`: motion profile, either a built-in symbol (`:linear`, `:min_jerk`, `:cosine`)
  or a custom function `s -> f(s)` mapping `[0,1] → [0,1]`
- `dt`: time step (seconds)

Returns a tuple `(moves, nsteps)` where `moves` is a vector of `MoveModifier`
objects and `nsteps` is the number of steps for the move segment.
"""
function move(atoms, beams, displacement, duration, sweep, dt)
    @assert duration ≥ 0 "duration must be non-negative, got $duration"
    @assert length(displacement) == 3 "displacement must be 3D vector"
    duration == 0.0 && return (MoveModifier[], 0)

    # Resolve user input:
    if sweep isa Symbol && haskey(BUILTIN_SWEEPS, sweep)
        schedule = BUILTIN_SWEEPS[sweep]
    elseif sweep isa Function
        if abs(sweep(0.0)) > 1e-10 || abs(sweep(1.0) - 1.0) > 1e-10
            @warn "Custom sweep function should map 0 → 0 and 1 → 1. Got f(0)=$(sweep(0.0)), f(1)=$(sweep(1.0))"
        end
        schedule = sweep
    else
        error("sweep must be a built-in Symbol ($(collect(keys(BUILTIN_SWEEPS))) ) or a function s->f(s)")
    end

    # Time grid: 0, dt, ..., duration
    tspan = collect(0.0:dt:duration)
    tspan = length(tspan) ≥ 2 ? tspan : [0.0, duration]  # ensure at least two points

    moves = MoveModifier[MoveModifier(beam, displacement, tspan; schedule = schedule) for beam in beams]
    return moves, length(tspan)
end


"""
    ramp(beams, amplitudes_final, ramp_time, dt) -> (ramps, tspan)

Linearly ramp each beam in `beams` from its current amplitude to the
corresponding value in `amplitudes_final` over `ramp_time` seconds.

Returns `(ramps, nsteps)`, where `ramps` is a vector of `AmplitudeModifier`
and `nsteps` is the number of time steps.
"""
function ramp(beams, amplitudes_final, ramp_time, dt)
    tspan = collect(dt:dt:ramp_time)
    nsteps = length(tspan)

    ramps = AmplitudeModifier[
        AmplitudeModifier(
            beam,
            # Linear interpolation from initial to final amplitude
            collect(range(beam._coeff[], stop=amp_final, length=nsteps))
        )
        for (beam, amp_final) in zip(beams, amplitudes_final)
    ]

    return ramps, nsteps
end

#-----------------------------------------------------------------------------
# Compile: common contract
#-----------------------------------------------------------------------------

"""
    compile(atoms, inst, dt; resolve_target = identity) -> (modifiers, boundary_modifiers, n_steps)

Lower an inert instruction spec `inst` into concrete modifiers and a step count.

- `modifiers`: `Vector{AbstractModifier}` — passed to `evolve!`, called every solver timestep
- `boundary_modifiers`: `Vector{AbstractBoundaryModifier}` — called once before/after `evolve!`
- `n_steps`: number of solver timesteps for this instruction

`begin_instruction!(m)` fires on each boundary modifier before `evolve!`;
`end_instruction!(m)` fires after. This keeps coupling set/reset out of the inner loop.
"""
function compile(atoms, inst, dt; resolve_target = identity)
    throw(ArgumentError(
        "No compile method for instruction type $(typeof(inst)). " *
        "Supported types: Wait, Pulse, On, Off, Parallel, MoveRow, MoveCol, " *
        "RampRow, RampCol, AmplRow, AmplCol, FreqRow, FreqCol."
    ))
end

const _NO_BMODS = AbstractBoundaryModifier[]

#-----------------------------------------------------------------------------
# Compile: motion and waits
#-----------------------------------------------------------------------------

"""
    compile(atoms, inst::MoveRow, dt; resolve_target = identity)

Lower a `MoveRow` instruction into position modifiers that move tweezers
in the specified rows along `y` according to the chosen sweep profile.
"""
function compile(atoms, inst::MoveRow, dt; resolve_target = identity)
    ta = resolve_target(inst.tweezers)
    Δy = ta.dy * inst.delta
    displacement = [0.0, Δy, 0.0]
    mods, n = move(atoms, tweezers_in_row(ta, inst.rows), displacement, inst.duration, inst.sweep, dt)
    return mods, _NO_BMODS, n
end

"""
    compile(atoms, inst::MoveCol, dt; resolve_target = identity)

Lower a `MoveCol` instruction into position modifiers that move tweezers
in the specified columns along `x` according to the chosen sweep profile.
"""
function compile(atoms, inst::MoveCol, dt; resolve_target = identity)
    ta = resolve_target(inst.tweezers)
    Δx = ta.dx * inst.delta
    displacement = [Δx, 0.0, 0.0]
    mods, n = move(atoms, tweezers_in_col(ta, inst.cols), displacement, inst.duration, inst.sweep, dt)
    return mods, _NO_BMODS, n
end

"""
    compile(atoms, inst::Wait, dt; resolve_target = identity)

Lower a `Wait` instruction into an idle time segment. No modifiers are
produced.
"""
function compile(atoms, inst::Wait, dt; resolve_target=identity)
    tsteps = Int(div(inst.duration, dt) + 1)
    return AbstractModifier[], _NO_BMODS, tsteps
end

#-----------------------------------------------------------------------------
# Compile: ramps and single-step amplitude/frequency changes
#-----------------------------------------------------------------------------

"""
    compile(atoms, inst::RampRow, dt; resolve_target = identity)

Lower a `RampRow` instruction into amplitude modifiers that linearly ramp
the amplitudes of tweezers in the selected columns over `inst.ramp_time`.
"""
function compile(atoms, inst::RampRow, dt; resolve_target = identity)
    ta = resolve_target(inst.tweezers)
    beams = tweezers_in_row(ta, inst.rows)
    nbeams = length(beams)
    amplitudes_final = inst.final_amplitude isa Number ? fill(inst.final_amplitude, nbeams) : inst.final_amplitude
    mods, n = ramp(beams, amplitudes_final, inst.ramp_time, dt)
    return mods, _NO_BMODS, n
end

"""
    compile(atoms, inst::RampCol, dt; resolve_target = identity)

Lower a `RampCol` instruction into amplitude modifiers that linearly ramp
the amplitudes of tweezers in the selected columns over `inst.ramp_time`.
"""
function compile(atoms, inst::RampCol, dt; resolve_target = identity)
    ta = resolve_target(inst.tweezers)
    beams = tweezers_in_col(ta, inst.cols)
    nbeams = length(beams)
    amplitudes_final = inst.final_amplitude isa Number ? fill(inst.final_amplitude, nbeams) : inst.final_amplitude
    mods, n = ramp(beams, amplitudes_final, inst.ramp_time, dt)
    return mods, _NO_BMODS, n
end

"""
    compile(atoms, inst::AmplCol, dt; resolve_target = identity)

Lower an `AmplCol` instruction into single-step amplitude modifiers for
all tweezers in the given column, enforcing row–column factorization.
"""
function compile(atoms, inst::AmplCol, dt; resolve_target = identity)
    ta = resolve_target(inst.tweezers)
    ta.col_amplitudes[inst.col] = inst.ampl
    col = inst.col
    modifiers = AmplitudeModifier[]
    for row in eachindex(ta.row_amplitudes)
        beam = ta[row, col]
        ampl = ta.row_amplitudes[row] * ta.col_amplitudes[col]
        push!(modifiers, AmplitudeModifier(beam, [ampl, ampl]))
    end
    return modifiers, _NO_BMODS, 2
end

"""
    compile(atoms, inst::AmplRow, dt; resolve_target = identity)

Lower an `AmplRow` instruction into single-step amplitude modifiers for
all tweezers in the given row, enforcing row–column factorization.
"""
function compile(atoms, inst::AmplRow, dt; resolve_target = identity)
    ta = resolve_target(inst.tweezers)
    ta.row_amplitudes[inst.row] = inst.ampl
    row = inst.row
    modifiers = AmplitudeModifier[]
    for col in eachindex(ta.col_amplitudes)
        beam = ta[row, col]
        ampl = ta.row_amplitudes[row] * ta.col_amplitudes[col]
        push!(modifiers, AmplitudeModifier(beam, [ampl, ampl]))
    end
    return modifiers, _NO_BMODS, 2
end

"""
    compile(atoms, inst::FreqCol, dt; resolve_target = identity)

Lower a `FreqCol` instruction into position modifiers that update the
effective position (e.g. optical frequency) of all beams in the given column.
"""
function compile(atoms, inst::FreqCol, dt; resolve_target = identity)
    ta = resolve_target(inst.tweezers)
    tweezers = tweezers_in_col(ta, inst.col)
    modifiers = [
        PositionModifier(
            beam,
            [[ta.dx * inst.freq, beam.r0[2], beam.r0[3]],
             [ta.dx * inst.freq, beam.r0[2], beam.r0[3]]],
            [0.0, dt];
            dims = [1]
        )
        for beam in tweezers
    ]
    return modifiers, _NO_BMODS, 2
end

"""
    compile(atoms, inst::FreqRow, dt; resolve_target = identity)

Lower a `FreqRow` instruction into position modifiers that update the
effective position (e.g. optical frequency) of all beams in the given row.
"""
function compile(atoms, inst::FreqRow, dt; resolve_target = identity)
    ta = resolve_target(inst.tweezers)
    tweezers = tweezers_in_row(ta, inst.row)
    modifiers = [
        PositionModifier(
            beam,
            [[beam.r0[1], ta.dy * inst.freq, beam.r0[3]],
             [beam.r0[1], ta.dy * inst.freq, beam.r0[3]]],
            [0.0, dt];
            dims = [2]
        )
        for beam in tweezers
    ]
    return modifiers, _NO_BMODS, 2
end

#-----------------------------------------------------------------------------
# Compile: pulse and simple on/off
#-----------------------------------------------------------------------------

# Helpers: build boundary modifier targeting the same coefficient reference
_reset_modifier(c::GaussianCoupling) = ResetModifier(c._amplitude)
_reset_modifier(c) = ResetModifier(c)
_set_modifier(c::GaussianCoupling, val) = SetModifier(c._amplitude, ComplexF64(val))
_set_modifier(c, val) = SetModifier(c, ComplexF64(val))

"""
    compile(atoms, inst::Pulse, dt; resolve_target = identity)

Constant-amplitude pulse: no inner-loop modifiers; boundary SetModifier turns on
the coupling before `evolve!` and ResetModifier zeros it after.

Shaped-amplitude pulse: AmplitudeModifiers run in the solver loop; a ResetModifier
at the boundary zeros the field after the instruction completes.
"""
function compile(atoms, inst::Pulse, dt; resolve_target = identity)
    resolved_couplings = [resolve_target(c) for c in inst.couplings]
    tsteps = round(Int, inst.duration / dt)
    bmods = AbstractBoundaryModifier[_reset_modifier(c) for c in resolved_couplings]

    if isempty(inst.amplitudes)
        # Constant pulse: set amplitude at boundary, no per-step modifiers
        prepend!(bmods, [_set_modifier(c, inst.ampl) for c in resolved_couplings])
        return AbstractModifier[], bmods, tsteps
    else
        # Shaped pulse: resample amplitude envelope onto tsteps points
        scaled = inst.ampl .* inst.amplitudes
        amplitude_vals = if inst.interp == :piecewise_constant
            interpolate_piecewise_constant(scaled, tsteps)
        else
            interpolate(scaled, collect(0.0:dt:inst.duration))[1:tsteps]
        end
        modifiers = AbstractModifier[AmplitudeModifier(c, amplitude_vals) for c in resolved_couplings]
        return modifiers, bmods, tsteps
    end
end

"""
    compile(atoms, inst::On, dt; resolve_target = identity)

Turn on couplings at instruction boundary (SetModifier). No inner-loop modifiers.
Amplitude persists until explicitly turned off.
"""
function compile(atoms, inst::On, dt; resolve_target = identity)
    resolved_couplings = [resolve_target(c) for c in inst.couplings]
    bmods = AbstractBoundaryModifier[_set_modifier(c, 1.0) for c in resolved_couplings]
    return AbstractModifier[], bmods, 2
end

"""
    compile(atoms, inst::Off, dt; resolve_target = identity)

Zero coupling amplitudes at instruction boundary (ResetModifier). No inner-loop modifiers.
"""
function compile(atoms, inst::Off, dt; resolve_target = identity)
    resolved_couplings = [resolve_target(c) for c in inst.couplings]
    bmods = AbstractBoundaryModifier[_reset_modifier(c) for c in resolved_couplings]
    return AbstractModifier[], bmods, 2
end

"""
    compile(atoms, inst::Parallel, dt; resolve_target = identity)

Compile several instructions to execute in parallel over a common time
interval equal to the longest internal instruction. Updates that exceed
`length(m.vals)` for modifier `m` are effectively ignored by the modifier.
"""
function compile(atoms, inst::Parallel, dt; resolve_target = identity)
    compiled = map(inst.parts) do subinst
        compile(atoms, subinst, dt; resolve_target = resolve_target)
    end
    all_mods  = reduce(vcat, getindex.(compiled, 1))
    all_bmods = reduce(vcat, getindex.(compiled, 2))
    n_steps   = maximum(getindex.(compiled, 3))
    return all_mods, all_bmods, n_steps
end