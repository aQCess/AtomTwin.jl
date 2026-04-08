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
    @assert duration > 0 "duration must be positive, got $duration"
    @assert length(displacement) == 3 "displacement must be 3D vector"

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
    compile(atoms, inst, dt; resolve_target = identity) -> (modifiers, tspan)

Lower an inert instruction spec `inst` into concrete modifiers and time spans
for a single instruction.

Common contract:

- Inputs:
  - `atoms`: current atom state (vector or model-specific container)
  - `inst`: instruction spec (e.g. `MoveCol`, `Wait`, `RampCol`, `Pulse`, …)
  - `dt`: time step (seconds, `Float64`)
  - `resolve_target`: function that binds objects captured by specs to runtime instances
- Output:
  - `(modifiers, tspan)` where `modifiers` is a vector of per-segment modifiers
    and `tspan` is the segment time vector
- Time convention:
  - `tspan` is local to the segment, typically `collect(dt:dt:duration)`
  - Downstream code accumulates absolute time across segments

- Notes:
  - `Off`, `Pulse` instructions set the amplitude of the target coupling to 0.0 at the end of the instruction
"""
function compile(atoms, inst, dt; resolve_target = identity)
    throw(ArgumentError(
        "No compile method for instruction type $(typeof(inst)). " *
        "Supported types: Wait, Pulse, On, Off, Parallel, MoveRow, MoveCol, " *
        "RampRow, RampCol, AmplRow, AmplCol, FreqRow, FreqCol."
    ))
end

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
    return move(atoms, tweezers_in_row(ta, inst.rows), displacement, inst.duration, inst.sweep, dt)
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
    return move(atoms, tweezers_in_col(ta, inst.cols), displacement, inst.duration, inst.sweep, dt)
end

"""
    compile(atoms, inst::Wait, dt; resolve_target = identity)

Lower a `Wait` instruction into an idle time segment. No modifiers are
produced.
"""
function compile(atoms, inst::Wait, dt; resolve_target=identity)

    tsteps = Int(div(inst.duration,dt)+1)

    return AbstractModifier[], tsteps
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

    # If a single amplitude is specified, use it for all; if a vector, use as-is
    amplitudes_final = inst.final_amplitude isa Number ? fill(inst.final_amplitude, nbeams) : inst.final_amplitude
    ramp_time = inst.ramp_time
    return ramp(beams, amplitudes_final, ramp_time, dt)
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

    # If a single amplitude is specified, use it for all; if a vector, use as-is
    amplitudes_final = inst.final_amplitude isa Number ? fill(inst.final_amplitude, nbeams) : inst.final_amplitude
    ramp_time = inst.ramp_time
    return ramp(beams, amplitudes_final, ramp_time, dt)
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

    # Loop over all rows for this column
    for row in eachindex(ta.row_amplitudes)
        beam = ta[row, col]
        ampl = ta.row_amplitudes[row] * ta.col_amplitudes[col]
        push!(modifiers, AmplitudeModifier(beam, [ampl, ampl]))
    end

    return modifiers, 2
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

    # Loop over all columns for this row
    for col in eachindex(ta.col_amplitudes)
        beam = ta[row, col]
        ampl = ta.row_amplitudes[row] * ta.col_amplitudes[col]
        push!(modifiers, AmplitudeModifier(beam, [ampl, ampl]))
    end

    return modifiers, 2
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
            [[ta.dx * inst.freq, beam.r0[2], beam.r0[3]],  # New position
             [ta.dx * inst.freq, beam.r0[2], beam.r0[3]]],  # Same (instantaneous)
            [0.0, dt];
            dims = [1]  # Only update x-coordinate
        )
        for beam in tweezers
    ]
    
    return modifiers, 2
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
            [[beam.r0[1], ta.dy * inst.freq, beam.r0[3]],  # New position
             [beam.r0[1], ta.dy * inst.freq, beam.r0[3]]],  # Same (instantaneous)
            [0.0, dt];
            dims = [2]  # Only update y-coordinate
        )
        for beam in tweezers
    ]
    
    return modifiers, 2
end

#-----------------------------------------------------------------------------
# Compile: pulse and simple on/off
#-----------------------------------------------------------------------------

# Build a ResetModifier targeting the same coefficient reference as an AmplitudeModifier would.
_reset_modifier(c::GaussianCoupling) = ResetModifier(c._amplitude)
_reset_modifier(c) = ResetModifier(c)

function compile(atoms, inst::Pulse, dt; resolve_target = identity)
    resolved_couplings = [resolve_target(c) for c in inst.couplings]

    tsteps = round(Int, inst.duration / dt)
    amplitude_vals = ComplexF64[]

    if isempty(inst.amplitudes)
        amplitude_vals = fill(inst.ampl, tsteps)
    else
        # Interpolate shaped pulse on [0, duration]; sample at tsteps start-of-step times.
        # Compute tsteps+1 points (grid [0, dt, ..., duration]) then drop the endpoint —
        # end_instruction! zeros the field after the loop, so no trailing zero is needed.
        scaled = inst.ampl .* inst.amplitudes
        amplitude_vals = interpolate(scaled, collect(0.0:dt:inst.duration))[1:tsteps]
    end

    modifiers = AbstractModifier[AmplitudeModifier(c, amplitude_vals) for c in resolved_couplings]
    append!(modifiers, [_reset_modifier(c) for c in resolved_couplings])
    return modifiers, tsteps
end

"""
    compile(atoms, inst::On, dt; resolve_target = identity)

Lower an `On` instruction into a single-step amplitude modifier that
sets the amplitude of the given couplings to 1 at `t = 0`.
"""
function compile(atoms, inst::On, dt; resolve_target = identity)
   
    # compile should only act on already resolved targets
    resolved_couplings = [resolve_target(coupling) for coupling in inst.couplings]

    # Create amplitude modifier at t=0
    modifiers = AmplitudeModifier[AmplitudeModifier(coupling, [1.0,1.0]) for coupling in resolved_couplings]

    return modifiers, 2
end

"""
    compile(atoms, inst::Off, dt; resolve_target = identity)

Lower an `Off` instruction into a single-step amplitude modifier that
sets the amplitude of the given couplings to 0 at `t = 0`.
"""
function compile(atoms, inst::Off, dt; resolve_target = identity)
    
    # Apply resolve_target to each coupling individually  
    resolved_couplings = [resolve_target(coupling) for coupling in inst.couplings]
    
    # Create amplitude modifier at t=0
    modifiers = AmplitudeModifier[AmplitudeModifier(coupling, [0.0,0.0]) for coupling in resolved_couplings]
    
    return modifiers, 2
end

"""
    compile(atoms, inst::Parallel, dt; resolve_target = identity)

Compile several instructions to execute in parallel over a common time
interval equal to the longest internal instruction. Updates that exceed
`length(m.vals)` for modifier `m` are effectively ignored by the modifier.
"""
function compile(atoms, inst::Parallel, dt; resolve_target = identity)
    # Compile all child instructions
    compiled = map(inst.parts) do subinst
        compile(atoms, subinst, dt; resolve_target = resolve_target)
    end

    mods_list  = first.(compiled)
    n_steps = maximum(last.(compiled))

    # Just concatenate all modifiers; shorter ones naturally stop updating
    all_mods = reduce(vcat, mods_list)

    return all_mods, n_steps
end