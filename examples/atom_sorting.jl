# # Atom Sorting
#
# In this example we simulate an atom sorting protocol using two tweezer
# arrays.
#
# A two-dimensional array is loaded with a particular arrangement of 
# atoms defined by an occupancy matrix. A second one-dimensional tweezer
# array is used to transfer atoms row-by-row into a dense arrangement.
# The final state is a dense 5×5 atom array.


using AtomTwin
# Explicit import: a blanket `using AtomTwin.Units` puts e, g and G into Main,
# and runtests.jl includes every file into the SAME Main — which then breaks
# every example using the `g, e = Level(...)` idiom. Everything except those
# three names is safe to bring in.
using AtomTwin.Units: MHz, kHz, GHz, Hz, s, ms, µs, ns, m, cm, mm, µm, nm, mW, µW, W, µK, mK, nK, K, hbar, kb, c, a0, amu
using GLMakie
using AtomTwin.Visualization: animate

# ## Parameters

occ =  [0  1  0  1  0  1  1  0  0  1;
        0  0  1  1  1  0  0  1  1  0;
        1  0  0  0  1  0  0  1  1  1;
        0  1  1  0  1  0  1  0  1  0;
        1  0  0  1  0  1  1  1  0  0] # initial occupancy matrix

temperature = 5e-6

dt     = 0.50e-6     # output grid: also quantises instruction durations, so it
                     # cannot simply be raised to the animation frame interval
T_sort = 100e-6      # sorting time within each row (horizontal)


# ## System definition
#
# `static` is a static 2D array holding the atoms.
# `dynamic` is a 1D array used as a movable “conveyor belt”.

# 10x5-site tweezer array
static = TweezerArray(
    λ         = 759nm,        
    w0        = 1.0µm,        
    P_total   = 500mW,         
    row_freqs = [-2:2...]*MHz,
    col_freqs = [-4.5:4.5...]*MHz,
    dx        = 3µm/MHz,      
    dy        = 3µm/MHz,
)

# 10x1-site tweezer array
dynamic = TweezerArray(
    λ         = 759nm,        
    w0        = 1.0µm,        
    P_total   = 100mW,         
    row_freqs = [-2.0]*MHz,   # single dynamic row
    col_freqs = [-4.5:4.5...]*MHz,
    dx        = 3µm/MHz,      
    dy        = 3µm/MHz,
)

# Place atoms according to occupancy matrix with a thermal velocity distribution
atoms = Ytterbium171Atom[]
for r in 1:size(occ,1)
    for c in 1:size(occ,2)
        if occ[r,c] == 1
            push!(atoms, Ytterbium171Atom(;
                levels = [Level("1S0")],
                x_init = getposition(static[r,c]),
                v_init = maxwellboltzmann(T = temperature)))
        end
    end
end

system = System(atoms, [static, dynamic])

# ## Detectors

for (i, atom) in enumerate(atoms)
    add_detector!(system, MotionDetectorSpec(atom; dims=[1,2], name = "atom_$i"))
end
# Traps are switched on and off during the sort, so record each beam's
# amplitude alongside its position. `animate` looks for the "<name>_ampl"
# companion detector and hides a trap in frames where it is off -- otherwise
# every trap is drawn at all times, including ones that do not currently exist.
for (i, beam) in enumerate(static)
    add_detector!(system, MotionDetectorSpec(beam; dims=[1,2], name = "static_$i"))
    add_detector!(system, FieldDetectorSpec(beam; name = "static_$(i)_ampl"))
end
for (i, beam) in enumerate(dynamic)
    add_detector!(system, MotionDetectorSpec(beam; dims=[1,2], name = "dynamic_$i"))
    add_detector!(system, FieldDetectorSpec(beam; name = "dynamic_$(i)_ampl"))
end

# ## Build sequence
#
# We bypass @sequence and construct the sequence procedurally.

# Helper functions are defined in a `let` block to keep them local to this
# script and avoid method-redefinition warnings when the script is run
# multiple times.
seq = let
    """
        row_moves(occ_row)

    From a 0/1 occupancy row, return (col, Δcol) moves that left-compress
    all occupied sites into contiguous columns starting at 1.
    """
    function row_moves(occ_row)
        src = findall(!iszero, occ_row)   # occupied columns
        k   = length(src)
        tgt = 1:k                         # target columns 1..k
        [(src[i], tgt[i] - src[i]) for i in eachindex(src)]
    end

    """
        sort_row!(seq, static, dynamic, row, occ_row;
                  sort_duration=T_sort)

    Sort static row `row` using the dynamic row:
    1. load static → dynamic
    2. horizontal MoveCol sorting moves
    3. hand back to static
    4. reset dynamic columns with FreqCol.
    """
    function sort_row!(seq, static, dynamic, row, occ_row;
                       sort_duration = T_sort)

        ncols = length(occ_row)

        "Step 1: load static row into dynamic row"
        for col in 1:ncols
            push!(seq, AmplCol(dynamic, col, occ_row[col]))
        end
        push!(seq, AmplRow(static,  row, 0.0))
        push!(seq, AmplRow(dynamic, 1,   1.0))

        "Step 2: horizontal sorting moves: Δcol from occupancy"
        moves = row_moves(occ_row)
        if !isempty(moves)
            push!(seq, Parallel([
                MoveCol(dynamic,
                        col,
                        Δcol * 1MHz,
                        sort_duration)
                for (col, Δcol) in moves
            ]))
        end

        "Step 3: hand atoms back to static row"
        push!(seq, AmplRow(static,  row, 1.0))
        push!(seq, AmplRow(dynamic, 1,   0.0))

        "Step 4: reset dynamic columns to their original frequencies using FreqCol"
        if !isempty(moves)
            push!(seq, Parallel([
                FreqCol(dynamic,
                        col,
                        dynamic.col_freqs[col])
                for (col, Δcol) in moves
            ]))
        end
    end

    """
        build_sort_sequence_from_occ(occ, static, dynamic, dt; row_step=1MHz)

    Build a row-by-row sorting sequence from `occ` using a single movable
    dynamic row. Uses FreqRow for vertical jumps, MoveCol for horizontal moves.
    """
    function build_sort_sequence_from_occ(occ, static, dynamic, dt;
                                          row_step = 1MHz)

        nrows, _ = size(occ)

        # `dt` is the output grid, not the accuracy knob -- the solver picks its
        # own sub-steps from `tol`. Without it a sequence records one sample per
        # instruction and the traps jump between frames.
        #
        # It cannot be set straight to the animation's frame interval, though:
        # instruction durations are rounded up to whole multiples of `dt`, and
        # the sort is built from many short `Wait`s and moves. Measured, raising
        # `dt` from 0.5 µs to the 20 µs frame interval padded the sequence from
        # 1.083 ms to 4.300 ms -- a different protocol, not a coarser recording
        # of the same one. So `dt` stays fine and the animation strides down.
        seq = Sequence(dt; tol = 1e-4)
        push!(seq, AmplRow(dynamic, 1, 0.0))
        push!(seq, Wait(0.25ms))

        current_static_row = 1  # which static row dynamic row 1 is aligned to

        for target_row in 1:nrows
            "Jump dynamic row 1 vertically to align with `target_row`"
            if target_row != current_static_row
                target_freq = dynamic.row_freqs[1] + (target_row - 1) * row_step
                push!(seq, FreqRow(dynamic, 1, target_freq))
                current_static_row = target_row
            end

            occ_row = view(occ, target_row, :)
            sort_row!(seq, static, dynamic, target_row, occ_row)
        end

        push!(seq, Wait(0.25ms))
        return seq
    end

    build_sort_sequence_from_occ(occ, static, dynamic, dt)
end


# ## Run simulations
#

out = play(system, seq)

# ## Plot results
#
tlist = out.times

# Order matters: groups are drawn in the order declared here, so later entries
# are painted on top. Static traps are the background, then the moving dynamic
# traps, then the atoms. (A Dict would not preserve this -- its key order is
# arbitrary, which previously let the large static markers hide the dynamic
# ones.)
plot_options = [
           "static"  => (marker=:rect, strokecolor=:gray, strokewidth=1,
                            color=:transparent, markersize=20),
           "dynamic" => (marker=:rect, strokecolor=:red, strokewidth=1,
                            color=:transparent, markersize=12),
           "atom"    => (marker=:circle, color=:black),
       ]

# Record the sorting sequence to a GIF. Pass `gifname = nothing` instead to
# animate interactively in a window (requires a display).
animate(out;
        options   = plot_options,
        limits    = ((-20, 20), (-12, 12)),
        gifname   = "atom_sorting.gif",
        framerate = 30,
        stride    = 40,)   # one frame per 20 µs of sort; see the `dt` note above
                           # for why the recording is finer than the animation
