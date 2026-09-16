# Shared plumbing for the solver benchmarks in this directory.
#
# Each benchmark reports, per case: wall time (best of `reps`), allocated bytes,
# and an accuracy figure against a reference. The refactor's contract is that
# none of the three moves — so the output is deliberately a flat, diffable table
# rather than anything pretty.
#
# Timing note: every case is run once before it is measured. Without that the
# first call pays AtomTwin's compilation and the number means nothing -- and the
# failure mode is not a uniformly slow run but a single inflated case, which
# reads exactly like a real anomaly. A 6.2 s "pathology" chased during the
# writing of these benchmarks was precisely this: the first case in a freshly
# restarted daemon, absorbing compilation that every later case was spared.
# If one case looks anomalous, re-run it on its own before believing it.

using Printf

"""
    bench(f; reps = 3) -> (best, bytes, value)

Run `f()` once to warm up, then `reps` more times, returning the best wall time,
the allocation count of a measured run, and the last returned value.

Best-of rather than mean: we are measuring the work the solver does, and a
slower run only ever means the machine was doing something else too.
"""
function bench(f; reps::Int = 3)
    value = f()                                  # warm up; result discarded
    best  = Inf
    bytes = 0
    for _ in 1:reps
        GC.gc()
        t0 = time_ns()
        b  = @allocated (value = f())
        dt = (time_ns() - t0) / 1e9
        dt < best && (best = dt)
        bytes = b
    end
    return (best = best, bytes = bytes, value = value)
end

"""
    relerr(a, b) -> Float64

Max absolute difference between two trajectories, normalised by the reference's
peak magnitude. Populations live on [0,1] and the peak is O(1), so this is an
absolute error in practice — which is what we want, since a relative error on a
population that passes through zero is meaningless.
"""
function relerr(a::AbstractVector, b::AbstractVector)
    n = min(length(a), length(b))
    n == 0 && return NaN
    scale = max(maximum(abs, view(b, 1:n)), 1e-12)
    return maximum(abs.(view(a, 1:n) .- view(b, 1:n))) / scale
end

"""
    report(title, rows)

Print one benchmark's results. `rows` are `(name, best, bytes, err, errlabel)`
tuples; `err` may be `nothing` where a case has no reference.
"""
function report(title::AbstractString, rows)
    println()
    println(title)
    println("-"^78)
    @printf("%-34s %9s %12s   %s\n", "case", "best[s]", "alloc", "accuracy")
    for (name, best, bytes, err, errlabel) in rows
        e = err === nothing ? "" : @sprintf("%.2e  (%s)", err, errlabel)
        @printf("%-34s %9.4f %12s   %s\n", name, best, _hsize(bytes), e)
    end
    println("-"^78)
end

function _hsize(b::Integer)
    b < 1024        && return string(b, " B")
    b < 1024^2      && return @sprintf("%.1f KiB", b / 1024)
    b < 1024^3      && return @sprintf("%.1f MiB", b / 1024^2)
    return @sprintf("%.2f GiB", b / 1024^3)
end

"""
    checkbudget(total, limit = 10.0)

Warn when a benchmark exceeds its time budget. These exist to be run often, so
one drifting past ten seconds is a defect in the benchmark, not a finding.
"""
function checkbudget(total::Float64, limit::Float64 = 10.0)
    @printf("total measured: %.2f s (budget %.0f s)%s\n",
            total, limit, total > limit ? "  ** OVER BUDGET **" : "")
end
