"""
Mark system components and controls as parametric using `Parameter`
and `ParametricExpression` so they can be varied at play time.

A `Parameter` represents a named scalar with a default (mean) value and,
optionally, a standard deviation for static disorder. Parameters can be
combined by arithmetic into `ParametricExpression` objects, which encode
symbolic expressions such as `2Ω + δ` without committing to concrete
numerical values.

Parameter objects appear in node constructors (`add_coupling!`, `add_detuning!`,
etc.) and are resolved at `play` time via the DAG compile/recompile lifecycle.
This enables:

- clean separation between model structure and parameter sampling,
- efficient parameter sweeps over grids of values, and
- Monte‑Carlo averaging over static noise (random draws of disordered
  couplings, detunings, etc.) without rebuilding the system definition.
"""

"""
    Parameter(name::Symbol, default; std = 0.0)

Scalar simulation parameter with an optional static-noise standard deviation.

`Parameter` stores a symbolic `name`, a `default` (mean) value and a
standard deviation `std`. The default `std = 0.0` corresponds to a
deterministic, noise‑free parameter; nonzero `std` values are typically
used to draw static random offsets for disorder or calibration errors.
switchable=true indicates that this parameter can be turned off for 
error budget calculations.

Parameters can be combined arithmetically (e.g. `2*Ω + δ`) to build
`ParametricExpression`s which are resolved to concrete numbers at
run time for each noise realization or sweep point.
"""
struct Parameter{T}
    name::Symbol
    default::T        # Mean value
    std::Float64      # Standard deviation (default 0 for deterministic)
    switchable::Bool  # Is parameter switchable
    Parameter(name::Symbol, default::T; std=0.0, switchable=false) where T = new{T}(name, default, std, switchable)
end

"""
    ParametricExpression

Symbolic expression tree built from `Parameter`s, numeric literals, and
basic arithmetic operations.

A `ParametricExpression` pairs an operator symbol (e.g. `:+`, `:*`, `:abs`)
with a tuple of argument nodes, which can themselves be `Parameter`s,
`ParametricExpression`s, or plain numbers. These expressions are not
evaluated immediately; instead they are resolved later given a concrete
dictionary of parameter values.
"""
struct ParametricExpression
    op::Symbol
    args::Tuple
end

#=============================================================================
ARITHMETIC OPERATIONS
=============================================================================#

"""
    Parameter / ParametricExpression arithmetic

Overloads `+`, `*`, `abs`, and `isless` so that combining `Parameter`s
with numbers or with other `Parameter` / `ParametricExpression` objects
builds new `ParametricExpression` trees instead of evaluating immediately.

These expression trees are later evaluated by the parameter-resolution
machinery, enabling concise definitions such as
```@example
Ω = Parameter(:Omega, 2π*1.0e6; std = 0.05e6)
δ = Parameter(:delta, 0.0; std = 0.1e6)
amp = 0.5 * Ω + δ
```
which can then be used in Hamiltonian terms or controls and sampled
for parametric sweeps or static-noise averaging.
"""

"""
    update!(obj, ::Val{name}, val)

Apply a resolved parameter value `val` to `obj` for the parameter named `name`.

Define methods of this function for each object type that accepts parameters.
Called by `compile` and `recompile!` for each registered parameter binding.
"""
function update! end

"""
    _apply_op(op, args)

Evaluate one `ParametricExpression` node on its resolved arguments. The single
table of operations shared by every resolver (build time, compile time, beams
and instructions), so an expression that builds also resolves.
"""
function _apply_op(op::Symbol, args)
    op === :* && return prod(args)
    op === :+ && return sum(args)
    op === :- && return length(args) == 1 ? -args[1] : args[1] - args[2]
    op === :inv && return inv(args[1])
    op === :abs && return abs(args[1])
    error("Unsupported op in ParametricExpression: $op. Supported ops: :+, :-, :*, :inv, :abs.")
end

Base.:-(p::Parameter)              = ParametricExpression(:-, (p,))
Base.:-(expr::ParametricExpression) = ParametricExpression(:-, (expr,))

Base.:*(p::Parameter, x::Number) = ParametricExpression(:*, (p, x))
Base.:*(x::Number, p::Parameter) = ParametricExpression(:*, (x, p))
Base.:*(p1::Parameter, p2::Parameter) = ParametricExpression(:*, (p1, p2))
Base.:+(p::Parameter, x::Number) = ParametricExpression(:+, (p, x))
Base.:+(x::Number, p::Parameter) = ParametricExpression(:+, (x, p))
Base.:+(p1::Parameter, p2::Parameter) = ParametricExpression(:+, (p1, p2))
Base.:-(p::Parameter, x::Number) = ParametricExpression(:-, (p, x))
Base.:-(x::Number, p::Parameter) = ParametricExpression(:-, (x, p))
Base.:-(p1::Parameter, p2::Parameter) = ParametricExpression(:-, (p1, p2))


# ParametricExpression arithmetic
Base.:*(p::Parameter, expr::ParametricExpression) = ParametricExpression(:*, (p, expr))
Base.:*(expr::ParametricExpression, p::Parameter) = ParametricExpression(:*, (expr, p))
Base.:*(x::Number, expr::ParametricExpression) = ParametricExpression(:*, (x, expr))
Base.:*(expr::ParametricExpression, x::Number) = ParametricExpression(:*, (expr, x))
Base.:*(expr1::ParametricExpression, expr2::ParametricExpression) = ParametricExpression(:*, (expr1, expr2))
Base.:+(p::Parameter, expr::ParametricExpression) = ParametricExpression(:+, (p, expr))
Base.:+(expr::ParametricExpression, p::Parameter) = ParametricExpression(:+, (expr, p))
Base.:+(x::Number, expr::ParametricExpression) = ParametricExpression(:+, (x, expr))
Base.:+(expr::ParametricExpression, x::Number) = ParametricExpression(:+, (expr, x))
Base.:+(expr1::ParametricExpression, expr2::ParametricExpression) = ParametricExpression(:+, (expr1, expr2))

Base.:-(p::Parameter, expr::ParametricExpression) = ParametricExpression(:-, (p, expr))
Base.:-(expr::ParametricExpression, p::Parameter) = ParametricExpression(:-, (expr, p))
Base.:-(x::Number, expr::ParametricExpression) = ParametricExpression(:-, (x, expr))
Base.:-(expr::ParametricExpression, x::Number) = ParametricExpression(:-, (expr, x))
Base.:-(expr1::ParametricExpression, expr2::ParametricExpression) = ParametricExpression(:-, (expr1, expr2))

Base.:/(p::Parameter, x::Number)              = ParametricExpression(:*, (p, 1/x))
Base.:/(x::Number, p::Parameter)              = ParametricExpression(:*, (x, inv(p)))
Base.:/(expr::ParametricExpression, x::Number) = ParametricExpression(:*, (expr, 1/x))
Base.:/(x::Number, expr::ParametricExpression) = ParametricExpression(:*, (x, inv(expr)))
Base.:/(e1::ParametricExpression, e2::ParametricExpression) = ParametricExpression(:*, (e1, inv(e2)))
Base.:/(p1::Parameter, p2::Parameter)             = ParametricExpression(:*, (p1, inv(p2)))
Base.:/(p::Parameter, expr::ParametricExpression) = ParametricExpression(:*, (p, inv(expr)))
Base.:/(expr::ParametricExpression, p::Parameter) = ParametricExpression(:*, (expr, inv(p)))

Base.:inv(p::Parameter)              = ParametricExpression(:inv, (p,))
Base.:inv(expr::ParametricExpression) = ParametricExpression(:inv, (expr,))

Base.:abs(expr::ParametricExpression) = ParametricExpression(:abs, (expr,))
Base.isless(::ParametricExpression, ::Number) = false  # treat as > any concrete threshold
Base.isless(::Number, ::ParametricExpression) = true   # treat as > any concrete threshold

