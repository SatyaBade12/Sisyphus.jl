"""
    heaviside(t::T) where {T <: Real}

Heaviside step function

```math
H(t) = 
\\begin{cases}
    1, & t > 0 \\\\
    0, & t \\leq 0
\\end{cases}.
```
"""
heaviside(t::T) where {T <: Real} = T(t > T(0))

"""
    interval(t::T, a::T, b::T) where {T <: Real}

Interval function

```math
    I(t; a, b) = H(t - a) - H(t - b).
```
"""
interval(t::T, a::T, b::T) where {T <: Real} = heaviside(t - a) - heaviside(t - b)

"""
    piecewise_const_interp(p, t; t0, t1)

Piecewise constant interpolation of equidistant samples `p`.
"""
function piecewise_const_interp(p::Vector{T}, t::T; t0=T(0), t1=T(1)) where {T <: Real}
    ts = collect(t0:(t1 - t0)/length(p):t1)
    sum(p[i] * interval(t, ts[i], ts[i + 1]) for i=1:length(ts) - 1)
end

"""
    linear_interp(p, t; t0, t1)

Linear interpolation of equidistant samples `p`.
"""
function linear_interp(p::Vector{T}, t::T; t0=T(0), t1=T(1)) where {T <: Real}
    Δt = (t1 - t0)/(length(p) - 1)
    ts = collect(t0:Δt:t1)
    sum((p[i] + (t - ts[i]) * (p[i + 1] - p[i]) / Δt) * interval(t, ts[i], ts[i + 1]) for i=1:length(ts) - 1)
end

"""
    cubic_spline_interp(p, t; t0, t1)

Cubic spline interpolation of equidistant samples `p` with natural boundary conditions.
"""
function cubic_spline_interp(p::Vector{T}, t::T; t0=T(0), t1=T(1)) where {T <: Real}
    Δt = (t1 - t0)/(length(p) - 1)
    ts = collect(t0:Δt:t1)
    
    v = 4 * ones(T, length(p) - 2)
    h = ones(T, length(p) - 3)
    b = [p[i + 1] - p[i] for i=1:length(p) - 1] / Δt
    u = 6 * [b[i] - b[i - 1] for i=2:length(b)]
    z = vcat([T(0)], inv(SymTridiagonal(v, h)) * u / Δt, [T(0)])
    
    sum(interval(t, ts[i], ts[i + 1]) * (z[i + 1] * (t - ts[i])^3 + z[i] * (ts[i + 1] - t)^3 + 
        (6 * p[i + 1] - z[i + 1] * Δt^2) * (t - ts[i]) + (6 * p[i] - z[i] * Δt^2) * (ts[i + 1] - t)) / (6Δt) for i=1:length(ts) - 1)
end