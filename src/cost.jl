"""
    CostFunction(distance, constraints)

Defines a cost function used for optimization.
# Arguments

 `distance` denotes a distance measure between quantum states, it should be a real valued function
for e.g. `d(x,y) = 1 - real(x'*y)` where `x` and `y` are two complex valued vectors.

 `constraints` (optional) denotes the constraints on the shapes of pulses, it should be a real valued function

NOTE: During the optimization we minimize the total cost given by the sum of average distance (considering all states in the transform) evaluated for a given set of parameters and the constraints. i.e. `d(x,y) + constraints(params)`.
"""
mutable struct CostFunction
    distance::Function
    constraints::Union{Function,Nothing}
    function CostFunction(
        distance::Function,
        constraints::Union{Function,Nothing} = nothing,
    )
        new(distance, constraints)
    end
end
