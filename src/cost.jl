"""
    CostFunction(distance, constraints)

Defines a cost function used for optimization separated into
two parts, distance measure between quantum states, and 
optional constraints on the shapes of pulses.
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
