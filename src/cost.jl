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
