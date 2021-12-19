mutable struct QOCProblem{T<:Real}
    operators::Vector{AbstractMatrix{Complex{T}}}
    drives::Function
    transform::Transform
    tspan::Tuple{T,T}
    cost::CostFunction

    function QOCProblem(
        ops::Vector{op},
        drives::Function,
        transform::Transform,
        tspan::Tuple{T,T},
        cost::CostFunction,
    ) where {T<:Real,op<:AbstractOperator}

        p = new{T}()
        p.operators = [op.data for op in ops]
        p.drives = drives
        p.transform = transform
        p.tspan = tspan
        p.cost = cost
        p

    end

end

function CuQOCProblem(prob::QOCProblem{T}) where {T<:Real}
    cu_ops = [sparse(CuArray(op)) for op in prob.operators]
    cu_trans = CuTransform(prob.transform)
    QOCProblem(cu_ops, prob.drives, cu_trans, prob.tspan, prob.cost)
end
