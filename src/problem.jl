mutable struct QOCProblem{T<:Real}
    operators::Tuple{AbstractMatrix{Complex{T}}, Vector{AbstractMatrix{Complex{T}}}}
    drives::Function
    transform::Transform
    tspan::Tuple{T,T}
    cost::CostFunction

    function QOCProblem(
        operators::Tuple{op, Vector{op}},
        drives::Function,
        transform::Transform,
        tspan::Tuple{T,T},
        cost::CostFunction,
    ) where {T<:Real,op<:AbstractOperator}

        p = new{T}()
        p.operators = (operators[1].data, [op.data for op in operators[2]])
        p.drives = drives
        p.transform = transform
        p.tspan = tspan
        p.cost = cost
        p

    end

end

function CuQOCProblem(prob::QOCProblem{T}) where {T<:Real}
    cu_ops = (sparse(CuArray(prob.operators[1].data)), [sparse(CuArray(op.data)) for op in prob.operators[2]])
    cu_trans = CuTransform(prob.transform)
    QOCProblem(cu_ops, prob.drives, cu_trans, prob.tspan, prob.cost)
end
