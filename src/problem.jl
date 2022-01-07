struct QOCProblem{T<:Real}
    hamiltonian::Hamiltonian{T}
    transform::Transform
    tspan::Tuple{T,T}
    cost::CostFunction
end

cu(prob::QOCProblem) =
    QOCProblem(cu(prob.hamiltonian), cu(prob.transform), prob.tspan, prob.cost)
