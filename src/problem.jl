struct QOCProblem{T<:Real}
    hamiltonian::Hamiltonian{T}
    transform::Transform
    tspan::Tuple{T,T}
    cost::CostFunction
end

CuQOCProblem(prob::QOCProblem) = QOCProblem(
    CuHamiltonian(prob.hamiltonian),
    CuTransform(prob.transform),
    prob.tspan,
    prob.cost,
)
