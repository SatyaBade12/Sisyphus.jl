struct QOCProblem{T<:Real}
    hamiltonian::Hamiltonian{T}
    transform::Transform
    tspan::Tuple{T,T}
    cost::CostFunction
end

cu(prob::QOCProblem) =
    QOCProblem(cu(prob.hamiltonian), cu(prob.transform), prob.tspan, prob.cost)

Base.convert(::Type{Float32}, prob::QOCProblem) = QOCProblem(
    convert(Float32, prob.hamiltonian),
    convert(ComplexF32, prob.transform),
    (Float32(prob.tspan[1]), Float32(prob.tspan[2])),
    prob.cost,
)
