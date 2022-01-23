"""
    QOCProblem(hamiltonian::Hamiltonian{T},
               transform::Transform,
               tspan::Tuple{T,T}},
               cost::CostFunction) where {T<:Real}

Defines a quantum optimal control problem to be solved.
"""
struct QOCProblem{T<:Real}
    hamiltonian::Hamiltonian{T}
    transform::Transform
    tspan::Tuple{T,T}
    cost::CostFunction
end

"""
    cu(prob)

Turns a quantum optimal control problem into a form
suitable for running on GPU.
"""
cu(prob::QOCProblem) =
    QOCProblem(cu(prob.hamiltonian), cu(prob.transform), prob.tspan, prob.cost)



"""
    convert(::Type{Float32}, prob::QOCProblem)

Returns a QOCProblem with all data in single precision.
"""
Base.convert(::Type{Float32}, prob::QOCProblem) = QOCProblem(
    convert(Float32, prob.hamiltonian),
    convert(ComplexF32, prob.transform),
    (Float32(prob.tspan[1]), Float32(prob.tspan[2])),
    prob.cost,
)
