mutable struct QOCProblem{T<:Real}
    hamiltonian::Hamiltonian
    drive::ParametricDrive{T}
    transform::Transform
    cost::CostFunction
end

function CuProblem(prob::QOCProblem{T}) where {T<:Real}
    cu_h = CuHamiltonian(prob.hamiltonian)
    cu_trans = CuTransform(prob.transform)
    QOCProblem(cu_h, prob.drive, cu_trans, prob.cost)
end