mutable struct Hamiltonian
    basis_l::Basis
    basis_r::Basis
    constOp::AbstractMatrix
    operators::Vector{AbstractMatrix}

    function Hamiltonian(op::AbstractOperator)
        new(op.basis_l, op.basis_r, op.data, AbstractOperator[])
    end

end

function Base.:+(H::Hamiltonian, op::AbstractOperator)
    (H.basis_r == op.basis_r && H.basis_l == op.basis_l) ? nothing :
    throw(IncompatibleBases)
    push!(H.operators, op.data)
    H
end

function CuOperator(op::AbstractOperator)
    cu_op = copy(op)
    cu_op.data = CuArray(op.data)
    cu_op
end

function CuHamiltonian(H::Hamiltonian)
    CuH = Hamiltonian(SparseOperator(H.basis_l, H.basis_r, H.constOp))
    CuH.constOp = sparse(CuArray(H.constOp))
    for i = 1:length(H.operators)
        push!(CuH.operators, sparse(CuArray(H.operators[i])))
    end
    CuH
end
