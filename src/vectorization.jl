"""
    vectorize(H, J, rates)

Vectorizes the Hamiltonian and jump operators so that
they can be submitted to the Schroedinger time evolution solver.
Uses the vectorization identity ``\\text{vec}(ABC) = (C^T \\otimes A)\\text{vec}(B)``.
"""
function vectorize(
    H::Hamiltonian,
    J::Vector{<:AbstractOperator},
    rates::Vector{T},
) where {T<:Real}

    for op in J
        op.basis_r == H.basis_r ? nothing : throw(IncompatibleBases)
        op.basis_l == H.basis_l ? nothing : throw(IncompatibleBases)
    end
    const_op = vectorize_constant_op(H.const_op, [op.data for op in J], rates)
    ops = [vectorize_operator(op) for op in H.operators]
    Hamiltonian(const_op, J[1].basis_l, J[1].basis_r, ops, H.drives)

end

function vectorize_operator(op::AbstractMatrix)

    id = one(op)
    sparse(kron(id, op) - kron(transpose(op), id))

end

function vectorize_jump_operators(
    ops::Vector{<:AbstractMatrix{Complex{T}}},
    rates::Vector{T},
) where {T<:Real}

    id = one(ops[1])
    res = T(0) * kron(id, id)
    for (Op, r) in zip(ops, rates)
        res += r * kron(conj(Op), Op)
        res -= r * T(0.5) * kron(id, Op' * Op)
        res -= r * T(0.5) * kron(transpose(Op) * conj(Op), id)
    end
    T(1)im * sparse(res)

end

function vectorize_constant_op(
    H0::op,
    ops::Vector{op},
    rates::Vector{T},
) where {T<:Real,op<:AbstractMatrix{Complex{T}}}

    Op = vectorize_operator(H0)
    sparse(Op + vectorize_jump_operators(ops, rates))

end
