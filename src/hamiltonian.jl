"""
    Hamiltonian(const_op, ops, drives)

Contains the information about the time-dependent Hamiltonian.
"""
mutable struct Hamiltonian{T<:Real}
    const_op::AbstractMatrix{Complex{T}}
    basis_l::Basis
    basis_r::Basis
    operators::Vector{AbstractMatrix{Complex{T}}}
    drives::Function
    function Hamiltonian(
        const_op::op,
        ops::Vector{op},
        drives::Function,
    ) where {op<:AbstractOperator}
        h = new{real(eltype(const_op.data))}()
        h.basis_l = const_op.basis_l
        h.basis_r = const_op.basis_r
        h.const_op = const_op.data
        h.operators = [op.data for op in ops]
        h.drives = drives
        h
    end
    function Hamiltonian(
        const_op::mat,
        basis_l::Basis,
        basis_r::Basis,
        operators::Vector{mat},
        drives::Function,
    ) where {T<:Real,mat<:AbstractMatrix{Complex{T}}}
        h = new{T}()
        h.basis_l = basis_l
        h.basis_r = basis_r
        h.const_op = const_op
        h.operators = operators
        h.drives = drives
        h
    end
end

"""
    cu(h)

Converts the Hamiltonian into a sparse form suitable for running on GPU.
"""
cu(h::Hamiltonian) = Hamiltonian(
    CuSparseMatrixCSC(h.const_op),
    h.basis_l,
    h.basis_r,
    [CuSparseMatrixCSC(op) for op in h.operators],
    h.drives,
)

Base.convert(::Type{Float32}, h::Hamiltonian) = Hamiltonian(
    SparseMatrixCSC{ComplexF32,Int64}(h.const_op),
    h.basis_l,
    h.basis_r,
    [SparseMatrixCSC{ComplexF32,Int64}(op) for op in h.operators],
    h.drives,
)
