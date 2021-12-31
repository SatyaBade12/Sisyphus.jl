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
        applicable(drives, rand(real(eltype(const_op.data)), length(operators + 1))...) ?
        nothing : throw(ArgumentError)
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

CuHamiltonian(h::Hamiltonian) = Hamiltonian(
    sparse(CuArray(h.const_op)),
    h.basis_l,
    h.basis_r,
    [sparse(CuArray(op)) for op in h.operators],
    h.drives,
)
