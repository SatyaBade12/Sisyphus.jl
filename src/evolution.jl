function schroedinger_dynamic(
    tspan,
    psi::Ket,
    h::Hamiltonian{T},
    params::Vector{T};
    kwargs...,
) where {T<:Real}
    n_coeffs = length(h.operators)
    const_op = Operator(h.basis_l, h.basis_r, h.const_op)
    ops = [Operator(h.basis_l, h.basis_r, op) for op in h.operators]

    func(t, psi) =
        let coeffs = h.drives(params, t)
            const_op + sum([coeffs[i] * ops[i] for i = 1:n_coeffs])
        end
    timeevolution.schroedinger_dynamic(tspan, psi, func; kwargs...)
end

function master_dynamic(
    tspan,
    psi,
    h::Hamiltonian,
    params::Vector{T},
    J::Vector{<:AbstractOperator},
    rates::Vector{T};
    kwargs...,
) where {T<:Real}

    n_coeffs = length(h.operators)
    const_op = Operator(h.basis_l, h.basis_r, h.const_op)
    ops = [Operator(h.basis_l, h.basis_r, op) for op in h.operators]
    Jdagger = [dagger(op) for op in J]

    func(t, psi) =
        let coeffs = h.drives(params, t)
            (const_op + sum([coeffs[i] * ops[i] for i = 1:n_coeffs]), J, Jdagger, rates)
        end

    timeevolution.master_dynamic(tspan, psi, func; kwargs...)
end
