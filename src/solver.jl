mutable struct Solution{T<:Real}
    params::Vector{T}
    trace::Vector{T}
end

function adjoint_system!(dpsi::AbstractMatrix{Complex{T}},
                         psi::AbstractMatrix{Complex{T}}, 
                         p::Tuple{Hamiltonian, Tuple{Function, Function}, Integer, Integer},
                         t::T) where T<:Real

    H, (_ct, _gt), n_dim, n_params = p
    n_ops = length(H.operators)
    mul!(dpsi, H.constOp, psi, -T(1)im, T(0)im)
    ct = _ct(t)
    @inbounds for i=1:n_ops
        mul!(dpsi, H.operators[i], psi, -T(1)im*T(ct[i]), T(1)+T(0)im)
    end
    gt = _gt(t)
    _psi = @view psi[:, 1:n_dim]
    @inbounds for i=1:n_params
        _dpsi = @view dpsi[:, n_dim*i+1:n_dim*(i+1)]
        @inbounds for j=1:n_ops
            mul!(_dpsi, H.operators[j], _psi, -T(1)im*T(gt[j, i]), T(1)+T(0)im)
        end
    end

end

function augmented_ode_problem(prob::QOCProblem{T}) where T<:Real
    n_coeffs = length(prob.hamiltonian.operators)
    n_params = length(prob.drive.params)
    n_dim = length(prob.transform.inputs)
    wf_size = length(prob.transform.inputs[1])
    psi = hcat([elm.data for elm in prob.transform.inputs]...)
    psi = hcat([psi, zeros(Complex{T}, wf_size, n_dim*n_params)]...)
    ODEProblem(adjoint_system!, psi, (prob.drive.t0, prob.drive.t1),
               (prob.hamiltonian, prob.drive, n_dim, n_params))

end

function evaluate_gradient(cost::CostFunction, result::AbstractMatrix{Complex{T}}, 
                           target::AbstractMatrix{Complex{T}}, n_dim::Integer,
                           n_params::Integer) where T<:Real

    gradients = []
    c = real(sum(diag(cost.distance(target, result[:, 1:n_dim]))))/n_dim
    @inbounds for i in 1:n_params
        res = @view result[:, n_dim*i+1:n_dim*(i+1)]
        push!(gradients, -real(sum(diag(cost.distance(target, res)))))
    end
    (gradients, c)

end

function optimize(prob::QOCProblem{T}, opt; alg=DP5(), n_iter::Integer=1, kwargs...) where T<:Real

    ode_prob = augmented_ode_problem(prob)
    sol = Solution(prob.drive.params, T[])
    drive = prob.drive
    n_params = length(drive.params)
    n_dim = length(prob.transform.inputs)
    target = hcat([k.data for k in prob.transform.outputs]...)
    constraint_gradient(θ) = gradient(ps->prob.cost.constraints(ps), θ)[1]
    p = Progress(n_iter)
    for i in 1:n_iter
        coeff = drive.coefficients(drive.params)
        grad = drive.gradient(drive.params)
        res = DifferentialEquations.solve(ode_prob, alg, p=(prob.hamiltonian, (coeff, grad), n_dim, n_params),
                    save_start=false, save_everystep=false; kwargs...)
        grads, c = evaluate_gradient(prob.cost, res.u[1], target, n_dim, n_params)
        push!(sol.trace, c)
        grads .+= constraint_gradient(drive.params)
        Flux.Optimise.update!(opt, drive.params, grads)
        next!(p,  showvalues = [(:cost, c)])
        GC.gc()
    end
    sol

end

