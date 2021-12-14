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

function input_data(t::UnitaryTransform, n_params::Integer)
    n_dim = length(t.inputs)
    wf_size = length(t.inputs[1])
    psi = hcat([elm.data for elm in t.inputs]...)
    psi = hcat([psi, zeros(eltype(t.inputs[1].data), wf_size, n_dim*n_params)]...)
    psi, n_dim
end

function input_data(t::StateTransform, n_params::Integer)
    n_dim = 1
    wf_size = length(t.input)
    psi = hcat([t.input.data]...)
    psi = hcat([psi, zeros(eltype(t.input.data), wf_size, n_params)]...)
    psi, n_dim
end

output_data(t::StateTransform) = hcat([t.output.data]...)
output_data(t::UnitaryTransform) = hcat([k.data for k in t.outputs]...)

function augmented_ode_problem(prob::QOCProblem{T}) where T<:Real
    n_coeffs = length(prob.hamiltonian.operators)
    n_params = length(prob.drive.params)
    psi, n_dim = input_data(prob.transform, n_params)
    coeffs = prob.drive.coefficients(prob.drive.params)
    grads = prob.drive.gradient(prob.drive.params)
    ODEProblem{true}(adjoint_system!, psi, (prob.drive.t0, prob.drive.t1),
               (prob.hamiltonian, (coeffs, grads), n_dim, n_params))

end

function evaluate_gradient(cost::CostFunction, result::AbstractMatrix{Complex{T}}, 
                           target::AbstractMatrix{Complex{T}}, n_dim::Integer,
                           n_params::Integer) where T<:Real

    gradients = []
    c = T(0)
    for (x, y) in zip(eachcol(target), eachcol(result[:, 1:n_dim]))
        c += real(cost.distance(x, y)/T(n_dim))
    end
    #c = sum(real(diag(cost.distance(target, result[:, 1:n_dim]))))
    @inbounds for i in 1:n_params
        res = @view result[:, n_dim*i+1:n_dim*(i+1)]
        _c = T(0)
        for (x, y) in zip(eachcol(target), eachcol(res))
            _c += -real(cost.distance(x, y))
        end
        #_c = -sum(real(diag(cost.distance(target, res))))
        push!(gradients, _c/T(n_dim))
    end
    (gradients, c)

end

dimension(t::UnitaryTransform) = length(t.inputs)
dimension(t::StateTransform) = 1

function optimize(prob::QOCProblem{T}, opt; alg=DP5(), n_iter::Integer=1, kwargs...) where T<:Real

    ode_prob = augmented_ode_problem(prob)
    n_params = length(prob.drive.params)
    target = output_data(prob.transform)
    n_dim = dimension(prob.transform)
    flux_optimize(prob, ode_prob, opt, alg, target, n_dim, n_iter, kwargs...)

end

function optimize(prob::QOCProblem{T}, opt::Opt; alg=DP5(), n_iter::Integer=1, kwargs...) where T<:Real
    
    ode_prob = augmented_ode_problem(prob)
    n_params = length(prob.drive.params)
    target = output_data(prob.transform)
    n_dim = dimension(prob.transform)
    nlopt_optimize(prob, ode_prob, opt, alg, target, n_dim, n_iter, kwargs...)

end

function nlopt_optimize(qoc_prob::QOCProblem{T}, ode_prob::ODEProblem, opt, alg,
                        target::AbstractMatrix{Complex{T}},
                        n_dim::Integer, n_iter::Integer, kwargs...) where T<: Real

    drive = qoc_prob.drive
    H = qoc_prob.hamiltonian
    cost = qoc_prob.cost
    n_params = length(drive.params)
    opt.maxeval = n_iter
    constraint_gradient(θ) = gradient(ps->cost.constraints(ps), θ)[1]
    sol = Solution(drive.params, T[])

    function opt_function(x::Vector{T}, g::Vector{T})
        drive.params[:] = x
        coeff = drive.coefficients(x)
        grad = drive.gradient(x)
        res = DifferentialEquations.solve(ode_prob, alg, p=(H, (coeff, grad), n_dim, n_params),
                    save_start=false, save_everystep=false; kwargs...)
        grads, c = evaluate_gradient(cost, res.u[1], target, n_dim, n_params)
        grads .+= constraint_gradient(x)
        if length(g)>0
            for i in 1:n_params
                g[i] = grads[i]
            end
        end
        push!(sol.trace, c)
        println(c)
        c + cost.constraints(x)
    end
    opt.min_objective = opt_function
    (minf,minx,ret) = NLopt.optimize(opt, drive.params)
    
    sol
end


function flux_optimize(qoc_prob::QOCProblem{T}, ode_prob::ODEProblem, opt, alg, target::AbstractMatrix{Complex{T}},
                       n_dim::Integer, n_iter::Integer, kwargs...) where T<: Real
    
    drive = qoc_prob.drive
    H = qoc_prob.hamiltonian
    cost = qoc_prob.cost
    constraint_gradient(θ) = gradient(ps->cost.constraints(ps), θ)[1]
    sol = Solution(drive.params, T[])
    p = Progress(n_iter)
    n_params = length(drive.params)
    for i in 1:n_iter
        coeff = drive.coefficients(drive.params)
        grad = drive.gradient(drive.params)
        res = DifferentialEquations.solve(ode_prob, alg, p=(H, (coeff, grad), n_dim, n_params),
                    save_start=false, save_everystep=false; kwargs...)
        grads, c = evaluate_gradient(cost, res.u[1], target, n_dim, n_params)
        push!(sol.trace, c)
        grads .+= constraint_gradient(drive.params)
        Flux.Optimise.update!(opt, drive.params, grads)
        next!(p, showvalues = [(:cost, c)])
        GC.gc()
    end
    sol

end

function optimize2(prob::QOCProblem{T}, opt; alg=DP5(), n_iter::Integer=1, kwargs...) where T<:Real

    ode_prob = augmented_ode_problem(prob)
    sol = Solution(prob.drive.params, T[])
    drive = prob.drive
    n_params = length(drive.params)
    target = output_data(prob.transform)
    n_dim = dimension(prob.transform)
    constraint_gradient(θ) = gradient(ps->prob.cost.constraints(ps), θ)[1]
    p = Progress(n_iter)
    coeff = drive.coefficients(drive.params)
    grad = drive.gradient(drive.params)
    integrator = DiffEqBase.__init(ode_prob, alg, p=(prob.hamiltonian, (coeff, grad), n_dim, n_params),
     save_start=false, save_everystep=false; kwargs...)
    for i in 1:n_iter
        coeff = drive.coefficients(drive.params)
        grad = drive.gradient(drive.params)
        integrator.p = (prob.hamiltonian, (coeff, grad), n_dim, n_params)
        DiffEqBase.solve!(integrator)
        grads, c = evaluate_gradient(prob.cost, integrator.sol.u[1], target, n_dim, n_params)
        push!(sol.trace, c)
        grads .+= constraint_gradient(drive.params)
        Flux.Optimise.update!(opt, drive.params, grads)
        next!(p, showvalues = [(:cost, c)])
        GC.gc()
        DiffEqBase.reinit!(integrator)
    end
    sol

end

